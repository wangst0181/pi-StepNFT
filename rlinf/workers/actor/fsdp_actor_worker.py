# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import os
from contextlib import nullcontext
from functools import partial

import numpy as np
import torch
from omegaconf import DictConfig, open_dict
from torch import nn
from torch.distributed.tensor import DTensor
from torch.multiprocessing.reductions import reduce_tensor

import rlinf.algorithms  # noqa: F401
from rlinf.algorithms.registry import calculate_adv_and_returns, policy_loss
from rlinf.algorithms.utils import (
    kl_penalty,
)
from rlinf.config import SupportedModel
from rlinf.data.io_struct import BatchResizingIterator, RolloutResult
from rlinf.hybrid_engines.fsdp import FSDP
from rlinf.hybrid_engines.fsdp.fsdp_model_manager import (
    FSDPModelManager,
)
from rlinf.models import get_model
from rlinf.scheduler import Channel, Cluster, CollectiveGroupOptions, Worker
from rlinf.utils.data_iter_utils import get_iterator_k_split
from rlinf.utils.distributed import all_reduce_dict, masked_normalization
from rlinf.utils.distributed import (
    compute_rollout_metrics as compute_math_rollout_metrics,
)
from rlinf.utils.metric_utils import (
    append_to_dict,
    compute_loss_mask,
    compute_rollout_metrics,
    compute_split_num,
    compute_time_decay_weights,
)
from rlinf.utils.nested_dict_process import (
    cat_list_of_dict_tensor,
    put_tensor_device,
    split_dict_to_chunk,
)
from rlinf.utils.placement import (
    HybridComponentPlacement,
    ModelParallelComponentPlacement,
)
from rlinf.utils.utils import (
    clear_memory,
    compute_entropy_from_logits,
    compute_logprobs_from_logits,
    cpu_weight_swap,
    get_loss_agg_func,
    masked_mean,
    reshape_entropy,
    retrieve_model_state_dict_in_cpu,
)
from rlinf.workers.rollout.utils import RankMapper

_ADV_INPUT_LOG_STATE = {"count": 0}


def _log_terminal_binary_adv_inputs(
    rank: int,
    rewards: torch.Tensor | None,
    dones: torch.Tensor | None,
    success_once: torch.Tensor | None,
    loss_mask: torch.Tensor | None,
) -> None:
    if _ADV_INPUT_LOG_STATE["count"] >= 3:
        return
    _ADV_INPUT_LOG_STATE["count"] += 1
    with torch.no_grad():
        def _stats(t: torch.Tensor | None):
            if t is None:
                return ("<none>", 0, 0.0, 0.0)
            nz = int((t != 0).sum().item())
            t_min = float(t.min().item()) if t.numel() > 0 else 0.0
            t_max = float(t.max().item()) if t.numel() > 0 else 0.0
            return (tuple(t.shape), nz, t_min, t_max)

        r_shape, r_nz, r_min, r_max = _stats(rewards)
        d_shape, d_nz, d_min, d_max = _stats(dones)
        s_shape, s_nz, s_min, s_max = _stats(success_once)
        m_shape, m_nz, m_min, m_max = _stats(loss_mask)
        print(
            "[adv][input] "
            f"rank={rank} rewards_shape={r_shape} rewards_nz={r_nz} "
            f"rewards_min={r_min:.3f} rewards_max={r_max:.3f} "
            f"dones_shape={d_shape} dones_nz={d_nz} dones_min={d_min:.3f} dones_max={d_max:.3f} "
            f"success_once_shape={s_shape} success_once_nz={s_nz} "
            f"success_once_min={s_min:.3f} success_once_max={s_max:.3f} "
            f"loss_mask_shape={m_shape} loss_mask_nz={m_nz} "
            f"loss_mask_min={m_min:.3f} loss_mask_max={m_max:.3f}",
            flush=True,
        )

_TERMINAL_BINARY_LOSS_LOG = {"count": 0}


def _log_terminal_binary_loss_inputs(
    rank: int,
    advantages: torch.Tensor,
    returns: torch.Tensor | None,
    loss_mask: torch.Tensor | None,
    adv_clip_max: float | None,
) -> None:
    if _TERMINAL_BINARY_LOSS_LOG["count"] >= 3:
        return
    _TERMINAL_BINARY_LOSS_LOG["count"] += 1
    with torch.no_grad():
        mask = loss_mask if loss_mask is not None else torch.ones_like(advantages)
        mask = mask.to(dtype=torch.bool)
        if mask.shape != advantages.shape:
            mask = mask.expand_as(advantages)
        masked_adv = advantages[mask]
        adv_min = float(masked_adv.min().item()) if masked_adv.numel() > 0 else 0.0
        adv_max = float(masked_adv.max().item()) if masked_adv.numel() > 0 else 0.0
        adv_mean = float(masked_adv.mean().item()) if masked_adv.numel() > 0 else 0.0
        if returns is not None:
            masked_ret = returns[mask]
            ret_min = float(masked_ret.min().item()) if masked_ret.numel() > 0 else 0.0
            ret_max = float(masked_ret.max().item()) if masked_ret.numel() > 0 else 0.0
            ret_unique = (
                torch.unique(masked_ret).detach().cpu().tolist()
                if masked_ret.numel() > 0
                else []
            )
        else:
            ret_min = ret_max = 0.0
            ret_unique = []
        print(
            "[loss][terminal-binary] "
            f"adv_clip_max={adv_clip_max} adv_min={adv_min:.3f} "
            f"adv_max={adv_max:.3f} adv_mean={adv_mean:.3f} "
            f"ret_min={ret_min:.3f} ret_max={ret_max:.3f} ret_unique={ret_unique} "
            f"src={__file__}",
            flush=True,
        )


def nft_return_decay(
    step: int, total_steps: int, base: float = 0.1, target: float = 0.8
) -> float:
    if total_steps <= 0:
        return 1.0

    progress = min(max(step / total_steps, 0.0), 1.0)
    cosine_val = math.cos(progress * math.pi)
    decay = target - (target - base) * 0.5 * (1 + cosine_val)
    return float(decay)


def process_nested_dict_for_adv(nested_dict, rollout_epoch):
    """
    original shape: [rollout_epoch x n_chunk_steps, bsz, num_action_chunks, ...]
    target shape: [n_chunk_steps, rollout_epoch x bsz, num_action_chunks, ...]
    """
    ret_dict = {}
    for key, value in nested_dict.items():
        if isinstance(value, torch.Tensor):
            new_value = value.reshape(
                rollout_epoch, -1, *value.shape[1:]
            )  # [rollout_epoch, n_chunk_step, bsz, ...]
            new_value = new_value.transpose(
                0, 1
            )  # [n_chunk_step, rollout_epoch, bsz, ...]
            new_value = new_value.reshape(new_value.shape[0], -1, *new_value.shape[3:])
            ret_dict[key] = new_value
        elif isinstance(value, dict):
            ret_dict[key] = process_nested_dict_for_adv(value, rollout_epoch)
    return ret_dict


def process_nested_dict_for_train(nested_dict, shuffle_id):
    ret_dict = {}
    for key, value in nested_dict.items():
        if key in ["dones", "terminations", "truncations", "prev_values"]:
            value = value[:-1]
        if "env_info" in key:
            raise NotImplementedError
        if value is None:
            ret_dict[key] = None
        if isinstance(value, torch.Tensor):
            ret_dict[key] = value.reshape(-1, *value.shape[2:])[shuffle_id]
        elif isinstance(value, dict):
            ret_dict[key] = process_nested_dict_for_train(value, shuffle_id)
    return ret_dict


def get_nested_k_split_for_specific_keys(nested_dict, num_splits, key_list):
    """
    Get k-split iterator for some keys in nested_dict.
    """
    extra_dict = {}
    for key in key_list:
        if key not in nested_dict.keys():
            continue
        value = nested_dict[key]
        if isinstance(value, dict):
            extra_dict[key] = split_dict_to_chunk(value, num_splits)
        elif isinstance(value, torch.Tensor):
            continue
        else:
            raise NotImplementedError(
                f"Only support dict and tensor type, but got {type(value)}"
            )
    # {key1: [d1, d2, ...], key2: [d1, d2, ...]} -> [{key1: d1, key2: d1}, {key1: d2, key2: d2}, ...]
    extra_list = [
        {k: extra_dict[k][i] for k in extra_dict.keys()} for i in range(num_splits)
    ]
    return extra_list


class FSDPActor(FSDPModelManager, Worker):
    def __init__(
        self, cfg: DictConfig, placement: ModelParallelComponentPlacement
    ) -> None:
        """
        FSDPActor worker used to train the model with data from rollout workers.

        Args:
            cfg (DictConfig): The global yaml configuration.
            placement (ModelParallelComponentPlacement): The accelerator placement for actor worker.
        """
        Worker.__init__(self)
        super().__init__(cfg.actor, self._world_size, self._rank)

        self.cfg = cfg

        self.response_len = (
            self.cfg.actor.model.encoder_seq_length - self.cfg.data.max_prompt_length
        )
        self.calculate_entropy = self.cfg.algorithm.calculate_entropy
        self.calculate_entropy_loss = (
            self.cfg.algorithm.entropy_bonus > 0 and self.calculate_entropy
        )
        self.kl_beta = self.cfg.algorithm.kl_beta
        self.kl_penalty_type = self.cfg.algorithm.kl_penalty_type

        self.total_batch_size_per_dp = (
            self.cfg.data.rollout_batch_size
            * self.cfg.algorithm.group_size
            // self._world_size
        )

        self._rollout_group_name = cfg.rollout.group_name
        self._component_placement = placement
        self.is_pipeline = self._component_placement.is_disaggregated
        self.ref_policy_state_dict = None
        if self.is_pipeline:
            self._inference_group_name = cfg.inference.group_name
            self._inference_world_size = self._component_placement.get_world_size(
                "inference"
            )
            self._inference_dst_map: dict[int, list[str]] = {}
        else:
            self._inference_group_name = None
            self._inference_world_size = 0
            self._inference_dst_map = None
        self.loss_agg_func = get_loss_agg_func(self.cfg.algorithm.loss_agg_func)
        self.enable_offload = (
            self.cfg.actor.get("enable_offload", False) and not self.is_pipeline
        )
        self.micro_batch_size = self.cfg.actor.micro_batch_size
        self.n_mini_batches = self.cfg.algorithm.n_minibatches
        self.task_type = self.cfg.runner.task_type
        self.entropy_op_type = self.cfg.algorithm.get("entropy_op_type", "liger_kernel")

    def init_worker(self) -> None:
        """
        Initialize the actor worker. build the model and use corresponding training backend
        (FSDP/FSDP2) to wrap it. If needed, offload model parameters and optimizer states to CPU.
        If kl_beta > 0, retrieve the reference policy model state dict to CPU.
        If mode is disaggregated, setup which inference ranks it needs to sync weights to by
        doing a handshake with inference workers.
        """
        self.setup_model_and_optimizer()
        if self.cfg.algorithm.kl_beta > 0 and self.cfg.actor.get(
            "combine_reference_model", True
        ):
            self.ref_policy_state_dict = retrieve_model_state_dict_in_cpu(self.model)

        if self.enable_offload and not self.is_pipeline:
            self.offload_param_and_grad()
            self.offload_optimizer()
        self._setup_rollout_weight_dst_ranks()

    def _setup_rollout_weight_dst_ranks(self) -> None:
        """Setup destination ranks for token and weight communication."""
        rank_map = RankMapper.get_actor_rank_to_rollout_rank_map(
            self._component_placement
        )
        self._weight_dst_rank_in_rollout = rank_map[self._rank]
        self.log_info(
            f"Actor rank {self._rank} will send weights to {self._weight_dst_rank_in_rollout}"
        )

    def del_reshard_state_dict(self) -> None:
        """Just for interface compatibility with MegatronActor."""
        if hasattr(self, "rollout_state_dict"):
            del self.rollout_state_dict
        clear_memory(sync=False)

    def sync_model_to_inference(self) -> None:
        """
        Sync the model's full state dict to the inference worker.
        The model state_dict is the reference of actor's model
        parameters(by setting cpu_offload=False).
        """
        if not hasattr(self._strategy, "setup_actor_sync_inference_ranks"):
            if self._rank == 0:
                self.log_info("[sync] inference sync disabled; skipping.")
            return
        if not self._inference_dst_map:
            self._strategy.setup_actor_sync_inference_ranks(self)

        if self.is_optimizer_offloaded:
            self.offload_optimizer()

        if self.is_weight_offloaded:
            self.load_param_and_grad(self.device, False)

        inference_state_dict = self.get_model_state_dict(
            cpu_offload=False, full_state_dict=False
        )
        # NOTE: we have already know which inference rank needs which params
        # by calling _strategy.setup_actor_sync_inference_ranks() to do handshake
        # with each inference rank. just send them accordingly.
        for rank, needed_params in self._inference_dst_map.items():
            sended_params = {}
            for name in needed_params:
                if name in inference_state_dict:
                    # mentioned again, no ShardedTensor here.
                    sended_params[name] = (
                        inference_state_dict[name].to_local()
                        if isinstance(inference_state_dict[name], DTensor)
                        else inference_state_dict[name]
                    )
            self.send(
                object=sended_params,
                dst_group_name=self._inference_group_name,
                dst_rank=rank,
                async_op=True,
            )

        if self.enable_offload and not self.is_weight_offloaded:
            self.offload_param_and_grad()

        torch.distributed.barrier()

    def sync_model_to_rollout(self) -> None:
        """
        Sync the model's full state dict to the rollout worker.
        """
        if self.enable_offload and not self.is_optimizer_offloaded:
            self.offload_optimizer()

        if self.enable_offload and self.is_weight_offloaded:
            self.load_param_and_grad(self.device, True)

        self.rollout_state_dict = self.get_model_state_dict(
            cpu_offload=False, full_state_dict=True
        )

        has_visual = any("visual." in k for k in self.rollout_state_dict.keys())

        state_dict = {}

        if self._weight_dst_rank_in_rollout is not None:
            for k, v in self.rollout_state_dict.items():
                name = k
                if has_visual:
                    if name.startswith("model.language_model."):
                        name = "model." + name[21:]
                    # NOTE:
                    # if transformers version is 4.56.1 or older(not tested),
                    # the following line should be uncommented

                    # elif name.startswith("model."):
                    #     name = name[6:]
                state_dict[name] = reduce_tensor(v) if not self.is_pipeline else v
            if not self.is_pipeline:
                self.send(
                    state_dict,
                    self._rollout_group_name,
                    self._weight_dst_rank_in_rollout,
                )
            else:
                for weight_dst_rank in self._weight_dst_rank_in_rollout:
                    self.send(
                        state_dict,
                        self._rollout_group_name,
                        weight_dst_rank,
                    )

        state_dict.clear()
        if self.enable_offload and not self.is_weight_offloaded:
            self.offload_param_and_grad()

    def get_batch(
        self, channel: Channel
    ) -> tuple[dict[str, torch.Tensor], RolloutResult]:
        result: RolloutResult = channel.get()

        batch = result.to_actor_batch(
            self.cfg.data.max_prompt_length,
            self.cfg.actor.model.encoder_seq_length,
            self.tokenizer.eos_token_id,
        )
        return batch, result

    def _load_weight_and_optimizer(self) -> None:
        # Acquire the GPUs to ensure that no one is using them before loading models
        # Otherwise, it may lead to OOM
        with self.device_lock:
            if not self.enable_offload:
                return
            if self.is_weight_offloaded:
                self.load_param_and_grad(self.device)
            if self.is_optimizer_offloaded:
                self.load_optimizer(self.device)

    @torch.no_grad()
    def inference_step(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        self.model.eval()
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        position_ids = batch["position_ids"]

        multi_modal_inputs = {}
        if "multi_modal_inputs" in batch.keys():
            for key in batch["multi_modal_inputs"][0].keys():
                multi_modal_inputs[key] = torch.cat(
                    [inputs[key] for inputs in batch["multi_modal_inputs"]],
                    dim=0,
                ).cuda()

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            **multi_modal_inputs,
        )

        logits = outputs.logits
        logits = logits[:, -self.response_len - 1 : -1, :]
        logits = logits / self.cfg.algorithm.sampling_params.temperature

        responses = input_ids[:, -self.response_len :]
        logprobs = compute_logprobs_from_logits(
            logits=logits, target=responses, op_type=self.entropy_op_type
        )
        return logprobs

    def run_inference(
        self,
        input_channel: Channel,
        output_channel: Channel,
        compute_ref_logprobs: bool,
    ) -> None:
        """
        Compute prev/ref logprobs using the actor Model's forward.

        Args:
            input_channel: The input channel to read from.
            output_channel: The output channel to send results to.
            compute_ref_logprobs: Whether to compute reference logprobs.
        """
        recv_batch_size = 0
        while recv_batch_size < self.total_batch_size_per_dp:
            batch, rollout_result = self.get_batch(input_channel)
            recv_batch_size += rollout_result.num_sequence
            self._load_weight_and_optimizer()

            num_splits = (
                rollout_result.num_sequence
                // self.cfg.algorithm.logprob_forward_micro_batch_size
            )
            micro_batches_iter = get_iterator_k_split(
                batch,
                num_splits=num_splits,
            )
            micro_batches = list(micro_batches_iter)

            prev_logprobs = []
            with self.worker_timer():
                for micro_batch in micro_batches:
                    prev_logprobs.append(self.inference_step(micro_batch).cpu())

                if rollout_result.rollout_logprobs is not None:
                    # Rollout has returned logprobs, store the recomputed logprobs in recompute_prev_logprobs
                    rollout_result.recompute_prev_logprobs = torch.cat(prev_logprobs)
                else:
                    # Otherwise, directly store the logprobs in prev_logprobs (the final logprobs used for training)
                    rollout_result.prev_logprobs = torch.cat(prev_logprobs)

            if compute_ref_logprobs:
                assert self.ref_policy_state_dict is not None, (
                    "Reference policy state dict is None but compute_ref_logprobs is True"
                )
                ref_logprobs = []
                with cpu_weight_swap(self.model, self.ref_policy_state_dict):
                    for micro_batch in micro_batches:
                        ref_logprobs.append(self.inference_step(micro_batch).cpu())
                    rollout_result.ref_logprobs = torch.cat(ref_logprobs)

            output_channel.put(rollout_result)

        assert recv_batch_size == self.total_batch_size_per_dp, (
            f"Expected {self.total_batch_size_per_dp} sequences from channel, but got {recv_batch_size}"
        )

    def training_step(
        self, batch: dict[str, torch.Tensor] | BatchResizingIterator
    ) -> tuple[dict[str, torch.Tensor], float, list[float]]:
        if isinstance(batch, dict):
            global_batch_size = batch["input_ids"].shape[0]
            assert global_batch_size % self.micro_batch_size == 0, (
                f"global batch size {global_batch_size} can not divide micro_batch_size {self.micro_batch_size}"
            )
            micro_batch_cnt = global_batch_size // self.micro_batch_size
            self.gradient_accumulation = micro_batch_cnt
            micro_batches = get_iterator_k_split(batch, micro_batch_cnt)
            micro_batches_iter = iter(micro_batches)
        else:
            global_batch_size = self.total_batch_size_per_dp // self.n_mini_batches
            micro_batch_cnt = global_batch_size // self.micro_batch_size
            self.gradient_accumulation = micro_batch_cnt

            def iterator_wrapper():
                for _ in range(micro_batch_cnt):
                    yield next(batch)

            micro_batches_iter = iterator_wrapper()
        self.optimizer.zero_grad()
        mbs_metrics_list = {}
        for idx, m_batch in enumerate(micro_batches_iter):
            backward_ctx = self.before_micro_batch(
                self.model,
                is_last_micro_batch=(idx + 1) == self.gradient_accumulation,
            )
            for k, v in m_batch.items():
                m_batch[k] = v.cuda() if isinstance(v, torch.Tensor) else v

            multi_modal_inputs = {}
            if "multi_modal_inputs" in m_batch.keys():
                for key in m_batch["multi_modal_inputs"][0].keys():
                    multi_modal_inputs[key] = torch.cat(
                        [inputs[key] for inputs in m_batch["multi_modal_inputs"]],
                        dim=0,
                    ).cuda()

            input_ids = m_batch["input_ids"]
            attention_mask = m_batch["attention_mask"]
            position_ids = m_batch["position_ids"]
            prev_logprobs = m_batch["prev_logprobs"]
            advantages = m_batch["advantages"]
            ref_logprobs = None
            if "ref_logprobs" in m_batch:
                ref_logprobs = m_batch["ref_logprobs"]

            loss_mask = m_batch["response_mask"][:, -self.response_len :]

            clip_ratio = self.cfg.algorithm.ratio_clip_eps
            clip_ratio_low = self.cfg.algorithm.get("clip_ratio_low", None)
            clip_ratio_high = self.cfg.algorithm.get("clip_ratio_high", None)
            clip_ratio_low = (
                clip_ratio_low if clip_ratio_low is not None else clip_ratio
            )
            clip_ratio_high = (
                clip_ratio_high if clip_ratio_high is not None else clip_ratio
            )
            clip_ratio_c = self.cfg.algorithm.get("clip_ratio_c", 3.0)

            with self.amp_context:
                output = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                )

                logits: torch.Tensor = output.logits

                logits.div_(self.cfg.algorithm.sampling_params.temperature)

                responses = input_ids[:, -self.response_len :]
                logits = logits[
                    :, -self.response_len - 1 : -1, :
                ]  # (bsz, response_length, vocab_size)
                logprobs = compute_logprobs_from_logits(
                    logits, responses, self.entropy_op_type
                )

                if self.cfg.algorithm.get("importance_sampling_fix", False):
                    rollout_prev_logprobs = prev_logprobs
                    recompute_prev_logprobs = m_batch["recompute_prev_logprobs"]
                    advantages = advantages * torch.clamp(
                        (recompute_prev_logprobs - rollout_prev_logprobs).exp(),
                        min=self.cfg.algorithm.importance_sampling_clip,
                    )

                if self.cfg.algorithm.adv_type == "terminal-binary":
                    _log_terminal_binary_loss_inputs(
                        rank=self._rank,
                        advantages=advantages,
                        returns=m_batch.get("returns", None),
                        loss_mask=loss_mask,
                        adv_clip_max=self.cfg.algorithm.get(
                            "clip_ratio_high", clip_ratio_high
                        ),
                    )

                loss, mbs_metrics_data = policy_loss(
                    loss_type=self.cfg.algorithm.loss_type,
                    loss_agg_func=self.loss_agg_func,
                    logprobs=logprobs,
                    old_logprobs=prev_logprobs,
                    advantages=advantages,
                    clip_ratio_low=clip_ratio_low,
                    clip_ratio_high=clip_ratio_high,
                    clip_ratio_c=clip_ratio_c,
                    loss_mask=loss_mask,
                    task_type=self.task_type,
                )

                entropy_loss = torch.tensor(0.0, device=torch.cuda.current_device())
                if self.calculate_entropy:
                    entropy = compute_entropy_from_logits(
                        logits,
                    )

                    entropy_loss = self.loss_agg_func(entropy, mask=loss_mask)
                    if self.calculate_entropy_loss:
                        loss = loss - self.cfg.algorithm.entropy_bonus * entropy_loss

                kl_loss = torch.tensor(0.0, device=torch.cuda.current_device())
                if self.kl_beta > 0 and ref_logprobs is not None:
                    kld = kl_penalty(ref_logprobs, logprobs, self.kl_penalty_type)
                    kl_loss = self.loss_agg_func(kld, loss_mask)
                    loss = loss + kl_loss * self.kl_beta

                # add to log
                # scale loss for gradient accumulation and backprop
                loss = loss / self.gradient_accumulation
                with backward_ctx:
                    self.grad_scaler.scale(loss).backward()

            mbs_metrics_data.update(
                {
                    "actor/final_loss": loss.detach(),
                    "actor/entropy_loss": entropy_loss.detach(),
                    "actor/kl_loss": kl_loss.detach(),
                }
            )

            append_to_dict(mbs_metrics_list, mbs_metrics_data)

        grad_norm, lr_list = self.optimizer_step()
        return mbs_metrics_list, grad_norm, lr_list

    def run_training_pipeline(self, input_channel: Channel) -> tuple[dict, list]:
        self.model.train()
        train_batch_iterator = BatchResizingIterator(
            cfg=self.cfg,
            get_batch_fn=partial(self.get_batch, input_channel),
            micro_batch_size=self.micro_batch_size,
            total_batch_size=self.total_batch_size_per_dp,
            num_global_batches=self.n_mini_batches,
            forward_only=False,
        )
        train_batch_iterator.register_get_batch_handler(
            self.compute_advantages_and_returns
        )

        if self.cfg.algorithm.normalize_advantages:

            def normalize_advantages(batch: dict[str, torch.Tensor]):
                mask = batch["response_mask"][:, -self.response_len :]
                batch["advantages"] = masked_normalization(batch["advantages"], mask)
                return batch

            train_batch_iterator.register_global_batch_handler(normalize_advantages)

        self._load_weight_and_optimizer()
        training_metrics_list = []
        with self.worker_timer():
            for _ in range(self.n_mini_batches):
                metrics, grad_norm, lr_list = self.training_step(
                    batch=train_batch_iterator
                )

                # aggregate metrics across micro-batches
                mean_metric_dict = {
                    key: torch.mean(torch.stack(value))
                    for key, value in metrics.items()
                }
                mean_metric_dict = all_reduce_dict(
                    mean_metric_dict, op=torch.distributed.ReduceOp.AVG
                )

                mean_metric_dict["actor/grad_norm"] = float(grad_norm)
                mean_metric_dict["actor/lr"] = lr_list[0]
                training_metrics_list.append(mean_metric_dict)

        # put lr scheduler step here
        self.lr_scheduler.step()

        # Rollout metrics
        batch = train_batch_iterator.get_all_batches()
        rollout_metrics, _, _ = compute_math_rollout_metrics(
            batch, self.cfg.data.max_prompt_length, self.response_len
        )

        return rollout_metrics, training_metrics_list

    def run_training(self, input_channel: Channel) -> tuple[dict, list]:
        # Get all batches for this DP
        if self.is_pipeline:
            with self.worker_timer():
                return self.run_training_pipeline(input_channel)

        batches = []
        recv_batch_size = 0
        while recv_batch_size < self.total_batch_size_per_dp:
            batch, rollout_result = self.get_batch(input_channel)
            batches.append(batch)
            recv_batch_size += rollout_result.num_sequence
        assert recv_batch_size == self.total_batch_size_per_dp, (
            f"Expected {self.total_batch_size_per_dp} sequences from channel, but got {recv_batch_size}"
        )
        global_batch = RolloutResult.merge_batches(batches)

        # Compute advantages and returns
        global_batch = self.compute_advantages_and_returns(global_batch)

        if self.cfg.algorithm.normalize_advantages:
            mask = global_batch["response_mask"][:, -self.response_len :]
            global_batch["advantages"] = masked_normalization(
                global_batch["advantages"], mask
            )

        # Must be called after batch is retrieved, which is when rollout has stopped
        # Otherwise, loading model might cause OOM
        self._load_weight_and_optimizer()

        mini_batches = get_iterator_k_split(
            global_batch,
            num_splits=self.cfg.algorithm.n_minibatches,
            shuffle=self.cfg.algorithm.get("shuffle_rollout", True),
            shuffle_seed=self.cfg.actor.seed,
        )

        self.model.train()
        assert (
            self.cfg.actor.global_batch_size
            % (self.cfg.actor.micro_batch_size * self._world_size)
            == 0
        )

        training_metrics_list = []
        # Global batch iterations
        with self.worker_timer():
            for mini_batch in mini_batches:
                metrics, grad_norm, lr_list = self.training_step(batch=mini_batch)

                # aggregate metrics across micro-batches
                mean_metric_dict = {
                    key: torch.mean(torch.stack(value))
                    for key, value in metrics.items()
                }
                mean_metric_dict = all_reduce_dict(
                    mean_metric_dict, op=torch.distributed.ReduceOp.AVG
                )

                mean_metric_dict["actor/grad_norm"] = float(grad_norm)
                mean_metric_dict["actor/lr"] = lr_list[0]
                training_metrics_list.append(mean_metric_dict)

        # put lr scheduler step here
        self.lr_scheduler.step()

        # Rollout metrics
        rollout_metrics, _, _ = compute_math_rollout_metrics(
            global_batch, self.cfg.data.max_prompt_length, self.response_len
        )

        return rollout_metrics, training_metrics_list

    # Advantages and returns
    def compute_advantages_and_returns(self, batch: dict[str, torch.Tensor]):
        """Compute the advantages and returns.

        Args:
            batch (Dict[str, torch.Tensor]): The rollout batch.
        """
        with self.worker_timer():
            if batch.get("advantages", None) is None:
                mask = batch["response_mask"][:, -self.response_len :]
                advantages, _ = calculate_adv_and_returns(
                    task_type=self.task_type,
                    adv_type=self.cfg.algorithm.adv_type,
                    rewards=batch["rewards"].cuda(),
                    loss_mask=mask.cuda(),
                    group_size=self.cfg.algorithm.group_size,
                    kl_beta=self.cfg.algorithm.get("reinpp_kl_beta", 0.0),
                    kl_penalty_type=self.kl_penalty_type,
                    logprob=batch["prev_logprobs"].cuda()
                    if "prev_logprobs" in batch
                    else None,
                    ref_logprob=batch["ref_logprobs"].cuda()
                    if "ref_logprobs" in batch
                    else None,
                    use_reinpp_baseline=self.cfg.algorithm.get(
                        "use_reinpp_baseline", False
                    ),
                )
                batch["advantages"] = advantages

        return batch


class EmbodiedFSDPActor(FSDPModelManager, Worker):
    def __init__(self, cfg: DictConfig):
        import warnings

        warnings.filterwarnings(
            "ignore",
            message=".*When using ``NO_SHARD`` for ``ShardingStrategy``.*",
        )
        Worker.__init__(self)
        super().__init__(cfg.actor, self._world_size, self._rank)
        self.cfg = cfg
        self.global_step = 0
        self._env_group_name = cfg.env.group_name
        self._rollout_group_name = cfg.rollout.group_name
        self._component_placement = HybridComponentPlacement(cfg, Cluster())

        # stage_num: default to 2, use for pipeline rollout process
        self.stage_num = cfg.rollout.pipeline_stage_num

        self.enable_offload = self.cfg.actor.get("enable_offload", False)
        self.entropy_op_type = self.cfg.algorithm.get("entropy_op_type", "torch")

        self.ref_model = None
        self._value_head_sync_ready = False
        self._shared_ref_param_names: set[str] = set()
        self._enable_mem_log = bool(getattr(self.cfg.actor, "enable_mem_log", False))
        self._update_ready = False
        self._student_param_snapshot = None
        self._student_param_snapshot_init = None
        self._watch_param_names = [
            "paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.0.layer_norm1.weight",
        ]
        self._logged_terminal_binary_loss = False

        # Sync weight comm options
        max_ctas = cfg.rollout.get("sync_weight_nccl_max_ctas", None)
        min_ctas = cfg.rollout.get("sync_weight_nccl_min_ctas", None)
        self._sync_weight_comm_options = CollectiveGroupOptions(
            accel_max_ctas=max_ctas, accel_min_ctas=min_ctas
        )

    def _log_cuda_memory(self, tag: str) -> None:
        if not self._enable_mem_log or not torch.cuda.is_available():
            return
        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        max_alloc = torch.cuda.max_memory_allocated() / (1024**3)
        print(
            f"[Mem] {tag} allocated_gb={allocated:.2f} reserved_gb={reserved:.2f} "
            f"max_allocated_gb={max_alloc:.2f}"
        )

    def _log_vlm_paths(self, model: torch.nn.Module, tag: str) -> None:
        candidates = [
            "paligemma_with_expert",
            "paligemma_with_expert.paligemma",
            "paligemma_with_expert.paligemma.language_model",
            "paligemma_with_expert.paligemma.vision_model",
            "paligemma_with_expert.paligemma.vision_tower",
            "paligemma_with_expert.gemma_expert",
        ]
        found = []
        for path in candidates:
            cur = model
            ok = True
            for part in path.split("."):
                if not hasattr(cur, part):
                    ok = False
                    break
                cur = getattr(cur, part)
            if ok:
                found.append(path)
        print(f"[VLM Path] {tag} candidates: {found}")

        likely = []
        for name, module in model.named_modules():
            cls_name = module.__class__.__name__.lower()
            if any(k in cls_name for k in ("paligemma", "siglip", "vision")):
                likely.append(name)
        if likely:
            print(f"[VLM Path] {tag} named_modules (sample): {likely[:10]}")

    def _maybe_log_terminal_binary_loss_inputs(
        self,
        advantages: torch.Tensor,
        returns: torch.Tensor | None,
        loss_mask: torch.Tensor | None,
        adv_clip_max: float | None = None,
    ) -> None:
        if self._logged_terminal_binary_loss:
            return
        if self._rank != 0:
            return
        with torch.no_grad():
            mask = loss_mask if loss_mask is not None else torch.ones_like(advantages)
            mask = mask.to(dtype=torch.bool)
            if mask.shape != advantages.shape:
                mask = mask.expand_as(advantages)
            masked_adv = advantages[mask]
            adv_min = float(masked_adv.min().item()) if masked_adv.numel() > 0 else 0.0
            adv_max = float(masked_adv.max().item()) if masked_adv.numel() > 0 else 0.0
            adv_mean = (
                float(masked_adv.mean().item()) if masked_adv.numel() > 0 else 0.0
            )
            if returns is not None:
                masked_ret = returns[mask]
                ret_min = (
                    float(masked_ret.min().item()) if masked_ret.numel() > 0 else 0.0
                )
                ret_max = (
                    float(masked_ret.max().item()) if masked_ret.numel() > 0 else 0.0
                )
                ret_unique = (
                    torch.unique(masked_ret).detach().cpu().tolist()
                    if masked_ret.numel() > 0
                    else []
                )
            else:
                ret_min = ret_max = 0.0
                ret_unique = []
            print(
                "[loss][terminal-binary] "
                f"adv_clip_max={adv_clip_max} adv_min={adv_min:.3f} "
                f"adv_max={adv_max:.3f} adv_mean={adv_mean:.3f} "
                f"ret_min={ret_min:.3f} ret_max={ret_max:.3f} ret_unique={ret_unique}",
                flush=True,
            )
        self._logged_terminal_binary_loss = True


    def _setup_rollout_weight_dst_ranks(self) -> None:
        """
        Setup destination ranks for weight communication.
        It can support any topology between actor and rollout workers.
        Assuming there are M actor ranks and N rollout ranks, each actor rank
        will send weights to most ceil(N/M) rollout ranks according to the modulo rule.
        """
        rollout_world_size = self._component_placement.get_world_size("rollout")
        actor_world_size = self._world_size
        rank = self._rank
        self._weight_dst_rank_in_rollout = []
        rollout_ranks_per_actor = (
            rollout_world_size + actor_world_size - 1
        ) // actor_world_size
        for i in range(rollout_ranks_per_actor):
            if i * actor_world_size + rank < rollout_world_size:
                self._weight_dst_rank_in_rollout.append(i * actor_world_size + rank)

    def init_worker(self) -> None:
        """
        Initialize the actor worker. build the model and use corresponding training backend,
        if needed, offload model parameters and optimizer states to CPU.
        """
        if self.cfg.algorithm.loss_type.startswith("nft"):
            with open_dict(self.cfg):
                if self.cfg.actor.model.get("add_value_head", False):
                    if self.cfg.actor.fsdp_config.get("wrap_value_head", True):
                        self.cfg.actor.fsdp_config.wrap_value_head = False
                        if self._rank == 0:
                            print(
                                "[FSDP] Disabling value_head auto-wrap to avoid shape writeback errors."
                            )

        self.setup_model_and_optimizer()
        self._log_cuda_memory("init/after_setup")

        if self.cfg.algorithm.kl_beta > 0 or self.cfg.algorithm.loss_type.startswith(
            "nft"
        ):
            import gc

            ref_model = get_model(self.cfg.actor.model)
            if ref_model is None:
                ref_model = super().model_provider_func()

            if self.cfg.runner.get("ckpt_path", None):
                model_dict = torch.load(self.cfg.runner.ckpt_path)
                ref_model.load_state_dict(model_dict)

            gc.collect()
            print("[Memory Opt] Moving Action Expert to GPU...")
            ref_model.to(self.device)
            self._log_cuda_memory("init/after_ref_model_to_gpu")

            ref_model.eval()
            for p in ref_model.parameters():
                p.requires_grad = False

            self.ref_model = ref_model

            share_vlm = bool(getattr(self.cfg.actor, "share_vlm_with_ref", True))
            if share_vlm:
                use_orig_params = bool(
                    getattr(self.cfg.actor.fsdp_config, "use_orig_params", False)
                )
                fsdp_use_orig = bool(getattr(self.model, "_use_orig_params", False))
                if isinstance(self.model, FSDP) and not (use_orig_params or fsdp_use_orig):
                    share_ctx = FSDP.summon_full_params(
                        self.model, writeback=False, recurse=True
                    )
                    if self._rank == 0:
                        print(
                            "[Memory Opt] VLM share via summon_full_params "
                            "(use_orig_params=False)."
                        )
                else:
                    share_ctx = nullcontext()

                with share_ctx:
                    student_inner = (
                        self.model.module if hasattr(self.model, "module") else self.model
                    )
                    if (
                        hasattr(student_inner, "paligemma_with_expert")
                        and hasattr(ref_model, "paligemma_with_expert")
                        and hasattr(student_inner.paligemma_with_expert, "paligemma")
                    ):
                        ref_vlm = ref_model.paligemma_with_expert.paligemma

                        student_params_alias: dict[str, torch.nn.Parameter] = {}
                        for name, param in student_inner.named_parameters():
                            if not name.startswith("paligemma_with_expert.paligemma."):
                                continue
                            suffix = name[len("paligemma_with_expert.paligemma.") :]
                            if suffix.startswith("_fsdp_wrapped_module."):
                                suffix = suffix[len("_fsdp_wrapped_module.") :]
                            suffix = suffix.replace("._fsdp_wrapped_module.", ".")
                            suffix = suffix.replace("._fsdp_wrapped_module", "")
                            student_params_alias.setdefault(suffix, param)
                            if suffix.startswith("model."):
                                student_params_alias.setdefault(
                                    suffix[len("model.") :], param
                                )
                            else:
                                student_params_alias.setdefault(f"model.{suffix}", param)
                        tied = 0
                        missing = 0
                        missing_names = []
                        for name, ref_param in ref_vlm.named_parameters():
                            src_param = student_params_alias.get(name, None)
                            if src_param is None:
                                missing += 1
                                if len(missing_names) < 5:
                                    missing_names.append(name)
                                continue
                            if src_param.shape != ref_param.shape:
                                missing += 1
                                if len(missing_names) < 5:
                                    missing_names.append(name)
                                continue
                            ref_param.data = src_param.data
                            tied += 1

                        shared_ptrs = {
                            p.data_ptr() for p in student_params_alias.values()
                        }
                        self._shared_ref_param_names = {
                            name
                            for name, p in self.ref_model.named_parameters()
                            if p.data_ptr() in shared_ptrs
                        }
                        print(
                            "[Memory Opt] Shared VLM weights between student/ref. "
                            f"shared_vlm_params={len(self._shared_ref_param_names)} "
                            f"tied={tied} missing_or_mismatch={missing} "
                            f"student_alias_keys={len(student_params_alias)}"
                        )
                        if missing_names:
                            print(
                                "[Memory Opt] VLM share missing/mismatch sample: "
                                f"{missing_names}"
                            )
                        if len(self._shared_ref_param_names) == 0:
                            print(
                                "[Warning] VLM share attempted but no shared params detected."
                            )
                        else:
                            sample_names = sorted(self._shared_ref_param_names)[:5]
                            print(
                                "[Memory Opt] Shared VLM param sample: "
                                f"{sample_names}"
                            )
                    else:
                        print(
                            "[Warning] Could not share VLM weights. "
                            "Missing paligemma_with_expert.paligemma."
                        )
            else:
                print("[Memory Opt] VLM share disabled via actor.share_vlm_with_ref.")
            self._log_cuda_memory("init/after_vlm_share")

            torch.cuda.empty_cache()
            self._log_cuda_memory("init/after_empty_cache")
            try:
                student_param_ptrs = {p.data_ptr() for p in self.model.parameters()}
                ref_param_ptrs = {p.data_ptr() for p in self.ref_model.parameters()}
                shared_param_count = len(student_param_ptrs & ref_param_ptrs)
                print(
                    "[Memory Opt] Reference Model loaded. "
                    f"Shared parameter count with student: {shared_param_count}"
                )
            except Exception as e:
                print(f"[Memory Opt] Reference Model loaded. Share check failed: {e}")
            self._log_vlm_paths(self.model, "student")
            self._log_vlm_paths(self.ref_model, "ref")

        if self.enable_offload:
            self.offload_param_and_grad()
            self.offload_optimizer()
        self._setup_rollout_weight_dst_ranks()

    def _ref_checkpoint_path(self, base_path: str) -> str:
        return os.path.join(base_path, "ref_model.pt")

    def _resume_signature_path(self, base_path: str) -> str:
        return os.path.join(base_path, "resume_signature.pt")

    def _build_state_signature(self, state_dict: dict, max_keys: int = 8) -> dict:
        signature = {}
        if not state_dict:
            return signature
        keys = sorted(state_dict.keys())[:max_keys]
        for name in keys:
            value = state_dict[name]
            if not torch.is_tensor(value):
                continue
            tensor = value.detach().float().cpu()
            signature[name] = {
                "shape": tuple(tensor.shape),
                "mean": float(tensor.mean().item()),
                "std": float(tensor.std().item()),
            }
        return signature

    def _compare_signature(self, saved: dict, current: dict, tol: float = 1e-3) -> list:
        mismatches = []
        for name, saved_stats in saved.items():
            cur_stats = current.get(name)
            if cur_stats is None:
                mismatches.append(f"{name}: missing_current")
                continue
            if saved_stats.get("shape") != cur_stats.get("shape"):
                mismatches.append(f"{name}: shape")
                continue
            for field in ("mean", "std"):
                saved_val = saved_stats.get(field)
                cur_val = cur_stats.get(field)
                if saved_val is None or cur_val is None:
                    mismatches.append(f"{name}: {field}_missing")
                    break
                if abs(saved_val - cur_val) > tol:
                    mismatches.append(f"{name}: {field}")
                    break
        return mismatches

    def _filter_shared_ref_state(self, state_dict: dict) -> dict:
        if not self._shared_ref_param_names:
            return state_dict
        return {
            name: value
            for name, value in state_dict.items()
            if name not in self._shared_ref_param_names
        }

    def save_checkpoint(self, save_path: str, global_steps: int) -> None:
        super().save_checkpoint(save_path, global_steps)
        if not self.cfg.algorithm.loss_type.startswith("nft"):
            return
        if self.ref_model is None:
            return
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        if self._rank == 0:
            ref_state = {
                k: v.detach().cpu() if torch.is_tensor(v) else v
                for k, v in self.ref_model.state_dict().items()
            }
            torch.save(ref_state, self._ref_checkpoint_path(save_path))
            actor_state = self.model.state_dict()
            signature = {
                "actor": self._build_state_signature(actor_state),
                "ref_model": self._build_state_signature(ref_state),
            }
            torch.save(signature, self._resume_signature_path(save_path))
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

    def load_checkpoint(self, load_path: str) -> None:
        super().load_checkpoint(load_path)
        if not self.cfg.algorithm.loss_type.startswith("nft"):
            return
        if self.ref_model is None:
            return
        ref_path = self._ref_checkpoint_path(load_path)
        if not os.path.exists(ref_path):
            if self._rank == 0:
                self.log_info(f"[resume] ref_model checkpoint not found: {ref_path}")
            return
        ref_state = torch.load(ref_path, map_location="cpu")
        ref_state = self._filter_shared_ref_state(ref_state)
        missing, unexpected = self.ref_model.load_state_dict(ref_state, strict=False)
        if self._rank == 0 and (missing or unexpected):
            self.log_info(
                "[resume] ref_model state dict mismatch: "
                f"missing={len(missing)} unexpected={len(unexpected)}"
            )
            shared_cnt = len(self._shared_ref_param_names)
            extra_missing = max(len(missing) - shared_cnt, 0)
            self.log_info(
                "[resume] ref_model load stats: "
                f"ref_state_keys={len(ref_state)} "
                f"ref_model_params={sum(1 for _ in self.ref_model.parameters())} "
                f"shared_vlm_params={shared_cnt} extra_missing={extra_missing}"
            )
        signature_path = self._resume_signature_path(load_path)
        if self._rank == 0 and os.path.exists(signature_path):
            saved_signature = torch.load(signature_path, map_location="cpu")
            actor_sig = self._build_state_signature(self.model.state_dict())
            ref_sig = self._build_state_signature(self.ref_model.state_dict())
            actor_mismatches = self._compare_signature(
                saved_signature.get("actor", {}), actor_sig
            )
            ref_mismatches = self._compare_signature(
                saved_signature.get("ref_model", {}), ref_sig
            )
            if actor_mismatches or ref_mismatches:
                self.log_info(
                    "[resume] signature mismatch: "
                    f"actor={actor_mismatches[:5]} ref_model={ref_mismatches[:5]}"
                )
            else:
                self.log_info("[resume] signature check passed for actor/ref_model")

    def model_provider_func(self) -> nn.Module:
        model = get_model(self.cfg.actor.model)
        if model is None:
            model = super().model_provider_func()

        if self.cfg.runner.get("ckpt_path", None):
            model_dict = torch.load(self.cfg.runner.ckpt_path)
            model.load_state_dict(model_dict)

        return model

    def sync_model_to_rollout(self) -> None:
        """
        Sync the model's full state dict to the rollout worker.
        """
        self._log_cuda_memory("sync/before_send")
        if self.enable_offload and not self.is_optimizer_offloaded:
            self.offload_optimizer()

        if self.enable_offload and self.is_weight_offloaded:
            self.load_param_and_grad(self.device)

        if (
            getattr(self.cfg.algorithm, "loss_type", "") == "nft-actor-critic"
            and self.ref_model is not None
        ):
            if self._value_head_sync_ready:
                if isinstance(self.model, FSDP) and not getattr(
                    self.model, "_is_root", False
                ):
                    if self._rank == 0:
                        self.log_info(
                            "[sync] skip value_head hard-copy (FSDP root not initialized yet)"
                        )
                else:
                    student_inner = (
                        self.model.module if hasattr(self.model, "module") else self.model
                    )
                    student_vh = getattr(student_inner, "value_head", None)
                    ref_vh = getattr(self.ref_model, "value_head", None)
                    if student_vh is not None and ref_vh is not None:
                        ref_vh.load_state_dict(student_vh.state_dict())
                        self.log_info(
                            "[sync] hard-copied value_head parameters from student to ref_model"
                        )
            else:
                if self._rank == 0:
                    self.log_info(
                        "[sync] value_head hard-copy skipped (training not started yet)"
                    )

        if (
            self.cfg.algorithm.loss_type.startswith("nft")
            and self.ref_model is not None
        ):
            state_dict = self.ref_model.state_dict()
        else:
            state_dict = self.get_model_state_dict(
                cpu_offload=False, full_state_dict=True
            )

        sync_to_cpu = bool(getattr(self.cfg.rollout, "sync_weights_to_cpu", True))
        if sync_to_cpu:
            state_dict = {
                k: v.detach().to(device="cpu", non_blocking=True).contiguous()
                if torch.is_tensor(v)
                else v
                for k, v in state_dict.items()
            }
        for rank in self._weight_dst_rank_in_rollout:
            self.send(
                state_dict,
                self._rollout_group_name,
                rank,
                async_op=True,
                options=self._sync_weight_comm_options,
            )
        if self.enable_offload and not self.is_weight_offloaded:
            self.offload_param_and_grad()
        self._log_cuda_memory("sync/after_send")

    def recv_rollout_batch(self, input_channel: Channel) -> None:
        """
        Receive rollout batch from rollout workers.

        Args:
            input_channel: The input channel to read from.
        """
        send_num = self._component_placement.get_world_size("rollout") * self.stage_num
        recv_num = self._component_placement.get_world_size("actor")
        split_num = compute_split_num(send_num, recv_num)

        self.rollout_batch = {}
        recv_list = []
        for _ in range(split_num):
            recv_list.append(input_channel.get())

        # shape [num_chunk, bsz, chunk_size], cat dim 1
        self.rollout_batch = cat_list_of_dict_tensor(recv_list, dim=1)

        self.rollout_batch = self._process_received_rollout_batch(self.rollout_batch)

    def _process_received_rollout_batch(
        self, rollout_batch: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """
        original shape: [rollout_epoch x n_chunk_steps, bsz, num_action_chunks, ...]
        target shape: [n_chunk_steps, rollout_epoch x bsz, num_action_chunks, ...]
        """
        rollout_epoch = self.cfg.algorithm.rollout_epoch
        rollout_batch = process_nested_dict_for_adv(rollout_batch, rollout_epoch)

        if (
            not self.cfg.env.train.auto_reset
            and not self.cfg.env.train.ignore_terminations
        ):
            dones = rollout_batch[
                "dones"
            ]  # [n_chunk_step, rollout_epoch x bsz, num_action_chunks]
            loss_mask, loss_mask_sum = compute_loss_mask(dones)

            if self.cfg.algorithm.reward_type == "chunk_level":
                loss_mask = loss_mask.any(dim=-1, keepdim=True)
                loss_mask_sum = loss_mask_sum[..., -1:]

            rollout_batch["loss_mask"] = loss_mask
            rollout_batch["loss_mask_sum"] = loss_mask_sum

        # filter data by rewards
        if self.cfg.algorithm.get("filter_rewards", False):
            rewards = rollout_batch[
                "rewards"
            ]  # [n_chunk_step, batch, num_action_chunks]
            if rollout_batch.get("loss_mask", None) is not None:
                rewards = rewards * rollout_batch["loss_mask"]
            n_chunk_step, batch_size, num_action_chunks = rewards.shape

            group_size = self.cfg.algorithm.group_size
            assert batch_size % group_size == 0, (
                f"batch {batch_size} not divisible by group_size {group_size}"
            )
            n_prompts = batch_size // group_size

            # calculate rewards by prompt
            rewards = rewards.transpose(
                0, 1
            )  # [batch, n_chunk_step, num_action_chunks]
            rewards = rewards.reshape(rewards.shape[0], -1)  # [batch, n_step]
            reward_matrix = rewards.reshape(
                n_prompts, group_size, rewards.shape[-1]
            )  # [n_prompts, group_size, n_step]
            reward_matrix = reward_matrix.sum(dim=-1)  # [n_prompts, group_size]
            mean_reward_in_group = reward_matrix.mean(dim=1)  # [n_prompts]

            # mask
            reward_filter_mask = (
                mean_reward_in_group >= self.cfg.algorithm.rewards_lower_bound
            ) & (
                mean_reward_in_group <= self.cfg.algorithm.rewards_upper_bound
            )  # [n_prompts]

            # extend mask dimension
            reward_filter_mask = reward_filter_mask.repeat_interleave(
                group_size
            )  # [batch]
            reward_filter_mask = (
                reward_filter_mask.unsqueeze(0).expand(n_chunk_step, -1).unsqueeze(-1)
            )  # [n_chunk_step, batch, 1]

            # update loss_mask
            if rollout_batch.get("loss_mask", None) is not None:
                rollout_batch["loss_mask"] = (
                    reward_filter_mask & rollout_batch["loss_mask"]
                )
            else:
                rollout_batch["loss_mask"] = reward_filter_mask

        use_time_decay = (
            self.cfg.algorithm.loss_type.startswith("nft")
            and self.cfg.algorithm.get("use_nft_time_decay", False)
        )
        if use_time_decay and rollout_batch.get("loss_mask", None) is not None:
            gamma = float(self.cfg.algorithm.get("nft_time_decay_gamma", 0.9))
            epsilon = float(self.cfg.algorithm.get("nft_time_decay_epsilon", 0.1))
            rollout_batch["time_decay_weights"] = compute_time_decay_weights(
                rollout_batch["loss_mask"], gamma=gamma, epsilon=epsilon
            )

        return rollout_batch

    def compute_advantages_and_returns(self) -> dict[str, torch.Tensor]:
        """
        Compute the advantages and returns.
        """

        if self.cfg.algorithm.adv_type == "terminal-binary":
            _log_terminal_binary_adv_inputs(
                rank=self._rank,
                rewards=self.rollout_batch.get("rewards", None),
                dones=self.rollout_batch.get("dones", None),
                success_once=self.rollout_batch.get("success_once", None),
                loss_mask=self.rollout_batch.get("loss_mask", None),
            )

        kwargs = {
            "task_type": self.cfg.runner.task_type,
            "adv_type": self.cfg.algorithm.adv_type,
            "rewards": self.rollout_batch["rewards"],
            "dones": self.rollout_batch["dones"],
            "values": self.rollout_batch.get("prev_values", None),
            "success_once": self.rollout_batch.get("success_once", None),
            "gamma": self.cfg.algorithm.get("gamma", 1),
            "gae_lambda": self.cfg.algorithm.get("gae_lambda", 1),
            "group_size": self.cfg.algorithm.get("group_size", 8),
            "reward_type": self.cfg.algorithm.reward_type,
            "loss_mask": self.rollout_batch.get("loss_mask", None),
            "loss_mask_sum": self.rollout_batch.get("loss_mask_sum", None),
            "rollout_epoch": self.cfg.algorithm.get("rollout_epoch", 1),
            "adv_clip_max": self.cfg.algorithm.get("clip_ratio_high", 1.0),
        }

        advantages_and_returns = calculate_adv_and_returns(**kwargs)

        self.rollout_batch.update(advantages_and_returns)
        if kwargs["loss_mask"] is not None:
            self.rollout_batch["loss_mask"] = kwargs["loss_mask"]
        else:
            self.rollout_batch.pop("loss_mask", None)
        if kwargs["loss_mask_sum"] is not None:
            self.rollout_batch["loss_mask_sum"] = kwargs["loss_mask_sum"]
        else:
            self.rollout_batch.pop("loss_mask_sum", None)

        rollout_metrics = compute_rollout_metrics(self.rollout_batch)
        if "avg_success_done_step" in self.rollout_batch:
            val = self.rollout_batch["avg_success_done_step"]
            if isinstance(val, torch.Tensor):
                val = val.item()
            rollout_metrics["avg_success_done_step"] = val

        return rollout_metrics

    def run_training(self) -> None:
        """
        Run the training process using the received rollout batch.
        """
        self._log_cuda_memory("train/start")
        if self.is_weight_offloaded:
            self.load_param_and_grad(self.device)
        if self.is_optimizer_offloaded:
            self.load_optimizer(self.device)

        self.model.train()
        rollout_size = (
            self.rollout_batch["prev_logprobs"].shape[0]
            * self.rollout_batch["prev_logprobs"].shape[1]
        )
        g = torch.Generator()
        g.manual_seed(self.cfg.actor.seed + self._rank)
        shuffle_id = torch.randperm(rollout_size, generator=g)

        with torch.no_grad():
            self.rollout_batch = process_nested_dict_for_train(
                self.rollout_batch, shuffle_id
            )

        assert (
            self.cfg.actor.global_batch_size
            % (self.cfg.actor.micro_batch_size * self._world_size)
            == 0
        ), "global_batch_size is not divisible by micro_batch_size * world_size"

        self.gradient_accumulation = (
            self.cfg.actor.global_batch_size
            // self.cfg.actor.micro_batch_size
            // self._world_size
        )

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        rollout_size = self.rollout_batch["prev_logprobs"].size(0)
        batch_size_per_rank = self.cfg.actor.global_batch_size // self._world_size
        assert rollout_size % batch_size_per_rank == 0, (
            f"{rollout_size} is not divisible by {batch_size_per_rank}"
        )
        metrics = {}
        update_epoch = self.cfg.algorithm.get("update_epoch", 1)
        for _ in range(update_epoch):
            rollout_dataloader_iter = get_iterator_k_split(
                self.rollout_batch,
                rollout_size // batch_size_per_rank,
            )
            for train_global_batch in rollout_dataloader_iter:
                # split batch into micro_batches
                train_global_batch_size = train_global_batch["prev_logprobs"].shape[0]
                assert (
                    train_global_batch_size
                    == self.cfg.actor.global_batch_size
                    // torch.distributed.get_world_size()
                )
                assert train_global_batch_size % self.cfg.actor.micro_batch_size == 0, (
                    f"{train_global_batch_size=}, {self.cfg.actor.micro_batch_size}"
                )

                train_micro_batch = get_iterator_k_split(
                    train_global_batch,
                    train_global_batch_size // self.cfg.actor.micro_batch_size,
                )

                self.optimizer.zero_grad()
                for idx, data in enumerate(train_micro_batch):
                    data = put_tensor_device(
                        data, f"cuda:{int(os.environ['LOCAL_RANK'])}"
                    )
                    backward_ctx = self.before_micro_batch(
                        self.model,
                        is_last_micro_batch=(idx + 1) == self.gradient_accumulation,
                    )
                    if self.cfg.algorithm.loss_type.startswith("nft"):
                        v_old = data.get("nft_v", None)
                        x_t_input = data.get("nft_xt", None)
                        x_next_input = data.get("nft_xnext", None)
                        step_indices = data.get("nft_step_index", None)
                        noise_level_for_loss = data.get("nft_noise_level", None)

                        num_steps = self.cfg.actor.model.num_steps
                        schedule = torch.linspace(
                            1,
                            0,
                            num_steps + 1,
                            device=self.device,
                            dtype=x_t_input.dtype,
                        )
                        t = schedule[step_indices.long()]

                        compute_values = self.cfg.actor.model.get(
                            "add_value_head", False
                        )
                        with self.amp_context:
                            output_dict = self.model(
                                data=data,
                                use_nft_loss=True,
                                compute_values=compute_values,
                                compute_noise_stats=True,
                                nft_explicit_inputs={"x_t": x_t_input, "timesteps": t},
                                use_cache=False,
                                shared_cache=None,
                            )

                        v_theta = output_dict["v_theta"]
                        values = output_dict.get("values", None)

                        chunk_size = v_theta.shape[1]
                        x_t_loss = x_t_input[:, :chunk_size, :]
                        x_next_loss = x_next_input[:, :chunk_size, :]

                        prev_values = data.get("prev_values", None)
                        if prev_values is not None and prev_values.dim() > 1:
                            prev_values = prev_values[:, :1]
                        returns = data.get("returns", None)
                        if returns is not None and returns.dim() > 1:
                            returns = returns[:, :1]

                        if self.cfg.algorithm.adv_type == "terminal-binary":
                            self._maybe_log_terminal_binary_loss_inputs(
                                advantages=data["advantages"],
                                returns=returns,
                                loss_mask=data.get("loss_mask", None),
                                adv_clip_max=self.cfg.algorithm.get(
                                    "clip_ratio_high", 5.0
                                ),
                            )

                        loss_type = (
                            "nft-actor-critic" if compute_values else "nft-actor"
                        )
                        kwargs = {
                            "loss_type": loss_type,
                            "task_type": self.cfg.runner.task_type,
                            "v_theta": v_theta,
                            "v_old": v_old,
                            "x_t": x_t_loss,
                            "x_next": x_next_loss,
                            "schedule": schedule,
                            "step_indices": step_indices,
                            "total_denoise_steps": num_steps,
                            "noise_level": noise_level_for_loss,
                            "advantages": data["advantages"],
                            "loss_mask": data.get("loss_mask", None),
                            "loss_mask_sum": data.get("loss_mask_sum", None),
                            "time_decay_weights": data.get("time_decay_weights", None),
                            "beta": self.cfg.algorithm.get("nft_beta", 1.0),
                            "kl_beta": self.cfg.algorithm.get("kl_beta", 0.0),
                            "adv_clip_max": self.cfg.algorithm.get(
                                "clip_ratio_high", 1.0
                            ),
                            "task_ids": data.get("task_ids", None),
                            "values": values,
                            "returns": returns,
                            "prev_values": prev_values,
                            "value_clip": self.cfg.algorithm.get("value_clip", None),
                            "huber_delta": self.cfg.algorithm.get("huber_delta", None),
                            "max_episode_steps": self.cfg.env.train.max_episode_steps,
                            "critic_warmup": self._is_in_critic_warmup(),
                        }
                        loss, metrics_data = policy_loss(**kwargs)
                    else:
                        advantages = data["advantages"]
                        prev_logprobs = data["prev_logprobs"]
                        returns = data.get("returns", None)
                        prev_values = data.get("prev_values", None)
                        loss_mask = data.get("loss_mask", None)
                        loss_mask_sum = data.get("loss_mask_sum", None)

                        if SupportedModel(self.cfg.actor.model.model_type) in [
                            SupportedModel.OPENVLA,
                            SupportedModel.OPENVLA_OFT,
                        ]:
                            data["temperature"] = (
                                self.cfg.algorithm.sampling_params.temperature_train
                            )
                            data["top_k"] = self.cfg.algorithm.sampling_params.top_k

                        compute_values = (
                            True if self.cfg.algorithm.adv_type == "gae" else False
                        )

                        with self.amp_context:
                            output_dict = self.model(
                                data=data,
                                compute_logprobs=True,
                                compute_entropy=self.cfg.algorithm.entropy_bonus > 0,
                                compute_values=compute_values,
                                use_cache=False,
                            )

                        if SupportedModel(self.cfg.actor.model.model_type) in [
                            SupportedModel.GR00T
                        ]:
                            prev_logprobs = output_dict["prev_logprobs"]

                        if self.cfg.algorithm.adv_type == "terminal-binary":
                            self._maybe_log_terminal_binary_loss_inputs(
                                advantages=advantages,
                                returns=returns,
                                loss_mask=loss_mask,
                                adv_clip_max=self.cfg.algorithm.clip_ratio_high,
                            )

                        kwargs = {
                            "loss_type": self.cfg.algorithm.loss_type,
                            "logprob_type": self.cfg.algorithm.logprob_type,
                            "reward_type": self.cfg.algorithm.reward_type,
                            "single_action_dim": self.cfg.actor.model.get(
                                "action_dim", 7
                            ),
                            "logprobs": output_dict["logprobs"],
                            "values": output_dict.get("values", None),
                            "old_logprobs": prev_logprobs,
                            "advantages": advantages,
                            "returns": returns,
                            "prev_values": prev_values,
                            "clip_ratio_high": self.cfg.algorithm.clip_ratio_high,
                            "clip_ratio_low": self.cfg.algorithm.clip_ratio_low,
                            "value_clip": self.cfg.algorithm.get("value_clip", None),
                            "huber_delta": self.cfg.algorithm.get("huber_delta", None),
                            "loss_mask": loss_mask,
                            "loss_mask_sum": loss_mask_sum,
                            "max_episode_steps": self.cfg.env.train.max_episode_steps,
                            "task_type": self.cfg.runner.task_type,
                            "critic_warmup": self._is_in_critic_warmup(),
                        }
                        loss, metrics_data = policy_loss(**kwargs)

                        entropy_loss = torch.tensor(
                            0.0, device=torch.cuda.current_device()
                        )
                        if (
                            self.cfg.algorithm.entropy_bonus > 0
                            and not kwargs["critic_warmup"]
                        ):
                            entropy = output_dict["entropy"]
                            entropy = reshape_entropy(
                                entropy,
                                entropy_type=self.cfg.algorithm.entropy_type,
                                action_dim=self.cfg.actor.model.get("action_dim", 7),
                                batch_size=output_dict["logprobs"].shape[0],
                            )
                            entropy_loss = masked_mean(entropy, mask=loss_mask)
                            loss -= self.cfg.algorithm.entropy_bonus * entropy_loss
                        metrics_data["entropy_loss"] = entropy_loss.detach().item()

                    loss /= self.gradient_accumulation
                    with backward_ctx:
                        self.grad_scaler.scale(loss).backward()

                    metrics_data["loss"] = loss.detach().item()
                    append_to_dict(metrics, metrics_data)

                torch.cuda.empty_cache()

                grad_norm, lr_list = self.optimizer_step()
                self._update_ready = True
                self._log_cuda_memory("train/after_optimizer_step")
                data = {
                    "actor/grad_norm": grad_norm,
                    "actor/lr": lr_list[0],
                }
                if len(lr_list) > 1:
                    data["critic/lr"] = lr_list[1]
                append_to_dict(metrics, data)
        # put LR scheduler step here
        self.lr_scheduler.step()
        self.optimizer.zero_grad()
        self._log_cuda_memory("train/after_lr_step")

        decay = 1.0
        if (
            self.cfg.algorithm.loss_type.startswith("nft")
            and self.ref_model is not None
        ):
            with torch.no_grad():
                total_steps = self.cfg.algorithm.get(
                    "decay_epochs",
                    self.cfg.runner.get("max_epochs", self.cfg.runner.get("max_steps", 0)),
                )
                base_decay = self.cfg.algorithm.get("base", 0.1)
                target_decay = self.cfg.algorithm.get("target", 0.8)
                decay = nft_return_decay(
                    self.global_step,
                    total_steps,
                    base=base_decay,
                    target=target_decay,
                )

                if decay < 1.0:
                    alpha = 1.0 - decay
                    try:
                        student_sd = self.get_model_state_dict(
                            cpu_offload=False, full_state_dict=True
                        )
                    except AssertionError as exc:
                        if self._rank == 0:
                            self.log_info(
                                "[EMA] state_dict assertion, falling back to named_parameters. "
                                f"error={exc}"
                            )

                        def _normalize_name(name: str) -> str:
                            if name.startswith("_fsdp_wrapped_module."):
                                name = name[len("_fsdp_wrapped_module.") :]
                            name = name.replace("._fsdp_wrapped_module.", ".")
                            name = name.replace("._fsdp_wrapped_module", "")
                            return name

                        student_sd = {}
                        for name, param in self.model.named_parameters():
                            norm = _normalize_name(name)
                            student_sd.setdefault(norm, param.detach())
                            if norm.startswith("model."):
                                student_sd.setdefault(norm[len("model.") :], param.detach())
                            else:
                                student_sd.setdefault(f"model.{norm}", param.detach())

                    shape_mismatch_cnt = 0
                    updated = 0
                    mismatch_samples = []
                    for name, tgt in self.ref_model.named_parameters():
                        if (
                            name in self._shared_ref_param_names
                            or "value_head" in name
                            or "paligemma." in name
                            or "u_encoder" in name
                        ):
                            continue
                        src = student_sd.get(name, None)
                        if src is None:
                            continue
                        if src.shape != tgt.shape:
                            shape_mismatch_cnt += 1
                            if len(mismatch_samples) < 5:
                                mismatch_samples.append(
                                    f"{name}: src={tuple(src.shape)} tgt={tuple(tgt.shape)}"
                                )
                            continue
                        tgt.data.lerp_(src.to(tgt.device), alpha)
                        updated += 1

                    try:
                        student_ptrs = {p.data_ptr() for p in self.model.parameters()}
                        leaked_shared = [
                            n
                            for n, p in self.ref_model.named_parameters()
                            if p.data_ptr() in student_ptrs
                            and n not in self._shared_ref_param_names
                        ]
                    except Exception:
                        leaked_shared = []
                    if updated > 0:
                        print(
                            f"[EMA] updated {updated} policy parameters with decay={decay:.4f}",
                            flush=True,
                        )
                    else:
                        print(
                            "[EMA] no policy parameter was updated in this step "
                            f"(decay={decay:.4f}, optimizer_steps={self.optimizer_steps}, "
                            f"shape_mismatch={shape_mismatch_cnt})",
                            flush=True,
                        )
                    if shape_mismatch_cnt > 0 and mismatch_samples:
                        print(
                            "[EMA] shape mismatch samples: " + "; ".join(mismatch_samples),
                            flush=True,
                        )
                    if leaked_shared:
                        print(
                            "[EMA] shared params not tracked (sample): "
                            + ", ".join(leaked_shared[:5]),
                            flush=True,
                        )


        clear_memory()

        def _to_scalar_list(val):
            if isinstance(val, list):
                items = val
            else:
                items = [val]
            scalars = []
            for x in items:
                if torch.is_tensor(x):
                    scalars.append(float(x.detach().mean().item()))
                else:
                    scalars.append(float(np.mean(x)))
            return scalars

        mean_metric_dict = {}
        for key, value in metrics.items():
            scalars = _to_scalar_list(value)
            mean_metric_dict[key] = (
                float(np.mean(scalars)) if len(scalars) > 0 else 0.0
            )
        mean_metric_dict = all_reduce_dict(
            mean_metric_dict, op=torch.distributed.ReduceOp.AVG
        )
        if self.cfg.algorithm.loss_type.startswith("nft"):
            mean_metric_dict["actor/nft_decay"] = decay

        return mean_metric_dict

    def set_global_step(self, global_step) -> None:
        """
        Set the global step for the model, if needed.
        """
        super().set_global_step(global_step)
        self.global_step = int(global_step)
        if hasattr(self.model, "set_global_step"):
            self.model.set_global_step(global_step)
        if self.global_step > 0:
            self._value_head_sync_ready = True
