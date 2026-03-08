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

import multiprocessing
import traceback
import warnings
from multiprocessing import connection
from typing import Any, Callable, Optional, Union

import gym
import numpy as np
from libero.libero.envs import OffScreenRenderEnv

from rlinf.envs.venv import (
    BaseVectorEnv,
    CloudpickleWrapper,
    EnvWorker,
    ShArray,
    SubprocEnvWorker,
    SubprocVectorEnv,
    SubprocError,
    _setup_buf,
)

gym_old_venv_step_type = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
gym_new_venv_step_type = tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]
warnings.simplefilter("once", DeprecationWarning)


def _worker(
    parent: connection.Connection,
    p: connection.Connection,
    env_fn_wrapper: CloudpickleWrapper,
    obs_bufs: Optional[Union[dict, tuple, ShArray]] = None,
) -> None:
    def _encode_obs(
        obs: Union[dict, tuple, np.ndarray], buffer: Union[dict, tuple, ShArray]
    ) -> None:
        if isinstance(obs, np.ndarray) and isinstance(buffer, ShArray):
            buffer.save(obs)
        elif isinstance(obs, tuple) and isinstance(buffer, tuple):
            for o, b in zip(obs, buffer):
                _encode_obs(o, b)
        elif isinstance(obs, dict) and isinstance(buffer, dict):
            for k in obs.keys():
                _encode_obs(obs[k], buffer[k])
        return None

    parent.close()
    init_err: Optional[str] = None
    env: Optional[gym.Env] = None
    try:
        env = env_fn_wrapper.data()
    except Exception:
        init_err = traceback.format_exc()
    try:
        while True:
            try:
                cmd, data = p.recv()
            except EOFError:  # the pipe has been closed
                p.close()
                break
            if init_err is not None:
                try:
                    p.send(SubprocError(where="init", tb=init_err))
                except Exception:
                    pass
                p.close()
                break
            if cmd == "step":
                try:
                    assert env is not None
                    env_return = env.step(data)
                except Exception:
                    try:
                        p.send(SubprocError(where="step", tb=traceback.format_exc()))
                    except Exception:
                        pass
                    p.close()
                    break
                if obs_bufs is not None:
                    _encode_obs(env_return[0], obs_bufs)
                    env_return = (None, *env_return[1:])
                p.send(env_return)
            elif cmd == "reset":
                try:
                    assert env is not None
                    retval = env.reset(**data)
                except Exception:
                    try:
                        p.send(SubprocError(where="reset", tb=traceback.format_exc()))
                    except Exception:
                        pass
                    p.close()
                    break
                reset_returns_info = (
                    isinstance(retval, (tuple, list))
                    and len(retval) == 2
                    and isinstance(retval[1], dict)
                )
                if reset_returns_info:
                    obs, info = retval
                else:
                    obs = retval
                if obs_bufs is not None:
                    _encode_obs(obs, obs_bufs)
                    obs = None
                if reset_returns_info:
                    p.send((obs, info))
                else:
                    p.send(obs)
            elif cmd == "close":
                try:
                    assert env is not None
                    p.send(env.close())
                except Exception:
                    try:
                        p.send(SubprocError(where="close", tb=traceback.format_exc()))
                    except Exception:
                        pass
                p.close()
                break
            elif cmd == "render":
                try:
                    assert env is not None
                    p.send(env.render(**data) if hasattr(env, "render") else None)
                except Exception:
                    try:
                        p.send(SubprocError(where="render", tb=traceback.format_exc()))
                    except Exception:
                        pass
                    p.close()
                    break
            elif cmd == "seed":
                try:
                    assert env is not None
                    if hasattr(env, "seed"):
                        p.send(env.seed(data))
                    else:
                        env.reset(seed=data)
                        p.send(None)
                except Exception:
                    try:
                        p.send(SubprocError(where="seed", tb=traceback.format_exc()))
                    except Exception:
                        pass
                    p.close()
                    break
            elif cmd == "getattr":
                try:
                    assert env is not None
                    p.send(getattr(env, data) if hasattr(env, data) else None)
                except Exception:
                    try:
                        p.send(SubprocError(where="getattr", tb=traceback.format_exc()))
                    except Exception:
                        pass
                    p.close()
                    break
            elif cmd == "setattr":
                try:
                    assert env is not None
                    setattr(env.unwrapped, data["key"], data["value"])
                except Exception:
                    try:
                        p.send(SubprocError(where="setattr", tb=traceback.format_exc()))
                    except Exception:
                        pass
                    p.close()
                    break
            elif cmd == "check_success":
                try:
                    assert env is not None
                    p.send(env.check_success())
                except Exception:
                    try:
                        p.send(
                            SubprocError(where="check_success", tb=traceback.format_exc())
                        )
                    except Exception:
                        pass
                    p.close()
                    break
            elif cmd == "get_segmentation_of_interest":
                try:
                    assert env is not None
                    p.send(env.get_segmentation_of_interest(data))
                except Exception:
                    try:
                        p.send(
                            SubprocError(
                                where="get_segmentation_of_interest",
                                tb=traceback.format_exc(),
                            )
                        )
                    except Exception:
                        pass
                    p.close()
                    break
            elif cmd == "get_sim_state":
                try:
                    assert env is not None
                    p.send(env.get_sim_state())
                except Exception:
                    try:
                        p.send(
                            SubprocError(where="get_sim_state", tb=traceback.format_exc())
                        )
                    except Exception:
                        pass
                    p.close()
                    break
            elif cmd == "set_init_state":
                try:
                    assert env is not None
                    obs = env.set_init_state(data)
                    p.send(obs)
                except Exception:
                    try:
                        p.send(
                            SubprocError(where="set_init_state", tb=traceback.format_exc())
                        )
                    except Exception:
                        pass
                    p.close()
                    break
            elif cmd == "reconfigure":
                try:
                    assert env is not None
                    env.close()
                    seed = data.pop("seed")
                    env = OffScreenRenderEnv(**data)
                    env.seed(seed)
                    p.send(None)
                except Exception:
                    try:
                        p.send(
                            SubprocError(where="reconfigure", tb=traceback.format_exc())
                        )
                    except Exception:
                        pass
                    p.close()
                    break
            else:
                p.close()
                raise NotImplementedError
    except KeyboardInterrupt:
        p.close()


class ReconfigureSubprocEnvWorker(SubprocEnvWorker):
    def __init__(self, env_fn: Callable[[], gym.Env], share_memory: bool = False):
        ctx = multiprocessing.get_context("spawn")
        self.parent_remote, self.child_remote = ctx.Pipe()
        self.share_memory = share_memory
        self.buffer: Optional[Union[dict, tuple, ShArray]] = None
        if self.share_memory:
            dummy = env_fn()
            obs_space = dummy.observation_space
            dummy.close()
            del dummy
            self.buffer = _setup_buf(obs_space)
        args = (
            self.parent_remote,
            self.child_remote,
            CloudpickleWrapper(env_fn),
            self.buffer,
        )
        self.process = ctx.Process(target=_worker, args=args, daemon=True)
        self.process.start()
        self.child_remote.close()
        EnvWorker.__init__(self, env_fn)

    def reconfigure_env_fn(self, env_fn_param):
        self.parent_remote.send(["reconfigure", env_fn_param])
        result = self.parent_remote.recv()
        if isinstance(result, SubprocError):
            raise RuntimeError(
                f"[ReconfigureSubprocEnvWorker] subprocess env error ({result.where}):\n{result.tb}"
            )
        return result


class ReconfigureSubprocEnv(SubprocVectorEnv):
    def __init__(self, env_fns: list[Callable[[], gym.Env]], **kwargs: Any) -> None:
        def worker_fn(fn: Callable[[], gym.Env]) -> ReconfigureSubprocEnvWorker:
            return ReconfigureSubprocEnvWorker(fn, share_memory=False)

        BaseVectorEnv.__init__(self, env_fns, worker_fn, **kwargs)

    def reconfigure_env_fns(self, env_fns, id=None):
        self._assert_is_not_closed()
        id = self._wrap_id(id)
        if self.is_async:
            self._assert_id(id)

        for j, i in enumerate(id):
            self.workers[i].reconfigure_env_fn(env_fns[j])
