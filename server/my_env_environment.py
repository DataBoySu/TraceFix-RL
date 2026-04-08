# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""OpenEnv adapter around the PythonDebuggingGym core environment."""

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..environment import PythonDebuggingGym
    from ..models import CodeAction, CodeObservation
except ImportError:
    from environment import PythonDebuggingGym
    from models import CodeAction, CodeObservation


class MyEnvironment(Environment):
    """Environment implementation compatible with OpenEnv's server interface."""

    # Enable concurrent WebSocket sessions.
    # Set to True if your environment isolates state between instances.
    # When True, multiple WebSocket clients can connect simultaneously, each
    # getting their own environment instance (when using factory mode in app.py).
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._gym = PythonDebuggingGym()
        self._state = State(episode_id="", step_count=0)

    def reset(self) -> CodeObservation:
        obs, system_prompt = self._gym.reset()
        self._state = State(
            episode_id=obs.info.get("episode_id", ""),
            step_count=obs.step_count,
        )
        metadata = dict(obs.metadata or {})
        metadata["system_prompt"] = system_prompt
        obs.metadata = metadata
        return obs

    def step(self, action: CodeAction) -> CodeObservation:  # type: ignore[override]
        obs, reward, done, info = self._gym.step(action)
        obs.reward = reward
        obs.done = done
        metadata = dict(obs.metadata or {})
        metadata.update(info)
        obs.metadata = metadata
        self._state = State(
            episode_id=obs.info.get("episode_id", ""),
            step_count=obs.step_count,
        )
        return obs

    @property
    def state(self) -> State:
        """
        Get the current environment state.

        Returns:
            Current State with episode_id and step_count
        """
        return self._state
