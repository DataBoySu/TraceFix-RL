"""OpenEnv adapter around the TraceFix-RL core environment."""

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..environment import TraceFixRLGym
    from ..models import CodeAction, CodeObservation
except ImportError:
    from environment import TraceFixRLGym
    from models import CodeAction, CodeObservation


class TraceFixRLEnvironment(Environment):
    """Environment implementation compatible with OpenEnv's server interface."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._gym = TraceFixRLGym()
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
        return self._state
