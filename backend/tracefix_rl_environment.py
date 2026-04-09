"""OpenEnv adapter around the TraceFix-RL core environment."""

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from core.environment import TraceFixRLGym
    from core.models import CodeAction, CodeObservation
except ImportError:
    from core.environment import TraceFixRLGym
    from core.models import CodeAction, CodeObservation


class TraceFixRLEnvironment(Environment):
    """Environment implementation compatible with OpenEnv's server interface."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._gym = TraceFixRLGym()
        self._state = State(episode_id="", step_count=0)

    def reset(self, difficulty: str | None = None, task_name: str | None = None) -> CodeObservation:
        if difficulty == "easy":
            self._gym.training_step = 1
        elif difficulty == "medium":
            self._gym.training_step = 2000
        elif difficulty == "hard":
            self._gym.training_step = 6000

        task_dict = None
        if task_name and task_name != "tracefix_rl":
            try:
                from tasks.tasks import ALL_TASKS
                for t in ALL_TASKS:
                    if t.get("name") == task_name:
                        task_dict = t
                        break
            except ImportError:
                pass

        obs, system_prompt = self._gym.reset(task_index=task_dict)
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
