"""Client for TraceFix-RL."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

try:
    from .models import CodeAction, CodeObservation, TestResult
except ImportError:
    from core.models import CodeAction, CodeObservation, TestResult


class TraceFixRLEnv(
    EnvClient[CodeAction, CodeObservation, State]
):
    """Typed OpenEnv client."""

    def _step_payload(self, action: CodeAction) -> Dict:
        return action.model_dump(exclude_none=True)

    def _parse_result(self, payload: Dict) -> StepResult[CodeObservation]:
        obs_data = payload.get("observation", {})
        raw_code_dict = obs_data.get("code_dict", {})
        code_dict = {
            int(k): v for k, v in raw_code_dict.items()
        } if isinstance(raw_code_dict, dict) else {}
        observation = CodeObservation(
            code_dict=code_dict,
            localized_context=obs_data.get("localized_context", ""),
            last_execution_output=obs_data.get("last_execution_output", ""),
            syntax_error=obs_data.get("syntax_error", False),
            test_results=[
                TestResult(**item) for item in obs_data.get("test_results", [])
            ],
            step_count=obs_data.get("step_count", 0),
            steps_remaining=obs_data.get("steps_remaining", 0),
            reward_last_step=obs_data.get("reward_last_step", 0.0),
            info=obs_data.get("info", {}),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )


MyEnv = TraceFixRLEnv
