"""Pydantic schema layer for TraceFix-RL."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from openenv.core.env_server.types import Action, Observation
from pydantic import BaseModel, Field, model_validator


ActionType = Literal[
    "VIEW_CODE",
    "RUN_TESTS",
    "REPLACE_LINES",
    "UNDO_EDIT",
    "RESET_TO_ORIGINAL",
    "SUBMIT",
]


class CodeAction(Action):
    """Structured action consumed by the environment."""

    thought: str = Field(
        ...,
        description="Mandatory reasoning string before selecting an action.",
    )
    action_type: ActionType = Field(
        ...,
        description="One of VIEW_CODE, RUN_TESTS, REPLACE_LINES, UNDO_EDIT, RESET_TO_ORIGINAL, SUBMIT.",
    )
    start_line: Optional[int] = Field(default=None)
    end_line: Optional[int] = Field(default=None)
    new_code_block: Optional[str] = Field(default=None)

    @model_validator(mode="before")
    @classmethod
    def validate_and_normalize(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        action_type = data.get("action_type")

        def _coerce_optional_int(value: Any) -> Optional[int]:
            if value is None:
                return None
            if isinstance(value, int):
                return value
            if isinstance(value, str):
                raw = value.strip()
                if raw == "":
                    return None
                try:
                    return int(raw)
                except ValueError:
                    return None
            return None

        data = dict(data)
        data["start_line"] = _coerce_optional_int(data.get("start_line"))
        data["end_line"] = _coerce_optional_int(data.get("end_line"))

        if action_type == "REPLACE_LINES":
            start_line = data.get("start_line")
            end_line = data.get("end_line")
            new_code_block = data.get("new_code_block")

            if start_line is None:
                raise ValueError("REPLACE_LINES requires start_line.")
            if end_line is None:
                raise ValueError("REPLACE_LINES requires end_line.")
            if new_code_block is None:
                raise ValueError("REPLACE_LINES requires new_code_block.")
            if start_line < 1 or end_line < 1:
                raise ValueError("REPLACE_LINES requires start_line and end_line >= 1.")
            if start_line > end_line:
                raise ValueError("REPLACE_LINES requires start_line <= end_line.")
        else:
            # Web UI often sends default line fields for non-edit actions.
            data["start_line"] = None
            data["end_line"] = None
            data["new_code_block"] = None

        return data


class TestResult(BaseModel):
    """Per-test execution outcome."""

    test_name: str
    passed: bool
    error_message: Optional[str] = None


class CodeObservation(Observation):
    """Full observation returned after each step."""

    code_dict: Dict[int, str] = Field(default_factory=dict)
    localized_context: str = Field(default="")
    last_execution_output: str = Field(default="")
    syntax_error: bool = Field(default=False)
    test_results: List[TestResult] = Field(default_factory=list)
    step_count: int = Field(default=0)
    steps_remaining: int = Field(default=0)
    reward_last_step: float = Field(default=0.0)
    info: Dict[str, Any] = Field(default_factory=dict)

    def render_code(self) -> str:
        """Render source with 1-indexed line numbers for prompts."""
        if not self.code_dict:
            return "<empty>"
        return "\n".join(
            f"{line_num:>3} | {self.code_dict[line_num]}"
            for line_num in sorted(self.code_dict.keys())
        )
