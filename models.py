"""Pydantic schema layer for the Python Debugging Gym OpenEnv environment."""

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

    thought: Optional[str] = Field(
        default=None,
        description="Optional reasoning string for debugging/traceability.",
    )
    action_type: ActionType = Field(
        ...,
        description="One of VIEW_CODE, RUN_TESTS, REPLACE_LINES, UNDO_EDIT, RESET_TO_ORIGINAL, SUBMIT.",
    )
    start_line: Optional[int] = Field(default=None, ge=1)
    end_line: Optional[int] = Field(default=None, ge=1)
    new_code_block: Optional[str] = Field(default=None)

    @model_validator(mode="after")
    def validate_replace_fields(self) -> "CodeAction":
        if self.action_type == "REPLACE_LINES":
            if self.start_line is None:
                raise ValueError("REPLACE_LINES requires start_line.")
            if self.end_line is None:
                raise ValueError("REPLACE_LINES requires end_line.")
            if self.new_code_block is None:
                raise ValueError("REPLACE_LINES requires new_code_block.")
        return self


class TestResult(BaseModel):
    """Per-test execution outcome."""

    test_name: str
    passed: bool
    error_message: Optional[str] = None


class CodeObservation(Observation):
    """Full observation returned after each step."""

    code_lines: List[str] = Field(default_factory=list)
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
        if not self.code_lines:
            return "<empty>"
        return "\n".join(
            f"{idx + 1:>3} | {line}" for idx, line in enumerate(self.code_lines)
        )
