"""Pydantic schema layer for TraceFix-RL."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from openenv.core.env_server.types import Action, Observation
from pydantic import BaseModel, ConfigDict, Field


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

    model_config = ConfigDict(strict=True)

    thought: str = Field(
        ...,
        description=(
            "MANDATORY. Analyze the localized_context and last_execution_output. "
            "If tests failed, identify the error line and root cause. Explicitly plan "
            "your next action before executing it."
        ),
    )
    action_type: ActionType = Field(
        ...,
        description=(
            "The specific tool to use. VIEW_CODE to read. RUN_TESTS to execute and get "
            "tracebacks. REPLACE_LINES to apply a fix. UNDO_EDIT to revert your last "
            "change if it failed. RESET_TO_ORIGINAL reverts the entire codebase back "
            "to its initial state; use this as a last resort if your edits have "
            "severely corrupted the code and UNDO_EDIT is not enough. SUBMIT only "
            "when all tests pass."
        ),
    )
    start_line: Optional[int] = Field(
        default=None,
        description=(
            "The inclusive start line number for REPLACE_LINES. You MUST use the exact "
            "integer keys provided in the code_dict observation."
        ),
    )
    end_line: Optional[int] = Field(
        default=None,
        description=(
            "The inclusive end line number for REPLACE_LINES. You MUST use the exact "
            "integer keys provided in the code_dict observation."
        ),
    )
    new_code_block: Optional[str] = Field(
        default=None,
        description=(
            "The exact replacement Python code. Must be properly indented to match the "
            "surrounding code. Do not include markdown formatting or backticks."
        ),
    )


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
