"""Task graders for TraceFix-RL.

The online validator expects importable grader callables for each task entry.
These graders execute the real task tests against the final code state so the
judge can verify actual solution quality instead of a canned lookup.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Optional

from core.sandbox import run_code_with_tests
from tasks.tasks import ALL_TASKS


MIN_SCORE = 0.01
MAX_SCORE = 0.98

def _clamp(score: float) -> float:
    return round(min(max(score, MIN_SCORE), MAX_SCORE), 4)


def _as_mapping(value: Any) -> Optional[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        return value
    if hasattr(value, "model_dump"):
        try:
            dumped = value.model_dump()
        except Exception:
            return None
        if isinstance(dumped, Mapping):
            return dumped
    if hasattr(value, "dict"):
        try:
            dumped = value.dict()
        except Exception:
            return None
        if isinstance(dumped, Mapping):
            return dumped
    return None


def _find_score_value(payload: Any) -> Optional[float]:
    mapping = _as_mapping(payload)
    if mapping is not None:
        for key in ("final_score", "grader_score", "score", "reward", "total_reward"):
            value = mapping.get(key)
            if isinstance(value, (int, float)):
                return float(value)
        for nested_key in ("metadata", "info", "observation", "state"):
            nested_value = mapping.get(nested_key)
            nested_score = _find_score_value(nested_value)
            if nested_score is not None:
                return nested_score
        return None

    for attr in ("final_score", "grader_score", "score", "reward", "total_reward"):
        if hasattr(payload, attr):
            value = getattr(payload, attr)
            if isinstance(value, (int, float)):
                return float(value)

    for attr in ("metadata", "info", "observation", "state"):
        if hasattr(payload, attr):
            nested_score = _find_score_value(getattr(payload, attr))
            if nested_score is not None:
                return nested_score

    return None


def _find_task(task_name: str) -> Optional[dict[str, Any]]:
    for task in ALL_TASKS:
        if task.get("name") == task_name:
            return task
    return None


def _extract_final_observation(payload: Any) -> Any:
    if payload is None:
        return None

    mapping = _as_mapping(payload)
    if mapping is not None:
        for key in ("final_observation", "observation", "state", "last_observation"):
            if key in mapping:
                candidate = mapping.get(key)
                if candidate is not None:
                    nested = _extract_final_observation(candidate)
                    if nested is not None:
                        return nested
        if "trajectory" in mapping:
            return _extract_final_observation(mapping.get("trajectory"))
        return payload

    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        if not payload:
            return None
        last_item = payload[-1]
        if isinstance(last_item, Sequence) and not isinstance(last_item, (str, bytes, bytearray)) and len(last_item) >= 2:
            return _extract_final_observation(last_item[1])
        if isinstance(last_item, Mapping) or hasattr(last_item, "model_dump") or hasattr(last_item, "dict"):
            return _extract_final_observation(last_item)
        return last_item

    return payload


def _observation_to_source(observation: Any) -> Optional[str]:
    if observation is None:
        return None

    mapping = _as_mapping(observation)
    if mapping is not None:
        source = mapping.get("source")
        if isinstance(source, str) and source.strip():
            return source

        code_lines = mapping.get("code_lines") or mapping.get("code")
        if isinstance(code_lines, Sequence) and not isinstance(code_lines, (str, bytes, bytearray)):
            lines = [str(line) for line in code_lines]
            return "\n".join(lines)

        code_dict = mapping.get("code_dict")
        if isinstance(code_dict, Mapping) and code_dict:
            ordered_lines: list[tuple[int, str]] = []
            for key, value in code_dict.items():
                try:
                    line_no = int(key)
                except Exception:
                    continue
                ordered_lines.append((line_no, str(value)))
            if ordered_lines:
                ordered_lines.sort(key=lambda item: item[0])
                return "\n".join(line for _, line in ordered_lines)

    for attr in ("source", "code", "code_lines", "code_dict"):
        if hasattr(observation, attr):
            value = getattr(observation, attr)
            if isinstance(value, str) and value.strip():
                return value
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
                return "\n".join(str(line) for line in value)
            if isinstance(value, Mapping) and value:
                ordered_lines = []
                for key, line in value.items():
                    try:
                        ordered_lines.append((int(key), str(line)))
                    except Exception:
                        continue
                if ordered_lines:
                    ordered_lines.sort(key=lambda item: item[0])
                    return "\n".join(line for _, line in ordered_lines)

    return None


def _evaluate_task(task_name: str, payload: Any) -> float:
    task = _find_task(task_name)
    if task is None:
        return MIN_SCORE

    final_observation = _extract_final_observation(payload)
    source = _observation_to_source(final_observation)
    if not source or not source.strip():
        return MIN_SCORE

    try:
        _, results, syntax_err = run_code_with_tests(
            source=source,
            test_callables=task["tests"],
        )
    except Exception:
        return MIN_SCORE

    if syntax_err:
        return MIN_SCORE

    if results and all(test_result.passed for test_result in results):
        return MAX_SCORE

    return MIN_SCORE


def grade(payload: Any = None, *args: Any, task_name: str = "", **kwargs: Any) -> float:
    """Execute the task's real tests against the final code state."""

    if payload is None and args:
        payload = args[0]

    if not task_name:
        task_name = str(kwargs.get("task_id") or kwargs.get("name") or "")

    if task_name:
        active_payload = payload if payload is not None else kwargs
        return _evaluate_task(task_name, active_payload)

    return MIN_SCORE


def grade_valid_parentheses_wrong_mapping(*args: Any, **kwargs: Any) -> float:
    task_kwargs = dict(kwargs)
    task_kwargs["task_name"] = "valid_parentheses_wrong_mapping"
    return grade(*args, **task_kwargs)


def grade_binary_search_off_by_one(*args: Any, **kwargs: Any) -> float:
    task_kwargs = dict(kwargs)
    task_kwargs["task_name"] = "binary_search_off_by_one"
    return grade(*args, **task_kwargs)


def grade_reverse_string_returns_original(*args: Any, **kwargs: Any) -> float:
    task_kwargs = dict(kwargs)
    task_kwargs["task_name"] = "reverse_string_returns_original"
    return grade(*args, **task_kwargs)


__all__ = [
    "grade",
    "grade_valid_parentheses_wrong_mapping",
    "grade_binary_search_off_by_one",
    "grade_reverse_string_returns_original",
]