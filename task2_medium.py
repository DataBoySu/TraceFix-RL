"""Submission task grader for binary_search_off_by_one."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Optional

from core.sandbox import run_code_with_tests
from tasks.tasks import TASK_BS_OFF_BY_ONE


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
			return "\n".join(str(line) for line in code_lines)

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


def grade(payload: Any = None, *args: Any, **kwargs: Any) -> float:
	final_payload = payload if payload is not None else (args[0] if args else kwargs)
	final_observation = _extract_final_observation(final_payload)
	source = _observation_to_source(final_observation)
	if not source or not source.strip():
		return MIN_SCORE

	try:
		_, results, syntax_err = run_code_with_tests(
			source=source,
			test_callables=TASK_BS_OFF_BY_ONE["tests"],
		)
	except Exception:
		return MIN_SCORE

	if syntax_err:
		return MIN_SCORE
	if results and all(test_result.passed for test_result in results):
		return MAX_SCORE
	return MIN_SCORE


__all__ = ["grade"]