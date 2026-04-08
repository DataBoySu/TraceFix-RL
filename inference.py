"""
Inference script for TraceFix-RL.

Mandatory env vars expected in deployment config:
  API_BASE_URL
  MODEL_NAME
  HF_TOKEN
  LOCAL_IMAGE_NAME  (required if using MyEnv.from_docker_image)

This script prints exactly:
  [START] ...
  [STEP] ...
  [END] ...
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Optional

from openai import OpenAI
from pydantic import ValidationError

try:
    from tracefix_rl import CodeAction, TraceFixRLEnv
except Exception:
    ROOT_DIR = Path(__file__).resolve().parent
    if str(ROOT_DIR) not in sys.path:
        sys.path.insert(0, str(ROOT_DIR))
    from client import TraceFixRLEnv
    from models import CodeAction


API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:1234/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "openai/gpt-oss-20b")
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or "lm-studio"
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860")
TASK_NAME = os.getenv("TASK_NAME", "tracefix_rl")
BENCHMARK = os.getenv("BENCHMARK", "tracefix_rl")
MAX_STEPS = int(os.getenv("MAX_STEPS", "50"))
SUCCESS_SCORE_THRESHOLD = float(os.getenv("SUCCESS_SCORE_THRESHOLD", "0.99"))

SYSTEM_PROMPT = """\
You are a deterministic debugging policy agent.
You must output exactly one valid CodeAction JSON object per turn and nothing else.

Primary failures to avoid:
1) Invalid JSON or wrong field types.
2) Misreading last_execution_output and submitting before tests are truly passing.

Output contract (strict):
- Return a single JSON object, not an array.
- Allowed keys only: thought, action_type, start_line, end_line, new_code_block.
- No markdown, no code fences, no commentary outside JSON, no extra keys.
- thought must be a plain string.
- action_type must be one of: VIEW_CODE, RUN_TESTS, REPLACE_LINES, UNDO_EDIT, RESET_TO_ORIGINAL, SUBMIT.
- start_line and end_line must be integer or null.
- new_code_block must be string or null.
- If action_type is not REPLACE_LINES, set start_line=null, end_line=null, new_code_block=null.
- If action_type is REPLACE_LINES, set start_line and end_line to exact integer keys from code_dict and provide new_code_block as replacement code only.

Mandatory thought format:
Observation: summarize concrete evidence from localized_context and/or last_execution_output.
Diagnosis: identify the most likely root cause and exact line numbers to edit when applicable.
Plan: choose the next action_type and justify it briefly.

How to read last_execution_output correctly:
- Prefer traceback and assertion text over assumptions.
- Extract failing test name, exception type, file path, and line number when present.
- If output is truncated or ambiguous, run RUN_TESTS before editing.
- Treat syntax errors as highest priority and fix them before semantic issues.
- Never claim success unless output clearly indicates complete pass status.

Terminal decision rule (no waiting):
- If last_execution_output contains both a full pass count pattern (for example, "Tests Passed: N/N")
    and the success marker "SUCCESS: ALL TESTS PASSED", the next action must be SUBMIT.
- If all_tests_pass_signal=true in the observation, the next action must be SUBMIT.
- Once this pass signal is present, RUN_TESTS is no longer a valid next action.
- Do not wait for extra confirmation, additional logs, or another RUN_TESTS cycle after this signal.

Action policy:
- VIEW_CODE when line mapping or surrounding context is insufficient.
- RUN_TESTS to collect fresh evidence after edits or when uncertain.
- REPLACE_LINES for minimal, line-accurate fixes tied to observed failures.
- UNDO_EDIT if latest change worsened results or introduced new failures.
- RESET_TO_ORIGINAL only as last-resort recovery.
- SUBMIT only when last_execution_output explicitly and unambiguously indicates all tests passed.
- After RUN_TESTS, do not choose RUN_TESTS again immediately unless test evidence is genuinely missing.
- Treat "no output" as invalid reasoning when pass_count_summary or traceback text is present.

Submit gate (hard rule):
- If any failure, error, traceback, xfailed/unfinished signal, or uncertainty remains, do not SUBMIT.
- If all-tests-passed signal is present, do SUBMIT immediately on this turn.

Self-check before finalizing response:
- Is this valid JSON?
- Are all values schema-valid primitive types?
- Are nulls set correctly for non-REPLACE_LINES actions?
- Does the thought have exactly 3 sentences in the required Observation/Diagnosis/Plan structure?
"""


class ModelParseError(Exception):
    """Raised when model output cannot be parsed into CodeAction."""

    def __init__(self, message: str, raw_response: str = "") -> None:
        super().__init__(message)
        self.raw_response = raw_response


def _decode_action_json(raw_text: str) -> dict[str, Any]:
    stripped = raw_text.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        first_newline = stripped.find("\n")
        if first_newline == -1:
            raise ValueError("Invalid fenced JSON response.")
        stripped = stripped[first_newline + 1 : -3].strip()
    return json.loads(stripped)


def _clean_validation_error(exc: ValidationError) -> str:
    """Return a concise, user-facing schema violation summary."""
    first_error = exc.errors()[0] if exc.errors() else {}
    loc = first_error.get("loc", ["Unknown"])
    field_name = loc[0] if isinstance(loc, (list, tuple)) and loc else "Unknown"
    return (
        f"JSON Schema Violation on field '{field_name}': Must be a flat string/integer. "
        "Do not use nested objects or arrays."
    )


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_value = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_value}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


def _extract_pass_signal_fields(last_execution_output: str) -> tuple[str, bool]:
    pass_count_match = re.search(r"Tests Passed:\s*(\d+)\s*/\s*(\d+)", last_execution_output)
    pass_count_text = pass_count_match.group(0) if pass_count_match else "unknown"
    all_tests_pass_signal = (
        ("SUCCESS: ALL TESTS PASSED" in last_execution_output)
        and bool(pass_count_match)
        and (pass_count_match.group(1) == pass_count_match.group(2))
    )
    return pass_count_text, all_tests_pass_signal


def _build_observation_text(observation: Any) -> str:
    last_execution_output = str(getattr(observation, "last_execution_output", "") or "")
    pass_count_text, all_tests_pass_signal = _extract_pass_signal_fields(last_execution_output)

    code_dict = getattr(observation, "code_dict", {}) or {}
    sorted_items = sorted(
        ((int(line_num), text) for line_num, text in code_dict.items()),
        key=lambda x: x[0],
    )
    code_preview = "\n".join(
        f"{line_num} | {text}"
        for line_num, text in sorted_items[:30]
    )
    return (
        f"step_count={observation.step_count}\n"
        f"steps_remaining={observation.steps_remaining}\n"
        f"syntax_error={observation.syntax_error}\n"
        f"pass_count_summary={pass_count_text}\n"
        f"all_tests_pass_signal={str(all_tests_pass_signal).lower()}\n"
        f"localized_context=\n{observation.localized_context}\n\n"
        f"last_execution_output=\n{last_execution_output}\n\n"
        f"code_preview=\n{code_preview}"
    )


def _get_model_action(
    client: OpenAI,
    history_messages: list[dict[str, str]],
) -> tuple[CodeAction, str]:
    request_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history_messages
    try:
        completion = client.beta.chat.completions.parse(
            model=MODEL_NAME,
            messages=request_messages,
            temperature=0.0,
            response_format=CodeAction,
        )
        message = completion.choices[0].message
        refusal_text = getattr(message, "refusal", None)
        if refusal_text:
            raise ModelParseError(f"Model refusal: {refusal_text}", raw_response=str(refusal_text))

        parsed = getattr(message, "parsed", None)
        if parsed is None:
            content = getattr(message, "content", "")
            if isinstance(content, str):
                raw_response = content
            else:
                raw_response = json.dumps(content, ensure_ascii=True, default=str)
            raise ModelParseError(
                "Model response was not parsed into CodeAction.",
                raw_response=raw_response,
            )

        try:
            action = CodeAction.model_validate(parsed)
        except ValidationError as exc:
            content = getattr(message, "content", "")
            raw_response = content if isinstance(content, str) else json.dumps(content, ensure_ascii=True, default=str)
            raise ModelParseError(_clean_validation_error(exc), raw_response=raw_response) from exc
        assistant_json = action.model_dump_json(exclude_none=False)
        return action, assistant_json
    except Exception as parse_exc:
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=request_messages,
                temperature=0.0,
                stream=False,
            )
            raw_text = (completion.choices[0].message.content or "").strip()
            parsed_dict = _decode_action_json(raw_text)
            try:
                action = CodeAction.model_validate(parsed_dict)
            except ValidationError as exc:
                raise ModelParseError(_clean_validation_error(exc), raw_response=raw_text) from exc
            assistant_json = action.model_dump_json(exclude_none=False)
            return action, assistant_json
        except ModelParseError:
            raise
        except Exception as fallback_exc:
            raise ModelParseError(
                (
                    f"Model parse call failed: {str(parse_exc).strip()} | "
                    f"fallback create path failed: {str(fallback_exc).strip()}"
                )
            ) from fallback_exc


def _print_thought(action: CodeAction, raw_response: str) -> None:
    thought_text = (action.thought or "").strip()
    print("[THOUGHT]", file=sys.stderr, flush=True)
    print(thought_text if thought_text else raw_response, file=sys.stderr, flush=True)


def _compute_score(step_result: Any, rewards: list[float]) -> float:
    meta = step_result.observation.metadata or {}
    raw = meta.get("final_score")
    if raw is None:
        info = step_result.observation.info or {}
        raw = info.get("final_score")
    if raw is None:
        raw = sum(rewards)
    return max(0.0, min(1.0, float(raw)))


async def run(difficulty: Optional[str] = None, show_thought: bool = False) -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    env: Optional[TraceFixRLEnv] = None
    rewards: list[float] = []
    history: list[str] = []
    history_messages: list[dict[str, str]] = []
    action_trajectory: list[str] = []
    steps_taken = 0
    score = 0.0
    success = False
    started = False
    kill_switch_triggered = False
    last_action_type: Optional[str] = None
    consecutive_same_action_count = 0
    consecutive_parse_error_count = 0

    try:
        if LOCAL_IMAGE_NAME:
            env = await TraceFixRLEnv.from_docker_image(LOCAL_IMAGE_NAME)
        else:
            env = TraceFixRLEnv(base_url=ENV_BASE_URL)

        if difficulty:
            reset_kwargs = {"difficulty": difficulty}
            result = await env.reset(**reset_kwargs)
        else:
            result = await env.reset()
        task_name = result.observation.info.get("task_name") or TASK_NAME
        log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)
        started = True

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            action: Optional[CodeAction] = None
            parse_error_note: Optional[str] = None
            if step == 1:
                action = CodeAction(
                    action_type="VIEW_CODE",
                    thought="First step policy: inspect source before testing or editing.",
                )
                if show_thought:
                    print("[THOUGHT]", file=sys.stderr, flush=True)
                    print(action.thought, file=sys.stderr, flush=True)
            else:
                obs_text = _build_observation_text(result.observation)
                obs_last_output = str(getattr(result.observation, "last_execution_output", "") or "")
                pass_count_text, all_tests_pass_signal = _extract_pass_signal_fields(obs_last_output)
                last_action = action_trajectory[-1] if action_trajectory else "none"
                history_messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Pick the single best next action and return only one valid CodeAction JSON object. "
                            "Use localized_context/last_execution_output as evidence, and do not SUBMIT unless all tests explicitly pass. "
                            "If all_tests_pass_signal=true, you must choose SUBMIT now and must not choose RUN_TESTS again. "
                            "Do not wait for additional test output when all_tests_pass_signal=true. "
                            "If last_action was RUN_TESTS and all_tests_pass_signal=false, choose REPLACE_LINES or VIEW_CODE next, not RUN_TESTS again.\n\n"
                            f"decision_guard: last_action={last_action}, pass_count_summary={pass_count_text}, all_tests_pass_signal={str(all_tests_pass_signal).lower()}\n\n"
                            f"action_trajectory={(' -> '.join(action_trajectory) if action_trajectory else 'none')}\n\n"
                            f"{obs_text}"
                        ),
                    }
                )
                try:
                    action, assistant_json = _get_model_action(client, history_messages)
                    consecutive_parse_error_count = 0
                    history_messages.append({"role": "assistant", "content": assistant_json})
                    if show_thought:
                        _print_thought(action, assistant_json)
                except ModelParseError as exc:
                    cause = str(exc).replace("\n", " ")
                    parse_error_note = cause
                    consecutive_parse_error_count += 1
                    raw_response = (exc.raw_response or "").strip()
                    if raw_response:
                        history_messages.append({"role": "assistant", "content": raw_response})
                    history_messages.append(
                        {
                            "role": "user",
                            "content": (
                                f"PARSE_ERROR: {cause}. "
                                "Return one valid CodeAction object only. "
                                "Include thought and ensure strict field types."
                            ),
                        }
                    )
                    history.append(f"PARSE_ERROR: {cause}")
                    if consecutive_parse_error_count >= 3:
                        kill_switch_triggered = True
                        history.append(
                            "KILL_SWITCH: PARSE_ERROR occurred 3 times consecutively. "
                            "Terminating episode early to prevent token burn."
                        )
                        steps_taken = step
                        success = False
                        score = 0.0
                        break
                    action = CodeAction(
                        action_type="RUN_TESTS",
                        thought=(
                            "PARSE_ERROR recovery step: run tests so the step is explicit and "
                            "collect fresh traceback context for the next valid action."
                        ),
                    )

            if kill_switch_triggered:
                break

            current_action_type = action.action_type
            if current_action_type == last_action_type:
                consecutive_same_action_count += 1
            else:
                consecutive_same_action_count = 1
                last_action_type = current_action_type

            if (
                current_action_type == "RUN_TESTS"
                and consecutive_same_action_count >= 3
            ):
                kill_switch_triggered = True
                history.append(
                    "KILL_SWITCH: RUN_TESTS selected 3 times consecutively. "
                    "Terminating episode early to prevent looping."
                )
                steps_taken = step
                success = False
                score = 0.0
                break

            result = await env.step(action)

            reward = float(result.reward or 0.0)
            done = bool(result.done)
            action_str = action.action_type

            obs_meta = result.observation.metadata or {}
            error = obs_meta.get("last_action_error")
            if error is not None:
                error = str(error).replace("\n", " ")
            if parse_error_note:
                error = f"PARSE_ERROR: {parse_error_note}"

            rewards.append(reward)
            steps_taken = step
            action_thought = (action.thought or "").strip()
            history.append(
                f"Action {action_str}; reward {reward:.2f}; error {error or 'null'}."
                + (f" Thought: {action_thought}" if action_thought else "")
            )
            action_trajectory.append(action_str)
            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

            if done:
                break

        if not kill_switch_triggered:
            score = _compute_score(result, rewards)
            success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        if not started:
            log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)
            started = True
        score = 0.0
        success = False
    finally:
        if env is not None:
            try:
                await env.close()
            except Exception:
                pass
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run TraceFix-RL inference baseline.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--easy", action="store_true", help="Run on easy curriculum tier.")
    group.add_argument("--medium", action="store_true", help="Run on medium curriculum tier.")
    group.add_argument("--hard", action="store_true", help="Run on hard curriculum tier.")
    parser.add_argument("--thought", action="store_true", help="Print LLM thought trace to stderr only.")
    args = parser.parse_args()

    difficulty: Optional[str] = None
    if args.easy:
        difficulty = "easy"
    elif args.medium:
        difficulty = "medium"
    elif args.hard:
        difficulty = "hard"

    asyncio.run(run(difficulty=difficulty, show_thought=args.thought))
