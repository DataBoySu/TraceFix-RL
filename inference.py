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
import sys
from pathlib import Path
from typing import Any, Optional

from openai import OpenAI

try:
    from tracefix_rl import CodeAction, TraceFixRLEnv
except Exception:
    ROOT_DIR = Path(__file__).resolve().parent
    if str(ROOT_DIR) not in sys.path:
        sys.path.insert(0, str(ROOT_DIR))
    from client import TraceFixRLEnv
    from models import CodeAction


API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:1234/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "nvidia/nemotron-3-nano-4b")
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or "lm-studio"
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860")
TASK_NAME = os.getenv("TASK_NAME", "tracefix_rl")
BENCHMARK = os.getenv("BENCHMARK", "tracefix_rl")
MAX_STEPS = int(os.getenv("MAX_STEPS", "50"))
SUCCESS_SCORE_THRESHOLD = float(os.getenv("SUCCESS_SCORE_THRESHOLD", "0.99"))

SYSTEM_PROMPT = """\
You are a debugging policy agent. Output exactly one CodeAction JSON object per turn.

Use Action Trajectory on every turn. If an action repeats without progress, change strategy.
PARSE_ERROR means your previous output was invalid; fix formatting immediately.

Mandatory thought format (exactly 3 sentences):
Observation: what you see in localized_context or last_execution_output.
Diagnosis: root cause and exact line(s) to change.
Plan: the next action_type and why.

Action policy:
- VIEW_CODE to inspect full line mapping.
- RUN_TESTS to get fresh traceback evidence.
- REPLACE_LINES for focused fixes using exact code_dict keys.
- UNDO_EDIT if the latest edit made things worse.
- RESET_TO_ORIGINAL as last-resort recovery.
- SUBMIT ONLY when last_execution_output explicitly contains the success signal that all tests passed.

Return only JSON keys: thought, action_type, start_line, end_line, new_code_block.
No markdown. No extra keys.
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


def _coerce_legacy_action_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """
    Normalize common legacy output shapes into strict CodeAction fields.

    This keeps runtime resilient across weaker model backends while still
    validating the final payload with strict Pydantic rules.
    """
    normalized = dict(payload)

    if "action_type" not in normalized and isinstance(normalized.get("type"), str):
        normalized["action_type"] = normalized["type"]

    if "thought" not in normalized or normalized.get("thought") in (None, ""):
        normalized["thought"] = (
            "Recovered malformed action payload and mapped legacy fields "
            "to strict CodeAction schema."
        )

    if "lines" in normalized and isinstance(normalized["lines"], list):
        line_items = []
        for item in normalized["lines"]:
            if not isinstance(item, dict):
                continue
            line_no = item.get("line")
            code_text = item.get("code")
            if isinstance(line_no, int) and isinstance(code_text, str):
                line_items.append((line_no, code_text))
        if line_items:
            line_items.sort(key=lambda x: x[0])
            if "start_line" not in normalized:
                normalized["start_line"] = line_items[0][0]
            if "end_line" not in normalized:
                normalized["end_line"] = line_items[-1][0]
            if "new_code_block" not in normalized:
                normalized["new_code_block"] = "\n".join(code for _, code in line_items)

    normalized.pop("type", None)
    normalized.pop("lines", None)
    normalized.pop("source", None)
    return normalized


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


def _build_observation_text(observation: Any) -> str:
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
        f"localized_context=\n{observation.localized_context}\n\n"
        f"last_execution_output=\n{observation.last_execution_output}\n\n"
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

        action = CodeAction.model_validate(parsed)
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
            parsed_dict = _coerce_legacy_action_payload(parsed_dict)
            action = CodeAction.model_validate(parsed_dict)
            assistant_json = action.model_dump_json(exclude_none=False)
            return action, assistant_json
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
                history_messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Pick the single best next action and return only a CodeAction JSON object.\n\n"
                            f"action_trajectory={(' -> '.join(action_trajectory) if action_trajectory else 'none')}\n\n"
                            f"{obs_text}"
                        ),
                    }
                )
                try:
                    action, assistant_json = _get_model_action(client, history_messages)
                    history_messages.append({"role": "assistant", "content": assistant_json})
                    if show_thought:
                        _print_thought(action, assistant_json)
                except ModelParseError as exc:
                    cause = str(exc).replace("\n", " ")
                    parse_error_note = cause
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
                    action = CodeAction(
                        action_type="RUN_TESTS",
                        thought=(
                            "PARSE_ERROR recovery step: run tests so the step is explicit and "
                            "collect fresh traceback context for the next valid action."
                        ),
                    )

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
