"""
Inference script for SWE-Gym - Software Engineer Gym.

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

import asyncio
import json
import os
import re
from typing import Any

from openai import OpenAI

try:
    from swe_gym import CodeAction, SWEGymEnv
except ImportError:
    from client import SWEGymEnv
    from models import CodeAction


API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN", "")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME", "")

ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860")
TASK_NAME = os.getenv("TASK_NAME", "swe_gym")
BENCHMARK = os.getenv("BENCHMARK", "swe_gym")
MAX_STEPS = int(os.getenv("MAX_STEPS", "50"))
SUCCESS_SCORE_THRESHOLD = float(os.getenv("SUCCESS_SCORE_THRESHOLD", "0.99"))

SYSTEM_PROMPT = (
    "You are controlling a Python debugging RL environment. "
    "Return only JSON for one action.\n"
    'Allowed action_type values: VIEW_CODE, RUN_TESTS, REPLACE_LINES, UNDO_EDIT, RESET_TO_ORIGINAL, SUBMIT.\n'
    "For REPLACE_LINES include start_line, end_line, new_code_block.\n"
    "Prefer RUN_TESTS after edits and SUBMIT only when all tests pass."
)


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: str | None) -> None:
    error_value = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_value}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


def _extract_json(text: str) -> dict[str, Any]:
    stripped = text.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    fence = re.search(r"```(?:json)?\s*({.*?})\s*```", stripped, re.DOTALL)
    if fence:
        try:
            return json.loads(fence.group(1))
        except json.JSONDecodeError:
            pass

    block = re.search(r"({.*?})", stripped, re.DOTALL)
    if block:
        try:
            return json.loads(block.group(1))
        except json.JSONDecodeError:
            pass

    return {"action_type": "RUN_TESTS"}


def _build_observation_text(observation: Any) -> str:
    code_preview = "\n".join(observation.code_lines[:30]) if observation.code_lines else ""
    return (
        f"step_count={observation.step_count}\n"
        f"steps_remaining={observation.steps_remaining}\n"
        f"syntax_error={observation.syntax_error}\n"
        f"localized_context=\n{observation.localized_context}\n\n"
        f"last_execution_output=\n{observation.last_execution_output}\n\n"
        f"code_preview=\n{code_preview}"
    )


def _get_model_action(client: OpenAI, observation: Any, history: list[str]) -> dict[str, Any]:
    obs_text = _build_observation_text(observation)
    user_prompt = (
        "Pick the single best next action and return only JSON.\n\n"
        f"{obs_text}\n\n"
        f"history:\n{chr(10).join(history[-5:]) if history else 'none'}"
    )
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=300,
            stream=False,
        )
        response_text = (completion.choices[0].message.content or "").strip()
        action = _extract_json(response_text)
    except Exception:
        action = {"action_type": "RUN_TESTS"}

    if action.get("action_type") not in {
        "VIEW_CODE",
        "RUN_TESTS",
        "REPLACE_LINES",
        "UNDO_EDIT",
        "RESET_TO_ORIGINAL",
        "SUBMIT",
    }:
        action = {"action_type": "RUN_TESTS"}

    return action


def _to_code_action(action_dict: dict[str, Any]) -> CodeAction:
    payload = {
        "action_type": action_dict.get("action_type", "RUN_TESTS"),
        "thought": action_dict.get("thought"),
        "start_line": action_dict.get("start_line"),
        "end_line": action_dict.get("end_line"),
        "new_code_block": action_dict.get("new_code_block"),
    }
    try:
        return CodeAction(**payload)
    except Exception:
        return CodeAction(action_type="RUN_TESTS")


def _compute_score(step_result: Any, rewards: list[float]) -> float:
    meta = step_result.observation.metadata or {}
    raw = meta.get("final_score")
    if raw is None:
        info = step_result.observation.info or {}
        raw = info.get("final_score")
    if raw is None:
        raw = sum(rewards)
    return max(0.0, min(1.0, float(raw)))


async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    env: SWEGymEnv | None = None
    rewards: list[float] = []
    history: list[str] = []
    steps_taken = 0
    score = 0.0
    success = False
    started = False

    try:
        if LOCAL_IMAGE_NAME:
            env = await SWEGymEnv.from_docker_image(LOCAL_IMAGE_NAME)
        else:
            env = SWEGymEnv(base_url=ENV_BASE_URL)

        result = await env.reset()
        task_name = result.observation.info.get("task_name") or TASK_NAME
        log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)
        started = True

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            action_dict = _get_model_action(client, result.observation, history)
            action = _to_code_action(action_dict)
            result = await env.step(action)

            reward = float(result.reward or 0.0)
            done = bool(result.done)
            action_str = action.action_type

            obs_meta = result.observation.metadata or {}
            error = obs_meta.get("last_action_error")
            if error is not None:
                error = str(error).replace("\n", " ")

            rewards.append(reward)
            steps_taken = step
            history.append(f"step={step} action={action_str} reward={reward:.2f}")
            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

            if done:
                break

        score = _compute_score(result, rewards)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        if not started:
            log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)
            started = True
        msg = str(exc).replace("\n", " ")
        if steps_taken == 0:
            log_step(step=1, action="RUN_TESTS", reward=0.0, done=False, error=msg)
            steps_taken = 1
            rewards.append(0.0)
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
    asyncio.run(main())
