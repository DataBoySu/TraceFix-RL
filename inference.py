"""
inference.py — Baseline Agent for Python Debugging Gym
=======================================================
Hackathon-compliant baseline script.  Connects to the PythonDebuggingGym
WebSocket server and drives an OpenAI-compatible LLM to find and fix bugs.

Required environment variables:
  HF_TOKEN       API key / HuggingFace token passed as Bearer auth
  MODEL_NAME     Model identifier             (default: nvidia/nemotron-3-nano-4b)
  API_BASE_URL   OpenAI-compatible base URL   (default: https://api.openai.com/v1)

Optional environment variables:
  ENV_WS_URL     WebSocket URL for the gym    (default: ws://localhost:8000/ws)

Mandatory stdout log lines (zero deviation in spacing or formatting):
  [START] task=<task_name> env=PythonDebuggingGym model=<model_name>
  [STEP]  step=<n> action=<action_type> reward=<r.rr> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<s.sss> rewards=<r1,r2,...,rn>
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from typing import Any

import websockets
from openai import OpenAI


# ---------------------------------------------------------------------------
# Config  (all readable from environment at import time)
# ---------------------------------------------------------------------------

API_BASE_URL: str = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME:   str = os.getenv("MODEL_NAME",   "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4")
HF_TOKEN:     str = os.getenv("HF_TOKEN",     "")
ENV_WS_URL:   str = os.getenv("ENV_WS_URL",   "ws://localhost:8000/ws")

# ---------------------------------------------------------------------------
# OpenAI client
# ---------------------------------------------------------------------------

_client = OpenAI(
    api_key=HF_TOKEN or "sk-placeholder",   # placeholder keeps the client from raising at init
    base_url=API_BASE_URL,
)

# ---------------------------------------------------------------------------
# Agent instruction appended after the environment's own system prompt
# ---------------------------------------------------------------------------

_AGENT_SUFFIX = """\

=======================================================================
RESPONSE FORMAT (MANDATORY)
=======================================================================
Respond with ONLY a valid JSON object. No markdown, no code fences,
no explanation text — just the raw JSON.

Valid action schemas (choose exactly one per turn):
  {"action_type": "VIEW_CODE"}
  {"action_type": "RUN_TESTS"}
  {"action_type": "REPLACE_LINES", "start_line": N, "end_line": M, "new_code_block": "line1\\nline2"}
  {"action_type": "UNDO_EDIT"}
  {"action_type": "RESET_TO_ORIGINAL"}
  {"action_type": "SUBMIT"}

Rules for REPLACE_LINES:
  - new_code_block: join multiple lines with \\n (literal backslash-n in the JSON string)
  - Include exact Python indentation (leading spaces) on every line
  - Do NOT include a trailing \\n character
  - After REPLACE_LINES, call VIEW_CODE to re-orient before the next edit

Rules for UNDO_EDIT / RESET_TO_ORIGINAL:
  - UNDO_EDIT reverts the last REPLACE_LINES. Use when an edit made things worse.
  - RESET_TO_ORIGINAL restores the original broken code. Last resort only.
  - Both cost -0.10. Prefer fixing forward over backtracking.
"""


# ---------------------------------------------------------------------------
# Observation formatter
# ---------------------------------------------------------------------------

def _format_obs(obs: dict[str, Any]) -> str:
    """Convert a CodeObservation dict into a compact string for the LLM."""
    parts: list[str] = []

    if obs.get("syntax_error"):
        parts.append("⚠ SYNTAX ERROR in current code — fix indentation/brackets first.\n")

    localized = obs.get("localized_context", "")
    if localized:
        parts.append(f"[Context around last edit]\n{localized}\n")

    last_out = obs.get("last_execution_output", "")
    if last_out:
        parts.append(f"[Last execution output]\n{last_out}\n")

    test_results: list[dict] = obs.get("test_results", [])
    if test_results:
        lines = []
        for t in test_results:
            status = "PASS" if t.get("passed") else "FAIL"
            msg    = t.get("error_message") or ""
            name   = t.get("test_name", "?")
            lines.append(f"  {status}  {name}" + (f": {msg}" if msg else ""))
        parts.append("[Test results]\n" + "\n".join(lines) + "\n")

    remaining = obs.get("steps_remaining", 0)
    parts.append(f"[Steps remaining: {remaining}]")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

_ACTION_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "CodeAction",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "thought": {
                    "type": "string",
                    "description": "Mandatory reasoning before selecting action_type.",
                },
                "action_type": {
                    "type": "string",
                    "enum": [
                        "VIEW_CODE", "RUN_TESTS", "REPLACE_LINES",
                        "UNDO_EDIT", "RESET_TO_ORIGINAL", "SUBMIT",
                    ],
                },
                "start_line":     {"type": ["integer", "null"]},
                "end_line":       {"type": ["integer", "null"]},
                "new_code_block": {"type": ["string",  "null"]},
            },
            "required": ["thought", "action_type"],
            "additionalProperties": False,
        },
    },
}


def _call_llm(system_prompt: str, messages: list[dict]) -> str:
    """
    Call the configured LLM and return the raw text reply.

    Tries json_schema structured output first (LM Studio / vLLM / newer
    llama.cpp all support this).  Falls back to a plain call if the backend
    raises an error for the response_format parameter — _extract_json()
    then handles extraction from free-form text.
    """
    base_kwargs: dict = dict(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt + _AGENT_SUFFIX},
            *messages,
        ],
        temperature=0.0,
    )
    try:
        response = _client.chat.completions.create(
            **base_kwargs,
            response_format=_ACTION_SCHEMA,
        )
    except Exception:
        # Backend doesn't support json_schema — fall back to free-form
        response = _client.chat.completions.create(**base_kwargs)
    
    msg = response.choices[0].message
    content = msg.content
    
    # Fallback for reasoning models (e.g., via LM Studio) that place their
    # entire output in the reasoning_content field instead of content.
    if not content:
        try:
            msg_dict = msg.model_dump()
            content = msg_dict.get("reasoning_content", "") or ""
        except AttributeError:
            pass
            
    return content or ""


# ---------------------------------------------------------------------------
# Constrained JSON extraction  (works with any local or cloud model)
# ---------------------------------------------------------------------------

def _extract_json(text: str) -> dict:
    """
    Best-effort JSON extraction from raw LLM output.

    Tries in order:
      1. Direct json.loads  (model produced clean JSON)
      2. Strip ```json ... ``` / ``` ... ``` markdown fences
      3. Regex: grab first {...} block in the text
      4. Safe fallback: {"action_type": "VIEW_CODE"}
    """
    import re

    # 1. Direct parse
    stripped = text.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    # 2. Markdown code fence  ```json\n{...}\n```
    fence = re.search(r"```(?:json)?\s*({.*?})\s*```", stripped, re.DOTALL)
    if fence:
        try:
            return json.loads(fence.group(1))
        except json.JSONDecodeError:
            pass

    # 3. First {...} block anywhere in the text
    brace = re.search(r"({.*?})", stripped, re.DOTALL)
    if brace:
        try:
            return json.loads(brace.group(1))
        except json.JSONDecodeError:
            pass

    # All extraction attempts failed.
    # Return an invalid action_type so Pydantic rejects it at the server,
    # the server returns an error envelope, and THAT error is fed back to
    # the LLM on the next turn — breaking the silent mask loop.
    # DO NOT default to VIEW_CODE here.
    return {"action_type": "PARSE_ERROR", "thought": f"Failed to parse LLM output as JSON: {text[:120]}"}


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

async def run_episode(difficulty: str = None, show_thought: bool = False) -> None:
    """
    Connect to the gym, run one full episode with an LLM agent,
    and emit the three required log lines.
    """
    rewards:      list[float] = []
    step:         int         = 0
    system_prompt: str        = ""
    task_name:    str         = "unknown"
    messages:     list[dict]  = []
    success:      bool        = False
    obs:          dict        = {}

    ws_url = ENV_WS_URL
    if difficulty:
        separator = "&" if "?" in ws_url else "?"
        ws_url = f"{ws_url}{separator}difficulty={difficulty}"

    async with websockets.connect(ws_url) as ws:

        # ── Receive initial observation + system prompt ──────────────────
        raw  = await ws.recv()
        data = json.loads(raw)

        system_prompt = data.get("info", {}).get("system_prompt", "")
        obs           = data.get("observation", {})
        task_name     = obs.get("info", {}).get("task_name", "unknown")

        # ── [START] log line ─────────────────────────────────────────────
        print(
            f"[START] task={task_name} env=PythonDebuggingGym model={MODEL_NAME}",
            flush=True,
        )

        # ── RL loop ──────────────────────────────────────────────────────
        while True:
            step     += 1
            error_str  = "null"
            action_type = "VIEW_CODE"   # will be overwritten by a real parse

            # Build observation message for the LLM
            obs_text = _format_obs(obs)
            messages.append({"role": "user", "content": obs_text})

            # Call LLM
            try:
                llm_reply   = _call_llm(system_prompt, messages)
                if os.getenv("DEBUG_LOG") == "1":
                    print(f"\n[DEBUG RAW LLM]: {llm_reply}\n", flush=True)  # see what model actually outputs
                action_json = _extract_json(llm_reply)
                action_type = action_json.get("action_type", "VIEW_CODE")
                messages.append({"role": "assistant", "content": llm_reply})
            except Exception as exc:
                # LLM call itself failed — surface error in log, do NOT mask as VIEW_CODE.
                # Send a harmless VIEW_CODE this turn but pass the error text back as
                # the next user message so the model sees what went wrong.
                error_str   = str(exc).replace("\n", " ")[:200]
                action_type = "VIEW_CODE"
                action_json = {"action_type": "VIEW_CODE"}
                messages.append({"role": "user", "content": f"[SYSTEM ERROR] {error_str}"})

            if show_thought:
                thought = action_json.get("thought", "")
                if thought:
                    print(f"\n[THOUGHT]: {thought}\n", flush=True)

            # Send action to the environment
            await ws.send(json.dumps({"action": action_json}))

            # Receive response
            raw  = await ws.recv()
            data = json.loads(raw)

            # Server may return a validation-error envelope (no "observation" key)
            if "observation" not in data:
                error_str = str(data.get("error", "server_error"))[:200]
                reward, done = 0.0, False
            else:
                reward = float(data.get("reward", 0.0))
                done   = bool(data.get("done", False))
                obs    = data.get("observation", {})

                if done:
                    test_results = obs.get("test_results", [])
                    total        = len(test_results)
                    passes       = sum(1 for t in test_results if t.get("passed"))
                    success      = (total > 0 and passes == total)

            rewards.append(reward)

            # ── [STEP] log line ──────────────────────────────────────────
            done_str = "true" if done else "false"
            print(
                f"[STEP] step={step} action={action_type} "
                f"reward={reward:.2f} done={done_str} error={error_str}",
                flush=True,
            )

            if done:
                break   # server will auto-reset, but we exit after one episode

    # ── [END] log line ───────────────────────────────────────────────────────
    success_str = "true" if success else "false"
    # Pull clamped final_score from info dict if available, else derive from rewards
    final_score = data.get("info", {}).get("final_score", None) if done else None
    if final_score is None:
        final_score = max(0.0, min(1.0, sum(rewards)))
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={success_str} steps={step} score={final_score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Run the Python debugging agent.")
    parser.add_argument("--easy", action="store_const", dest="difficulty", const="easy", help="Run an easy task.")
    parser.add_argument("--medium", action="store_const", dest="difficulty", const="medium", help="Run a medium task.")
    parser.add_argument("--hard", action="store_const", dest="difficulty", const="hard", help="Run a hard task.")
    parser.add_argument("--thought", action="store_true", dest="show_thought", help="Print the agent's chain-of-thought reasoning.")
    
    args = parser.parse_args()
    asyncio.run(run_episode(difficulty=args.difficulty, show_thought=args.show_thought))


if __name__ == "__main__":
    main()
