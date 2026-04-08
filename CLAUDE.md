# CLAUDE.md - TraceFix-RL (RL_ENV_FINAL)

Current, code-backed notes for assistants working in this repository.
Last updated: 2026-04-08

## Project Status Snapshot

- Repo: `code_reasoner_rl_env`
- Branch: `master`
- Working tree: dirty
  - Modified: `.gitignore`, `inference.py`, `models.py`, `__pycache__/models.cpython-312.pyc`
  - Untracked: `.hfignore`
- Last recorded pre-validation command in terminal:
  - `./pre-val.sh https://sus-human-tracefix-rl.hf.space .`
  - Exit code: `1`

This file describes the current implementation in `RL_ENV_FINAL` only.

## High-Level Architecture

- `environment.py`: core gym-style state machine (`TraceFixRLGym`)
- `server/tracefix_rl_environment.py`: OpenEnv adapter (`Environment` interface)
- `server/app.py`: FastAPI app creation and uvicorn entrypoint
- `models.py`: action/observation schemas (`CodeAction`, `CodeObservation`, `TestResult`)
- `sandbox.py`: isolated code execution + test running + timeout handling
- `tasks.py`: static task registry (easy/medium/hard)
- `context.py`: localized context windowing around last edit
- `client.py`: typed OpenEnv client (`TraceFixRLEnv` / `MyEnv`)
- `inference.py`: baseline agent runner with OpenAI-compatible API
- `openenv.yaml`: OpenEnv runtime metadata (`app: server.app:app`, `port: 7860`)

## Runtime and Entry Points

- Local server via project script:
  - `uv run --project . server`
- Container command in `Dockerfile`:
  - `uvicorn server.app:app --host 0.0.0.0 --port 7860`
- OpenEnv spec points to:
  - `server.app:app`

## Environment Behavior (`environment.py`)

Action space:

- `VIEW_CODE`
- `RUN_TESTS`
- `REPLACE_LINES`
- `UNDO_EDIT`
- `RESET_TO_ORIGINAL`
- `SUBMIT`

Reward constants currently defined:

- `R_STEP_COST = -0.01`
- `R_RUN_TESTS = +0.10`
- `R_PER_NEW_PASS = +0.05`
- `R_SYNTAX_ERROR = -0.10`
- `R_INVALID_LINE = -0.02`
- `R_DESTRUCTIVE_PENALTY = -0.20`
- `R_UNDO_RESET = -0.10`
- `MAX_STEPS = 50`

Episode internals include:

- code snapshotting (`_original_code`, `_edit_history`)
- anti-loop penalty for repeated identical `action_type`
- contextual anchor (`_last_edited_line`) for localized context
- cumulative step-cost tracking (`_accumulated_step_costs`)

Submit scoring model:

- `proportion = passing_tests / total_tests` (or `0` on syntax error)
- `raw_score = proportion - _accumulated_step_costs`
- `final_score = clamp(raw_score, 0.0, 1.0)`
- same clamp model used on max-step timeout auto-evaluation

Task sampling policy:

- `training_step == 0`: random from `ALL_TASKS`
- `< 1000`: easy
- `< 5000`: medium
- `>= 5000`: hard
- fallback to first non-empty bucket

## Schema Notes (`models.py`)

Important: current code uses Pydantic v2-style validation APIs.

- `CodeAction` uses `@model_validator(mode="before")`
- Non-`REPLACE_LINES` actions force `start_line`, `end_line`, `new_code_block` to `None`
- `REPLACE_LINES` enforces required fields and 1-indexed positive range constraints

This is not compatible with Pydantic v1-only assumptions.

## Sandbox Notes (`sandbox.py`)

`run_code_with_tests(...)` returns a strict 3-tuple:

- `output_str`
- `List[TestResult>`
- `had_syntax_error: bool`

Execution safeguards:

- subprocess isolation via `multiprocessing.Process`
- timeout terminate/kill path
- tail truncation (`MAX_OUTPUT_CHARS = 1000`)
- restricted builtins to block risky operations

## Tasks Registry (`tasks.py`)

- Static hardcoded registry grouped by difficulty
- Exports:
  - `TASKS_BY_DIFFICULTY`
  - `ALL_TASKS`
- Expected total currently: 16 tasks
  - easy: 4
  - medium: 6
  - hard: 6

## OpenEnv Adapter and Client

`server/tracefix_rl_environment.py`:

- Maps optional reset difficulty to `training_step` hints
- Writes `system_prompt` into observation metadata
- Sets observation reward/done from gym step output

`client.py`:

- Sends actions using `model_dump(exclude_none=True)`
- Parses OpenEnv payloads into typed `CodeObservation`

## Inference Runner (`inference.py`)

Key defaults:

- `API_BASE_URL = https://router.huggingface.co/v1`
- `MODEL_NAME = Qwen/Qwen2.5-72B-Instruct`
- `MAX_STEPS = 50`
- `SUCCESS_SCORE_THRESHOLD = 0.99`
- `THINKING_TOKEN_LIMIT = 512`

Behavior:

- Logs in strict sequence: `[START]`, repeated `[STEP]`, then `[END]`
- Uses JSON extraction fallback path from model text
- Falls back to `RUN_TESTS` on parse or validation failure
- Supports `--easy`, `--medium`, `--hard`, `--debug`

## Drift and Risk Notes

1. `requirements.txt` currently pins `pydantic==1.10.17`, but code in `models.py` uses v2 APIs (`model_validator`).
2. `pyproject.toml` is the active dependency source for `uv sync`; `requirements.txt` appears stale relative to runtime assumptions.
3. `environment.py` defines `R_SUBMIT_ALL_PASS` and `R_SUBMIT_FAIL`, but submit currently uses clamped proportion-minus-step-cost scoring instead of those constants.
4. `server/tracefix_rl_environment.py` advertises concurrent sessions support, while `create_app(..., max_concurrent_envs=1)` constrains server-level concurrency.

## Practical Checklist Before Validation

1. Confirm dependency source of truth (`pyproject.toml` vs `requirements.txt`) and align Pydantic version expectations.
2. Re-run pre-validation and capture the first failing check/output.
3. Remove tracked cache artifacts from version control if unintended (for example `__pycache__/*.pyc`).
4. Keep stdout format in `inference.py` unchanged, as validator parsing depends on it.
