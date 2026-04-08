---
title: TraceFix-RL
emoji: 🧑‍💻
colorFrom: blue
colorTo: cyan
sdk: docker
pinned: false
app_port: 7860
base_path: /web
tags:
  - openenv
  - reinforcement-learning
  - software-engineering
---

# TraceFix-RL

TraceFix-RL is an OpenEnv-compatible environment designed to teach agent behavior
that looks like real software engineering work. Instead of one-shot answers,
the agent must inspect code, form a hypothesis, run tests, patch the code,
verify outcomes, and only then submit. The loop rewards disciplined debugging
and penalizes random edits, forcing the model to learn an engineering workflow.

## Core Design

- Action space:
`VIEW_CODE`, `RUN_TESTS`, `REPLACE_LINES`, `UNDO_EDIT`, `RESET_TO_ORIGINAL`, `SUBMIT`
- Observation includes full code, localized edit context, execution output, syntax status, and per-test outcomes.
- Dense rewards:
`RUN_TESTS` bonus, per-test progress bonus, step-cost penalty, invalid-edit penalties, and final clamped score in `[0, 1]`.
- Curriculum-ready task sampling:
easy/medium/hard buckets with safe random fallback for evaluator runs.

## State Machine Training Pattern

The environment prompt in `environment.py` encodes a fixed operating pattern
the agent is expected to follow:

1. ORIENT: inspect code (`VIEW_CODE`)
2. DIAGNOSE: run tests and read failures (`RUN_TESTS`)
3. FIX: patch one region (`REPLACE_LINES`)
4. VERIFY: rerun tests (`RUN_TESTS`)
5. REPEAT: continue until all failures are resolved
6. SUBMIT: finalize only after tests pass

This structure is intentional: the environment trains planning, controlled
editing, and verification behavior, not just raw code generation.

## Task Tiers And Test Structure

Tasks are organized in `tasks.py` into three tiers.

- Easy: 4 tasks, each with 4 unit tests.
  Focus: basic operators, indexing, and simple string/array logic.
- Medium: 6 tasks, each with 4 unit tests.
  Focus: recursive behavior, branching correctness, and text normalization edge cases.
- Hard: 6 tasks, each with 3-4 unit tests.
  Focus: data-structure invariants, eviction/promotion logic, bracket mapping, and interval merging edge behavior.

Every task follows the same schema:

- `name`, `description`, `difficulty`, `bug_type`
- `code`: buggy implementation (line list)
- `solution`: reference implementation
- `tests`: callable assertions executed in the sandbox

This gives consistent training signals while scaling complexity across tiers.

## Environment Files

- `models.py`: action/observation schemas
- `tasks.py`: static curated task registry
- `sandbox.py`: isolated timed execution
- `environment.py`: reset/step/reward logic
- `context.py`: localized code windowing
- `server/app.py`: FastAPI OpenEnv server entry
- `inference.py`: baseline OpenAI-client inference script

## Local Run

```bash
uv sync
uv run --project . server
```

Server endpoints:

- `POST /reset`
- `POST /step`
- `GET /health`
- `WS /ws`
- `GET /web`

## Inference Script Compliance

- Script location/name: `inference.py` at repo root.
- Uses OpenAI client with:
`API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN`.
- Supports local container mode with:
`LOCAL_IMAGE_NAME`.
- Emits standardized stdout lines in this order:
`[START]`, one or more `[STEP]`, then `[END]`.
- Final score is clamped to `[0, 1]`.

## Inference Flags

`inference.py` supports:

- `--easy`: run episode using easy-tier curriculum sampling.
- `--medium`: run episode using medium-tier curriculum sampling.
- `--hard`: run episode using hard-tier curriculum sampling.
- `--debug`: print raw model response snippets for troubleshooting.

Example:

```bash
python inference.py --medium --debug
```

The script also enforces a model-thinking/output cap:

- `THINKING_TOKEN_LIMIT` (default `512`) is used as `max_tokens` in model calls.
- `thought` content is hard-truncated before action validation to prevent oversized payloads.

## Docker + Deployment

Build locally:

```bash
docker build -t swe-gym:test -f Dockerfile .
```

Run locally:

```bash
docker run --rm -p 7860:7860 swe-gym:test
```

Deploy to Hugging Face:

```bash
openenv push
```

Validate before submit:

```bash
bash ./pre-val.sh https://<your-space>.hf.space .
```
