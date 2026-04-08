---
title: SWE-Gym - Software Engineer Gym
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

# SWE-Gym - Software Engineer Gym

SWE-Gym is an OpenEnv-compatible RL environment where an agent must debug broken
Python code by iteratively inspecting source, running tests, editing lines, and
submitting once all tests pass.

## Core Design

- Action space:
`VIEW_CODE`, `RUN_TESTS`, `REPLACE_LINES`, `UNDO_EDIT`, `RESET_TO_ORIGINAL`, `SUBMIT`
- Observation includes full code, localized edit context, execution output, syntax status, and per-test outcomes.
- Dense rewards:
`RUN_TESTS` bonus, per-test progress bonus, step-cost penalty, invalid-edit penalties, and final clamped score in `[0, 1]`.
- Curriculum-ready task sampling:
easy/medium/hard buckets with safe random fallback for evaluator runs.

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
