---
title: Python Debugging Gym
emoji: 🐛
colorFrom: blue
colorTo: cyan
sdk: docker
pinned: false
app_port: 7860
base_path: /web
tags:
  - openenv
  - reinforcement-learning
  - code-generation
---

# Python Debugging Gym

An OpenEnv-compatible RL environment where agents debug broken Python code by
iteratively viewing, editing, and testing code snippets until all tests pass.

## Environment Overview

- Action space:
`VIEW_CODE`, `RUN_TESTS`, `REPLACE_LINES`, `UNDO_EDIT`, `RESET_TO_ORIGINAL`, `SUBMIT`
- Observation includes:
`code_lines`, `localized_context`, `last_execution_output`, `syntax_error`, `test_results`
- Dense reward with step cost and final score on submit.

## Local Run

```bash
uv sync
uv run --project . server --port 7860
```

Server endpoints:
- `POST /reset`
- `POST /step`
- `GET /health`
- `WS /ws`
- `GET /web` (OpenEnv web UI)

## Deploy to Hugging Face Spaces

```bash
openenv push
```

## Validate Submission

From repo:

```bash
./pre-val.sh https://<your-space>.hf.space .
```
