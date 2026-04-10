---
title: TraceFix-RL
emoji: 🧑‍💻
colorFrom: blue
sdk: docker
pinned: false
app_port: 7860
base_path: /web
tags:
  - openenv
  - reinforcement-learning
  - software-engineering
---

## TraceFix-RL

TraceFix-RL is an OpenEnv-compatible environment designed to teach agent behavior
that looks like real software engineering work. Instead of one-shot answers,
the agent must inspect code, form a hypothesis, run tests, patch the code,
verify outcomes, and only then submit. The loop rewards disciplined debugging
and penalizes random edits, forcing the model to learn an engineering workflow.

## Core Design

- **Action space:** `VIEW_CODE`, `RUN_TESTS`, `REPLACE_LINES`, `UNDO_EDIT`, `RESET_TO_ORIGINAL`, `SUBMIT`
- **Observations:** The full code snapshot, localized edit context, execution output, syntax status, and per-test outcomes.
- **Dense Rewards:** `RUN_TESTS` bonus, per-test progress bonus, step-cost penalty, invalid-edit penalties, and a final clamped score bounded within `[0.01, 0.98]`.
- **Curriculum-ready Tasks:** Includes Easy, Medium, and Hard buckets that the OpenEnv trainer can sequence, alongside random fallback for evaluators.

## State Machine Training Pattern

The environment prompt in `environment.py` encodes a strict operating pattern the agent is expected to follow:

1. **ORIENT:** Inspect code (`VIEW_CODE`)
2. **DIAGNOSE:** Run tests and read failures (`RUN_TESTS`)
3. **FIX:** Patch one localized region (`REPLACE_LINES`)
4. **VERIFY:** Rerun tests (`RUN_TESTS`)
5. **REPEAT:** Continue until all failures are resolved
6. **SUBMIT:** Finalize only after tests pass

This sequence naturally guides reinforcement learning toward robust planning, controlled editing, and verification behavior.

## Task Tiers And Test Structure

The registry in `tasks.py` acts as a static curated set of coding challenges (16 tasks total):

- **Easy (4 tasks):** Focuses on basic operators, indexing, and simple string/array logic.
- **Medium (6 tasks):** Focuses on recursive behavior, branching correctness, and text normalization edges.
- **Hard (6 tasks):** Focuses on data-structure invariants, bracket mapping, interval merging, and eviction logic.

Every task contains: `name`, `description`, `difficulty`, `bug_type`, `code` (buggy implementation), `solution`, and executable `tests`. All tests are safely run inside isolated sandboxes via `sandbox.py` using `multiprocessing`.

## Tech Stack & Project Files

This environment enforces strict typing and uses standard modern tooling:

- **`uv`:** Handles dependency management (see `pyproject.toml`).
- **FastAPI:** Provides the `server.app` integration layer for OpenEnv compliance.
- **Pydantic (v2):** Provides strong validation layers for `models.py` (e.g., `CodeAction`, `CodeObservation`).
- **OpenEnv Config:** See `openenv.yaml` which specifies `tracefix_rl` to run the FastAPI app on port `7860`.

**File Layout:**

- `models.py` / `context.py`: Domain and schema logic.
- `tasks.py`: Task metadata definitions.
- `sandbox.py`: Subprocess runtime and output tracking.
- `environment.py`: Reset/step/reward core RL loop logic (`TraceFixRLGym`).
- `server/tracefix_rl_environment.py` / `server/app.py`: Maps the OpenAI/OpenEnv network interface to the core environment.
- `inference.py`: Baseline OpenAI-client inference script to evaluate agents.

## Local Development

You must install [`uv`](https://github.com/astral-sh/uv) on your system.

```bash
# Sync dependencies
uv sync

# Run the OpenEnv server on port 7860
uv run --project . server
```

Server endpoints available:

- `POST /reset`
- `POST /step`
- `GET /health`
- `WS /ws`

## Baseline Scores

Baseline scores are intended to be recorded from the bundled `inference.py` runner against the three validator tasks.
The current environment intentionally squashes scores into the open interval `[0.01, 0.98]`, so benchmark output should be
reported with that convention in mind.

| Task | Baseline Score |
| --- | --- |
| `valid_parentheses_wrong_mapping` | Pending first benchmark run |
| `binary_search_off_by_one` | Pending first benchmark run |
| `reverse_string_returns_original` | Pending first benchmark run |

## Docker + Hugging Face Spaces Deployment

The space runs via Docker. The container is securely configured to run as a non-root `appuser` (UID base `1000`) for Spaces compliance.

### Testing Locally in Docker

```bash
docker build -t tracefix-rl:test -f Dockerfile .
docker run --rm -p 7860:7860 tracefix-rl:test
```

### Deploy to Hugging Face Spaces

This project uses the OpenEnv CLI for seamless Hugging Face Space deployments.

```bash
# Push directly to your specified HF Space
openenv push
```

### Server Pre-validation

Before committing to training, you can validate your deployed server or local space:

```bash
bash ./pre-val.sh https://<your-space>.hf.space .
```

## Inference & Evaluation (`inference.py`)

The baseline inference runner evaluates agents against the environment using an OpenAI-compatible interface.

**Requirements for Inference:**

- `API_BASE_URL` (Defaults to `https://router.huggingface.co/v1`)
- `MODEL_NAME` (Defaults to `Qwen/Qwen2.5-72B-Instruct`)
- `HF_TOKEN`

**Usage Flags:**

- `--easy`, `--medium`, `--hard`: Lock the environment to a specific task bucket.
- `--thought`: Send `<thought>` token blocks back to the payload to train chain-of-thought capabilities.

Example execution tracking thoughts in medium tasks:

```bash
python inference.py --medium --thought
```
