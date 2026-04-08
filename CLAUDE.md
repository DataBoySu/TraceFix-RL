# CLAUDE.md — TraceFix-RL

Codebase knowledge for AI assistants. Read before making changes.

**Phase status:**
- Phase 1 — REPLACE_LINES action space ✅
- Phase 1B — Hackathon compliance (clamper + inference.py) ✅
- Phase 2 — Mini-Git (UNDO_EDIT + RESET_TO_ORIGINAL) ✅
- Phase 3 — Curriculum learning + static task registry ✅
- Phase 4 — Constrained decoding + Chain-of-Thought + "Reasoning Model" support ✅

---

## File Map

```
models.py           ← Pydantic v1 schema (action + observation spaces)
tasks.py            ← Static task registry (16 hardcoded curated tasks)
sandbox.py          ← Isolated multiprocessing executor
environment.py      ← RL state machine (reset/step/reward/clamping/curriculum)
context.py          ← ±10-line localized view around last edit
server.py           ← FastAPI WebSocket + HTTP server
inference.py        ← Hackathon baseline agent (OpenAI client)
openenv.yaml        ← OpenEnv metadata (validate-submission.sh)
Dockerfile          ← 2-stage HuggingFace Spaces build (production)
requirements.txt

── Offline tools (NOT in Docker image) ────────────────────────────────────
mutation_engine.py  ← Bug injection operators — run locally to generate tasks
dataset_generator.py← Validate + build task dicts from base solutions
```

**Critical:** `mutation_engine.py` and `dataset_generator.py` are **not copied into the Docker image**. They are local data-science tools only. `tasks.py` must not import them.

---

## `models.py`

- **Pydantic v1** (`pydantic==1.10.17`). Never upgrade — `.dict()`, `.parse_raw()`, `.json()`, `@validator` are v1 APIs used everywhere.
- `ActionType`: exactly **6** strings — `"VIEW_CODE"`, `"RUN_TESTS"`, `"REPLACE_LINES"`, `"UNDO_EDIT"`, `"RESET_TO_ORIGINAL"`, `"SUBMIT"`.
- `CodeAction(extra="forbid")` — any extra JSON key raises `ValidationError`.

### CodeAction fields

| Field | Type | Required for |
|---|---|---|
| `thought` | `Optional[str]` | Always (Chain-of-thought scratchpad) |
| `action_type` | `ActionType` | always |
| `start_line` | `Optional[int]` (ge=1) | `REPLACE_LINES` |
| `end_line` | `Optional[int]` (ge=1) | `REPLACE_LINES` |
| `new_code_block` | `Optional[str]` | `REPLACE_LINES` |

No extra fields for `UNDO_EDIT` or `RESET_TO_ORIGINAL`.

### CodeObservation key fields

| Field | Notes |
|---|---|
| `code_lines: List[str]` | Complete current source (authoritative) |
| `localized_context: str` | ±10 lines around last edit; empty until first REPLACE_LINES |
| `last_execution_output: str` | Tail of stdout+stderr from last RUN_TESTS/SUBMIT |
| `syntax_error: bool` | `ast.parse()` check, updated every step |
| `test_results: List[TestResult]` | Per-test pass/fail + error_message |
| `step_count / steps_remaining` | Progress vs MAX_STEPS=50 |
| `reward_last_step: float` | Per-step RL signal |
| `done: bool` | Episode ended |
| `info: dict` | `episode_id`, `task_name`, `task_difficulty` |

---

## `tasks.py` — Static Registry

**This file is a dumb registry.** It contains only hardcoded dicts — no imports from `mutation_engine` or `dataset_generator`. Zero cold-start cost; fully deterministic for evaluators.

To add new tasks: run `mutation_engine.py` + `dataset_generator.py` locally, curate the best outputs, paste them in as hardcoded dicts.

### Exported symbols

| Symbol | Type | Description |
|---|---|---|
| `TASKS_BY_DIFFICULTY` | `Dict[str, List[Dict]]` | Tasks grouped by difficulty tier |
| `ALL_TASKS` | `List[Dict]` | Flat list of all tasks (for random sampling) |

**Current registry size:** `easy=4`, `medium=6`, `hard=6` → 16 tasks total.

### Task dict schema

```python
{
  "name":        str,           # e.g. "binary_search_off_by_one"
  "description": str,
  "code":        List[str],     # buggy version, lines without trailing \n
  "solution":    List[str],     # correct version
  "tests":       List[Callable],# accept (namespace_dict), raise AssertionError
  "difficulty":  str,           # "easy" | "medium" | "hard"
  "bug_type":    str,           # e.g. "wrong_operator" or "logic_inversion"
}
```

### Task catalogue

| Name | Bug | Difficulty |
|---|---|---|
| `sum_even_wrong_condition` | `!= 0` instead of `== 0` | easy |
| `sum_even_missing_accumulator` | `-=` instead of `+=` | easy |
| `reverse_string_wrong_step` | `[::-2]` instead of `[::-1]` | easy |
| `reverse_string_returns_original` | `[::1]` instead of `[::-1]` | easy |
| `binary_search_off_by_one` | `right = len(arr)` instead of `len(arr)-1` | medium |
| `binary_search_wrong_mid` | `left + right` instead of `(left + right) // 2` | medium |
| `flatten_missing_recursion` | `append` instead of `extend(flatten(item))` | medium |
| `flatten_inverted_branch` | `not isinstance` inverts the recursive branch | medium |
| `word_count_no_lower` | missing `text = text.lower()` | medium |
| `word_count_no_punct_strip` | missing punctuation stripping | medium |
| `lru_cache_wrong_eviction` | `pop(-1)` instead of `pop(0)` — evicts MRU | hard |
| `lru_cache_no_promotion` | `get()` doesn't move key to most-recently-used | hard |
| `valid_parentheses_wrong_mapping` | all three bracket mappings are wrong | hard |
| `valid_parentheses_no_empty_check` | missing `not stack or` guard before `pop()` | hard |
| `merge_intervals_strict_overlap` | `<` instead of `<=` — touching intervals not merged | hard |
| `merge_intervals_missing_sort` | missing `intervals.sort()` | hard |

---

## `environment.py`

### Interface
```python
obs, system_prompt = env.reset()
obs, reward, done, info = env.step(action: CodeAction)
```

### Reward constants
```python
R_STEP_COST    = -0.01   # every step (RL signal only)
R_RUN_TESTS    = +0.10
R_PER_NEW_PASS = +0.05   # per newly passing test
R_INVALID_LINE = -0.02
R_SYNTAX_ERROR = -0.10   # inside _act_run_tests on syntax failure
R_UNDO_RESET   = -0.10   # UNDO_EDIT and RESET_TO_ORIGINAL
MAX_STEPS      = 50
```

### Episode state (ALL reset in `reset()`)

- **System Prompt**: Enforces SOP (Standard Operating Procedure: ORIENT → DIAGNOSE → FIX → VERIFY → REPEAT → SUBMIT) and strictly forbids consecutive `VIEW_CODE` calls.

| Field | Description |
|---|---|
| `_code_lines` | Working copy of buggy code |
| `_task` | Current task dict |
| `_step_count` | Steps this episode |
| `_prev_pass_count` | Test passes at last RUN_TESTS |
| `_last_test_results` | From last RUN_TESTS/SUBMIT |
| `_last_output` | Text output for observation |
| `_last_edited_line` | 1-indexed anchor for context.py |
| `_episode_id` | 8-char UUID prefix |
| `_done` | Episode finished |
| `_cumulative_reward` | Sum of all step rewards |
| `_accumulated_step_costs` | `count × 0.01` — used by hackathon clamper |
| `_original_code` | Deep copy of episode-start code; never mutated |
| `_edit_history` | Stack of `List[str]` snapshots; one pushed before each REPLACE_LINES |

`training_step: int = 0` — **not reset by `reset()`**. Persists across episodes. Set externally by trainer.

### `_sample_task()` — Evaluation-safe curriculum sampler

Priority order:

1. **`task_override=dict`** → return it directly (eval/test pinning)
2. **`training_step == 0`** → random pick from `ALL_TASKS` ← **judge-safe default**
   - The Meta evaluator calls `reset()` without setting `training_step`, so this must not crash or bias to one bucket
3. **`training_step > 0`** → curriculum bucketing:
   - `< 1000` → easy
   - `1000 – 4999` → medium
   - `>= 5000` → hard
   - Falls back to any non-empty bucket if the target is empty

### Action handlers

| Method | Delta reward | Key behavior |
|---|---|---|
| `_act_view_code()` | 0.0 | Sets `_last_output` with numbered source |
| `_act_run_tests()` | `R_RUN_TESTS` ± syntax ± new passes | Updates `_prev_pass_count` |
| `_act_replace_lines(s, e, block)` | 0.0 or `R_INVALID_LINE` | Snapshots before mutating; slice assign; anchor = end of new block; blocks deletion of >5 lines (`R_DESTRUCTIVE_PENALTY`) |
| `_act_undo_edit()` | `R_UNDO_RESET` (-0.10) | Pops `_edit_history`; sets `_last_edited_line = None` |
| `_act_reset_to_original()` | `R_UNDO_RESET` (-0.10) | Restores `_original_code`; clears `_edit_history`; sets `_last_edited_line = None` |
| `_act_submit()` | clamped [0.0, 1.0] | Hackathon score formula |

**Action Penalties**:
- **Anti-Loop**: `step()` applies an escalating `-0.05 * n` penalty if the agent chooses the exact same `action_type` repeatedly.
- **Escape Hatch Rule**: The prompt explicitly warns against manual space-fixing on syntax/indent errors, directing the agent to use `UNDO_EDIT` or `RESET_TO_ORIGINAL`.

### Hackathon Reward Clamper (`_act_submit` & Timeout)

```python
proportion  = passes / total        # 0.0 on syntax error
raw_score   = proportion - self._accumulated_step_costs
final_score = max(0.0, min(1.0, raw_score))
```

- **Deterministic Evaluation**: Floor ≥0.0 and <=1.0 guaranteed.
- **Trigger**: Runs on `SUBMIT` **or** when hitting `MAX_STEPS` timeout. Never trusts the LLM to call `SUBMIT`.
- Stored in `info["final_score"]` when `done=True`.

---

## `context.py`

`get_localized_context(code_lines, anchor_line, window=10) -> str`
- Returns `""` if `anchor_line is None` or `code_lines` is empty.
- Uses `len(code_lines)` dynamically — handles REPLACE_LINES growth/shrink correctly.
- Hard cap: `MAX_CONTEXT_CHARS = 2_000`.

---

## `sandbox.py`

`run_code_with_tests(source: str, callables, timeout=5) -> (output_str, List[TestResult], had_syntax_error)`

- **Always a 3-tuple.** Never access as an object (no `.all_pass`, no `.test_results`).
- `source` must be a `str` — call `"\n".join(code_lines)` before passing.
- Isolation: `multiprocessing.Process`, SIGTERM → SIGKILL on timeout.
- Output tail-truncated to `MAX_OUTPUT_CHARS = 1_000`.

---

## `server.py`

FastAPI WebSocket layer. Port: `os.environ.get("PORT", 7860)`.

| Endpoint | Notes |
|---|---|
| `GET /health` | Liveness probe |
| `GET /info` | Env metadata + `CodeAction.schema()` |
| `POST /reset` | Stateless, new env per request |
| `WS /ws` | Primary RL channel — auto-resets on `done=True`. Append `?difficulty=easy|medium|hard` to set tier. |

---

## `inference.py`

Config from `os.getenv`:

| Variable | Default | Notes |
|---|---|---|
| `API_BASE_URL` | `https://api.openai.com/v1` | OpenEnv compatible proxy URL |
| `MODEL_NAME` | `gpt-4o` | Robust fallback model if missing |
| `HF_TOKEN` | `""` | Optional HuggingFace Token |
| `ENV_WS_URL` | `ws://localhost:7860/ws` | Connecting environment URL |
| `DEBUG_LOG` | `0` | Set to `1` to print raw LLM output |

**CLI Flags:**
- `python inference.py --easy` (or `--medium`, `--hard`) appends `?difficulty=...` parameter to the WS URL to override `training_step` bucketing.

### Decoding & Fallbacks

- **Structured Output**: Uses `json_schema` protocol with strict `CodeAction` forcing `thought` generation before `action_type`.
- **Reasoning Models**: Directly parses `.model_dump()["reasoning_content"]` if `content` is empty (e.g. DeepSeek-R1 / Nemotron in LM Studio).
- **Mask-Free Parser**: Invalid JSON explicitly returns `PARSE_ERROR` to the server (preventing silent `VIEW_CODE` loops), forcing LLM self-correction.

**Exact stdout log format (regex-parsed by validation judge):**
```
[START] task=<task_name> env=TraceFixRL model=<model_name>
[STEP] step=<n> action=<action_type> reward=<r.rr> done=<true|false> error=<msg|null>
[END] success=<true|false> steps=<n> score=<s.sss> rewards=<r1,r2,...,rn>
```

- `reward` → `:.2f`; `done` → lowercase; `error` → `"null"` on success.
- `score` → `:.3f` — pulled from `info["final_score"]` (the clamped [0,1] value).
- `rewards` → comma-separated, no spaces.

---

## `openenv.yaml`

Consumed by `openenv validate` step in `validate-submission.sh`.

Key fields: `reward_range: [0.0, 1.0]`, `inference_script: inference.py`, `websocket_path: /ws`, `port: 7860`.

---

## `Dockerfile`

Two-stage build. Runtime COPY (all with `--chown=appuser:appuser`):
```
models.py  environment.py  sandbox.py  tasks.py
server.py  context.py      inference.py
```

**`mutation_engine.py` and `dataset_generator.py` are NOT copied.** They are offline tools.

---

## Offline Tools (local only)

### `mutation_engine.py`

`MutationEngine(seed).mutate(code_lines, difficulty, max_attempts=10)`
→ `(List[str], {"bug_type": str, "num_bugs": int})` or `(None, None)`

Operator sets:

| Difficulty | Operators |
|---|---|
| easy | `_var_name_error`, `_wrong_operator` |
| medium | easy + `_off_by_one`, `_logic_inversion`, `_index_error` |
| hard | medium + `_mutable_default`, `_remove_return`, `_wrong_function_call` |

### `dataset_generator.py`

`validate_task(original, mutated, tests)` — original must pass all tests; mutated must fail ≥ 1.
`generate_task(base_task, mutator)` — calls mutate + validate; returns task dict or `None`.

**Workflow to add new tasks:**
```bash
python -c "
from mutation_engine import MutationEngine
from dataset_generator import generate_task
# define base_task with solution + tests
# run generate_task, inspect output, paste into tasks.py
"
```

---

## Dependencies

```
fastapi==0.111.0
uvicorn[standard]==0.30.1
pydantic==1.10.17          ← v1 ONLY
websockets==12.0
openai>=1.30.0             ← inference.py only
```

IDE lint warnings for these packages are expected false-positives — they live in Docker, not system Python.

---

## Invariants

1. **Pydantic v1 only.** Never upgrade.
2. **1-indexed lines in public API**; 0-indexed in `_code_lines`.
3. `reset()` wipes every mutable field including `_accumulated_step_costs`, `_original_code`, `_edit_history`. `training_step` is NOT reset.
4. Reward delta model — handlers return delta; `R_STEP_COST` applied once per step before routing.
5. REPLACE_LINES anchor = `min(start + len(new_lines) - 1, file_length)`.
6. SUBMIT reward clamped `[0.0, 1.0]` — this is the grader score. Floor guaranteed ≥ 0.0.
7. `_act_run_tests()` updates `_prev_pass_count`; `_act_submit()` does not.
8. Task `code` strings have no trailing `\n`; `_source()` joins with `\n`.
9. `context.py` is already fully dynamic — no changes needed for REPLACE_LINES growth/shrink.
10. Output truncation is **tail-based** (end of traceback = actionable info).
11. **Mini-Git snapshot timing**: snapshot pushed **before** slice assignment. Rejected edits (OOB, inverted range) produce no snapshot.
12. **Context desync invariant**: Both rollback handlers set `_last_edited_line = None`. Without this, `context.py` anchors to a ghost line after revert.
13. **`_original_code` is immutable**: set once in `reset()`, only read in `_act_reset_to_original()`.
14. **`sandbox.run_code_with_tests` returns a 3-tuple**: `(output_str, List[TestResult], had_syntax_error)`. Never treat as object.
15. **`tasks.py` must not import `mutation_engine` or `dataset_generator`**: those are offline tools not in the Docker image.
16. **`training_step == 0` → random from ALL_TASKS**: the judge calls `reset()` with default `training_step=0`, so this path must work correctly and not bias to one difficulty bucket.
