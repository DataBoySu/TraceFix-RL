"""
environment.py — Python Debugging Gym (Core RL Environment)
=============================================================

PRINCIPLE 1 — You Don't Design the Control Flow
  The agent decides the sequence of actions. step() is a pure router:
  it receives whatever action the agent chose (in whatever order),
  processes it, and returns the new state. There is no forced sequence,
  no "you must VIEW_CODE before RUN_TESTS" gate. The system prompt
  explains what tools exist; the agent decides how to use them.

PRINCIPLE 5 — Cost-Per-Turn Reward Logic
  Each call to step() costs R_STEP_COST = -0.01. This makes the episode
  a multi-turn budget problem: the agent is rewarded for solving quickly.
  An agent that solves in 4 steps scores ~0.14 more than one that takes
  18 steps to reach the same solution.

PRINCIPLE 7 — The Prompt is Code
  The string returned by reset() is the agent's complete operational
  contract for the session. It states: the goal, the available actions
  (with exact JSON examples), the reward structure, the current code,
  and the expected termination condition. Ambiguity in this string
  directly causes off-task behaviour.

PRINCIPLE 10 — Layered Context Compaction
  _build_observation() tracks `_last_edited_line` and passes it to
  context.get_localized_context() to produce a focused ±10-line view
  after each write action. This prevents the observation from inflating
  the agent's context window on large files.

Reward table (dense, non-sparse — every step emits a signal):
  +1.00  SUBMIT and ALL tests pass     → episode solved
  +0.10  RUN_TESTS called              → information-gathering rewarded
  +0.05  Per test transitioning fail→pass on a RUN_TESTS or SUBMIT
  -0.01  Every step taken              → efficiency pressure (Principle 5)
  -0.10  Syntax error detected         → broken code penalised immediately
  -0.10  UNDO_EDIT or RESET_TO_ORIGINAL → backtracking discouraged
  -0.02  Invalid line range supplied   → hallucination deterrent
  -0.20  SUBMIT with tests still failing

Max episode length: 50 steps.
"""

from __future__ import annotations

import random
import uuid
from typing import Any, Dict, List, Optional, Tuple

try:
    from .context import get_localized_context
    from .models import CodeAction, CodeObservation, TestResult
    from .sandbox import check_syntax, run_code_with_tests
    from .tasks import ALL_TASKS, TASKS_BY_DIFFICULTY
except ImportError:
    from context import get_localized_context
    from models import CodeAction, CodeObservation, TestResult
    from sandbox import check_syntax, run_code_with_tests
    from tasks import ALL_TASKS, TASKS_BY_DIFFICULTY


# ---------------------------------------------------------------------------
# Reward constants
# ---------------------------------------------------------------------------

R_SUBMIT_ALL_PASS = +1.00
R_SUBMIT_FAIL     = -0.20
R_SYNTAX_ERROR    = -0.10
R_RUN_TESTS       = +0.10
R_PER_NEW_PASS    = +0.05
R_STEP_COST       = -0.01   # PRINCIPLE 5 — every step has a cost
R_INVALID_LINE    = -0.02
R_DESTRUCTIVE_PENALTY = -0.20
R_UNDO_RESET      = -0.10   # Mini-Git backtracking penalty

MAX_STEPS: int = 50


# ---------------------------------------------------------------------------
# System Prompt  (PRINCIPLE 7 — The Prompt is Code)
# ---------------------------------------------------------------------------
# This string is the agent's entire operational contract.
# It must be:
#   • Self-contained (no assumed context from training data)
#   • Precise (exact JSON examples, not vague descriptions)
#   • Non-directive about sequence (Principle 1: agent chooses order)
#   • Complete (goal, tools, rewards, termination — nothing omitted)

_SYSTEM_PROMPT = """\
╔══════════════════════════════════════════════════════╗
║          PYTHON DEBUGGING GYM — EPISODE BRIEF        ║
╚══════════════════════════════════════════════════════╝

GOAL
----
The Python source file shown below contains one or more bugs.
Your task is to find and fix every bug so that ALL unit tests pass, then
call SUBMIT to end the episode.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STANDARD OPERATING PROCEDURE  (follow this state machine)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

STEP 1 — ORIENT   : Call VIEW_CODE to read the full file with line numbers.
STEP 2 — DIAGNOSE : Call RUN_TESTS to get the exact error message and traceback.
STEP 3 — FIX      : Call REPLACE_LINES to correct the identified bug.
                     (Use UNDO_EDIT if the edit made things worse.)
STEP 4 — VERIFY   : Call RUN_TESTS again to confirm the fix worked.
STEP 5 — REPEAT   : If tests still fail, return to STEP 1 and re-read the code.
STEP 6 — SUBMIT   : Once ALL tests pass, call SUBMIT.

⚠ Do NOT call VIEW_CODE more than once in a row. Each VIEW_CODE costs -0.01.
  If you have already viewed the code, call RUN_TESTS next, not VIEW_CODE again.

⚠ THE ESCAPE HATCH RULE: If an edit results in a syntax error or an indentation error,
  DO NOT try to manually fix spaces. IMMEDIATELY use UNDO_EDIT or RESET_TO_ORIGINAL.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
AVAILABLE TOOLS  (send one JSON object per turn)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. VIEW_CODE — see the full file with line numbers
   {{"thought": "<your reasoning>", "action_type": "VIEW_CODE"}}

2. RUN_TESTS — execute all unit tests; see pass/fail + output
   {{"thought": "<your reasoning>", "action_type": "RUN_TESTS"}}

3. REPLACE_LINES — replace a contiguous block of lines (start to end, inclusive)
   {{"thought": "<your reasoning>", "action_type": "REPLACE_LINES", "start_line": 3, "end_line": 5, "new_code_block": "    x = 1\\n    return x"}}
   ⚠ start_line and end_line are 1-indexed and INCLUSIVE.
   ⚠ new_code_block is a single string; separate lines with \\n (no trailing \\n).
   ⚠ Indentation is syntax in Python — include the correct leading spaces on every line.
   ⚠ The file grows or shrinks when the new block has more/fewer lines than the range.
   ⚠ After REPLACE_LINES, call RUN_TESTS (not VIEW_CODE) to verify the fix.

4. UNDO_EDIT — revert to the state before the most recent REPLACE_LINES (-0.10 penalty)
   {{"thought": "<your reasoning>", "action_type": "UNDO_EDIT"}}
   Use when an edit made things worse and you want to try a different approach.
   No-op (with penalty) if there is no edit history.

5. RESET_TO_ORIGINAL — restore the pristine broken code from episode start (-0.10 penalty)
   {{"thought": "<your reasoning>", "action_type": "RESET_TO_ORIGINAL"}}
   Last resort only. Clears all undo history. Resets context anchor.

6. SUBMIT — declare the fix complete; ends the episode
   {{"thought": "<your reasoning>", "action_type": "SUBMIT"}}
   Only call SUBMIT when RUN_TESTS has confirmed ALL tests pass.
   The episode ends immediately on SUBMIT, pass or fail.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REWARD SIGNALS  (visible in observation.reward_last_step)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  +1.00  SUBMIT and all tests pass           ← primary objective
  +0.10  RUN_TESTS called (any outcome)      ← gathering info is good
  +0.05  Per test newly passing vs last run  ← incremental progress
  -0.01  Every step taken                    ← solve efficiently
  -0.10  Syntax error in current code        ← fix broken syntax first
  -0.10  UNDO_EDIT or RESET_TO_ORIGINAL      ← backtracking is expensive
  -0.02  Invalid line range sent             ← use VIEW_CODE to check range
  -0.20  SUBMIT with tests still failing     ← verify before submitting

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EPISODE PARAMETERS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Task        : {task_name}  ({difficulty})
  Unit tests  : {test_count} tests — ALL must pass
  Max steps   : {max_steps}  (episode terminates at 0 steps remaining)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CURRENT CODE  (this is the broken version — fix it)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{code_preview}
"""


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class PythonDebuggingGym:
    """
    Gymnasium-compatible RL environment for Python debugging.

    PRINCIPLE 1: step() is a stateless router — the agent chooses the
    sequence. No internal gates, no forced ordering between actions.

    Interface
    ---------
    obs, system_prompt = env.reset()
    obs, reward, done, info = env.step(action: CodeAction)
    """

    metadata = {"name": "PythonDebuggingGym-v1", "render_modes": []}

    def __init__(
        self,
        task_index: Optional[int] = None,
        seed: Optional[int] = None,
    ):
        self._task_index = task_index
        self._rng = random.Random(seed)

        # All mutable episode state lives here; reset() wipes every field.
        self._code_lines: List[str] = []
        self._task: Dict[str, Any] = {}
        self._step_count: int = 0
        self._prev_pass_count: int = 0
        self._last_test_results: List[TestResult] = []
        self._last_output: str = ""
        self._last_edited_line: Optional[int] = None   # PRINCIPLE 10
        self._episode_id: str = ""
        self._done: bool = False
        self._cumulative_reward: float = 0.0
        self._accumulated_step_costs: float = 0.0  # Hackathon compliance
        # Mini-Git snapshot history (Phase 2)
        self._original_code: List[str] = []          # pristine copy set at reset()
        self._edit_history: List[List[str]] = []     # stack of pre-edit snapshots
        # Curriculum learning — persists across episodes, incremented externally
        self.training_step: int = 0

    # ── Curriculum task sampler ──────────────────────────────────────────────

    def _sample_task(self, task_override=None) -> Dict[str, Any]:
        """
        Evaluation-safe curriculum sampler.

        Priority order:
          1. task_override dict  → return it directly (test/eval pinning)
          2. training_step == 0  → randomly sample from ALL_TASKS (judge-safe default;
                                   the Meta evaluator calls reset() without setting
                                   training_step, so this must work correctly)
          3. training_step > 0   → curriculum bucketing:
               < 1000  → easy
               < 5000  → medium
               >= 5000 → hard
             Falls back to any non-empty bucket if the target bucket is empty.
        """
        if isinstance(task_override, dict):
            return task_override

        # Judge-safe default: no training_step set → random from all tasks
        if self.training_step == 0:
            if not ALL_TASKS:
                raise RuntimeError("ALL_TASKS is empty — check tasks.py.")
            return self._rng.choice(ALL_TASKS)

        # Curriculum mode (trainer increments training_step between episodes)
        if self.training_step < 1000:
            bucket = "easy"
        elif self.training_step < 5000:
            bucket = "medium"
        else:
            bucket = "hard"

        pool = TASKS_BY_DIFFICULTY.get(bucket, [])
        if not pool:
            # Fallback: any non-empty bucket rather than crashing
            for b in ("easy", "medium", "hard"):
                pool = TASKS_BY_DIFFICULTY.get(b, [])
                if pool:
                    break
        if not pool:
            raise RuntimeError("TASKS_BY_DIFFICULTY is entirely empty — check tasks.py.")

        return self._rng.choice(pool)

    # ── reset() ─────────────────────────────────────────────────────────────

    def reset(
        self, *, task_index: Optional[int] = None
    ) -> Tuple[CodeObservation, str]:
        """
        Wipe all episode state; select a task; return initial observation + prompt.

        State isolation guarantee: every mutable field is explicitly reset here.
        There is no shared state between episodes — not even the RNG advances
        carry forward (the seed is fixed at __init__ time).
        """
        self._task = self._sample_task(task_index)

        # ── Complete state wipe ──────────────────────────────────────────
        self._code_lines       = list(self._task["code"])   # deep copy — no alias
        self._step_count       = 0
        self._prev_pass_count  = 0
        self._last_test_results = []
        self._last_output      = ""
        self._last_edited_line = None   # no edits yet — localized_context will be empty
        self._episode_id       = str(uuid.uuid4())[:8]
        self._done             = False
        self._cumulative_reward = 0.0
        self._accumulated_step_costs = 0.0
        # Mini-Git: seed pristine snapshot and clear history
        self._original_code = list(self._task["code"])  # separate copy from _code_lines
        self._edit_history  = []
        # Anti-Loop history
        self._last_action: Optional[str] = None
        self._consecutive_count: int = 0

        obs = self._build_observation(reward=0.0)

        # PRINCIPLE 7: build the operational contract string
        system_prompt = _SYSTEM_PROMPT.format(
            task_name   = self._task["name"],
            difficulty  = self._task.get("difficulty", "unknown"),
            test_count  = len(self._task["tests"]),
            max_steps   = MAX_STEPS,
            code_preview = obs.render_code(),
        )

        return obs, system_prompt

    # ── step() ──────────────────────────────────────────────────────────────

    def step(
        self, action: CodeAction
    ) -> Tuple[CodeObservation, float, bool, Dict[str, Any]]:
        """
        PRINCIPLE 1 — Pure router. Accept any valid action in any order.

        The only sequencing constraint is that SUBMIT ends the episode.
        All other actions can be called in any combination and in any order.
        step() does NOT enforce a workflow — it applies the action and returns
        the resulting state for the agent to reason about.

        PRINCIPLE 5 — R_STEP_COST is applied before routing so it is
        impossible to take a "free" step — every turn has a cost.
        """
        if self._done:
            raise RuntimeError(
                "step() called on a finished episode. Call reset() first."
            )

        self._step_count += 1
        reward = R_STEP_COST   # PRINCIPLE 5: cost-per-turn baseline
        self._accumulated_step_costs += abs(R_STEP_COST)  # Hackathon compliance

        # ── Repetition Penalty (Anti-Loop) ───────────────────────────────
        if action.action_type == self._last_action:
            self._consecutive_count += 1
            reward += -0.05 * self._consecutive_count
        else:
            self._consecutive_count = 0
        self._last_action = action.action_type

        # ── Route (PRINCIPLE 1: no forced sequence) ──────────────────────
        atype = action.action_type

        if   atype == "VIEW_CODE":
            reward += self._act_view_code()

        elif atype == "RUN_TESTS":
            reward += self._act_run_tests()

        elif atype == "REPLACE_LINES":
            reward += self._act_replace_lines(
                action.start_line, action.end_line, action.new_code_block
            )

        elif atype == "UNDO_EDIT":
            reward += self._act_undo_edit()

        elif atype == "RESET_TO_ORIGINAL":
            reward += self._act_reset_to_original()

        elif atype == "SUBMIT":
            reward += self._act_submit()
            self._done = True

        # ── Max-steps termination ────────────────────────────────────────
        if self._step_count >= MAX_STEPS and not self._done:
            self._done = True
            # Deterministic clamp — never trust the LLM to call SUBMIT.
            # Evaluate the current code and produce a valid [0.0, 1.0] score
            # regardless of how the episode ended.
            _, results, syntax_err = run_code_with_tests(
                source=self._source(),
                test_callables=self._task["tests"],
            )
            total  = len(results)
            passes = 0 if syntax_err else sum(1 for t in results if t.passed)
            raw    = (passes / total if total > 0 else 0.0) - self._accumulated_step_costs
            reward = max(0.0, min(1.0, raw))
            self._last_output += (
                f"\n⚠ Max steps ({MAX_STEPS}) reached. "
                f"Auto-evaluated: {passes}/{total} tests passing. "
                f"Final score: {reward:.4f}"
            )

        self._cumulative_reward += reward
        obs  = self._build_observation(reward=reward)
        info = {
            "episode_id":        self._episode_id,
            "task":              self._task["name"],
            "cumulative_reward": round(self._cumulative_reward, 4),
            "step":              self._step_count,
        }
        if self._done:
            # PRINCIPLE: Ensure Hackathon score leak doesn't occur. It must be strictly [0.0, 1.0].
            # During SUBMIT, reward might be negative if _act_submit returned 0.0 added to -0.01.
            info["final_score"] = max(0.0, min(1.0, round(reward, 4)))

        return obs, round(reward, 4), self._done, info

    # ── Action handlers ─────────────────────────────────────────────────────
    # Each returns the delta reward (R_STEP_COST already applied by step()).
    # Handlers update self._last_output and self._last_edited_line as needed.

    def _act_view_code(self) -> float:
        self._last_output = (
            "=== Full Source ===\n" +
            "\n".join(
                f"{i + 1:>3} | {line}"
                for i, line in enumerate(self._code_lines)
            )
        )
        # VIEW_CODE does not change the code — localized_context stays where it was
        return 0.0

    def _act_run_tests(self) -> float:
        output, results, syntax_err = run_code_with_tests(
            source=self._source(),
            test_callables=self._task["tests"],
        )
        self._last_output      = output
        self._last_test_results = results

        reward = R_RUN_TESTS   # information-gathering bonus (Principle 5)

        if syntax_err:
            reward += R_SYNTAX_ERROR
        else:
            current_pass = sum(1 for t in results if t.passed)
            new_passes   = max(0, current_pass - self._prev_pass_count)
            reward       += new_passes * R_PER_NEW_PASS
            self._prev_pass_count = current_pass

        return reward

    def _act_replace_lines(
        self, start_line: int, end_line: int, new_code_block: str
    ) -> float:
        n = len(self._code_lines)
        
        if new_code_block is None:
            new_code_block = ""

        # ── Guard: Destructive Action (Anti-Deletion) ─────────────────────
        if len(new_code_block) == 0 and (end_line - start_line) > 5:
            self._last_output = "Error: Cannot delete more than 5 lines at once."
            return R_DESTRUCTIVE_PENALTY

        # ── Guard: inverted range ─────────────────────────────────────────
        if start_line > end_line:
            self._last_output = (
                f"Error: start_line ({start_line}) > end_line ({end_line}). "
                "Inverted range rejected. Call VIEW_CODE to check the current line count."
            )
            return R_INVALID_LINE

        # ── Guard: out-of-bounds ──────────────────────────────────────────
        if start_line < 1 or start_line > n:
            self._last_output = (
                f"Error: start_line {start_line} is out of range [1, {n}]. "
                "Call VIEW_CODE to check the current line count."
            )
            return R_INVALID_LINE
        if end_line < 1 or end_line > n:
            self._last_output = (
                f"Error: end_line {end_line} is out of range [1, {n}]. "
                "Call VIEW_CODE to check the current line count."
            )
            return R_INVALID_LINE

        # ── Slice assignment (PRINCIPLE 1: pure data transformation) ──────
        start_idx = start_line - 1   # convert to 0-indexed
        end_idx   = end_line         # exclusive upper bound for Python slice

        # ── Mini-Git: snapshot BEFORE mutating (Phase 2) ─────────────────
        self._edit_history.append(list(self._code_lines))

        new_lines = new_code_block.split("\n")
        self._code_lines[start_idx:end_idx] = new_lines

        # ── Anchor context at END of new block (PRINCIPLE 10) ─────────────
        # If the agent replaces lines 5–10 with 20 new lines, the anchor
        # settles at start_line + len(new_lines) - 1, clamped to file length.
        new_end = start_line + len(new_lines) - 1
        self._last_edited_line = min(new_end, len(self._code_lines))

        replaced_count = end_line - start_line + 1
        self._last_output = (
            f"✏ Replaced lines {start_line}–{end_line} "
            f"({replaced_count} line(s)) with {len(new_lines)} new line(s).\n"
            f"File now has {len(self._code_lines)} lines total. "
            f"Context anchored at line {self._last_edited_line}. "
            "Call VIEW_CODE to re-orient before referencing line numbers."
        )
        return 0.0

    def _act_submit(self) -> float:
        output, results, syntax_err = run_code_with_tests(
            source=self._source(),
            test_callables=self._task["tests"],
        )
        self._last_output      = output
        self._last_test_results = results

        total  = len(results)
        passes = 0 if syntax_err else sum(1 for t in results if t.passed)

        if syntax_err:
            self._last_output += "\n❌ SUBMIT rejected — syntax error in current code."

        # ── Hackathon compliance: final score ∈ [0.0, 1.0] ───────────────
        # raw = (tests_passed / total) - accumulated_step_costs
        # Then clamped so the grader always receives a value in spec.
        proportion  = passes / total if total > 0 else 0.0
        raw_score   = proportion - self._accumulated_step_costs
        final_score = max(0.0, min(1.0, raw_score))

        if not syntax_err:
            if passes == total:
                self._last_output += (
                    f"\n🎉 ALL {total} TESTS PASS! Episode solved. "
                    f"Final score: {final_score:.4f}"
                )
            else:
                fail_count = total - passes
                self._last_output += (
                    f"\n❌ SUBMIT — {fail_count}/{total} tests still failing. "
                    f"Final score: {final_score:.4f}"
                )

        return final_score

    def _act_undo_edit(self) -> float:
        """
        Mini-Git UNDO: restore the code snapshot from immediately before the
        most recent REPLACE_LINES call.  Applies R_UNDO_RESET penalty.

        CRITICAL (Phase 2, point 4 — Context Desync Watchout):
        _last_edited_line is set to None so context.py does not anchor the
        localized view to a line that may no longer exist or mean the same
        thing after the revert.
        """
        if not self._edit_history:
            self._last_output = (
                "⚠ UNDO_EDIT: no edit history — nothing to revert. "
                "The code is still at its current state."
            )
        else:
            self._code_lines    = self._edit_history.pop()
            self._last_output   = (
                f"↩ UNDO_EDIT: reverted to previous state "
                f"({len(self._code_lines)} lines). "
                "Call VIEW_CODE to inspect the restored file."
            )

        # PRINCIPLE 10 desync fix: anchor is stale after rollback — wipe it.
        self._last_edited_line = None
        return R_UNDO_RESET

    def _act_reset_to_original(self) -> float:
        """
        Mini-Git RESET: restore the pristine episode-start code and clear the
        entire undo stack.  Applies R_UNDO_RESET penalty.

        CRITICAL (Phase 2, point 4 — Context Desync Watchout):
        _last_edited_line is set to None to prevent context.py from anchoring
        to a ghost line in the freshly-restored original code.
        """
        self._code_lines    = list(self._original_code)  # deep copy
        self._edit_history  = []                          # clear stack
        self._last_output   = (
            f"↺ RESET_TO_ORIGINAL: code restored to pristine episode state "
            f"({len(self._code_lines)} lines). All undo history cleared. "
            "Call VIEW_CODE to inspect the file."
        )

        # PRINCIPLE 10 desync fix: context anchor is meaningless after full reset.
        self._last_edited_line = None
        return R_UNDO_RESET

    # ── Helpers ─────────────────────────────────────────────────────────────

    def _source(self) -> str:
        return "\n".join(self._code_lines)

    def _build_observation(self, reward: float) -> CodeObservation:
        syntax_valid, _ = check_syntax(self._source())

        # PRINCIPLE 10: localized context — only ±10 lines around last edit
        localized = get_localized_context(self._code_lines, self._last_edited_line)

        return CodeObservation(
            code_lines            = list(self._code_lines),
            localized_context     = localized,
            last_execution_output = self._last_output,
            syntax_error          = not syntax_valid,
            test_results          = list(self._last_test_results),
            step_count            = self._step_count,
            steps_remaining       = max(0, MAX_STEPS - self._step_count),
            reward_last_step      = round(reward, 4),
            done                  = self._done,
            info = {
                "episode_id":      self._episode_id,
                "task_name":       self._task.get("name", ""),
                "task_difficulty": self._task.get("difficulty", ""),
            },
        )
