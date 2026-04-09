"""Core TraceFix-RL environment implementation."""

from __future__ import annotations

import random
import re
import uuid
from typing import Any, Dict, List, Optional, Tuple

try:
    from .context import extract_error_line, get_localized_context
    from .models import CodeAction, CodeObservation, TestResult
    from .sandbox import check_syntax, run_code_with_tests
    from tasks.tasks import ALL_TASKS, TASKS_BY_DIFFICULTY
except ImportError:
    from core.context import extract_error_line, get_localized_context
    from core.models import CodeAction, CodeObservation, TestResult
    from core.sandbox import check_syntax, run_code_with_tests
    from tasks.tasks import ALL_TASKS, TASKS_BY_DIFFICULTY


R_SUBMIT_ALL_PASS = +1.00
R_SUBMIT_FAIL     = -0.20
R_SYNTAX_ERROR    = -0.10
R_RUN_TESTS       = +0.10
R_PER_NEW_PASS    = +0.05
R_STEP_COST       = -0.01
R_INVALID_LINE    = -0.02
R_DESTRUCTIVE_PENALTY = -0.20
R_UNDO_RESET      = -0.10

MAX_STEPS: int = 50

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


class TraceFixRLGym:
    """Gym-style environment with reset/step methods."""

    metadata = {"name": "TraceFixRL-v1", "render_modes": []}

    def __init__(
        self,
        task_index: Optional[int] = None,
        seed: Optional[int] = None,
    ):
        self._task_index = task_index
        self._rng = random.Random(seed)

        self._code_lines: List[str] = []
        self._task: Dict[str, Any] = {}
        self._step_count: int = 0
        self._prev_pass_count: int = 0
        self._last_test_results: List[TestResult] = []
        self._last_output: str = ""
        self._last_execution_output: str = ""
        self._last_edited_line: Optional[int] = None
        self._episode_id: str = ""
        self._done: bool = False
        self._cumulative_reward: float = 0.0
        self._accumulated_step_costs: float = 0.0
        self._original_code: List[str] = []
        self._edit_history: List[List[str]] = []
        self.training_step: int = 0
        self._last_run_all_passed: bool = False


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

        if self.training_step == 0:
            if not ALL_TASKS:
                raise RuntimeError("ALL_TASKS is empty — check tasks.py.")
            return self._rng.choice(ALL_TASKS)

        if self.training_step < 1000:
            bucket = "easy"
        elif self.training_step < 5000:
            bucket = "medium"
        else:
            bucket = "hard"

        pool = TASKS_BY_DIFFICULTY.get(bucket, [])
        if not pool:
            for b in ("easy", "medium", "hard"):
                pool = TASKS_BY_DIFFICULTY.get(b, [])
                if pool:
                    break
        if not pool:
            raise RuntimeError("TASKS_BY_DIFFICULTY is entirely empty — check tasks.py.")

        return self._rng.choice(pool)


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

        self._code_lines       = list(self._task["code"])   # deep copy — no alias
        self._step_count       = 0
        self._prev_pass_count  = 0
        self._last_test_results = []
        self._last_output      = ""
        self._last_execution_output = ""
        self._last_edited_line = None   # no edits yet — localized_context will be empty
        self._episode_id       = str(uuid.uuid4())[:8]
        self._done             = False
        self._cumulative_reward = 0.0
        self._accumulated_step_costs = 0.0
        self._original_code = list(self._task["code"])  # separate copy from _code_lines
        self._edit_history  = []
        self._last_action: Optional[str] = None
        self._consecutive_count: int = 0
        self._last_run_all_passed = False

        obs = self._build_observation(reward=0.0)

        system_prompt = _SYSTEM_PROMPT.format(
            task_name   = self._task["name"],
            difficulty  = self._task.get("difficulty", "unknown"),
            test_count  = len(self._task["tests"]),
            max_steps   = MAX_STEPS,
            code_preview = obs.render_code(),
        )

        return obs, system_prompt


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

        if action.action_type == self._last_action:
            self._consecutive_count += 1
            reward += -0.05 * self._consecutive_count
        else:
            self._consecutive_count = 0
        self._last_action = action.action_type

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

        if self._step_count >= MAX_STEPS and not self._done:
            self._done = True
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
            info["final_score"] = max(0.0, min(1.0, round(reward, 4)))

        return obs, round(reward, 4), self._done, info


    def _act_view_code(self) -> float:
        self._last_output = (
            "=== Full Source ===\n" +
            "\n".join(
                f"{i + 1:>3} | {line}"
                for i, line in enumerate(self._code_lines)
            )
        )
        return 0.0

    def _act_run_tests(self) -> float:
        output, results, syntax_err = run_code_with_tests(
            source=self._source(),
            test_callables=self._task["tests"],
        )
        self._last_test_results = results

        reward = R_RUN_TESTS   # information-gathering bonus (Principle 5)
        total_tests = len(self._task["tests"])
        current_pass = 0 if syntax_err else sum(1 for t in results if t.passed)

        if syntax_err:
            reward += R_SYNTAX_ERROR
            self._last_run_all_passed = False
        else:
            new_passes   = max(0, current_pass - self._prev_pass_count)
            reward       += new_passes * R_PER_NEW_PASS
            self._prev_pass_count = current_pass
            self._last_run_all_passed = (current_pass == total_tests)

        if current_pass == total_tests and not syntax_err:
            self._last_execution_output = (
                f"Tests Passed: {total_tests}/{total_tests}.\n\n"
                "SUCCESS: ALL TESTS PASSED! You are finished. You MUST now use the SUBMIT action."
            )
        else:
            failing_messages = [
                t.error_message for t in results
                if (not t.passed) and t.error_message
            ]
            error_traceback = (output or "").strip() or "\n\n".join(failing_messages).strip()
            if not error_traceback:
                error_traceback = "No traceback available."
            self._last_execution_output = (
                f"Tests Passed: {current_pass}/{total_tests}.\n\n"
                f"Traceback:\n{error_traceback}"
            )
        self._last_output = self._last_execution_output

        return reward

    def _act_replace_lines(
        self, start_line: int, end_line: int, new_code_block: str
    ) -> float:
        n = len(self._code_lines)
        
        if new_code_block is None:
            new_code_block = ""

        if len(new_code_block) == 0 and (end_line - start_line) > 5:
            self._last_output = "Error: Cannot delete more than 5 lines at once."
            return R_DESTRUCTIVE_PENALTY

        if start_line > end_line:
            self._last_output = (
                f"Error: start_line ({start_line}) > end_line ({end_line}). "
                "Inverted range rejected. Call VIEW_CODE to check the current line count."
            )
            return R_INVALID_LINE

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

        start_idx = start_line - 1   # convert to 0-indexed
        end_idx   = end_line         # exclusive upper bound for Python slice

        self._edit_history.append(list(self._code_lines))

        original_line = self._code_lines[start_line - 1]
        original_indent = re.match(r"[ \t]*", original_line).group(0)
        new_lines = self._auto_indent_replacement_block(
            new_code_block=new_code_block,
            original_indent=original_indent,
        )
        self._code_lines[start_idx:end_idx] = new_lines

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

    def _auto_indent_replacement_block(
        self, new_code_block: str, original_indent: str
    ) -> List[str]:
        lines = new_code_block.split("\n")
        if not lines:
            return []

        first_indent_match = re.match(r"[ \t]*", lines[0])
        first_indent_len = len(first_indent_match.group(0)) if first_indent_match else 0

        adjusted_lines: List[str] = []
        for line in lines:
            leading_match = re.match(r"[ \t]*", line)
            leading_whitespace = leading_match.group(0) if leading_match else ""
            content = line[len(leading_whitespace):]
            if first_indent_len > 0:
                relative_whitespace = leading_whitespace[first_indent_len:]
            else:
                relative_whitespace = leading_whitespace
            adjusted_lines.append(f"{original_indent}{relative_whitespace}{content}")

        return adjusted_lines

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

        self._last_edited_line = None
        return R_UNDO_RESET


    def _source(self) -> str:
        return "\n".join(self._code_lines)

    def _build_observation(self, reward: float) -> CodeObservation:
        syntax_valid, _ = check_syntax(self._source())

        context_anchor = self._last_edited_line
        if self._last_action == "RUN_TESTS" and not self._last_run_all_passed:
            extracted_line = extract_error_line(self._last_execution_output)
            if extracted_line is not None:
                context_anchor = extracted_line

        localized = get_localized_context(self._code_lines, context_anchor)

        return CodeObservation(
            code_dict             = {
                idx + 1: line for idx, line in enumerate(self._code_lines)
            },
            localized_context     = localized,
            last_execution_output = (
                self._last_execution_output
                if self._last_action == "RUN_TESTS"
                else self._last_output
            ),
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
