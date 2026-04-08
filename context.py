"""
context.py — Layered Context Compaction
=========================================

PRINCIPLE 10 — Layered Context Compaction
  For large files, returning the full source on every observation would rapidly
  fill the agent's context window, leaving no room for reasoning.

  Instead we return a *localized* view: a ±WINDOW_LINES slice of the code
  centred on the last line that was edited. This gives the agent exactly the
  context it needs — the neighbourhood of its most recent change — without
  flooding the context with unrelated code.

  This module is intentionally pure (no environment state dependencies) so
  it can be unit-tested independently and reused across environment versions.
"""

from __future__ import annotations

from typing import List, Optional

# How many lines above and below the anchor to include
WINDOW_LINES: int = 10

# Maximum characters for the localized context block
# (Principle 9: all outputs must be bounded)
MAX_CONTEXT_CHARS: int = 2_000


def get_localized_context(
    code_lines: List[str],
    anchor_line: Optional[int],
    window: int = WINDOW_LINES,
) -> str:
    """
    Return a ±`window`-line slice of `code_lines` centred on `anchor_line`.

    Parameters
    ----------
    code_lines  : Full list of source lines (0-indexed internally).
    anchor_line : The 1-indexed line number of the most recent edit.
                  If None (no edits yet) returns an empty string.
    window      : Number of lines to show above and below the anchor.

    Returns
    -------
    A formatted string with line numbers, bounded to MAX_CONTEXT_CHARS,
    annotated with the visible range and an anchor marker (▶).

    Example output
    --------------
    [Showing lines 3–13 of 20, anchor ▶ line 7]
      3 |     left, right = 0, len(arr)
      4 |     while left <= right:
      5 |         mid = (left + right) // 2
      6 |         if arr[mid] == target:
      7 ▶         return mid          ← last edit
      8 |         elif arr[mid] < target:
      9 |             left = mid + 1
     10 |         else:
     11 |             right = mid - 1
     12 |     return -1
    """
    if anchor_line is None or not code_lines:
        return ""

    total = len(code_lines)

    # Clamp anchor into valid range
    anchor_0 = max(0, min(anchor_line - 1, total - 1))

    # Compute slice bounds (inclusive on both ends, 0-indexed)
    start_0 = max(0, anchor_0 - window)
    end_0   = min(total - 1, anchor_0 + window)

    # Build header
    start_1 = start_0 + 1
    end_1   = end_0   + 1
    header  = f"[Showing lines {start_1}–{end_1} of {total}, anchor ▶ line {anchor_line}]"

    # Build body
    body_lines = []
    for i in range(start_0, end_0 + 1):
        line_num = i + 1
        marker   = "▶" if i == anchor_0 else "|"
        body_lines.append(f"{line_num:>4} {marker} {code_lines[i]}")

    result = header + "\n" + "\n".join(body_lines)

    # PRINCIPLE 9 — hard cap on output size
    if len(result) > MAX_CONTEXT_CHARS:
        result = result[:MAX_CONTEXT_CHARS] + "\n... [context truncated]"

    return result