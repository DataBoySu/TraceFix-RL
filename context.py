"""Localized context helpers for TraceFix-RL."""

from __future__ import annotations

from typing import List, Optional

WINDOW_LINES: int = 10

MAX_CONTEXT_CHARS: int = 2_000


def get_localized_context(
    code_lines: List[str],
    anchor_line: Optional[int],
    window: int = WINDOW_LINES,
) -> str:
    """Return a bounded ±window slice around the latest edited line."""
    if anchor_line is None or not code_lines:
        return ""

    total = len(code_lines)

    anchor_0 = max(0, min(anchor_line - 1, total - 1))
    start_0 = max(0, anchor_0 - window)
    end_0   = min(total - 1, anchor_0 + window)
    start_1 = start_0 + 1
    end_1   = end_0   + 1
    header  = f"[Showing lines {start_1}–{end_1} of {total}, anchor ▶ line {anchor_line}]"

    body_lines = []
    for i in range(start_0, end_0 + 1):
        line_num = i + 1
        marker   = "▶" if i == anchor_0 else "|"
        body_lines.append(f"{line_num:>4} {marker} {code_lines[i]}")

    result = header + "\n" + "\n".join(body_lines)

    if len(result) > MAX_CONTEXT_CHARS:
        result = result[:MAX_CONTEXT_CHARS] + "\n... [context truncated]"

    return result
