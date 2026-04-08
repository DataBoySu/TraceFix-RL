"""Localized context helpers for TraceFix-RL."""

from __future__ import annotations

import re
from typing import List, Optional

WINDOW_LINES: int = 10

MAX_CONTEXT_CHARS: int = 2_000


_TRACEBACK_FILE_LINE_RE = re.compile(r'File "([^"]+)", line (\d+)')
_SYNTAX_LINE_RE = re.compile(r"SyntaxError at line (\d+)")


def extract_error_line(traceback_str: str) -> Optional[int]:
    """
    Extract the most relevant crashing line number from sandbox output.

    Preference order:
    1) Last frame pointing to agent code pseudo-files (<agent_code>, <string>).
    2) Last traceback frame line number.
    3) "SyntaxError at line N" fallback.
    """
    if not traceback_str:
        return None

    matches = _TRACEBACK_FILE_LINE_RE.findall(traceback_str)
    if matches:
        preferred_files = {"<agent_code>", "<string>"}
        for file_name, line_str in reversed(matches):
            if file_name in preferred_files:
                return int(line_str)
        return int(matches[-1][1])

    syntax_match = _SYNTAX_LINE_RE.search(traceback_str)
    if syntax_match:
        return int(syntax_match.group(1))

    return None


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
