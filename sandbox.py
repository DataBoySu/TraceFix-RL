"""
sandbox.py — Safe Code Execution Sandbox
=========================================

PRINCIPLE 2 — Errors are Data, Not Control Flow
  If the agent writes code that throws SyntaxError, AssertionError, TypeError,
  RecursionError, or ANY other exception, the environment must NOT crash or
  propagate that exception to the server loop. Every possible failure mode is
  caught inside the child process, serialized into a string, and returned as
  structured data in the CodeObservation. The agent then reads this error text
  and adapts on its next turn.

PRINCIPLE 8 — Security is Per Invocation
  The sandbox executes arbitrary LLM-generated Python code. Two defences:
  1. TIMEOUT: The worker process is hard-killed (SIGKILL after SIGTERM) after
     EXEC_TIMEOUT_SECONDS. This stops while-True loops and CPU-exhaustion.
  2. RESTRICTED BUILTINS: exec() receives a controlled __builtins__ dict with
     dangerous callables (open, __import__, eval, exec, compile, breakpoint,
     input) replaced with safe stubs that raise RuntimeError. This prevents
     the agent from escaping the sandbox via filesystem or subprocess access.

PRINCIPLE 9 — Optimizations are MVP Requirements
  Python tracebacks can be thousands of lines. We tail-truncate to the last
  MAX_OUTPUT_CHARS characters. The tail of a traceback is the most actionable
  part (it contains the actual exception, not the call stack preamble).
  Prefix '[...truncated N chars...]' is added so the agent knows output was cut.
"""

from __future__ import annotations

import ast
import io
import inspect
import multiprocessing
import signal
import sys
import textwrap
import traceback
from typing import Any, Callable, Dict, List, Tuple

try:
    from .models import TestResult
except ImportError:
    from models import TestResult



EXEC_TIMEOUT_SECONDS: int = 5    # Hard wall-clock kill limit (Principle 8)
MAX_OUTPUT_CHARS: int = 1_000    # Tail-truncate limit (Principle 9)



def _make_safe_stub(name: str) -> Callable:
    """Return a callable that raises RuntimeError — used to block dangerous builtins."""
    def _stub(*args, **kwargs):
        raise RuntimeError(
            f"'{name}' is disabled in the sandbox. "
            "Do not attempt to access the filesystem, import modules dynamically, "
            "or execute arbitrary code within your solution."
        )
    _stub.__name__ = name
    return _stub


_SAFE_BUILTINS: Dict[str, Any] = {
    "int": int, "float": float, "str": str, "bool": bool,
    "list": list, "dict": dict, "set": set, "tuple": tuple,
    "bytes": bytes, "bytearray": bytearray, "frozenset": frozenset,
    "complex": complex,
    "len": len, "range": range, "enumerate": enumerate, "zip": zip,
    "map": map, "filter": filter, "reversed": reversed, "sorted": sorted,
    "iter": iter, "next": next, "sum": sum, "min": min, "max": max,
    "abs": abs, "round": round, "divmod": divmod, "pow": pow,
    "isinstance": isinstance, "issubclass": issubclass, "type": type,
    "hasattr": hasattr, "getattr": getattr, "setattr": setattr,
    "callable": callable, "repr": repr, "hash": hash, "id": id,
    "print": print,
    "Exception": Exception, "ValueError": ValueError, "TypeError": TypeError,
    "KeyError": KeyError, "IndexError": IndexError, "AttributeError": AttributeError,
    "StopIteration": StopIteration, "RuntimeError": RuntimeError,
    "AssertionError": AssertionError, "NotImplementedError": NotImplementedError,
    "OverflowError": OverflowError, "ZeroDivisionError": ZeroDivisionError,
    "RecursionError": RecursionError, "MemoryError": MemoryError,
    "KeyboardInterrupt": KeyboardInterrupt,
    "BaseException": BaseException,
    "any": any, "all": all,
    "chr": chr, "ord": ord, "hex": hex, "oct": oct, "bin": bin,
    "format": format,
    "object": object, "property": property, "staticmethod": staticmethod,
    "classmethod": classmethod, "super": super,
    "open":        _make_safe_stub("open"),
    "__import__":  _make_safe_stub("__import__"),
    "eval":        _make_safe_stub("eval"),
    "exec":        _make_safe_stub("exec"),
    "compile":     _make_safe_stub("compile"),
    "breakpoint":  _make_safe_stub("breakpoint"),
    "input":       _make_safe_stub("input"),
    "globals":     _make_safe_stub("globals"),
    "locals":      _make_safe_stub("locals"),
    "vars":        _make_safe_stub("vars"),
    "dir":         _make_safe_stub("dir"),
    "__loader__":  None,
    "__spec__":    None,
}



def _tail_truncate(s: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    """
    Return the TAIL of `s`, bounded to `limit` characters.

    Rationale: Python tracebacks print in chronological call order — the most
    actionable information (the actual exception type and message) appears at
    the very END of the traceback, not the beginning. Tail-truncation therefore
    preserves the signal the agent needs while discarding verbose call stacks.
    """
    if len(s) <= limit:
        return s
    dropped = len(s) - limit
    return f"[...truncated {dropped} chars...]\n" + s[-limit:]



def _worker(
    source: str,
    test_sources: List[str],
    result_queue: multiprocessing.Queue,
) -> None:
    """
    Isolated execution unit. Never raises — all failures become data.

    PRINCIPLE 2: Every exception path is caught and serialized.
    PRINCIPLE 8: exec() receives the restricted builtins dict.
    """
    buf = io.StringIO()
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout = sys.stderr = buf

    test_results: List[Dict] = []
    had_syntax_error = False
    fn_name = "<unknown>"

    try:
        try:
            code_obj = compile(source, "<agent_code>", "exec")
        except SyntaxError as exc:
            had_syntax_error = True
            sys.stdout, sys.stderr = old_stdout, old_stderr
            err = f"SyntaxError at line {exc.lineno}: {exc.msg}\n  >> {exc.text or ''}"
            result_queue.put((_tail_truncate(err), [], True))
            return

        namespace: Dict[str, Any] = {"__builtins__": __builtins__}
        try:
            exec(code_obj, namespace)  # noqa: S102
        except Exception:  # noqa: BLE001
            tb = traceback.format_exc()
            sys.stdout, sys.stderr = old_stdout, old_stderr
            result_queue.put((_tail_truncate(buf.getvalue() + "\n" + tb), [], False))
            return

        for test_src in test_sources:
            fn_name = "<unknown>"
            try:
                exec(test_src, namespace)  # noqa: S102

                fn_name = [
                    ln.split("(")[0].replace("def ", "").strip()
                    for ln in test_src.splitlines()
                    if ln.startswith("def ")
                ][-1]

                namespace[fn_name](namespace)
                test_results.append({"test_name": fn_name, "passed": True})

            except AssertionError as exc:
                test_results.append({
                    "test_name": fn_name,
                    "passed": False,
                    "error_message": _tail_truncate(
                        f"AssertionError: {exc}" if str(exc) else "AssertionError (no message)"
                    ),
                })
            except Exception:  # noqa: BLE001
                test_results.append({
                    "test_name": fn_name,
                    "passed": False,
                    "error_message": _tail_truncate(traceback.format_exc()),
                })

    except Exception:  # noqa: BLE001
        traceback.print_exc(file=buf)
    finally:
        sys.stdout, sys.stderr = old_stdout, old_stderr

    captured = _tail_truncate(buf.getvalue())
    result_queue.put((captured, test_results, had_syntax_error))



def check_syntax(source: str) -> Tuple[bool, str]:
    """
    Fast syntax check via ast.parse() — no execution, no subprocess overhead.

    Returns (is_valid, error_description).
    Called on every observation build to keep syntax_error field current.
    """
    try:
        ast.parse(source)
        return True, ""
    except SyntaxError as exc:
        return False, f"SyntaxError at line {exc.lineno}: {exc.msg}"


def run_code_with_tests(
    source: str,
    test_callables: List[Callable],
    timeout: int = EXEC_TIMEOUT_SECONDS,
) -> Tuple[str, List[TestResult], bool]:
    """
    Execute `source` with restricted builtins and run each test callable.

    PRINCIPLE 8 — hard timeout enforced via multiprocessing:
      proc.join(timeout) → if still alive → SIGTERM → SIGKILL → proceed.

    PRINCIPLE 2 — all outcomes return as data:
      timeout     → ("⏱ timed out", [], False)
      dead proc   → ("process exited unexpectedly", [], False)
      normal run  → (stdout_stderr, [TestResult...], had_syntax_error)

    Returns
    -------
    (output_str, test_results, had_syntax_error)
    """
    test_sources = [
        textwrap.dedent(inspect.getsource(fn))
        for fn in test_callables
    ]

    q: multiprocessing.Queue = multiprocessing.Queue()
    proc = multiprocessing.Process(
        target=_worker,
        args=(source, test_sources, q),
        daemon=True,  # Dies automatically if parent exits
    )
    proc.start()
    proc.join(timeout)

    if proc.is_alive():
        proc.terminate()
        proc.join(2)   # Give it 2s to handle SIGTERM gracefully
        if proc.is_alive():
            proc.kill()  # SIGKILL — unconditional
            proc.join()
        return (
            f"⏱ Execution timed out after {timeout}s. "
            "Your code contains an infinite loop or is too slow. "
            "Fix the logic and try again.",
            [],
            False,
        )

    if q.empty():
        return "Process exited unexpectedly with no output.", [], False

    raw_output, raw_results, syntax_err = q.get_nowait()
    test_results = [TestResult(**r) for r in raw_results]
    return raw_output, test_results, syntax_err
