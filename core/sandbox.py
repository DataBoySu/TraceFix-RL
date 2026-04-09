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
import importlib
import signal
import sys
import textwrap
import traceback
from typing import Any, Callable, Dict, List, Set, Tuple

try:
    from .models import TestResult
except ImportError:
    from core.models import TestResult



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


TEST_SUITE_ALLOWED_MODULES: Set[str] = {
    "bisect",
    "collections",
    "functools",
    "heapq",
    "itertools",
    "math",
    "re",
    "string",
    "typing",
}


SAFE_BUILTINS: Dict[str, Any] = {
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


def _sanitize_imports_and_prepare_bindings(
    source: str,
    allowed_modules: Set[str],
) -> Tuple[str, List[Tuple[str, str, str]], List[Tuple[str, str]]]:
    """
    Parse source, validate imports against allowlist, and strip import statements.

    Returns
    -------
    sanitized_source:
      Source with all import statements removed (so code never calls __import__).
    module_alias_bindings:
      List[(local_name, module_name, attribute_name)].
      `attribute_name == ""` means bind module object itself.
    modules_to_preload:
      List[(root_name, import_target)] pairs.
    """
    tree = ast.parse(source)
    blocked_lines: Set[int] = set()
    module_alias_bindings: List[Tuple[str, str, str]] = []
    modules_to_preload: Set[Tuple[str, str]] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                module_name = alias.name
                root_name = module_name.split(".")[0]
                if root_name not in allowed_modules:
                    raise ImportError(
                        f"Import of '{root_name}' is not allowed in this sandbox."
                    )
                local_name = alias.asname or root_name
                module_alias_bindings.append((local_name, module_name, ""))
                modules_to_preload.add((root_name, module_name))
            if hasattr(node, "lineno") and hasattr(node, "end_lineno"):
                blocked_lines.update(range(node.lineno, node.end_lineno + 1))

        if isinstance(node, ast.ImportFrom):
            if node.level != 0 or not node.module:
                raise ImportError(
                    "Relative imports are not allowed in this sandbox."
                )
            module_name = node.module
            root_name = module_name.split(".")[0]
            if root_name not in allowed_modules:
                raise ImportError(
                    f"Import of '{root_name}' is not allowed in this sandbox."
                )
            for alias in node.names:
                if alias.name == "*":
                    raise ImportError(
                        "Wildcard imports are not allowed in this sandbox."
                    )
                local_name = alias.asname or alias.name
                module_alias_bindings.append((local_name, module_name, alias.name))
            modules_to_preload.add((root_name, module_name))
            if hasattr(node, "lineno") and hasattr(node, "end_lineno"):
                blocked_lines.update(range(node.lineno, node.end_lineno + 1))

    sanitized_lines = [
        line
        for i, line in enumerate(source.splitlines(), start=1)
        if i not in blocked_lines
    ]
    return "\n".join(sanitized_lines), module_alias_bindings, sorted(modules_to_preload)


def _build_local_env_for_source(
    source: str,
    allowed_modules: Set[str],
) -> Tuple[str, Dict[str, Any]]:
    """
    Build a local env with preloaded authorized modules/symbols.
    """
    sanitized_source, bindings, modules_to_preload = _sanitize_imports_and_prepare_bindings(
        source, allowed_modules
    )
    local_env: Dict[str, Any] = {}
    loaded_modules: Dict[str, Any] = {}

    for root_name, import_target in modules_to_preload:
        if import_target not in loaded_modules:
            loaded_modules[import_target] = importlib.import_module(import_target)
        if root_name not in loaded_modules:
            loaded_modules[root_name] = importlib.import_module(root_name)

    for local_name, module_name, attribute_name in bindings:
        module_obj = loaded_modules[module_name]
        if attribute_name:
            local_env[local_name] = getattr(module_obj, attribute_name)
        else:
            local_env[local_name] = module_obj

    return sanitized_source, local_env



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

        try:
            sanitized_source, local_env = _build_local_env_for_source(
                source,
                TEST_SUITE_ALLOWED_MODULES,
            )
            exec_env: Dict[str, Any] = {"__builtins__": SAFE_BUILTINS}
            exec_env.update(local_env)
            code_obj = compile(sanitized_source, "<agent_code>", "exec")
            exec(code_obj, exec_env, exec_env)  # noqa: S102
        except Exception:  # noqa: BLE001
            tb = traceback.format_exc()
            sys.stdout, sys.stderr = old_stdout, old_stderr
            result_queue.put((_tail_truncate(buf.getvalue() + "\n" + tb), [], False))
            return

        for test_src in test_sources:
            fn_name = "<unknown>"
            try:
                sanitized_test_src, test_env_injections = _build_local_env_for_source(
                    test_src,
                    TEST_SUITE_ALLOWED_MODULES,
                )
                exec_env.update(test_env_injections)
                exec(
                    compile(sanitized_test_src, "<sandbox_test>", "exec"),
                    exec_env,
                    exec_env,
                )  # noqa: S102

                fn_name = [
                    ln.split("(")[0].replace("def ", "").strip()
                    for ln in sanitized_test_src.splitlines()
                    if ln.startswith("def ")
                ][-1]

                exec_env[fn_name](exec_env)
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
