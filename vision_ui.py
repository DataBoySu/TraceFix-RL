from __future__ import annotations

import html
import json
import os
import queue
import re
import subprocess
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Generator

import gradio as gr

try:
    from tasks import tasks
    ALL_TASKS = tasks.ALL_TASKS
except Exception:
    tasks = None
    ALL_TASKS = []


ROOT_DIR = Path(__file__).resolve().parent
INFERENCE_PATH = ROOT_DIR / "inference.py"
BACKEND_HOST = "127.0.0.1"
BACKEND_PORT = 8000
GRADIO_HOST = "0.0.0.0"
GRADIO_PORT = 7860

START_RE = re.compile(r"^\[START\]\s+task=(?P<task>\S+)\s+env=(?P<env>\S+)\s+model=(?P<model>.+)$")
STEP_RE = re.compile(
    r"^\[STEP\]\s+step=(?P<step>\d+)\s+action=(?P<action>[A-Z_]+)\s+"
    r"reward=(?P<reward>-?\d+(?:\.\d+)?)\s+done=(?P<done>true|false)\s+error=(?P<error>.*)$"
)
END_RE = re.compile(
    r"^\[END\]\s+success=(?P<success>true|false)\s+steps=(?P<steps>\d+)\s+"
    r"score=(?P<score>-?\d+(?:\.\d+)?)\s+rewards=(?P<rewards>.*)$"
)

TASK_MAP: dict[str, dict[str, Any]] = {
    str(task.get("name", "")): task
    for task in ALL_TASKS
    if isinstance(task, dict) and task.get("name")
}


CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

:root {
  --bg-top: #0f1115;
  --bg-bottom: #1a1e27;
  --panel: rgba(255, 255, 255, 0.04);
  --panel-border: rgba(255, 255, 255, 0.12);
  --text-main: #e7e9ef;
  --text-dim: #aab1c2;
  --accent: #91c6ff;
  --ok: #6ce7b5;
  --warn: #f9d78b;
  --err: #ff9b9b;
}

.gradio-container {
  font-family: 'Inter', sans-serif !important;
  background: radial-gradient(circle at 20% 0%, #202636 0%, transparent 40%),
              linear-gradient(180deg, var(--bg-top) 0%, var(--bg-bottom) 100%);
  color: var(--text-main);
}

#header-wrap {
  margin-bottom: 10px;
  border: 1px solid var(--panel-border);
  background: var(--panel);
  border-radius: 16px;
  padding: 16px 20px;
}

#header-wrap h1 {
  margin: 0;
  letter-spacing: 0.2px;
  font-weight: 600;
  color: #f5f7fb;
}

#header-wrap p {
  margin: 6px 0 0;
  color: var(--text-dim);
}

.panel {
  border: 1px solid var(--panel-border);
  border-radius: 16px;
  background: var(--panel);
  overflow: hidden;
}

.panel-title {
  padding: 10px 14px;
  border-bottom: 1px solid var(--panel-border);
  color: var(--text-dim);
  font-size: 12px;
  letter-spacing: 0.09em;
  text-transform: uppercase;
}

.code-panel * {
  font-family: 'JetBrains Mono', monospace !important;
}

.terminal-wrap {
  height: 620px;
  overflow-y: auto;
  padding: 12px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 12px;
  line-height: 1.55;
  background: #0c0f16;
}

.term-line {
  white-space: pre-wrap;
  word-break: break-word;
}

.term-step { color: var(--accent); }
.term-start { color: #c8d7ff; }
.term-end { color: var(--ok); font-weight: 600; }
.term-thought { color: #b9c7ff; }
.term-error { color: var(--err); }
.term-muted { color: var(--text-dim); }

.metric {
  border: 1px solid var(--panel-border);
  background: var(--panel);
  border-radius: 14px;
  padding: 12px;
}
"""


def _code_from_task_name(task_name: str) -> str:
    task = TASK_MAP.get((task_name or "").strip())
    if not task:
        return (
            "# Waiting for mission start...\n"
            "# Tip: Set TASK_NAME to one of the known tasks from tasks.py\n"
            "# so the buggy sandbox code can be previewed before launch."
        )
    return "\n".join(task.get("code", []))


def _normalize_base_url(base_url: str) -> str:
    candidate = (base_url or "").strip()
    if not candidate:
        return f"http://{BACKEND_HOST}:{BACKEND_PORT}"
    if not candidate.startswith(("http://", "https://")):
        candidate = f"http://{candidate}"
    return candidate.rstrip("/")


def _code_from_openenv(task_name: str, env_base_url: str) -> str | None:
    normalized_url = _normalize_base_url(env_base_url)
    task_key = (task_name or "").strip()
    if not task_key:
        return None

    candidates = [
        f"{normalized_url}/tasks/{task_key}/code",
        f"{normalized_url}/task/{task_key}/code",
        f"{normalized_url}/tasks/{task_key}",
        f"{normalized_url}/task/{task_key}",
    ]

    for url in candidates:
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=3) as response:
                if response.status != 200:
                    continue
                payload = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError):
            continue

        if isinstance(payload, dict):
            code = payload.get("code")
            if isinstance(code, list):
                return "\n".join(str(line) for line in code)
            if isinstance(code, str):
                return code

            task_data = payload.get("task")
            if isinstance(task_data, dict):
                task_code = task_data.get("code")
                if isinstance(task_code, list):
                    return "\n".join(str(line) for line in task_code)
                if isinstance(task_code, str):
                    return task_code
    return None


def load_code(task_name: str, env_base_url: str) -> str:
    local_code = _code_from_task_name(task_name)
    if "Waiting for mission start" not in local_code:
        return local_code

    api_code = _code_from_openenv(task_name, env_base_url)
    if api_code:
        return api_code

    return (
        "# Unable to load code for the selected task.\n"
        "# Verify Task / Bug Selection and confirm OpenEnv API is reachable."
    )


def _solution_from_task_name(task_name: str) -> str | None:
    task = TASK_MAP.get((task_name or "").strip())
    if not task:
        return None
    return "\n".join(task.get("solution", []))


def _terminal_html(lines: list[tuple[str, str]]) -> str:
    rendered: list[str] = []
    for css_class, text in lines:
        safe = html.escape(text)
        rendered.append(f"<div class='term-line {css_class}'>{safe}</div>")
    content = "\n".join(rendered) if rendered else "<div class='term-line term-muted'>Idle. Configure mission variables and press Run Agent.</div>"
    return (
        "<div id='terminal' class='terminal-wrap'>"
        f"{content}"
        "</div>"
        "<script>"
        "const t=document.getElementById('terminal'); if(t){t.scrollTop=t.scrollHeight;}"
        "</script>"
    )


def _metric_block(state: str, details: str) -> str:
    return (
        "<div class='metric'>"
        f"<div><strong>{html.escape(state)}</strong></div>"
        f"<div style='color:var(--text-dim); margin-top: 6px'>{html.escape(details)}</div>"
        "</div>"
    )


def _reader_thread(stream: Any, source: str, out_q: queue.Queue[tuple[str, str | None]]) -> None:
    try:
        for raw in iter(stream.readline, ""):
            out_q.put((source, raw.rstrip("\n")))
    finally:
        try:
            stream.close()
        except Exception:
            pass
        out_q.put((source, None))


def _build_env(
    hf_token: str,
    api_base_url: str,
    model_name: str,
    env_base_url: str,
    task_name: str,
    benchmark: str,
    max_steps: int,
    success_score_threshold: float,
    local_image_name: str,
) -> dict[str, str]:
    env = os.environ.copy()
    updates = {
        "HF_TOKEN": hf_token,
        "API_BASE_URL": api_base_url,
        "MODEL_NAME": model_name,
        "ENV_BASE_URL": _normalize_base_url(env_base_url),
        "TASK_NAME": task_name,
        "BENCHMARK": benchmark,
        "MAX_STEPS": str(int(max_steps)),
        "SUCCESS_SCORE_THRESHOLD": str(float(success_score_threshold)),
        "LOCAL_IMAGE_NAME": local_image_name,
    }
    for key, value in updates.items():
        cleaned = (value or "").strip()
        if cleaned:
            env[key] = cleaned
        elif key in env:
            env.pop(key, None)
    return env


def _reset_run_state(task_name: str) -> tuple[str, str, str, float, str]:
    return (
        _code_from_task_name(task_name),
        _terminal_html([]),
        _metric_block("Mission Ready", "Awaiting [START] from inference subprocess..."),
        0.0,
        "`Rewards:` pending",
    )


def run_agent(
    hf_token: str,
    api_base_url: str,
    model_name: str,
    env_base_url: str,
    task_name: str,
    benchmark: str,
    max_steps: int,
    success_score_threshold: float,
    local_image_name: str,
    difficulty: str,
    show_thought: bool,
) -> Generator[tuple[str, str, str, float, str], None, None]:
    code_view = _code_from_task_name(task_name)
    terminal_lines: list[tuple[str, str]] = []
    terminal_lines.append(("term-muted", "Boot sequence initialized."))

    status_html = _metric_block("Mission Ready", "Launching inference subprocess...")
    score_value = 0.0
    rewards_md = "`Rewards:` pending"
    yield code_view, _terminal_html(terminal_lines), status_html, score_value, rewards_md

    cmd = [sys.executable, str(INFERENCE_PATH)]
    if difficulty in {"easy", "medium", "hard"}:
        cmd.append(f"--{difficulty}")
    if show_thought:
        cmd.append("--thought")

    env = _build_env(
        hf_token,
        api_base_url,
        model_name,
        env_base_url,
        task_name,
        benchmark,
        max_steps,
        success_score_threshold,
        local_image_name,
    )

    process = subprocess.Popen(
        cmd,
        cwd=str(ROOT_DIR),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    out_q: queue.Queue[tuple[str, str | None]] = queue.Queue()
    stdout_thread = threading.Thread(target=_reader_thread, args=(process.stdout, "stdout", out_q), daemon=True)
    stderr_thread = threading.Thread(target=_reader_thread, args=(process.stderr, "stderr", out_q), daemon=True)
    stdout_thread.start()
    stderr_thread.start()

    ended_streams: set[str] = set()
    thought_mode = False
    active_task_name = (task_name or "").strip()
    final_steps = 0

    while True:
        try:
            source, line = out_q.get(timeout=0.15)
        except queue.Empty:
            if process.poll() is not None and ended_streams == {"stdout", "stderr"}:
                break
            continue

        if line is None:
            ended_streams.add(source)
            if process.poll() is not None and ended_streams == {"stdout", "stderr"}:
                break
            continue

        if source == "stderr":
            if line.strip() == "[THOUGHT]":
                thought_mode = True
                terminal_lines.append(("term-thought", "[THOUGHT]"))
            elif line.startswith("[") and line.endswith("]"):
                thought_mode = False
                terminal_lines.append(("term-muted", line))
            elif thought_mode:
                terminal_lines.append(("term-thought", line))
            else:
                terminal_lines.append(("term-error", line))
        else:
            start_match = START_RE.match(line)
            step_match = STEP_RE.match(line)
            end_match = END_RE.match(line)

            if start_match:
                active_task_name = start_match.group("task").strip()
                task_preview = _code_from_task_name(active_task_name)
                if "Waiting for mission start" not in task_preview:
                    code_view = task_preview
                terminal_lines.append(("term-start", line))
                status_html = _metric_block(
                    "Mission Running",
                    f"task={active_task_name} | env={start_match.group('env')} | model={start_match.group('model')}",
                )
            elif step_match:
                final_steps = int(step_match.group("step"))
                action = step_match.group("action")
                reward = float(step_match.group("reward"))
                done_flag = step_match.group("done") == "true"
                err = step_match.group("error")
                css = "term-step" if err == "null" else "term-error"
                terminal_lines.append((css, line))
                status_html = _metric_block(
                    "Mission Running",
                    f"step={final_steps} action={action} reward={reward:.2f} done={str(done_flag).lower()}",
                )
            elif end_match:
                success = end_match.group("success") == "true"
                final_steps = int(end_match.group("steps"))
                score_value = float(end_match.group("score"))
                rewards_raw = end_match.group("rewards").strip()
                rewards_md = f"`Rewards:` {rewards_raw or 'none'}"
                terminal_lines.append(("term-end", line))
                if success:
                    solved = _solution_from_task_name(active_task_name)
                    if solved:
                        code_view = solved
                    status_html = _metric_block(
                        "Mission Success",
                        f"score={score_value:.2f} | steps={final_steps}",
                    )
                else:
                    status_html = _metric_block(
                        "Mission Failed",
                        f"score={score_value:.2f} | steps={final_steps}",
                    )
            else:
                terminal_lines.append(("term-muted", line))

        if len(terminal_lines) > 500:
            terminal_lines = terminal_lines[-500:]

        yield code_view, _terminal_html(terminal_lines), status_html, score_value, rewards_md

    return_code = process.wait(timeout=2)
    if return_code != 0:
        terminal_lines.append(("term-error", f"Process exited with code {return_code}."))
        status_html = _metric_block(
            "Mission Error",
            f"inference.py exited non-zero (code={return_code})",
        )

    if len(terminal_lines) > 500:
        terminal_lines = terminal_lines[-500:]

    yield code_view, _terminal_html(terminal_lines), status_html, score_value, rewards_md


with gr.Blocks(title="TraceFix-RL Mission Control", css=CSS) as demo:
    gr.HTML(
        """
        <div id='header-wrap'>
          <h1>TraceFix-RL: Autonomous Debugging Agent</h1>
          <p>Mission Control UI for real-time agent orchestration on Hugging Face Spaces.</p>
        </div>
        """
    )

    if hasattr(gr, "Sidebar"):
        sidebar_context = gr.Sidebar()
    else:
        sidebar_context = gr.Column()

    with sidebar_context:
        gr.Markdown("### Runtime Inputs")
        hf_token = gr.Textbox(label="HF Token", type="password", placeholder="hf_xxx")
        task_choices = sorted(TASK_MAP.keys())
        selected_task = os.getenv("TASK_NAME", "")
        with gr.Row():
            task_name = gr.Dropdown(
                label="Task / Bug Selection",
                choices=task_choices,
                value=selected_task if selected_task else None,
                allow_custom_value=True,
                interactive=True,
            )
            load_code_button = gr.Button("Load Code")
        model_name = gr.Textbox(label="Model Name", value=os.getenv("MODEL_NAME", "openai/gpt-oss-20b"))
        api_base_url = gr.Textbox(label="API Base URL", value=os.getenv("API_BASE_URL", "https://router.huggingface.co/v1"))
        env_base_url = gr.Textbox(label="Env Base URL", value=os.getenv("ENV_BASE_URL", f"http://{BACKEND_HOST}:{BACKEND_PORT}"))
        benchmark = gr.Textbox(label="Benchmark", value=os.getenv("BENCHMARK", "tracefix_rl"))
        local_image_name = gr.Textbox(label="Local Image Name", value=os.getenv("LOCAL_IMAGE_NAME", ""), placeholder="optional")
        max_steps = gr.Number(label="Max Steps", value=int(os.getenv("MAX_STEPS", "50")), precision=0)
        success_score_threshold = gr.Number(
            label="Success Score Threshold",
            value=float(os.getenv("SUCCESS_SCORE_THRESHOLD", "0.99")),
            precision=2,
        )
        difficulty = gr.Dropdown(label="Difficulty", choices=["auto", "easy", "medium", "hard"], value="auto")
        show_thought = gr.Checkbox(label="Stream Thought Trace", value=True)
        run_button = gr.Button("Run Agent", variant="primary")

    with gr.Row(equal_height=True):
        with gr.Column(scale=1, elem_classes=["panel", "code-panel"]):
            gr.HTML("<div class='panel-title'>The Sandbox</div>")
            code_view = gr.Code(
                language="python",
                interactive=False,
                value=_code_from_task_name(selected_task),
                lines=30,
            )

        with gr.Column(scale=1, elem_classes=["panel"]):
            gr.HTML("<div class='panel-title'>The Terminal</div>")
            terminal = gr.HTML(_terminal_html([]))

    with gr.Row():
        metric = gr.HTML(_metric_block("Idle", "Waiting for launch."))
        score = gr.Number(label="Final Score", value=0.0, precision=3)
        rewards = gr.Markdown("`Rewards:` pending")

    load_code_button.click(load_code, inputs=[task_name, env_base_url], outputs=[code_view])

    run_event = run_button.click(
        _reset_run_state,
        inputs=[task_name],
        outputs=[code_view, terminal, metric, score, rewards],
        queue=False,
    )

    run_event.then(
        run_agent,
        inputs=[
            hf_token,
            api_base_url,
            model_name,
            env_base_url,
            task_name,
            benchmark,
            max_steps,
            success_score_threshold,
            local_image_name,
            difficulty,
            show_thought,
        ],
        outputs=[code_view, terminal, metric, score, rewards],
    )
