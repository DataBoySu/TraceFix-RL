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
    TASKS_BY_DIFFICULTY = tasks.TASKS_BY_DIFFICULTY
except Exception:
    tasks = None
    ALL_TASKS = []
    TASKS_BY_DIFFICULTY = {"easy": [], "medium": [], "hard": []}

EASY_CHOICES = [t.get("name") for t in TASKS_BY_DIFFICULTY.get("easy", []) if t.get("name")]
MEDIUM_CHOICES = [t.get("name") for t in TASKS_BY_DIFFICULTY.get("medium", []) if t.get("name")]
HARD_CHOICES = [t.get("name") for t in TASKS_BY_DIFFICULTY.get("hard", []) if t.get("name")]

ROOT_DIR = Path(__file__).resolve().parent
INFERENCE_PATH = ROOT_DIR / "inference.py"
BACKEND_HOST = "127.0.0.1"
BACKEND_PORT = 7860
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
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;900&family=JetBrains+Mono:wght@400;600;800&display=swap');

:root {
  --bg-top: #0a0a0a;
  --bg-bottom: #111111;
  --panel: #0a0a0a;
  --panel-border: rgba(255, 255, 255, 0.15);
  --text-main: #f5f5f5;
  --text-dim: #999;
  --accent: #E60012;
}

.gradio-container {
  font-family: 'Inter', sans-serif !important;
  background: var(--bg-top);
  color: var(--text-main);
  padding: 0px !important;
}

#header-wrap {
  margin-bottom: 2px;
  border: 1px solid var(--panel-border);
  background: #000;
  border-radius: 0px;
  padding: 8px 12px;
  text-transform: uppercase;
}

#header-wrap h1 {
  margin: 0;
  letter-spacing: 1px;
  font-weight: 700;
  color: #fff;
  font-size: 20px;
}

#header-wrap p {
  margin: 2px 0 0;
  color: var(--text-dim);
  font-weight: 500;
  font-size: 13px;
}

.panel {
  border: 1px solid var(--panel-border);
  border-radius: 0px !important;
  background: var(--panel) !important;
  overflow: hidden;
  padding: 0px !important;
}

.panel-title {
  padding: 6px 10px;
  border-bottom: 1px solid var(--panel-border);
  color: var(--text-dim);
  font-size: 12px;
  letter-spacing: 0.05em;
  text-transform: uppercase;
  font-weight: bold;
}

#execute-btn {
  background: #2b2b2b !important;
  color: #fff !important;
  border-radius: 0px !important;
  font-weight: 700 !important;
  font-size: 16px !important;
  text-transform: uppercase !important;
  border: 2px solid #fff !important;
  transition: all 0.2s ease !important;
  height: 40px !important;
}

#execute-btn:hover {
  background: #801a1a !important;
  border-color: #ff4a4a !important;
}

#execute-btn-running {
  background: #801a1a !important;
  color: #fff !important;
  border-radius: 0px !important;
  font-weight: 700 !important;
  font-size: 16px !important;
  text-transform: uppercase !important;
  border: 2px solid #ff4a4a !important;
  height: 40px !important;
}

.code-panel * {
  font-family: 'JetBrains Mono', monospace !important;
}

.terminal-wrap {
  height: 45vh;
  overflow-y: auto;
  padding: 8px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 13px;
  line-height: 1.5;
  background: #050505;
  border: 1px solid var(--panel-border);
}

.term-line {
  white-space: pre-wrap;
  word-break: break-word;
}

/* Base Log Colors */
.c-start { color: #fff; font-weight: bold; }
.c-end { color: #fff; font-weight: bold; }
.c-step { color: #39ff14; font-weight: bold; }
.c-thought { color: #5b7a96; font-style: italic; }
.c-error { color: #ff4a4a; }
.c-muted { color: var(--text-dim); }

.metric {
  background: #000;
  padding: 4px;
}

@keyframes pulse-border {
  0% { border-color: #ff4a4a; box-shadow: 0 0 10px #ff4a4a; }
  50% { border-color: #2b2b2b; box-shadow: none; }
  100% { border-color: #ff4a4a; box-shadow: 0 0 10px #ff4a4a; }
}
.token-alert > div > input {
  animation: pulse-border 1.5s infinite;
}
"""

def _code_from_task_name(task_name: str) -> str:
    task = TASK_MAP.get((task_name or "").strip())
    if not task:
        return (
            "# Waiting for selection...\n"
            "# Tip: Select a target from the Task Selection Grid\n"
        )
    return "\n".join(task.get("code", []))

def _normalize_base_url(base_url: str) -> str:
    candidate = (base_url or "").strip()
    if not candidate:
        return f"http://{BACKEND_HOST}:{BACKEND_PORT}"
    if not candidate.startswith(("http://", "https://")):
        candidate = f"http://{candidate}"
    return candidate.rstrip("/")

def load_code(task_name: str, env_base_url: str) -> str:
    local_code = _code_from_task_name(task_name)
    if "Waiting for selection" not in local_code:
        return local_code
    return (
        "# Unable to load code for the selected task.\n"
        "# Verify OpenEnv API is reachable."
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
    content = "\n".join(rendered) if rendered else "<div class='term-line c-muted'>Idle. Configure parameters and run agent.</div>"
    return (
        "<div id='terminal' class='terminal-wrap'>"
        f"{content}"
        "</div>"
        "<script>"
        "const t=document.getElementById('terminal'); if(t){t.scrollTop=t.scrollHeight;}"
        "</script>"
    )

def _update_hud_badge(task_name: str, difficulty: str) -> str:
    if not task_name:
        return "<div style='padding: 6px; color: var(--text-dim); text-align: center; font-size: 14px;'>Waiting for Task Selection...</div>"
    return f"""<div style='padding: 6px; color: #fff; font-weight: 700; font-size: 15px; text-transform: uppercase; text-align: center;'>
        Active Task: {html.escape(task_name)} | Difficulty: {difficulty.capitalize()}
    </div>"""

def _large_metric_html(success: bool, score: float, steps: int, reward: str) -> str:
    color = "#39ff14" if success else "#ff4a4a"
    status_text = "SUCCESS" if success else "FAILED"
    return f"""<div style='text-align: center; padding: 10px; border: 1px solid var(--panel-border); background: #000;'>
        <h1 style='color: {color}; margin: 0; font-size: 32px; font-weight: 900;'>{status_text}</h1>
        <h3 style='color: #fff; margin: 4px 0 0 0;'>Score: {score:.2f} | Steps: {steps}</h3>
        <p style='color: var(--text-dim); margin: 4px 0 0 0;'>Rewards: {html.escape(reward)}</p>
    </div>"""

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
    max_steps: int,
    success_score_threshold: float,
) -> dict[str, str]:
    env = os.environ.copy()
    updates = {
        "HF_TOKEN": hf_token,
        "API_BASE_URL": api_base_url,
        "MODEL_NAME": model_name,
        "ENV_BASE_URL": _normalize_base_url(env_base_url),
        "TASK_NAME": task_name,
        "MAX_STEPS": str(int(max_steps)),
        "SUCCESS_SCORE_THRESHOLD": str(float(success_score_threshold)),
    }
    for key, value in updates.items():
        cleaned = (value or "").strip()
        if cleaned:
            env[key] = cleaned
        elif key in env:
            env.pop(key, None)
    return env


def sync_tasks(selected, grid_name):
    if not selected:
        return (
            gr.skip(), 
            gr.skip(), 
            gr.skip(), 
            gr.skip(), 
            gr.skip(), 
            gr.skip(),
            gr.skip()
        )
    if grid_name == "easy":
        easy_val = selected
        med_val = None
        hard_val = None
        diff = "easy"
    elif grid_name == "medium":
        easy_val = None
        med_val = selected
        hard_val = None
        diff = "medium"
    else:
        easy_val = None
        med_val = None
        hard_val = selected
        diff = "hard"
        
    code_content = _code_from_task_name(selected)
    hud_content = _update_hud_badge(selected, diff)
    title_content = "<div class='panel-title'>Target Source Code (Buggy)</div>"
    
    return (
        selected, 
        gr.update(value=easy_val), 
        gr.update(value=med_val), 
        gr.update(value=hard_val), 
        hud_content, 
        title_content,
        code_content
    )

def validate_and_start(token):
    if not token or not token.strip():
        return (
            gr.update(elem_classes=["token-alert"]), 
            gr.update(value="ERROR: Token Required"), 
            False
        )
    return (
        gr.update(elem_classes=[]), 
        gr.update(value="RUNNING...", elem_id="execute-btn-running", interactive=False),
        True
    )

def _reset_run_state():
    return (
        _terminal_html([("c-muted", "Boot sequence initialized...")]),
        "<div style='text-align: center; color: var(--text-dim); padding: 20px;'>Running...</div>"
    )

def run_agent(
    task_name: str,
    hf_token: str,
    api_base_url: str,
    model_name: str,
    env_base_url: str,
    max_steps: int,
    success_score_threshold: float,
    show_thought: bool,
    proceed: bool
) -> Generator[tuple[Any, str, str, dict, Any], None, None]:
    
    if not proceed:
        yield (gr.skip(), gr.skip(), gr.skip(), gr.update(value="INITIATE TRACE RESOLUTION", interactive=True), gr.skip())
        return

    terminal_lines: list[tuple[str, str]] = []
    terminal_lines.append(("c-muted", "Agent initialized... infiltrating target."))

    result_html = "<div style='text-align: center; color: var(--text-dim); padding: 20px;'>Awaiting end...</div>"
    yield gr.skip(), _terminal_html(terminal_lines), result_html, gr.update(), gr.skip()

    cmd = [sys.executable, str(INFERENCE_PATH)]
    if show_thought:
        cmd.append("--thought")

    env = _build_env(
        hf_token, api_base_url, model_name, env_base_url,
        task_name, max_steps, success_score_threshold
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
    
    final_success = False
    final_solved_code = None

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
                if show_thought:
                    terminal_lines.append(("c-thought", "[THOUGHT]"))
            elif line.startswith("[") and line.endswith("]"):
                thought_mode = False
                terminal_lines.append(("c-muted", line))
            elif thought_mode:
                if show_thought:
                    terminal_lines.append(("c-thought", line))
            else:
                if not show_thought:
                    if not any(tag in line for tag in ["[START]", "[STEP]", "[END]"]):
                        continue
                terminal_lines.append(("c-error", line))
        else:
            if not show_thought:
                if not any(tag in line for tag in ["[START]", "[STEP]", "[END]"]):
                    continue

            start_match = START_RE.match(line)
            step_match = STEP_RE.match(line)
            end_match = END_RE.match(line)

            if start_match:
                terminal_lines.append(("c-start", line))
            elif step_match:
                err = step_match.group("error")
                css = "c-step" if err == "null" else "c-error"
                terminal_lines.append((css, line))
            elif end_match:
                success = end_match.group("success") == "true"
                final_steps = int(end_match.group("steps"))
                score_value = float(end_match.group("score"))
                rewards_raw = end_match.group("rewards").strip()
                terminal_lines.append(("c-end", line))
                
                result_html = _large_metric_html(success, score_value, final_steps, rewards_raw or 'none')
                
                if success:
                    final_success = True
                    solved = _solution_from_task_name(task_name)
                    if solved:
                        final_solved_code = solved
            else:
                terminal_lines.append(("c-muted", line))

        if len(terminal_lines) > 500:
            terminal_lines = terminal_lines[-500:]

        yield gr.skip(), _terminal_html(terminal_lines), result_html, gr.update(), gr.skip()

    return_code = process.wait(timeout=2)
    if return_code != 0:
        terminal_lines.append(("c-error", f"Process exited with code {return_code}."))
        result_html = _large_metric_html(False, 0.0, 0, f"Error code {return_code}")

    if len(terminal_lines) > 500:
        terminal_lines = terminal_lines[-500:]

    code_update = gr.skip()
    title_update = gr.skip()
    if final_success and final_solved_code is not None:
        code_update = final_solved_code
        title_update = "<div class='panel-title'>Target Source Code (Resolved)</div>"

    yield code_update, _terminal_html(terminal_lines), result_html, gr.update(value="INITIATE TRACE RESOLUTION", elem_id="execute-btn", interactive=True), title_update


with gr.Blocks(title="TraceFix-RL") as demo:
    gr.HTML(
        f"""
        <style>{CSS}</style>
        <div id='header-wrap'>
          <h1>TraceFix-RL: Auto SWE OpenEnv RL</h1>
          <p>Professional Autonomous Agent Trace Orchestration.</p>
        </div>
        """
    )
    
    selected_task_state = gr.State(value="")

    if hasattr(gr, "Sidebar"):
        sidebar_context = gr.Sidebar()
    else:
        sidebar_context = gr.Column()

    with sidebar_context:
        gr.Markdown("### Authentication")
        hf_token = gr.Textbox(label="HF Token", type="password", placeholder="hf_xxx", elem_classes=[])

        with gr.Accordion("Engine Parameters", open=False):
            model_name = gr.Textbox(label="Model Name", value=os.getenv("MODEL_NAME", "openai/gpt-oss-20b"))
            api_base_url = gr.Textbox(label="API Base URL", value=os.getenv("API_BASE_URL", "https://router.huggingface.co/v1"))
            env_base_url = gr.Textbox(label="Env Base URL", value=os.getenv("ENV_BASE_URL", f"http://127.0.0.1:{BACKEND_PORT}"))
            max_steps = gr.Number(label="Max Steps", value=int(os.getenv("MAX_STEPS", "50")), precision=0)
            success_score_threshold = gr.Number(
                label="Success Score Threshold",
                value=float(os.getenv("SUCCESS_SCORE_THRESHOLD", "0.99")),
                precision=2,
            )
            show_thought = gr.Checkbox(label="Stream Thought Trace", value=False)

    gr.HTML("<div class='panel-title'>Task Selection Grid</div>")
    with gr.Row(elem_classes=["panel"]):
        easy_radio = gr.Radio(choices=EASY_CHOICES, label="Easy Targets", elem_id="easy-radio")
        medium_radio = gr.Radio(choices=MEDIUM_CHOICES, label="Medium Targets", elem_id="medium-radio")
        hard_radio = gr.Radio(choices=HARD_CHOICES, label="Hard Targets", elem_id="hard-radio")

    hud_badge = gr.HTML(_update_hud_badge("", ""))
    run_button = gr.Button("INITIATE TRACE RESOLUTION", elem_id="execute-btn", variant="primary")

    with gr.Row(equal_height=True):
        with gr.Column(scale=1, elem_classes=["panel", "code-panel"]):
            code_panel_title = gr.HTML("<div class='panel-title'>Target Source Code (Buggy)</div>")
            code_view = gr.Code(
                language="python",
                interactive=False,
                value=_code_from_task_name(""),
            )
            # Override height via CSS
            gr.HTML("<style>.code-panel .cm-content { height: 45vh; overflow-y: auto; }</style>")

        with gr.Column(scale=1, elem_classes=["panel"]):
            gr.HTML("<div class='panel-title'>Terminal Trace</div>")
            terminal = gr.HTML(_terminal_html([]))

    with gr.Row(elem_classes=["panel"]):
        result_block = gr.HTML("<div style='text-align: center; color: var(--text-dim); padding: 20px;'>Awaiting Execution</div>")

    easy_radio.change(lambda x: sync_tasks(x, "easy"), inputs=[easy_radio], outputs=[selected_task_state, easy_radio, medium_radio, hard_radio, hud_badge, code_panel_title, code_view])
    medium_radio.change(lambda x: sync_tasks(x, "medium"), inputs=[medium_radio], outputs=[selected_task_state, easy_radio, medium_radio, hard_radio, hud_badge, code_panel_title, code_view])
    hard_radio.change(lambda x: sync_tasks(x, "hard"), inputs=[hard_radio], outputs=[selected_task_state, easy_radio, medium_radio, hard_radio, hud_badge, code_panel_title, code_view])

    # Run Sequence
    run_state = gr.State(value=True)

    validate_step = run_button.click(
        validate_and_start,
        inputs=[hf_token],
        outputs=[hf_token, run_button, run_state],
        queue=False
    )
    
    reset_step = validate_step.then(
        _reset_run_state,
        inputs=[],
        outputs=[terminal, result_block],
        queue=False,
    )

    reset_step.then(
        run_agent,
        inputs=[
            selected_task_state,
            hf_token,
            api_base_url,
            model_name,
            env_base_url,
            max_steps,
            success_score_threshold,
            show_thought,
            run_state
        ],
        outputs=[code_view, terminal, result_block, run_button, code_panel_title],
    )
