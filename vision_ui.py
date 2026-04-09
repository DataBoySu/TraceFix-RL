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
}

#header-wrap {
  margin-bottom: 5px;
  border: 1px solid var(--accent);
  background: #000;
  border-radius: 0px;
  padding: 16px 20px;
  text-transform: uppercase;
}

#header-wrap h1 {
  margin: 0;
  letter-spacing: 2px;
  font-weight: 900;
  color: #fff;
  font-style: italic;
  text-shadow: 2px 2px #E60012;
}

#header-wrap p {
  margin: 6px 0 0;
  color: #fff;
  font-weight: 500;
}

.panel {
  border: 1px solid var(--panel-border);
  border-radius: 0px !important;
  background: var(--panel) !important;
  overflow: hidden;
}

.panel-title {
  padding: 10px 14px;
  border-bottom: 1px solid var(--panel-border);
  color: var(--text-dim);
  font-size: 14px;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  font-weight: bold;
}

#execute-btn {
  background: var(--accent) !important;
  color: #fff !important;
  border-radius: 0px !important;
  font-weight: 900 !important;
  font-size: 18px !important;
  text-transform: uppercase !important;
  border: none !important;
  transition: all 0.2s ease !important;
  height: 60px !important;
}

#execute-btn:hover {
  background: #fff !important;
  color: var(--accent) !important;
  box-shadow: 0 0 15px var(--accent) !important;
}

.code-panel * {
  font-family: 'JetBrains Mono', monospace !important;
}

.terminal-wrap {
  height: 600px;
  overflow-y: auto;
  padding: 12px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 13px;
  line-height: 1.6;
  background: #050505;
  border: 2px solid var(--accent);
}

.term-line {
  white-space: pre-wrap;
  word-break: break-word;
}

/* Cyberpunk Log Colors */
.c-start { color: #E60012; font-weight: bold; }
.c-end { color: #E60012; font-weight: bold; }
.c-step { color: #39ff14; font-weight: bold; }
.c-thought { color: #5b7a96; font-style: italic; }
.c-error { color: #E60012; }
.c-muted { color: var(--text-dim); }

.metric {
  border: 1px solid var(--panel-border);
  background: #000;
  border-radius: 0px;
  padding: 12px;
  border-left: 4px solid var(--accent);
}
"""

def _code_from_task_name(task_name: str) -> str:
    task = TASK_MAP.get((task_name or "").strip())
    if not task:
        return (
            "# Waiting for mission start...\n"
            "# Tip: Select a target from the Mission Board\n"
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
    content = "\n".join(rendered) if rendered else "<div class='term-line c-muted'>Idle. Configure mission variables and press EXECUTE TRACEFIX.</div>"
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

def _update_hud_badge(task_name: str, difficulty: str) -> str:
    if not task_name:
        return "<div style='padding: 10px; color: var(--text-dim); border: 1px dashed var(--panel-border); text-align: center;'>WAITING FOR TARGET SELECTION...</div>"
    color = "#39ff14" if difficulty == "Easy" else ("#f9d78b" if difficulty == "Medium" else "#E60012")
    return f"""<div style='border: 2px solid {color}; padding: 12px; background: rgba(0,0,0,0.5); color: {color}; font-weight: 900; font-size: 16px; text-transform: uppercase; text-align: center; letter-spacing: 1.5px;'>
        >> TARGET ACQUIRED: {html.escape(task_name)} | THREAT LEVEL: {difficulty} <<
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

def get_active_task(easy, medium, hard):
    return (easy or medium or hard or "").strip()

def _reset_run_state(easy, medium, hard):
    task_name = get_active_task(easy, medium, hard)
    return (
        _code_from_task_name(task_name),
        _terminal_html([]),
        _metric_block("Mission Ready", "Awaiting [START] from inference subprocess..."),
        0.0,
        "`Rewards:` pending"
    )

def run_agent(
    easy_radio: str,
    medium_radio: str,
    hard_radio: str,
    hf_token: str,
    api_base_url: str,
    model_name: str,
    env_base_url: str,
    benchmark: str,
    max_steps: int,
    success_score_threshold: float,
    local_image_name: str,
    difficulty: str,
    show_thought: bool,
) -> Generator[tuple[str, str, str, float, str, dict], None, None]:
    
    task_name = get_active_task(easy_radio, medium_radio, hard_radio)
    code_view = _code_from_task_name(task_name)
    terminal_lines: list[tuple[str, str]] = []
    terminal_lines.append(("c-muted", "Boot sequence initialized... infiltrating target."))

    status_html = _metric_block("Mission Infiltration", "Launching inference subprocess...")
    score_value = 0.0
    rewards_md = "`Rewards:` pending"
    yield code_view, _terminal_html(terminal_lines), status_html, score_value, rewards_md, gr.update(value="INFILTRATING...", interactive=False)

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
    active_task_name = task_name
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
                    # Strict gatekeeper rules over stderr leakage too
                    if not any(tag in line for tag in ["[START]", "[STEP]", "[END]"]):
                        continue
                terminal_lines.append(("c-error", line))
        else:
            if not show_thought:
                if not any(tag in line for tag in ["[START]", "[STEP]", "[END]"]):
                    continue # Strict Gatekeeper skipping log

            start_match = START_RE.match(line)
            step_match = STEP_RE.match(line)
            end_match = END_RE.match(line)

            if start_match:
                active_task_name = start_match.group("task").strip()
                task_preview = _code_from_task_name(active_task_name)
                if "Waiting for mission start" not in task_preview:
                    code_view = task_preview
                terminal_lines.append(("c-start", line))
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
                css = "c-step" if err == "null" else "c-error"
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
                terminal_lines.append(("c-end", line))
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
                terminal_lines.append(("c-muted", line))

        if len(terminal_lines) > 500:
            terminal_lines = terminal_lines[-500:]

        yield code_view, _terminal_html(terminal_lines), status_html, score_value, rewards_md, gr.update(value="INFILTRATING...", interactive=False)

    return_code = process.wait(timeout=2)
    if return_code != 0:
        terminal_lines.append(("c-error", f"Process exited with code {return_code}."))
        status_html = _metric_block(
            "Mission Error",
            f"inference.py exited non-zero (code={return_code})",
        )

    if len(terminal_lines) > 500:
        terminal_lines = terminal_lines[-500:]

    yield code_view, _terminal_html(terminal_lines), status_html, score_value, rewards_md, gr.update(value="EXECUTE TRACEFIX", interactive=True)


with gr.Blocks(title="TraceFix-RL Mission Control") as demo:
    gr.HTML(
        f"""
        <style>{CSS}</style>
        <div id='header-wrap'>
          <h1>TraceFix-RL /// PHANTOM PROTOCOL</h1>
          <p>Real-time autonomous agent infiltration orchestration.</p>
        </div>
        """
    )

    if hasattr(gr, "Sidebar"):
        sidebar_context = gr.Sidebar()
    else:
        sidebar_context = gr.Column()

    with sidebar_context:
        # Zone 1: The Config Sidebar
        gr.Markdown("### CORE AUTHENTICATION")
        hf_token = gr.Textbox(label="HF Token", type="password", placeholder="hf_xxx")

        with gr.Accordion("Advanced Engine Parameters", open=False):
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

    # Zone 2: The Mission Board
    gr.HTML("<div class='panel-title' style='margin-top: 10px;'>MISSION BOARD /// TARGET SELECTION</div>")
    with gr.Row(elem_classes=["panel"]):
        easy_radio = gr.Radio(choices=EASY_CHOICES, label="Easy Targets", elem_id="easy-radio")
        medium_radio = gr.Radio(choices=MEDIUM_CHOICES, label="Medium Targets", elem_id="medium-radio")
        hard_radio = gr.Radio(choices=HARD_CHOICES, label="Hard Targets", elem_id="hard-radio")

    # Zone 3: The HUD 
    hud_badge = gr.HTML(_update_hud_badge("", ""))
    run_button = gr.Button("EXECUTE TRACEFIX", elem_id="execute-btn", variant="primary")

    # Radio change handlers for mutual exclusivity logic & HUD updates
    def select_easy(val):
        if not val:
            return gr.skip(), gr.skip(), gr.skip(), gr.skip()
        return None, None, _update_hud_badge(val, "Easy"), _code_from_task_name(val)

    def select_medium(val):
        if not val:
            return gr.skip(), gr.skip(), gr.skip(), gr.skip()
        return None, None, _update_hud_badge(val, "Medium"), _code_from_task_name(val)

    def select_hard(val):
        if not val:
            return gr.skip(), gr.skip(), gr.skip(), gr.skip()
        return None, None, _update_hud_badge(val, "Hard"), _code_from_task_name(val)

    # Zone 4: The Arena
    with gr.Row(equal_height=True):
        with gr.Column(scale=1, elem_classes=["panel", "code-panel"]):
            gr.HTML("<div class='panel-title'>SANDBOX CODE</div>")
            code_view = gr.Code(
                language="python",
                interactive=False,
                value=_code_from_task_name(""),
                lines=30,
            )

        with gr.Column(scale=1, elem_classes=["panel"]):
            gr.HTML("<div class='panel-title'>TERMINAL TRACE</div>")
            terminal = gr.HTML(_terminal_html([]))

    with gr.Row():
        metric = gr.HTML(_metric_block("Idle", "Awaiting target selection."))
        score = gr.Number(label="Final Score", value=0.0, precision=3)
        rewards = gr.Markdown("`Rewards:` pending")

    easy_radio.change(select_easy, inputs=[easy_radio], outputs=[medium_radio, hard_radio, hud_badge, code_view])
    medium_radio.change(select_medium, inputs=[medium_radio], outputs=[easy_radio, hard_radio, hud_badge, code_view])
    hard_radio.change(select_hard, inputs=[hard_radio], outputs=[easy_radio, medium_radio, hud_badge, code_view])

    # Run Sequence
    # First disable button to show immediate feedback
    run_immediate = run_button.click(
        lambda: gr.update(value="INFILTRATING...", interactive=False),
        inputs=[],
        outputs=[run_button],
        queue=False
    )
    
    # Then reset state
    run_event = run_immediate.then(
        _reset_run_state,
        inputs=[easy_radio, medium_radio, hard_radio],
        outputs=[code_view, terminal, metric, score, rewards],
        queue=False,
    )

    # Finally run generator (loads environment, streams stdout, then re-enables button upon END)
    run_event.then(
        run_agent,
        inputs=[
            easy_radio,
            medium_radio,
            hard_radio,
            hf_token,
            api_base_url,
            model_name,
            env_base_url,
            benchmark,
            max_steps,
            success_score_threshold,
            local_image_name,
            difficulty,
            show_thought,
        ],
        outputs=[code_view, terminal, metric, score, rewards, run_button],
    )
