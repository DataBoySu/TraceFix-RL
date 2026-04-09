"""FastAPI entry point for TraceFix-RL."""

import gradio as gr
from vision_ui import demo

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e

try:
    from core.models import CodeAction, CodeObservation
    from backend.tracefix_rl_environment import TraceFixRLEnvironment
except ImportError:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from core.models import CodeAction, CodeObservation
    from backend.tracefix_rl_environment import TraceFixRLEnvironment


app = create_app(
    TraceFixRLEnvironment,
    CodeAction,
    CodeObservation,
    env_name="tracefix_rl",
    max_concurrent_envs=1,
)

from fastapi.responses import RedirectResponse

@app.get("/", include_in_schema=False)
async def root_redirect():
    return RedirectResponse(url="/web/")

app = gr.mount_gradio_app(app, demo, path="/web")


def main() -> None:
    """Entry point for local and container execution."""
    import os
    import uvicorn

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "7860"))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
