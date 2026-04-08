"""FastAPI entry point for TraceFix-RL."""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e

try:
    from ..models import CodeAction, CodeObservation
    from .tracefix_rl_environment import TraceFixRLEnvironment
except ImportError:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from models import CodeAction, CodeObservation
    from server.tracefix_rl_environment import TraceFixRLEnvironment


app = create_app(
    TraceFixRLEnvironment,
    CodeAction,
    CodeObservation,
    env_name="tracefix_rl",
    max_concurrent_envs=1,
)


def main() -> None:
    """Entry point for local and container execution."""
    import os
    import uvicorn

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "7860"))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
