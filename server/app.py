# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""FastAPI entry point for SWE-Gym - Software Engineer Gym."""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e

try:
    from ..models import CodeAction, CodeObservation
    from .swe_gym_environment import SWEGymEnvironment
except ImportError:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from models import CodeAction, CodeObservation
    from server.swe_gym_environment import SWEGymEnvironment


# Create the app with web interface and README integration
app = create_app(
    SWEGymEnvironment,
    CodeAction,
    CodeObservation,
    env_name="swe_gym",
    max_concurrent_envs=1,  # increase this number to allow more concurrent WebSocket sessions
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
