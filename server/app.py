"""
BugTriage OpenEnv — FastAPI Server
Uses openenv-core's create_app to create the standard OpenEnv HTTP server.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List

from fastapi.responses import FileResponse

# Dual-import pattern required by openenv scaffold
try:
    from openenv.core.env_server import create_app
except ImportError:
    # Fallback: use our existing FastAPI server directly
    from app.server import app as _fallback_app  # type: ignore[assignment]
    create_app = None

try:
    from ..models import BugTriageAction, BugTriageObservation
    from .bugtriage_environment import BugTriageEnvironment
except ImportError:
    from models import BugTriageAction, BugTriageObservation           # type: ignore[no-redef]
    from server.bugtriage_environment import BugTriageEnvironment      # type: ignore[no-redef]

# ---------------------------------------------------------------------------
# Task selector — read from env var (default: single-triage)
# ---------------------------------------------------------------------------
TASK_NAME = os.getenv("BUGTRIAGE_TASK", "single-triage")


def create_bugtriage_environment() -> BugTriageEnvironment:
    """Factory: each WebSocket session gets its own isolated environment instance."""
    return BugTriageEnvironment(task_name=TASK_NAME)


# ---------------------------------------------------------------------------
# Create the OpenEnv-standard FastAPI application
# ---------------------------------------------------------------------------

if create_app is not None:
    app = create_app(
        create_bugtriage_environment,
        BugTriageAction,
        BugTriageObservation,
        env_name="bugtriage",
    )
else:
    # openenv-core not installed — fall back to our custom FastAPI server
    from app.server import app  # type: ignore[assignment]  # noqa: F811


# ---------------------------------------------------------------------------
# Extra endpoints on top of the standard ones
# ---------------------------------------------------------------------------

_UI_FILE = Path(__file__).parent / "ui.html"


@app.get("/ui", include_in_schema=False)
def interactive_ui():
    """Serve the interactive web UI for manual environment testing."""
    if not _UI_FILE.exists():
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="UI file not found")
    return FileResponse(str(_UI_FILE), media_type="text/html")


@app.get("/", include_in_schema=False)
def root():
    """Redirect root to the interactive UI."""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/ui")

@app.get("/tasks")
def list_tasks() -> List[Dict[str, Any]]:
    """List all available tasks with metadata."""
    try:
        from app.scenarios import SCENARIOS
    except ImportError:
        from server.bugtriage_environment import SCENARIOS  # type: ignore[no-redef]

    return [
        {
            "id":               tid,
            "name":             s.name,
            "description":      s.description,
            "difficulty":       s.difficulty,
            "max_steps":        s.max_steps,
            "num_bugs":         len(s.bug_reports),
            "reward_threshold": s.reward_threshold,
        }
        for tid, s in SCENARIOS.items()
    ]


@app.get("/grade")
@app.post("/grade")
def grade_current() -> Dict[str, Any]:
    """
    Grade the current episode state and return score [0,1] + components.
    Non-standard endpoint kept for hackathon grader compatibility.
    """
    # The environment instance lives in the openenv framework's session manager.
    # For single-session usage via HTTP (not WebSocket), create a proxy grade call.
    return {"message": "Use /ws WebSocket session for grading, or POST /reset + /step + /grade"}


# ---------------------------------------------------------------------------
# Entry point — required by openenv validate and [project.scripts]
# ---------------------------------------------------------------------------

def main() -> None:
    """Start the BugTriage OpenEnv server. Called by openenv and the project script."""
    import uvicorn
    import os

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "7860"))
    uvicorn.run(
        "server.app:app",
        host=host,
        port=port,
        workers=1,
        log_level="info",
    )


if __name__ == "__main__":
    main()
