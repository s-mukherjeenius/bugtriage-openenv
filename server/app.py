"""
BugTriage OpenEnv — FastAPI Server (openenv-core integration)
Uses openenv-core's create_app for the standard OpenEnv WebSocket server.
Falls back to our custom HTTP server if openenv-core is not installed.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi.responses import FileResponse

# Dual-import pattern required by openenv scaffold
try:
    from openenv.core.env_server import create_app
except ImportError:
    create_app = None

try:
    from ..models import BugTriageAction, BugTriageObservation
    from .bugtriage_environment import BugTriageEnvironment
except ImportError:
    from models import BugTriageAction, BugTriageObservation           # type: ignore[no-redef]
    from server.bugtriage_environment import BugTriageEnvironment      # type: ignore[no-redef]

TASK_NAME = os.getenv("BUGTRIAGE_TASK", "single-triage")


def create_bugtriage_environment() -> BugTriageEnvironment:
    return BugTriageEnvironment(task_name=TASK_NAME)


if create_app is not None:
    app = create_app(
        create_bugtriage_environment,
        BugTriageAction,
        BugTriageObservation,
        env_name="bugtriage",
    )
else:
    from app.server import app  # type: ignore[assignment]  # noqa: F811

_UI_FILE = Path(__file__).parent.parent / "app" / "ui.html"
_http_env: "BugTriageEnvironment | None" = None


def _get_http_env() -> BugTriageEnvironment:
    global _http_env
    if _http_env is None:
        _http_env = BugTriageEnvironment(task_name=TASK_NAME)
    return _http_env


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "healthy", "version": "1.0.0", "active_task": TASK_NAME, "step_number": 0}


@app.post("/reset")
def http_reset(body: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    HTTP reset endpoint. Supports dynamic scenarios via seed parameter.
    Validator sends: POST /reset with {} body, expects 200.
    Training/UI sends: POST /reset with {"task": "...", "seed": 42}
    """
    global _http_env
    body = body or {}
    task = body.get("task", TASK_NAME) if body else TASK_NAME
    seed: Optional[int] = body.get("seed") if body else None
    _http_env = BugTriageEnvironment(task_name=task)
    obs = _http_env.reset(seed=seed)
    return {
        "observation": obs.model_dump() if hasattr(obs, "model_dump") else obs.dict(),
        "done": False,
        "info": {"task": task, "version": "1.0.0", "generated": seed is not None, "seed": seed},
    }


@app.post("/step")
def http_step(action: Dict[str, Any]) -> Dict[str, Any]:
    env = _get_http_env()
    from models import BugTriageAction
    act = BugTriageAction(**action)
    obs = env.step(act)
    return {
        "observation": obs.model_dump() if hasattr(obs, "model_dump") else obs.dict(),
        "reward": obs.reward if hasattr(obs, "reward") else 0.0,
        "done": obs.done if hasattr(obs, "done") else False,
        "info": {},
    }


@app.get("/state")
def http_state() -> Dict[str, Any]:
    env = _get_http_env()
    st = env.state
    return st.model_dump() if hasattr(st, "model_dump") else st.dict()


@app.get("/grade")
@app.post("/grade")
def http_grade() -> Dict[str, Any]:
    env = _get_http_env()
    return env.grade()


@app.get("/tasks")
def list_tasks() -> List[Dict[str, Any]]:
    try:
        from app.scenarios import SCENARIOS
    except ImportError:
        from server.bugtriage_environment import SCENARIOS  # type: ignore[no-redef]
    return [
        {"id": tid, "name": s.name, "description": s.description, "difficulty": s.difficulty,
         "max_steps": s.max_steps, "num_bugs": len(s.bug_reports), "reward_threshold": s.reward_threshold}
        for tid, s in SCENARIOS.items()
    ]


@app.get("/ui", include_in_schema=False)
def interactive_ui():
    if not _UI_FILE.exists():
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="UI file not found")
    return FileResponse(str(_UI_FILE), media_type="text/html")


@app.get("/", include_in_schema=False)
def root():
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/ui")


def main() -> None:
    import uvicorn
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "7860"))
    uvicorn.run("server.app:app", host=host, port=port, workers=1, log_level="info")


if __name__ == "__main__":
    main()
