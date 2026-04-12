"""
BugTriage OpenEnv — FastAPI HTTP Server
=======================================
Implements the OpenEnv HTTP interface:
  POST /reset   → ResetResult    (optional: {"task": "...", "seed": 42, "session_id": "..."})
  POST /step    → StepResult
  GET  /state   → BugTriageState
  POST /grade   → score + components
  GET  /health  → HealthResponse
  GET  /tasks   → List[TaskInfo]

Supports concurrent sessions via optional session_id parameter.
Default session ("default") is used when no session_id is provided.
"""
from __future__ import annotations

import traceback
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.env import BugTriageEnv
from app.models import (
    BugTriageState,
    HealthResponse,
    ResetResult,
    StepResult,
    TaskInfo,
    TriageAction,
)
from app.scenarios import SCENARIOS

UI_FILE = Path(__file__).parent / "ui.html"

app = FastAPI(
    title="BugTriage OpenEnv",
    description="OpenEnv environment for Bug Report Triage & Priority Queue Management.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Session-based environment management (supports concurrent evaluation)
# ---------------------------------------------------------------------------

_sessions: Dict[str, BugTriageEnv] = {}
_DEFAULT_SESSION = "default"
_MAX_SESSIONS = 64  # Prevent unbounded memory growth


def _get_env(session_id: Optional[str] = None) -> BugTriageEnv:
    """Get or create an environment for the given session."""
    sid = session_id or _DEFAULT_SESSION
    if sid not in _sessions:
        _sessions[sid] = BugTriageEnv("single-triage")
    return _sessions[sid]


def _set_env(session_id: Optional[str], env: BugTriageEnv) -> str:
    """Register a new environment for the session. Returns the session ID."""
    sid = session_id or _DEFAULT_SESSION

    # Evict oldest session if at capacity (simple LRU would be better, but this is fine)
    if len(_sessions) >= _MAX_SESSIONS and sid not in _sessions:
        oldest_key = next(iter(_sessions))
        del _sessions[oldest_key]

    _sessions[sid] = env
    return sid


class ResetRequest(BaseModel):
    task: Optional[str] = "single-triage"
    seed: Optional[int] = None
    session_id: Optional[str] = None  # For concurrent evaluation


class StepRequest(BaseModel):
    """Step request that includes optional session_id alongside the action."""
    action_type: str
    bug_id: str
    severity: Optional[str] = None
    assigned_team: Optional[str] = None
    duplicate_of: Optional[str] = None
    info_requested: Optional[List[str]] = None
    escalation_reason: Optional[str] = None
    spam_reason: Optional[str] = None
    session_id: Optional[str] = None


class SessionRequest(BaseModel):
    session_id: Optional[str] = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    env = _get_env()
    try:
        st = env.state()
        return HealthResponse(
            status="healthy", version=BugTriageEnv.VERSION,
            active_task=st.task_name, step_number=st.step_number,
            active_sessions=len(_sessions),
        )
    except RuntimeError:
        return HealthResponse(
            status="healthy", version=BugTriageEnv.VERSION,
            active_task=None, step_number=0,
            active_sessions=len(_sessions),
        )


@app.get("/tasks", response_model=List[TaskInfo])
def list_tasks() -> List[TaskInfo]:
    return [
        TaskInfo(
            id=tid, name=s.name, description=s.description, difficulty=s.difficulty,
            max_steps=s.max_steps, num_bugs=len(s.bug_reports),
            reward_threshold=s.reward_threshold,
        )
        for tid, s in SCENARIOS.items()
    ]


@app.post("/reset", response_model=ResetResult)
def reset(request: ResetRequest = ResetRequest()) -> ResetResult:
    task_name = request.task or "single-triage"
    if task_name not in SCENARIOS:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown task '{task_name}'. Available: {list(SCENARIOS.keys())}",
        )

    env = BugTriageEnv(task_name)
    sid = _set_env(request.session_id, env)
    result = env.reset(seed=request.seed)

    # Include session_id in info so clients can track it
    result.info["session_id"] = sid
    return result


@app.post("/step", response_model=StepResult)
def step(action: TriageAction, session_id: Optional[str] = None) -> StepResult:
    env = _get_env(session_id)
    try:
        result = env.step(action)
        if result.done:
            grade_result = env.grade()
            result.info["final_score"] = grade_result["score"]
            result.info["score_components"] = grade_result["components"]
        return result
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        raise HTTPException(status_code=500, detail=traceback.format_exc())


@app.get("/state", response_model=BugTriageState)
def get_state(session_id: Optional[str] = None) -> BugTriageState:
    env = _get_env(session_id)
    try:
        return env.state()
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/grade")
def handle_grade(session_id: Optional[str] = None) -> Dict[str, Any]:
    env = _get_env(session_id)
    try:
        return env.grade()
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/sessions")
def list_sessions() -> Dict[str, Any]:
    """List active sessions with their state summaries."""
    sessions = {}
    for sid, env in _sessions.items():
        try:
            st = env.state()
            sessions[sid] = {
                "task": st.task_name,
                "step": st.step_number,
                "done": st.done,
                "total_reward": round(st.total_reward, 4),
            }
        except RuntimeError:
            sessions[sid] = {"task": env.task_name, "step": 0, "done": False}
    return {"active_sessions": len(sessions), "sessions": sessions}


@app.delete("/sessions/{session_id}")
def delete_session(session_id: str) -> Dict[str, str]:
    """Clean up a specific session."""
    if session_id in _sessions:
        del _sessions[session_id]
        return {"status": "deleted", "session_id": session_id}
    raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")


@app.get("/ui", include_in_schema=False)
def interactive_ui():
    if not UI_FILE.exists():
        raise HTTPException(status_code=404, detail="UI file not found")
    return FileResponse(str(UI_FILE), media_type="text/html")


@app.get("/", include_in_schema=False)
def root():
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/ui")
