"""
BugTriage OpenEnv — FastAPI HTTP Server
Implements the OpenEnv HTTP interface:
  POST /reset   → ResetResult    (optional: {"task": "...", "seed": 42})
  POST /step    → StepResult
  GET  /state   → BugTriageState
  POST /grade   → score + components
  GET  /health  → HealthResponse
  GET  /tasks   → List[TaskInfo]
"""
from __future__ import annotations

import traceback
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
# Single-session environment state (suitable for evaluation)
# ---------------------------------------------------------------------------

_env: Optional[BugTriageEnv] = None


def _get_env() -> BugTriageEnv:
    global _env
    if _env is None:
        _env = BugTriageEnv("single-triage")
    return _env


class ResetRequest(BaseModel):
    task: Optional[str] = "single-triage"
    seed: Optional[int] = None  # If provided, generates a dynamic scenario


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    env = _get_env()
    try:
        st = env.state()
        return HealthResponse(status="healthy", version=BugTriageEnv.VERSION,
                              active_task=st.task_name, step_number=st.step_number)
    except RuntimeError:
        return HealthResponse(status="healthy", version=BugTriageEnv.VERSION,
                              active_task=None, step_number=0)


@app.get("/tasks", response_model=List[TaskInfo])
def list_tasks() -> List[TaskInfo]:
    return [
        TaskInfo(id=tid, name=s.name, description=s.description, difficulty=s.difficulty,
                 max_steps=s.max_steps, num_bugs=len(s.bug_reports), reward_threshold=s.reward_threshold)
        for tid, s in SCENARIOS.items()
    ]


@app.post("/reset", response_model=ResetResult)
def reset(request: ResetRequest = ResetRequest()) -> ResetResult:
    global _env
    task_name = request.task or "single-triage"
    if task_name not in SCENARIOS:
        raise HTTPException(status_code=422,
                            detail=f"Unknown task '{task_name}'. Available: {list(SCENARIOS.keys())}")
    _env = BugTriageEnv(task_name)
    return _env.reset(seed=request.seed)


@app.post("/step", response_model=StepResult)
def step(action: TriageAction) -> StepResult:
    env = _get_env()
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
def get_state() -> BugTriageState:
    env = _get_env()
    try:
        return env.state()
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/grade")
def grade_episode() -> Dict[str, Any]:
    env = _get_env()
    try:
        return env.grade()
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/ui", include_in_schema=False)
def interactive_ui():
    if not UI_FILE.exists():
        raise HTTPException(status_code=404, detail="UI file not found")
    return FileResponse(str(UI_FILE), media_type="text/html")


@app.get("/", include_in_schema=False)
def root():
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/ui")
