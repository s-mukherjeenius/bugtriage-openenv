"""
BugTriage OpenEnv — FastAPI Server
Implements the OpenEnv HTTP interface:
  POST /reset          → ResetResult
  POST /step           → StepResult
  GET  /state          → BugTriageState
  GET  /tasks          → List[TaskInfo]
  POST /grade          → grade result dict
  GET  /health         → HealthResponse
"""
from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

UI_FILE = Path(__file__).parent / "ui.html"

from app.env import BugTriageEnv, init_env
from app.models import (
    BugTriageState,
    HealthResponse,
    ResetResult,
    StepResult,
    TaskInfo,
    TriageAction,
)
from app.scenarios import SCENARIOS

app = FastAPI(
    title="BugTriage OpenEnv",
    description=(
        "An OpenEnv-compliant environment for Bug Report Triage & Priority Queue Management. "
        "Simulates the real-world workflow of classifying, routing, and escalating software bugs."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Active environment state (single-session; suitable for evaluation)
# ---------------------------------------------------------------------------

_env: Optional[BugTriageEnv] = None


def _get_env() -> BugTriageEnv:
    global _env
    if _env is None:
        _env = BugTriageEnv("single-triage")
    return _env


# ---------------------------------------------------------------------------
# Request bodies
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task: Optional[str] = "single-triage"


class GradeRequest(BaseModel):
    pass  # grades current state in-place


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health() -> HealthResponse:
    env = _get_env()
    try:
        st = env.state()
        return HealthResponse(
            status="healthy",
            version=BugTriageEnv.VERSION,
            active_task=st.task_name,
            step_number=st.step_number,
        )
    except RuntimeError:
        return HealthResponse(
            status="healthy",
            version=BugTriageEnv.VERSION,
            active_task=None,
            step_number=0,
        )


@app.get("/tasks", response_model=List[TaskInfo], tags=["meta"])
def list_tasks() -> List[TaskInfo]:
    """List all available tasks with metadata."""
    results = []
    for task_id, scenario in SCENARIOS.items():
        results.append(
            TaskInfo(
                id=task_id,
                name=scenario.name,
                description=scenario.description,
                difficulty=scenario.difficulty,
                max_steps=scenario.max_steps,
                num_bugs=len(scenario.bug_reports),
                reward_threshold=scenario.reward_threshold,
            )
        )
    return results


@app.post("/reset", response_model=ResetResult, tags=["openenv"])
def reset(request: ResetRequest = ResetRequest()) -> ResetResult:
    """
    OpenEnv reset() — initialise or restart the episode.
    Optionally pass { "task": "single-triage" | "batch-triage" | "sla-crisis" }.
    """
    global _env
    task_name = request.task or "single-triage"
    if task_name not in SCENARIOS:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown task '{task_name}'. Available: {list(SCENARIOS.keys())}",
        )
    _env = BugTriageEnv(task_name)
    return _env.reset()


@app.post("/step", response_model=StepResult, tags=["openenv"])
def step(action: TriageAction) -> StepResult:
    """
    OpenEnv step() — execute one triage action.
    Returns updated observation, step reward, done flag, and info dict.
    """
    env = _get_env()
    try:
        result = env.step(action)

        # Append grade info to the final step
        if result.done:
            grade_result = env.grade()
            result.info["final_score"] = grade_result["score"]
            result.info["score_components"] = grade_result["components"]

        return result
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        raise HTTPException(status_code=500, detail=traceback.format_exc())


@app.get("/state", response_model=BugTriageState, tags=["openenv"])
def get_state() -> BugTriageState:
    """
    OpenEnv state() — return full internal state for inspection.
    Does NOT expose ground truth (severity / team labels).
    """
    env = _get_env()
    try:
        return env.state()
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/grade", tags=["openenv"])
def grade_episode() -> Dict[str, Any]:
    """
    Run the episode grader on current state and return score + components.
    Can be called at any point in the episode.
    """
    env = _get_env()
    try:
        return env.grade()
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/ui", tags=["meta"], include_in_schema=False)
def interactive_ui():
    """Serve the interactive web UI for manual environment testing."""
    if not UI_FILE.exists():
        raise HTTPException(status_code=404, detail="UI file not found")
    return FileResponse(str(UI_FILE), media_type="text/html")


@app.get("/", tags=["meta"])
def root() -> Dict[str, str]:
    return {
        "name": "BugTriage OpenEnv",
        "version": "1.0.0",
        "ui": "/ui",
        "docs": "/docs",
        "tasks": "/tasks",
        "reset": "POST /reset",
        "step": "POST /step",
        "state": "GET /state",
        "grade": "POST /grade",
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.server:app", host="0.0.0.0", port=7860, reload=False)
