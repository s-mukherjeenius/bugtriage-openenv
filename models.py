"""
BugTriage OpenEnv — Root models (openenv-core typed).
=====================================================
Canonical domain types (Severity, Team, ActionType, BugReport, TriageAction)
live in app/models.py. This module re-exports them and adds the openenv-core
typed wrappers (Action, Observation, State) required by the server layer.

This avoids duplicate enum/model definitions across the codebase.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

# ── Import canonical domain types from app/models.py ──
from app.models import (  # noqa: F401 — re-exported
    ActionType,
    BugReport,
    BugTriageObservation as InternalObservation,
    BugTriageReward,
    BugTriageState as InternalState,
    Severity,
    Team,
    TriageAction,
)

# ── openenv-core base classes (with fallback for local dev) ──
try:
    from openenv.core.env_server.types import Action, Observation, State
except ImportError:
    from pydantic import BaseModel as Action       # type: ignore[assignment]
    from pydantic import BaseModel as Observation   # type: ignore[assignment]
    from pydantic import BaseModel as State         # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Action — openenv-typed wrapper (used by server layer + client)
# ---------------------------------------------------------------------------

class BugTriageAction(Action):
    """
    A single triage action taken by the agent.

    action_type and bug_id are always required.
    Additional fields depend on action_type:
      classify        → severity
      assign          → assigned_team
      request_info    → info_requested
      mark_duplicate  → duplicate_of
      escalate        → escalation_reason
      flag_spam       → spam_reason
      submit          → (no extra fields)
    """

    action_type: ActionType = Field(..., description="Type of triage operation")
    bug_id:      str        = Field(..., description="Target bug report ID")

    severity:          Optional[Severity]    = Field(None, description="For 'classify' action")
    assigned_team:     Optional[Team]        = Field(None, description="For 'assign' action")
    duplicate_of:      Optional[str]         = Field(None, description="Original bug ID for 'mark_duplicate'")
    info_requested:    Optional[List[str]]   = Field(None, description="Info items for 'request_info'")
    escalation_reason: Optional[str]         = Field(None, description="Reason text for 'escalate'")
    spam_reason:       Optional[str]         = Field(None, description="Reason for flagging as spam (Task 4)")


# ---------------------------------------------------------------------------
# Observation — openenv-typed wrapper (returned by server to client)
# ---------------------------------------------------------------------------

class BugTriageObservation(Observation):
    """
    Full observation returned after reset() or step().
    Includes all bug reports, current triage state, and action history.
    reward and done are read by the openenv framework to build StepResult.
    """

    # ── openenv-required fields ──
    reward:  float = Field(0.0,   description="Step reward (read by framework)")
    done:    bool  = Field(False, description="Episode done flag (read by framework)")

    # ── task context ──
    step_number:      int    = Field(..., description="Current step index")
    task_name:        str    = Field(..., description="Active task identifier")
    task_description: str    = Field("",  description="Human-readable task goal")
    instructions:     str    = Field("",  description="Action format instructions")

    # ── bug report data ──
    bug_reports: List[Dict[str, Any]] = Field(
        default_factory=list, description="All bug reports in the episode"
    )

    # ── triage state ──
    unprocessed_bug_ids:     List[str]        = Field(default_factory=list)
    submitted_bug_ids:       List[str]        = Field(default_factory=list)
    flagged_spam_ids:        List[str]        = Field(default_factory=list)
    sla_breached_bug_ids:    List[str]        = Field(default_factory=list)
    current_classifications: Dict[str, str]   = Field(default_factory=dict)
    current_assignments:     Dict[str, str]   = Field(default_factory=dict)
    duplicate_map:           Dict[str, str]   = Field(default_factory=dict)
    escalated_bug_ids:       List[str]        = Field(default_factory=list)

    # ── meta ──
    action_history:    List[Dict[str, Any]] = Field(default_factory=list)
    available_teams:   List[str]            = Field(default_factory=list)
    steps_remaining:   int                  = Field(0)
    cumulative_reward: float                = Field(0.0)


# ---------------------------------------------------------------------------
# Reward — structured reward with component breakdown
# ---------------------------------------------------------------------------

# Re-export BugTriageReward from app/models.py (already defined there)
# Available as: from models import BugTriageReward


# ---------------------------------------------------------------------------
# State — episode metadata (returned by state())
# ---------------------------------------------------------------------------

class BugTriageState(State):
    """Episode-level metadata for inspection (openenv-typed)."""

    task_name:       str   = Field("", description="Active task name")
    step_number:     int   = Field(0,  description="Steps taken so far")
    max_steps:       int   = Field(0,  description="Maximum steps per episode")
    total_reward:    float = Field(0.0)
    done:            bool  = Field(False)
    submitted_count: int   = Field(0,  description="Number of bugs submitted")
    total_bugs:      int   = Field(0,  description="Total bugs in this episode")
