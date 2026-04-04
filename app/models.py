"""
BugTriage OpenEnv — Typed Pydantic Models
All Action, Observation, Reward, and State models.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class Severity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class Team(str, Enum):
    BACKEND = "backend"
    FRONTEND = "frontend"
    MOBILE = "mobile"
    INFRASTRUCTURE = "infrastructure"
    SECURITY = "security"
    DATABASE = "database"
    QA = "qa"


class ActionType(str, Enum):
    CLASSIFY = "classify"             # Assign severity to a bug
    ASSIGN = "assign"                 # Assign bug to a responsible team
    REQUEST_INFO = "request_info"     # Ask reporter for more details
    MARK_DUPLICATE = "mark_duplicate" # Mark bug as duplicate of another
    ESCALATE = "escalate"             # Escalate to engineering leadership
    SUBMIT = "submit"                 # Finalise triage for this bug


# ---------------------------------------------------------------------------
# Domain Objects
# ---------------------------------------------------------------------------

class BugReport(BaseModel):
    """A single raw bug report as received from the reporting system."""

    id: str = Field(..., description="Unique bug identifier, e.g. 'BUG-007'")
    title: str = Field(..., description="Short summary of the bug")
    description: str = Field(..., description="Full description provided by reporter")
    reporter: str = Field(..., description="Name or username of the reporter")
    timestamp: str = Field(..., description="ISO-8601 timestamp of submission")
    product: str = Field(..., description="Affected product/service name")
    version: str = Field(..., description="Affected product version")
    steps_to_reproduce: Optional[str] = Field(
        None, description="Reproduction steps, if provided"
    )
    expected_behavior: Optional[str] = Field(
        None, description="What the reporter expected to happen"
    )
    actual_behavior: Optional[str] = Field(
        None, description="What actually happened"
    )
    environment_info: Optional[Dict[str, str]] = Field(
        None, description="OS, browser, device info"
    )
    customer_tier: Optional[Literal["enterprise", "business", "starter", "free"]] = Field(
        None, description="Billing tier of the reporting customer"
    )
    sla_hours_remaining: Optional[float] = Field(
        None, description="Hours until SLA breach; None = no SLA"
    )
    linked_bug_ids: Optional[List[str]] = Field(
        None, description="IDs of bugs known to be related"
    )


# ---------------------------------------------------------------------------
# Action Model (what the agent sends)
# ---------------------------------------------------------------------------

class TriageAction(BaseModel):
    """
    A single triage action taken by the agent.

    Depending on action_type, different fields are required:
      classify        → bug_id + severity
      assign          → bug_id + assigned_team
      request_info    → bug_id + info_requested (non-empty list)
      mark_duplicate  → bug_id + duplicate_of
      escalate        → bug_id + escalation_reason
      submit          → bug_id  (finalises triage for this bug)
    """

    action_type: ActionType = Field(..., description="Type of triage action")
    bug_id: str = Field(..., description="Target bug report ID")

    # Optional fields — populated depending on action_type
    severity: Optional[Severity] = Field(None, description="Severity for 'classify' action")
    assigned_team: Optional[Team] = Field(None, description="Team for 'assign' action")
    duplicate_of: Optional[str] = Field(None, description="Original bug ID for 'mark_duplicate'")
    info_requested: Optional[List[str]] = Field(
        None, description="List of info items to request from reporter"
    )
    escalation_reason: Optional[str] = Field(
        None, description="Reason string for 'escalate' action"
    )


# ---------------------------------------------------------------------------
# Observation Model (what the agent receives)
# ---------------------------------------------------------------------------

class BugTriageObservation(BaseModel):
    """
    Full observation returned after reset() or step().
    Contains everything the agent needs to make its next decision.
    """

    step_number: int = Field(..., description="Current step index (0-based before first step)")
    task_name: str = Field(..., description="Active task identifier")
    task_description: str = Field(..., description="Human-readable task goal")
    instructions: str = Field(..., description="Detailed action format instructions")

    bug_reports: List[BugReport] = Field(..., description="All bug reports in this episode")

    action_history: List[Dict[str, Any]] = Field(
        default_factory=list, description="Sequence of past actions and their outcomes"
    )

    unprocessed_bug_ids: List[str] = Field(
        ..., description="Bug IDs that have not yet been submitted"
    )
    submitted_bug_ids: List[str] = Field(
        ..., description="Bug IDs whose triage has been finalised (submit action)"
    )

    current_classifications: Dict[str, str] = Field(
        default_factory=dict, description="bug_id → severity string assigned so far"
    )
    current_assignments: Dict[str, str] = Field(
        default_factory=dict, description="bug_id → team string assigned so far"
    )
    duplicate_map: Dict[str, str] = Field(
        default_factory=dict, description="bug_id → original_id for marked duplicates"
    )
    escalated_bug_ids: List[str] = Field(
        default_factory=list, description="Bug IDs that have been escalated"
    )

    available_teams: List[str] = Field(
        ..., description="Valid team names for assignment"
    )
    steps_remaining: int = Field(..., description="Steps left before episode truncation")
    cumulative_reward: float = Field(0.0, description="Total reward accumulated so far")


# ---------------------------------------------------------------------------
# Reward Model
# ---------------------------------------------------------------------------

class BugTriageReward(BaseModel):
    """
    Structured reward with component breakdown for interpretability.

    Step rewards range from -0.15 (wrong action) to +0.20 (correct action)
    to give the agent continuous learning signal at every step.
    Episode-level grader scores are in [0.0, 1.0].
    """

    value: float = Field(
        ...,
        ge=-1.0,
        le=1.0,
        description="Scalar reward for this step [-0.15, +0.20] or episode score [0, 1]",
    )
    components: Dict[str, float] = Field(
        default_factory=dict, description="Named sub-scores for interpretability"
    )
    message: str = Field("", description="Human-readable explanation of reward")


# ---------------------------------------------------------------------------
# State Model (full internal state, returned by state())
# ---------------------------------------------------------------------------

class BugTriageState(BaseModel):
    """Complete internal state of the environment (for debugging / inspection)."""

    task_name: str
    step_number: int
    max_steps: int
    bug_reports: List[BugReport]

    # Agent decisions recorded so far
    classifications: Dict[str, str]        # bug_id → severity
    assignments: Dict[str, str]            # bug_id → team
    duplicates: Dict[str, str]             # bug_id → original_id
    escalations: List[str]                 # escalated bug_ids
    info_requests: Dict[str, List[str]]    # bug_id → [requested items]
    submitted_bugs: List[str]              # finalised bug_ids

    total_reward: float
    done: bool
    episode_complete: bool                 # all bugs submitted OR max_steps hit


# ---------------------------------------------------------------------------
# API Response Models
# ---------------------------------------------------------------------------

class StepResult(BaseModel):
    observation: BugTriageObservation
    reward: float
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)


class ResetResult(BaseModel):
    observation: BugTriageObservation
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)


class TaskInfo(BaseModel):
    id: str
    name: str
    description: str
    difficulty: Literal["easy", "medium", "hard"]
    max_steps: int
    num_bugs: int
    reward_threshold: float


class HealthResponse(BaseModel):
    status: str
    version: str
    active_task: Optional[str]
    step_number: int
