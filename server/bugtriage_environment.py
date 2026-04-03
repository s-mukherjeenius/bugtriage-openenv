"""
BugTriage OpenEnv — Server Environment
Implements openenv.core.env_server.interfaces.Environment.
All actual game logic delegates to app.env.BugTriageEnv (our existing core).
"""
from __future__ import annotations

import copy
from typing import Any, Dict
from uuid import uuid4

try:
    from openenv.core.env_server.interfaces import Environment
    from openenv.core.env_server.types import State
except ImportError:
    from pydantic import BaseModel as Environment  # type: ignore[assignment]
    from pydantic import BaseModel as State        # type: ignore[assignment]

# Dual-import pattern: works in-repo AND in Docker
try:
    from ..models import BugTriageAction, BugTriageObservation, BugTriageState
    from ..app.scenarios import SCENARIOS
    from ..app.graders import compute_step_reward, grade_episode
except ImportError:
    from models import BugTriageAction, BugTriageObservation, BugTriageState  # type: ignore[no-redef]
    from app.scenarios import SCENARIOS                                         # type: ignore[no-redef]
    from app.graders import compute_step_reward, grade_episode                 # type: ignore[no-redef]


_AVAILABLE_TEAMS = [
    "backend", "frontend", "mobile",
    "infrastructure", "security", "database", "qa",
]

_TASK_DESCRIPTIONS = {
    "single-triage": (
        "A single critical bug report requires immediate triage. "
        "Classify severity, assign to the correct team, and submit your decision."
    ),
    "batch-triage": (
        "Eight bug reports need processing. Classify each, assign teams, detect "
        "duplicates, request info where needed, escalate security issues, then submit all."
    ),
    "sla-crisis": (
        "Fifteen bug reports have arrived simultaneously during a critical incident. "
        "Handle SLA-critical escalations first, identify duplicates, request missing "
        "info, then process and submit all remaining bugs within the step budget."
    ),
}


class BugTriageEnvironment(Environment):
    """
    BugTriage environment implementing the openenv Environment interface.
    Wraps our deterministic graders and scenario data.
    """

    def __init__(self, task_name: str = "single-triage") -> None:
        super().__init__()
        if task_name not in SCENARIOS:
            raise ValueError(
                f"Unknown task '{task_name}'. Available: {list(SCENARIOS.keys())}"
            )
        self._task_name = task_name
        self._scenario  = SCENARIOS[task_name]
        self._reset_internal_state()

    # ------------------------------------------------------------------
    # openenv API
    # ------------------------------------------------------------------

    def reset(self) -> BugTriageObservation:
        """Reset episode and return initial observation."""
        self._reset_internal_state()
        return self._make_observation(reward=0.0)

    def step(self, action: BugTriageAction) -> BugTriageObservation:
        """Execute one triage action and return updated observation + reward + done."""
        if self._done:
            raise RuntimeError("Episode done. Call reset() to start a new episode.")

        self._step_number += 1
        step_reward, reward_msg = self._apply_action(action)
        self._total_reward += step_reward

        # Record in history
        hist_entry: Dict[str, Any] = {
            "step":        self._step_number,
            "action_type": action.action_type.value,
            "bug_id":      action.bug_id,
            "reward":      round(step_reward, 4),
            "message":     reward_msg,
        }
        if action.severity:
            hist_entry["severity"] = action.severity.value
        if action.assigned_team:
            hist_entry["assigned_team"] = action.assigned_team.value
        if action.duplicate_of:
            hist_entry["duplicate_of"] = action.duplicate_of
        if action.info_requested:
            hist_entry["info_requested"] = action.info_requested
        self._action_history.append(hist_entry)

        done = self._check_done()
        self._done = done

        return self._make_observation(reward=step_reward, done=done)

    @property
    def state(self) -> BugTriageState:
        """Return episode-level metadata."""
        return BugTriageState(
            task_name=self._task_name,
            step_number=self._step_number,
            max_steps=self._scenario.max_steps,
            total_reward=round(self._total_reward, 4),
            done=self._done,
            submitted_count=len(self._submitted_bugs),
            total_bugs=len(self._scenario.bug_reports),
        )

    def grade(self) -> Dict[str, Any]:
        """Run episode grader — returns score [0,1] + component breakdown."""
        from app.models import BugTriageState as InternalState
        internal = InternalState(
            task_name=self._task_name,
            step_number=self._step_number,
            max_steps=self._scenario.max_steps,
            bug_reports=self._scenario.bug_reports,
            classifications=dict(self._classifications),
            assignments=dict(self._assignments),
            duplicates=dict(self._duplicates),
            escalations=list(self._escalations),
            info_requests=dict(self._info_requests),
            submitted_bugs=list(self._submitted_bugs),
            total_reward=self._total_reward,
            done=self._done,
            episode_complete=self._done,
        )
        return grade_episode(self._task_name, internal, self._scenario)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _reset_internal_state(self) -> None:
        self._step_number:   int           = 0
        self._total_reward:  float         = 0.0
        self._done:          bool          = False
        self._classifications: Dict[str, str]       = {}
        self._assignments:    Dict[str, str]        = {}
        self._duplicates:     Dict[str, str]        = {}
        self._escalations:    list                  = []
        self._info_requests:  Dict[str, list]       = {}
        self._submitted_bugs: list                  = []
        self._action_history: list                  = []

    def _apply_action(self, action: BugTriageAction):
        valid_ids = {b.id for b in self._scenario.bug_reports}
        bug_id    = action.bug_id
        atype     = action.action_type.value

        if bug_id not in valid_ids:
            return -0.05, f"Unknown bug_id '{bug_id}'"
        if bug_id in self._submitted_bugs and atype != "submit":
            return -0.03, f"Bug '{bug_id}' already submitted"

        if atype == "classify":
            if action.severity is None:
                return -0.05, "classify requires 'severity'"
            self._classifications[bug_id] = action.severity.value

        elif atype == "assign":
            if action.assigned_team is None:
                return -0.05, "assign requires 'assigned_team'"
            self._assignments[bug_id] = action.assigned_team.value

        elif atype == "request_info":
            items = action.info_requested or []
            if not items:
                return -0.04, "request_info requires non-empty 'info_requested'"
            self._info_requests[bug_id] = items

        elif atype == "mark_duplicate":
            if action.duplicate_of is None:
                return -0.05, "mark_duplicate requires 'duplicate_of'"
            if action.duplicate_of not in valid_ids:
                return -0.05, f"Unknown duplicate_of '{action.duplicate_of}'"
            if action.duplicate_of == bug_id:
                return -0.05, "Cannot mark bug as duplicate of itself"
            self._duplicates[bug_id] = action.duplicate_of

        elif atype == "escalate":
            if bug_id not in self._escalations:
                self._escalations.append(bug_id)

        elif atype == "submit":
            if bug_id not in self._submitted_bugs:
                self._submitted_bugs.append(bug_id)

        # Build a minimal state object for the step grader
        from app.models import BugTriageState as InternalState
        tmp = InternalState(
            task_name=self._task_name,
            step_number=self._step_number,
            max_steps=self._scenario.max_steps,
            bug_reports=self._scenario.bug_reports,
            classifications=dict(self._classifications),
            assignments=dict(self._assignments),
            duplicates=dict(self._duplicates),
            escalations=list(self._escalations),
            info_requests=dict(self._info_requests),
            submitted_bugs=list(self._submitted_bugs),
            total_reward=self._total_reward,
            done=self._done,
            episode_complete=self._done,
        )
        payload = {}
        if action.severity:
            payload["severity"] = action.severity.value
        if action.assigned_team:
            payload["assigned_team"] = action.assigned_team.value
        if action.duplicate_of:
            payload["duplicate_of"] = action.duplicate_of
        if action.info_requested:
            payload["info_requested"] = action.info_requested

        return compute_step_reward(atype, bug_id, self._scenario, tmp, payload)

    def _check_done(self) -> bool:
        all_submitted = len(self._submitted_bugs) >= len(self._scenario.bug_reports)
        out_of_steps  = self._step_number >= self._scenario.max_steps
        return all_submitted or out_of_steps

    def _make_observation(self, reward: float = 0.0, done: bool = False) -> BugTriageObservation:
        submitted_set = set(self._submitted_bugs)
        unprocessed   = [b.id for b in self._scenario.bug_reports if b.id not in submitted_set]

        bug_dicts = [b.model_dump() for b in self._scenario.bug_reports]

        return BugTriageObservation(
            reward=round(reward, 4),
            done=done,
            step_number=self._step_number,
            task_name=self._task_name,
            task_description=_TASK_DESCRIPTIONS.get(self._task_name, ""),
            instructions=self._scenario.instructions,
            bug_reports=bug_dicts,
            unprocessed_bug_ids=unprocessed,
            submitted_bug_ids=list(self._submitted_bugs),
            current_classifications=dict(self._classifications),
            current_assignments=dict(self._assignments),
            duplicate_map=dict(self._duplicates),
            escalated_bug_ids=list(self._escalations),
            action_history=self._action_history[-20:],
            available_teams=_AVAILABLE_TEAMS,
            steps_remaining=max(0, self._scenario.max_steps - self._step_number),
            cumulative_reward=round(self._total_reward, 4),
        )
