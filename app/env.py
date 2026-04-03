"""
BugTriage OpenEnv — Environment Core
Implements the OpenEnv interface: reset(), step(), state().
"""
from __future__ import annotations

import copy
from typing import Any, Dict, Optional

from app.graders import compute_step_reward, grade_episode
from app.models import (
    ActionType,
    BugTriageObservation,
    BugTriageState,
    ResetResult,
    StepResult,
    TriageAction,
)
from app.scenarios import SCENARIOS, TaskScenario

_AVAILABLE_TEAMS = [
    "backend", "frontend", "mobile",
    "infrastructure", "security", "database", "qa",
]

_TASK_DESCRIPTIONS = {
    "single-triage": (
        "A single critical bug report requires immediate triage. "
        "Classify severity, assign to the correct team, and submit your decision. "
        "Avoid unnecessary steps."
    ),
    "batch-triage": (
        "Eight bug reports need processing. Classify each, assign teams, detect "
        "any duplicates, request info where critical details are missing, escalate "
        "security issues, then submit all bugs."
    ),
    "sla-crisis": (
        "Fifteen bug reports have arrived simultaneously during a critical incident. "
        "Handle SLA-critical escalations first, identify duplicates, request missing "
        "info, then process and submit all remaining bugs within the step budget."
    ),
}


class BugTriageEnv:
    """
    BugTriage OpenEnv environment.

    Lifecycle:
        env = BugTriageEnv(task_name="single-triage")
        result = env.reset()                    # → ResetResult
        result = env.step(action)               # → StepResult
        current_state = env.state()             # → BugTriageState
    """

    VERSION = "1.0.0"

    def __init__(self, task_name: str = "single-triage") -> None:
        if task_name not in SCENARIOS:
            raise ValueError(
                f"Unknown task '{task_name}'. "
                f"Available: {list(SCENARIOS.keys())}"
            )
        self.task_name = task_name
        self._scenario: TaskScenario = SCENARIOS[task_name]
        self._state: Optional[BugTriageState] = None
        self._action_history: list = []

    # ------------------------------------------------------------------
    # Public OpenEnv API
    # ------------------------------------------------------------------

    def reset(self) -> ResetResult:
        """Reset the environment and return the initial observation."""
        self._state = BugTriageState(
            task_name=self.task_name,
            step_number=0,
            max_steps=self._scenario.max_steps,
            bug_reports=copy.deepcopy(self._scenario.bug_reports),
            classifications={},
            assignments={},
            duplicates={},
            escalations=[],
            info_requests={},
            submitted_bugs=[],
            total_reward=0.0,
            done=False,
            episode_complete=False,
        )
        self._action_history = []
        obs = self._make_observation()
        return ResetResult(
            observation=obs,
            done=False,
            info={"task": self.task_name, "version": self.VERSION},
        )

    def step(self, action: TriageAction) -> StepResult:
        """Execute one triage action and return the updated observation + reward."""
        if self._state is None:
            raise RuntimeError("Call reset() before step()")
        if self._state.done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")

        self._state.step_number += 1
        step_reward, reward_msg = self._apply_action(action)
        self._state.total_reward += step_reward

        # Record in history
        history_entry = {
            "step": self._state.step_number,
            "action_type": action.action_type.value,
            "bug_id": action.bug_id,
            "reward": round(step_reward, 4),
            "message": reward_msg,
        }
        if action.severity:
            history_entry["severity"] = action.severity.value
        if action.assigned_team:
            history_entry["assigned_team"] = action.assigned_team.value
        if action.duplicate_of:
            history_entry["duplicate_of"] = action.duplicate_of
        if action.info_requested:
            history_entry["info_requested"] = action.info_requested
        if action.escalation_reason:
            history_entry["escalation_reason"] = action.escalation_reason
        self._action_history.append(history_entry)

        # Check done conditions
        done = self._check_done()
        self._state.done = done
        if done:
            self._state.episode_complete = True

        obs = self._make_observation()
        return StepResult(
            observation=obs,
            reward=step_reward,
            done=done,
            info={
                "reward_message": reward_msg,
                "step": self._state.step_number,
                "submitted": len(self._state.submitted_bugs),
                "total_bugs": len(self._state.bug_reports),
            },
        )

    def state(self) -> BugTriageState:
        """Return the full internal environment state (for inspection)."""
        if self._state is None:
            raise RuntimeError("Call reset() before state()")
        return self._state

    def grade(self) -> Dict[str, Any]:
        """
        Run the episode grader and return final score + component breakdown.
        Can be called at any time (not just on done).
        """
        if self._state is None:
            raise RuntimeError("Call reset() before grade()")
        return grade_episode(self.task_name, self._state, self._scenario)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _apply_action(self, action: TriageAction) -> tuple[float, str]:
        """Validate action, update state, compute step reward."""
        st = self._state
        bug_id = action.bug_id
        action_type = action.action_type.value

        # Validate bug_id exists
        valid_ids = {b.id for b in st.bug_reports}
        if bug_id not in valid_ids:
            return -0.05, f"Unknown bug_id '{bug_id}'"

        # Validate bug not already submitted (can't act on closed bugs)
        if bug_id in st.submitted_bugs and action_type != "submit":
            return -0.03, f"Bug '{bug_id}' already submitted — action ignored"

        # Apply state changes
        if action_type == "classify":
            if action.severity is None:
                return -0.05, "classify action requires 'severity' field"
            st.classifications[bug_id] = action.severity.value

        elif action_type == "assign":
            if action.assigned_team is None:
                return -0.05, "assign action requires 'assigned_team' field"
            st.assignments[bug_id] = action.assigned_team.value

        elif action_type == "request_info":
            items = action.info_requested or []
            if not items:
                return -0.04, "request_info action requires non-empty 'info_requested' list"
            st.info_requests[bug_id] = items

        elif action_type == "mark_duplicate":
            if action.duplicate_of is None:
                return -0.05, "mark_duplicate action requires 'duplicate_of' field"
            if action.duplicate_of not in valid_ids:
                return -0.05, f"'duplicate_of' references unknown bug_id '{action.duplicate_of}'"
            if action.duplicate_of == bug_id:
                return -0.05, "A bug cannot be marked as duplicate of itself"
            st.duplicates[bug_id] = action.duplicate_of

        elif action_type == "escalate":
            if bug_id not in st.escalations:
                st.escalations.append(bug_id)

        elif action_type == "submit":
            if bug_id not in st.submitted_bugs:
                st.submitted_bugs.append(bug_id)

        # Get step reward from grader
        action_payload = {}
        if action.severity:
            action_payload["severity"] = action.severity.value
        if action.assigned_team:
            action_payload["assigned_team"] = action.assigned_team.value
        if action.duplicate_of:
            action_payload["duplicate_of"] = action.duplicate_of
        if action.info_requested:
            action_payload["info_requested"] = action.info_requested

        reward, msg = compute_step_reward(
            action_type=action_type,
            bug_id=bug_id,
            scenario=self._scenario,
            state=st,
            action_payload=action_payload,
        )
        return reward, msg

    def _check_done(self) -> bool:
        """Episode ends when all bugs submitted OR max_steps reached."""
        st = self._state
        all_submitted = len(st.submitted_bugs) >= len(st.bug_reports)
        out_of_steps = st.step_number >= st.max_steps
        return all_submitted or out_of_steps

    def _make_observation(self) -> BugTriageObservation:
        st = self._state
        submitted_set = set(st.submitted_bugs)
        unprocessed = [b.id for b in st.bug_reports if b.id not in submitted_set]

        return BugTriageObservation(
            step_number=st.step_number,
            task_name=self.task_name,
            task_description=_TASK_DESCRIPTIONS.get(self.task_name, ""),
            instructions=self._scenario.instructions,
            bug_reports=st.bug_reports,
            action_history=self._action_history[-20:],  # last 20 actions for context
            unprocessed_bug_ids=unprocessed,
            submitted_bug_ids=list(st.submitted_bugs),
            current_classifications=dict(st.classifications),
            current_assignments=dict(st.assignments),
            duplicate_map=dict(st.duplicates),
            escalated_bug_ids=list(st.escalations),
            available_teams=_AVAILABLE_TEAMS,
            steps_remaining=max(0, st.max_steps - st.step_number),
            cumulative_reward=round(st.total_reward, 4),
        )


# ---------------------------------------------------------------------------
# Global singleton (used by the FastAPI server)
# ---------------------------------------------------------------------------

_active_env: Optional[BugTriageEnv] = None


def get_env() -> BugTriageEnv:
    global _active_env
    if _active_env is None:
        _active_env = BugTriageEnv("single-triage")
    return _active_env


def init_env(task_name: str) -> BugTriageEnv:
    global _active_env
    _active_env = BugTriageEnv(task_name)
    return _active_env
