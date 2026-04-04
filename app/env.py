"""
BugTriage OpenEnv — Environment Core
Implements the OpenEnv interface: reset(), step(), state(), grade().

Supports both static scenarios (reproducible for evaluation) and
dynamic scenario generation (seed-based, for RL training).
"""
from __future__ import annotations

import copy
from typing import Any, Dict, Optional

from app.graders import compute_step_reward, grade_episode
from app.models import (
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
        env = BugTriageEnv("single-triage")
        result = env.reset()            # static scenario (reproducible)
        result = env.reset(seed=42)     # dynamic scenario (new bugs each seed)
        result = env.step(action)
        state  = env.state()
        grade  = env.grade()
    """

    VERSION = "1.0.0"

    def __init__(self, task_name: str = "single-triage") -> None:
        if task_name not in SCENARIOS:
            raise ValueError(f"Unknown task '{task_name}'. Available: {list(SCENARIOS.keys())}")
        self.task_name = task_name
        self._scenario: TaskScenario = SCENARIOS[task_name]
        self._state: Optional[BugTriageState] = None
        self._action_history: list = []

    def reset(self, seed: Optional[int] = None) -> ResetResult:
        """
        Reset the environment and return the initial observation.

        Args:
            seed: If provided, generates a fresh scenario from this seed.
                  If None, uses the static (default) scenario for reproducible evaluation.
        """
        if seed is not None:
            from app.generator import generate_scenario
            self._scenario = generate_scenario(self.task_name, seed)

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
        return ResetResult(
            observation=self._make_observation(),
            done=False,
            info={
                "task": self.task_name,
                "version": self.VERSION,
                "generated": seed is not None,
                "seed": seed,
            },
        )

    def step(self, action: TriageAction) -> StepResult:
        """Execute one triage action and return updated observation + reward."""
        if self._state is None:
            raise RuntimeError("Call reset() before step()")
        if self._state.done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")

        self._state.step_number += 1
        step_reward, reward_msg = self._apply_action(action)
        self._state.total_reward += step_reward

        history_entry: Dict[str, Any] = {
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

        done = self._check_done()
        self._state.done = done
        if done:
            self._state.episode_complete = True

        return StepResult(
            observation=self._make_observation(),
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
        """Return full internal environment state."""
        if self._state is None:
            raise RuntimeError("Call reset() before state()")
        return self._state

    def grade(self) -> Dict[str, Any]:
        """Run the episode grader. Returns score [0,1] + component breakdown."""
        if self._state is None:
            raise RuntimeError("Call reset() before grade()")
        return grade_episode(self.task_name, self._state, self._scenario)

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _apply_action(self, action: TriageAction) -> tuple[float, str]:
        st = self._state
        bug_id = action.bug_id
        action_type = action.action_type.value

        valid_ids = {b.id for b in st.bug_reports}
        if bug_id not in valid_ids:
            return -0.05, f"Unknown bug_id '{bug_id}'"
        if bug_id in st.submitted_bugs and action_type != "submit":
            return -0.03, f"Bug '{bug_id}' already submitted — action ignored"

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
                return -0.04, "request_info requires non-empty 'info_requested'"
            st.info_requests[bug_id] = items

        elif action_type == "mark_duplicate":
            if action.duplicate_of is None:
                return -0.05, "mark_duplicate requires 'duplicate_of'"
            if action.duplicate_of not in valid_ids:
                return -0.05, f"Unknown duplicate_of '{action.duplicate_of}'"
            if action.duplicate_of == bug_id:
                return -0.05, "A bug cannot be marked as duplicate of itself"
            st.duplicates[bug_id] = action.duplicate_of

        elif action_type == "escalate":
            if bug_id not in st.escalations:
                st.escalations.append(bug_id)

        elif action_type == "submit":
            if bug_id not in st.submitted_bugs:
                st.submitted_bugs.append(bug_id)

        payload = {}
        if action.severity:
            payload["severity"] = action.severity.value
        if action.assigned_team:
            payload["assigned_team"] = action.assigned_team.value
        if action.duplicate_of:
            payload["duplicate_of"] = action.duplicate_of
        if action.info_requested:
            payload["info_requested"] = action.info_requested

        return compute_step_reward(
            action_type=action_type, bug_id=bug_id,
            scenario=self._scenario, state=st, action_payload=payload,
        )

    def _check_done(self) -> bool:
        st = self._state
        return (len(st.submitted_bugs) >= len(st.bug_reports)
                or st.step_number >= st.max_steps)

    def _make_observation(self) -> BugTriageObservation:
        st = self._state
        submitted_set = set(st.submitted_bugs)
        return BugTriageObservation(
            step_number=st.step_number,
            task_name=self.task_name,
            task_description=_TASK_DESCRIPTIONS.get(self.task_name, ""),
            instructions=self._scenario.instructions,
            bug_reports=st.bug_reports,
            action_history=self._action_history[-20:],
            unprocessed_bug_ids=[b.id for b in st.bug_reports if b.id not in submitted_set],
            submitted_bug_ids=list(st.submitted_bugs),
            current_classifications=dict(st.classifications),
            current_assignments=dict(st.assignments),
            duplicate_map=dict(st.duplicates),
            escalated_bug_ids=list(st.escalations),
            available_teams=_AVAILABLE_TEAMS,
            steps_remaining=max(0, st.max_steps - st.step_number),
            cumulative_reward=round(st.total_reward, 4),
        )
