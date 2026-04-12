"""
BugTriage OpenEnv — Server Environment (Wrapper)
=================================================
Implements openenv.core.env_server.interfaces.Environment by delegating
to the canonical BugTriageEnv in app/env.py.

This is a thin adapter — all environment logic lives in app/env.py.
No duplicated state management, reward computation, or grading logic.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

try:
    from openenv.core.env_server.interfaces import Environment
except ImportError:
    from pydantic import BaseModel as Environment  # type: ignore[assignment]

try:
    from ..app.env import BugTriageEnv
    from ..app.models import BugTriageObservation as InternalObservation
    from ..app.models import TriageAction as InternalAction
    from ..models import BugTriageAction, BugTriageObservation, BugTriageState
except ImportError:
    from app.env import BugTriageEnv  # type: ignore[no-redef]
    from app.models import BugTriageObservation as InternalObservation  # type: ignore[no-redef]
    from app.models import TriageAction as InternalAction  # type: ignore[no-redef]
    from models import BugTriageAction, BugTriageObservation, BugTriageState  # type: ignore[no-redef]

# Re-export for server/app.py
try:
    from ..app.scenarios import SCENARIOS  # noqa: F401
except ImportError:
    from app.scenarios import SCENARIOS  # type: ignore[no-redef]  # noqa: F401

_AVAILABLE_TEAMS = [
    "backend", "frontend", "mobile",
    "infrastructure", "security", "database", "qa",
]


class BugTriageEnvironment(Environment):
    """
    OpenEnv Environment adapter for BugTriage.

    Wraps the canonical BugTriageEnv — all logic (rewards, SLA ticking,
    root-cause resolution, grading) is delegated, not reimplemented.
    """

    def __init__(self, task_name: str = "single-triage") -> None:
        super().__init__()
        self._env = BugTriageEnv(task_name)

    # ------------------------------------------------------------------
    # openenv API
    # ------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None) -> BugTriageObservation:
        result = self._env.reset(seed=seed)
        return self._to_openenv_obs(result.observation, reward=0.0, done=False)

    def step(self, action: BugTriageAction) -> BugTriageObservation:
        internal_action = InternalAction(
            action_type=action.action_type,
            bug_id=action.bug_id,
            severity=action.severity,
            assigned_team=action.assigned_team,
            duplicate_of=action.duplicate_of,
            info_requested=action.info_requested,
            escalation_reason=action.escalation_reason,
            spam_reason=getattr(action, "spam_reason", None),
        )
        result = self._env.step(internal_action)
        return self._to_openenv_obs(result.observation, reward=result.reward, done=result.done)

    @property
    def state(self) -> BugTriageState:
        st = self._env.state()
        return BugTriageState(
            task_name=st.task_name,
            step_number=st.step_number,
            max_steps=st.max_steps,
            total_reward=round(st.total_reward, 4),
            done=st.done,
            submitted_count=len(st.submitted_bugs),
            total_bugs=len(st.bug_reports),
        )

    def grade(self) -> Dict[str, Any]:
        return self._env.grade()

    # ------------------------------------------------------------------
    # Adapter: convert internal Observation → openenv Observation
    # ------------------------------------------------------------------

    def _to_openenv_obs(
        self, obs: InternalObservation, reward: float, done: bool,
    ) -> BugTriageObservation:
        return BugTriageObservation(
            reward=round(reward, 4),
            done=done,
            step_number=obs.step_number,
            task_name=obs.task_name,
            task_description=obs.task_description,
            instructions=obs.instructions,
            bug_reports=[
                b.model_dump() if hasattr(b, "model_dump") else b.dict()
                for b in obs.bug_reports
            ],
            unprocessed_bug_ids=obs.unprocessed_bug_ids,
            submitted_bug_ids=obs.submitted_bug_ids,
            flagged_spam_ids=obs.flagged_spam_ids,
            sla_breached_bug_ids=obs.sla_breached_bug_ids,
            current_classifications=obs.current_classifications,
            current_assignments=obs.current_assignments,
            duplicate_map=obs.duplicate_map,
            escalated_bug_ids=obs.escalated_bug_ids,
            action_history=obs.action_history,
            available_teams=obs.available_teams,
            steps_remaining=obs.steps_remaining,
            cumulative_reward=obs.cumulative_reward,
        )
