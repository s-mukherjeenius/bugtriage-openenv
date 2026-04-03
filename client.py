"""
BugTriage OpenEnv — Client
Implements openenv's EnvClient for typed interaction with the BugTriage server.
"""
from __future__ import annotations

from typing import Any, Dict

try:
    from openenv.core.env_client import EnvClient
    from openenv.core.client_types import StepResult
except ImportError:
    # Local dev fallback — minimal stub
    class EnvClient:  # type: ignore[no-redef]
        def __init__(self, base_url: str = "http://localhost:7860", **kw): ...
    class StepResult:  # type: ignore[no-redef]
        def __init__(self, **kw): ...

from models import BugTriageAction, BugTriageObservation, BugTriageState


class BugTriageEnv(EnvClient[BugTriageAction, BugTriageObservation, BugTriageState]):
    """
    Typed client for the BugTriage OpenEnv environment.

    Usage (async):
        async with BugTriageEnv(base_url="ws://localhost:7860") as env:
            result = await env.reset()
            action = BugTriageAction(action_type="classify", bug_id="PAY-001", severity="critical")
            result = await env.step(action)

    Usage (sync):
        with BugTriageEnv(base_url="ws://localhost:7860").sync() as env:
            result = env.reset()
            result = env.step(BugTriageAction(...))
    """

    def _step_payload(self, action: BugTriageAction) -> Dict[str, Any]:
        """Serialize action to JSON dict for the WebSocket message."""
        payload: Dict[str, Any] = {
            "action_type": action.action_type.value,
            "bug_id":      action.bug_id,
        }
        if action.severity:
            payload["severity"] = action.severity.value
        if action.assigned_team:
            payload["assigned_team"] = action.assigned_team.value
        if action.duplicate_of:
            payload["duplicate_of"] = action.duplicate_of
        if action.info_requested:
            payload["info_requested"] = action.info_requested
        if action.escalation_reason:
            payload["escalation_reason"] = action.escalation_reason
        return payload

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult:
        """Deserialize server response into a typed StepResult."""
        obs_data = payload.get("observation", {})
        obs      = BugTriageObservation(**obs_data)
        return StepResult(
            observation=obs,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> BugTriageState:
        """Deserialize state response."""
        return BugTriageState(**payload)
