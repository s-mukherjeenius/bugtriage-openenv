"""
BugTriage OpenEnv — Deterministic Graders
Each grader scores a completed (or partial) BugTriageState against ground truth.
All graders return float in [0.0, 1.0] and are fully reproducible.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Tuple

if TYPE_CHECKING:
    from app.models import BugTriageState
    from app.scenarios import TaskScenario


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def _severity_adjacent(predicted: str, truth: str) -> float:
    """
    Partial credit for severity close to ground truth.
    Correct → 1.0;  adjacent (off by 1) → 0.5;  off by 2+ → 0.0
    """
    order = ["low", "medium", "high", "critical"]
    if predicted not in order or truth not in order:
        return 0.0
    dist = abs(order.index(predicted) - order.index(truth))
    if dist == 0:
        return 1.0
    if dist == 1:
        return 0.5
    return 0.0


def _efficiency_factor(steps_used: int, max_steps: int, n_bugs: int) -> float:
    """
    Bonus multiplier (0.85 → 1.0) based on how efficiently the agent used steps.
    Minimum steps needed = 3 * n_bugs (classify + assign + submit per bug).
    At minimum steps → 1.0;  at max_steps → 0.85.
    Only meaningful when the episode has actually made progress.
    """
    min_steps = 3 * n_bugs
    if max_steps <= min_steps:
        return 1.0
    ratio = (steps_used - min_steps) / (max_steps - min_steps)
    return 1.0 - 0.15 * _clamp(ratio)


# ---------------------------------------------------------------------------
# Step-level reward (called from env.py on every step)
# ---------------------------------------------------------------------------

def compute_step_reward(
    action_type: str,
    bug_id: str,
    scenario: "TaskScenario",
    state: "BugTriageState",
    action_payload: Dict,
) -> Tuple[float, str]:
    """
    Returns (reward, message) for a single step action.
    Reward is in [-0.15, +0.20] so the agent gets continuous signal.
    """
    gt = scenario.ground_truth.get(bug_id)
    if gt is None:
        return -0.05, f"Unknown bug_id '{bug_id}'"

    reward = 0.0
    msg_parts = []

    if action_type == "classify":
        predicted = action_payload.get("severity", "")
        score = _severity_adjacent(predicted, gt.severity)
        if score == 1.0:
            reward = 0.10
            msg_parts.append(f"✓ Correct severity '{predicted}'")
        elif score == 0.5:
            reward = 0.04
            msg_parts.append(f"~ Adjacent severity '{predicted}' (correct: '{gt.severity}')")
        else:
            reward = -0.05
            msg_parts.append(f"✗ Wrong severity '{predicted}' (correct: '{gt.severity}')")

    elif action_type == "assign":
        predicted_team = action_payload.get("assigned_team", "")
        if predicted_team == gt.team:
            reward = 0.08
            msg_parts.append(f"✓ Correct team '{predicted_team}'")
        else:
            reward = -0.05
            msg_parts.append(f"✗ Wrong team '{predicted_team}' (correct: '{gt.team}')")

    elif action_type == "request_info":
        if gt.needs_info:
            reward = 0.06
            msg_parts.append("✓ Info request appropriate")
        else:
            reward = -0.04
            msg_parts.append("✗ Unnecessary info request")

    elif action_type == "mark_duplicate":
        claimed_original = action_payload.get("duplicate_of", "")
        if gt.is_duplicate_of and gt.is_duplicate_of == claimed_original:
            reward = 0.12
            msg_parts.append(f"✓ Correct duplicate: '{bug_id}' → '{claimed_original}'")
        elif gt.is_duplicate_of and gt.is_duplicate_of != claimed_original:
            reward = -0.05
            msg_parts.append("✗ Wrong duplicate mapping")
        else:
            reward = -0.08
            msg_parts.append(f"✗ False duplicate claim ('{bug_id}' is not a duplicate)")

    elif action_type == "escalate":
        if gt.should_escalate:
            reward = 0.08
            msg_parts.append(f"✓ Correct escalation for '{bug_id}'")
        else:
            reward = -0.03
            msg_parts.append(f"✗ Unnecessary escalation for '{bug_id}'")

    elif action_type == "submit":
        classified = bug_id in state.classifications
        assigned = bug_id in state.assignments
        if classified and assigned:
            reward = 0.05
            msg_parts.append(f"✓ Bug '{bug_id}' submitted with classify + assign")
        elif classified or assigned:
            reward = 0.01
            msg_parts.append(f"~ Partial: submitted '{bug_id}' but missing classify or assign")
        else:
            reward = -0.05
            msg_parts.append(f"✗ Submitted '{bug_id}' without classify or assign")

    return _clamp(reward, -0.15, 0.20), "; ".join(msg_parts)


# ---------------------------------------------------------------------------
# Episode-level grader: Task 1 — single-triage
# ---------------------------------------------------------------------------

def grade_single_triage(state: "BugTriageState", scenario: "TaskScenario") -> Dict:
    """
    Weights:
      severity correct          40%
      team correct              30%
      no unnecessary info req   15%  (only scored if bug was classified)
      escalation when expected  10%  (binary — no partial credit for inaction)
      efficiency bonus           5%  (only scored if bug was submitted)
    """
    bug_id = "PAY-001"
    gt = scenario.ground_truth[bug_id]
    components: Dict[str, float] = {}

    bug_classified = bug_id in state.classifications
    bug_submitted  = bug_id in state.submitted_bugs

    # Severity (40%)
    classified = state.classifications.get(bug_id, "")
    sev_score = _severity_adjacent(classified, gt.severity) if classified else 0.0
    components["severity"] = sev_score * 0.40

    # Team (30%)
    assigned = state.assignments.get(bug_id, "")
    components["team"] = (1.0 if assigned == gt.team else 0.0) * 0.30

    # No unnecessary info request (15%)
    # Only meaningful once the bug has been classified — inaction doesn't earn credit.
    if bug_classified:
        requested_info = bug_id in state.info_requests
        no_waste = 0.0 if (requested_info and not gt.needs_info) else 1.0
    else:
        no_waste = 0.0
    components["no_wasted_info"] = no_waste * 0.15

    # Escalation (10%) — binary, no partial credit for not escalating
    escalated = bug_id in state.escalations
    if escalated and gt.should_escalate:
        esc_score = 1.0
    elif escalated and not gt.should_escalate:
        esc_score = -0.5   # penalty for unwarranted escalation
    else:
        esc_score = 0.0    # not escalated → no credit, no penalty
    components["escalation"] = _clamp(esc_score) * 0.10

    # Efficiency (5%) — only meaningful once the bug has been submitted
    if bug_submitted and state.step_number > 0:
        eff = _efficiency_factor(state.step_number, scenario.max_steps, len(scenario.bug_reports))
    else:
        eff = 0.0
    components["efficiency"] = eff * 0.05

    total = _clamp(sum(components.values()))
    return {"score": round(total, 4), "components": {k: round(v, 4) for k, v in components.items()}}


# ---------------------------------------------------------------------------
# Episode-level grader: Task 2 — batch-triage
# ---------------------------------------------------------------------------

def grade_batch_triage(state: "BugTriageState", scenario: "TaskScenario") -> Dict:
    """
    Weights:
      severity accuracy (all 8)        30%
      team accuracy (all 8)            25%
      duplicate detection (1 pair)     20%
      info request for BUG-005         10%
      security escalation (BUG-007)    10%
      efficiency                        5%  (only if bugs submitted)
    """
    gt_map = scenario.ground_truth
    bug_ids = [b.id for b in scenario.bug_reports]
    n = len(bug_ids)
    components: Dict[str, float] = {}

    # Severity accuracy (30%) — partial credit for adjacent
    sev_scores = []
    for bid in bug_ids:
        gt = gt_map[bid]
        predicted = state.classifications.get(bid, "")
        sev_scores.append(_severity_adjacent(predicted, gt.severity) if predicted else 0.0)
    components["severity"] = (sum(sev_scores) / n) * 0.30

    # Team accuracy (25%)
    team_correct = sum(
        1 for bid in bug_ids
        if state.assignments.get(bid, "") == gt_map[bid].team
    )
    components["team"] = (team_correct / n) * 0.25

    # Duplicate detection (20%) — BUG-003 must be marked duplicate of BUG-006
    dup_detected = state.duplicates.get("BUG-003") == "BUG-006"
    false_dups = sum(
        1 for bid, orig in state.duplicates.items()
        if gt_map.get(bid) and gt_map[bid].is_duplicate_of is None
    )
    dup_score = (1.0 if dup_detected else 0.0) - false_dups * 0.25
    components["duplicate_detection"] = _clamp(dup_score) * 0.20

    # Info request for BUG-005 (10%)
    info_req_correct = "BUG-005" in state.info_requests and len(state.info_requests["BUG-005"]) > 0
    info_req_false = sum(
        1 for bid in bug_ids
        if bid != "BUG-005" and bid in state.info_requests and not gt_map[bid].needs_info
    )
    info_score = (1.0 if info_req_correct else 0.0) - info_req_false * 0.15
    components["info_request"] = _clamp(info_score) * 0.10

    # Security escalation for BUG-007 (10%)
    sec_escalated = "BUG-007" in state.escalations
    unnecessary_escalations = sum(
        1 for bid in bug_ids
        if bid in state.escalations and bid != "BUG-007" and not gt_map[bid].should_escalate
    )
    sec_score = (1.0 if sec_escalated else 0.0) - unnecessary_escalations * 0.2
    components["security_escalation"] = _clamp(sec_score) * 0.10

    # Efficiency (5%) — only if episode has made progress with submissions
    if state.submitted_bugs and state.step_number > 0:
        eff = _efficiency_factor(state.step_number, scenario.max_steps, n)
    else:
        eff = 0.0
    components["efficiency"] = eff * 0.05

    total = _clamp(sum(components.values()))
    return {"score": round(total, 4), "components": {k: round(v, 4) for k, v in components.items()}}


# ---------------------------------------------------------------------------
# Episode-level grader: Task 3 — sla-crisis
# ---------------------------------------------------------------------------

def grade_sla_crisis(state: "BugTriageState", scenario: "TaskScenario") -> Dict:
    """
    Weights:
      severity accuracy (15 bugs)      25%
      team accuracy (15 bugs)          20%
      duplicate detection (3 pairs)    20%
      SLA-critical escalations         20%
      info requests for incomplete     10%
      efficiency                        5%  (only if bugs submitted)
    """
    gt_map = scenario.ground_truth
    bug_ids = [b.id for b in scenario.bug_reports]
    n = len(bug_ids)
    components: Dict[str, float] = {}

    # Severity accuracy (25%)
    sev_scores = []
    for bid in bug_ids:
        gt = gt_map[bid]
        predicted = state.classifications.get(bid, "")
        sev_scores.append(_severity_adjacent(predicted, gt.severity) if predicted else 0.0)
    components["severity"] = (sum(sev_scores) / n) * 0.25

    # Team accuracy (20%)
    team_correct = sum(
        1 for bid in bug_ids
        if state.assignments.get(bid, "") == gt_map[bid].team
    )
    components["team"] = (team_correct / n) * 0.20

    # Duplicate detection (20%) — 3 correct pairs
    expected_dups = {
        "CRI-009": "CRI-003",
        "CRI-011": "CRI-004",
        "CRI-012": "CRI-007",
    }
    dup_hits = sum(
        1 for dup_id, orig_id in expected_dups.items()
        if state.duplicates.get(dup_id) == orig_id
    )
    false_dup_count = sum(
        1 for bid in state.duplicates
        if bid not in expected_dups
    )
    dup_score = (dup_hits / len(expected_dups)) - false_dup_count * 0.15
    components["duplicate_detection"] = _clamp(dup_score) * 0.20

    # SLA-critical escalations (20%)
    sla_critical_ids = [bid for bid, gt in gt_map.items() if gt.sla_critical and gt.should_escalate]
    escalated_count = 0
    for sla_id in sla_critical_ids:
        if sla_id in state.escalations:
            escalated_count += 1
        elif sla_id == "CRI-011" and "CRI-004" in state.escalations:
            escalated_count += 1  # duplicate resolved — original was escalated
    sla_score = escalated_count / len(sla_critical_ids) if sla_critical_ids else 1.0
    low_sev_escalated = sum(
        1 for bid in state.escalations
        if gt_map.get(bid) and gt_map[bid].severity in ("low", "medium")
        and not gt_map[bid].should_escalate
    )
    sla_score -= low_sev_escalated * 0.10
    components["sla_escalations"] = _clamp(sla_score) * 0.20

    # Info requests (10%) — CRI-005 and CRI-014 need info
    info_needed = [bid for bid, gt in gt_map.items() if gt.needs_info]
    info_hits = sum(1 for bid in info_needed if bid in state.info_requests)
    false_info = sum(1 for bid in state.info_requests if bid not in info_needed)
    info_score = (info_hits / len(info_needed)) - false_info * 0.10
    components["info_requests"] = _clamp(info_score) * 0.10

    # Efficiency (5%) — only if episode has made progress with submissions
    if state.submitted_bugs and state.step_number > 0:
        eff = _efficiency_factor(state.step_number, scenario.max_steps, n)
    else:
        eff = 0.0
    components["efficiency"] = eff * 0.05

    total = _clamp(sum(components.values()))
    return {"score": round(total, 4), "components": {k: round(v, 4) for k, v in components.items()}}


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

GRADERS = {
    "single-triage": grade_single_triage,
    "batch-triage":  grade_batch_triage,
    "sla-crisis":    grade_sla_crisis,
}


def grade_episode(task_id: str, state: "BugTriageState", scenario: "TaskScenario") -> Dict:
    """
    Entry point for episode grading. Returns dict with 'score' and 'components'.
    Score is in [0.0, 1.0].
    """
    grader = GRADERS.get(task_id)
    if grader is None:
        raise ValueError(f"No grader registered for task '{task_id}'")
    return grader(state, scenario)
