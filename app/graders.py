"""
BugTriage OpenEnv — Deterministic Graders
==========================================
Each grader scores a completed (or partial) BugTriageState against ground truth.
All graders return float in (0.0, 1.0) — strictly between 0 and 1, as
required by the OpenEnv validator — and are fully reproducible.

Graders are GENERIC — they derive all bug IDs and expected actions from
the scenario's ground_truth dict, not from hardcoded IDs. This allows
both static scenarios and dynamically generated scenarios to be graded
by the same code.

Step-level rewards:
  Correct actions   → positive reward (+0.10 to +0.20)
  Adjacent/partial  → small positive   (+0.03 to +0.06)
  Wrong actions     → negative reward  (-0.05 to -0.15)
  Invalid actions   → penalty          (-0.05 to -0.08)

Episode-level grading weights (v2 — rebalanced):
  Task 4 root-cause resolution raised from 2% → 8% (matched to feature prominence).
  Team accuracy now gives partial credit for adjacent teams.
  Efficiency scaled to 8-10% range to reward genuine strategic planning.
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
    Correct → 1.0;  adjacent (off by 1) → 0.4;  off by 2 → 0.1;  off by 3 → 0.0
    """
    order = ["low", "medium", "high", "critical"]
    if predicted not in order or truth not in order:
        return 0.0
    dist = abs(order.index(predicted) - order.index(truth))
    if dist == 0:
        return 1.0
    if dist == 1:
        return 0.4
    if dist == 2:
        return 0.1
    return 0.0


# Team adjacency map — teams with overlapping responsibilities get partial credit.
# This models real-world ambiguity: a Redis connection pool issue could reasonably
# be assigned to either infrastructure or backend.
_TEAM_ADJACENCY: Dict[str, set] = {
    "backend":        {"infrastructure", "database"},
    "infrastructure": {"backend", "database", "security"},
    "database":       {"backend", "infrastructure"},
    "security":       {"backend", "infrastructure"},
    "frontend":       {"mobile"},
    "mobile":         {"frontend"},
    "qa":             set(),
}


def _team_score(predicted: str, truth: str) -> float:
    """
    Team assignment accuracy with partial credit for adjacent teams.
    Correct → 1.0;  adjacent (related team) → 0.3;  wrong → 0.0
    """
    if predicted == truth:
        return 1.0
    adjacents = _TEAM_ADJACENCY.get(truth, set())
    if predicted in adjacents:
        return 0.3
    return 0.0


def _efficiency_factor(steps_used: int, max_steps: int, n_bugs: int) -> float:
    """
    Efficiency score (0.0 → 1.0) based on how efficiently the agent used steps.
    Minimum steps needed = 3 * n_bugs (classify + assign + submit per bug).

    Perfect efficiency (at minimum steps) → 1.0
    Using all max_steps → 0.3 (not zero — completing the task still has value)
    """
    min_steps = 3 * n_bugs
    if steps_used <= min_steps:
        return 1.0
    if max_steps <= min_steps:
        return 1.0
    ratio = (steps_used - min_steps) / (max_steps - min_steps)
    return 1.0 - 0.7 * _clamp(ratio)


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
    Already generic — reads from scenario.ground_truth.
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
            reward = 0.15
            msg_parts.append(f"✓ Correct severity '{predicted}'")
        elif score >= 0.4:
            reward = 0.06
            msg_parts.append(f"~ Adjacent severity '{predicted}' (correct: '{gt.severity}')")
        elif score >= 0.1:
            reward = -0.03
            msg_parts.append(f"~ Far severity '{predicted}' (correct: '{gt.severity}')")
        else:
            reward = -0.10
            msg_parts.append(f"✗ Wrong severity '{predicted}' (correct: '{gt.severity}')")

    elif action_type == "assign":
        predicted_team = action_payload.get("assigned_team", "")
        t_score = _team_score(predicted_team, gt.team)
        if t_score == 1.0:
            reward = 0.12
            msg_parts.append(f"✓ Correct team '{predicted_team}'")
        elif t_score > 0:
            reward = 0.04
            msg_parts.append(f"~ Adjacent team '{predicted_team}' (correct: '{gt.team}')")
        else:
            reward = -0.08
            msg_parts.append(f"✗ Wrong team '{predicted_team}' (correct: '{gt.team}')")

    elif action_type == "request_info":
        if gt.needs_info:
            reward = 0.10
            msg_parts.append("✓ Info request appropriate — critical details are missing")
        else:
            reward = -0.05
            msg_parts.append("✗ Unnecessary info request — all details already present")

    elif action_type == "mark_duplicate":
        claimed_original = action_payload.get("duplicate_of", "")
        if gt.is_duplicate_of and gt.is_duplicate_of == claimed_original:
            reward = 0.18
            msg_parts.append(f"✓ Correct duplicate: '{bug_id}' → '{claimed_original}'")
        elif gt.is_duplicate_of and gt.is_duplicate_of != claimed_original:
            reward = -0.08
            msg_parts.append(f"✗ Wrong duplicate target (correct: '{gt.is_duplicate_of}')")
        else:
            reward = -0.12
            msg_parts.append(f"✗ False duplicate claim ('{bug_id}' is not a duplicate)")

    elif action_type == "escalate":
        if gt.should_escalate:
            reward = 0.12
            msg_parts.append(f"✓ Correct escalation for '{bug_id}'")
        else:
            reward = -0.05
            msg_parts.append(f"✗ Unnecessary escalation for '{bug_id}'")

    elif action_type == "submit":
        classified = bug_id in state.classifications
        assigned = bug_id in state.assignments
        if classified and assigned:
            reward = 0.08
            msg_parts.append(f"✓ Bug '{bug_id}' submitted with classify + assign")
        elif classified or assigned:
            reward = 0.02
            msg_parts.append(f"~ Partial: submitted '{bug_id}' but missing classify or assign")
        else:
            reward = -0.08
            msg_parts.append(f"✗ Submitted '{bug_id}' without classify or assign")

    elif action_type == "flag_spam":
        if gt.is_spam:
            reward = 0.20
            msg_parts.append(f"✓ Correctly flagged '{bug_id}' as spam")
        else:
            reward = -0.15
            msg_parts.append(f"✗ Falsely flagged real bug '{bug_id}' as spam")

    return _clamp(reward, -0.15, 0.20), "; ".join(msg_parts)


# ---------------------------------------------------------------------------
# Helper stub for safe .needs_info access
# ---------------------------------------------------------------------------

class BugGroundTruthStub:
    needs_info = False


# ---------------------------------------------------------------------------
# Episode-level grader: Task 1 — single-triage (EASY)
# ---------------------------------------------------------------------------

def grade_single_triage(state: "BugTriageState", scenario: "TaskScenario") -> Dict:
    """
    Weights (v2):
      severity correct          35%
      team correct              25%
      no unnecessary info req   12%
      escalation when expected  10%
      efficiency bonus           8%
      prioritisation (submit     10%  ← new: did agent submit the bug?
        without wasting steps)
    """
    bug_id = scenario.bug_reports[0].id
    gt = scenario.ground_truth[bug_id]
    components: Dict[str, float] = {}

    bug_classified = bug_id in state.classifications
    bug_submitted = bug_id in state.submitted_bugs

    # Severity (35%)
    classified = state.classifications.get(bug_id, "")
    sev_score = _severity_adjacent(classified, gt.severity) if classified else 0.0
    components["severity"] = sev_score * 0.35

    # Team (25%) — now with adjacency partial credit
    assigned = state.assignments.get(bug_id, "")
    components["team"] = _team_score(assigned, gt.team) * 0.25

    # No unnecessary info request (12%)
    if bug_classified:
        requested_info = bug_id in state.info_requests
        no_waste = 0.0 if (requested_info and not gt.needs_info) else 1.0
    else:
        no_waste = 0.0
    components["no_wasted_info"] = no_waste * 0.12

    # Escalation (10%)
    escalated = bug_id in state.escalations
    if escalated and gt.should_escalate:
        esc_score = 1.0
    elif escalated and not gt.should_escalate:
        esc_score = -0.5
    else:
        esc_score = 0.0
    components["escalation"] = _clamp(esc_score) * 0.10

    # Efficiency (8%) — meaningful now
    if bug_submitted and state.step_number > 0:
        eff = _efficiency_factor(state.step_number, scenario.max_steps, len(scenario.bug_reports))
    else:
        eff = 0.0
    components["efficiency"] = eff * 0.08

    # Completion / prioritisation (10%) — did agent actually submit?
    if bug_submitted and bug_classified:
        comp_score = 1.0
    elif bug_submitted:
        comp_score = 0.5
    else:
        comp_score = 0.0
    components["completion"] = comp_score * 0.10

    total = _clamp(sum(components.values()))
    return {"score": round(total, 4), "components": {k: round(v, 4) for k, v in components.items()}}


# ---------------------------------------------------------------------------
# Episode-level grader: Task 2 — batch-triage (MEDIUM)
# ---------------------------------------------------------------------------

def grade_batch_triage(state: "BugTriageState", scenario: "TaskScenario") -> Dict:
    """
    Weights (v2):
      severity accuracy          25%
      team accuracy              20%
      duplicate detection        20%
      info request quality       10%
      escalation correctness     10%
      efficiency                  8%
      completion rate             7%  ← new
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

    # Team accuracy (20%) — with adjacency partial credit
    team_scores = []
    for bid in bug_ids:
        assigned = state.assignments.get(bid, "")
        team_scores.append(_team_score(assigned, gt_map[bid].team))
    components["team"] = (sum(team_scores) / n) * 0.20

    # Duplicate detection (20%)
    expected_dups = {bid: gt.is_duplicate_of for bid, gt in gt_map.items() if gt.is_duplicate_of}
    if expected_dups:
        dup_hits = sum(
            1 for dup_id, orig_id in expected_dups.items()
            if state.duplicates.get(dup_id) == orig_id
        )
        false_dups = sum(1 for bid in state.duplicates if bid not in expected_dups)
        dup_score = (dup_hits / len(expected_dups)) - false_dups * 0.25
    else:
        dup_score = 1.0
        if state.duplicates:
            dup_score = 0.0
    components["duplicate_detection"] = _clamp(dup_score) * 0.20

    # Info request quality (10%)
    info_needed = [bid for bid, gt in gt_map.items() if gt.needs_info]
    if info_needed:
        info_hits = sum(1 for bid in info_needed if bid in state.info_requests)
        false_info = sum(
            1 for bid in state.info_requests
            if bid not in info_needed and not gt_map.get(bid, BugGroundTruthStub()).needs_info
        )
        info_score = (info_hits / len(info_needed)) - false_info * 0.15
    else:
        info_score = 1.0
        if state.info_requests:
            false_count = sum(1 for bid in state.info_requests if not gt_map.get(bid, BugGroundTruthStub()).needs_info)
            info_score = max(0.0, 1.0 - false_count * 0.15)
    components["info_request"] = _clamp(info_score) * 0.10

    # Escalation correctness (10%)
    should_escalate = [bid for bid, gt in gt_map.items() if gt.should_escalate]
    if should_escalate:
        esc_hits = sum(1 for bid in should_escalate if bid in state.escalations)
        unnecessary = sum(1 for bid in state.escalations if bid not in should_escalate)
        esc_score = (esc_hits / len(should_escalate)) - unnecessary * 0.2
    else:
        esc_score = 1.0
        if state.escalations:
            esc_score = max(0.0, 1.0 - len(state.escalations) * 0.2)
    components["security_escalation"] = _clamp(esc_score) * 0.10

    # Efficiency (8%)
    if state.submitted_bugs and state.step_number > 0:
        eff = _efficiency_factor(state.step_number, scenario.max_steps, n)
    else:
        eff = 0.0
    components["efficiency"] = eff * 0.08

    # Completion rate (7%) — fraction of bugs actually submitted
    completion = len(state.submitted_bugs) / n if n > 0 else 0.0
    components["completion"] = completion * 0.07

    total = _clamp(sum(components.values()))
    return {"score": round(total, 4), "components": {k: round(v, 4) for k, v in components.items()}}


# ---------------------------------------------------------------------------
# Episode-level grader: Task 3 — sla-crisis (HARD)
# ---------------------------------------------------------------------------

def grade_sla_crisis(state: "BugTriageState", scenario: "TaskScenario") -> Dict:
    """
    Weights (v2):
      severity accuracy          22%
      team accuracy              18%
      duplicate detection        18%
      SLA-critical escalations   18%
      info requests               8%
      efficiency                  8%
      completion rate             8%  ← new
    """
    gt_map = scenario.ground_truth
    bug_ids = [b.id for b in scenario.bug_reports]
    n = len(bug_ids)
    components: Dict[str, float] = {}

    # Severity accuracy (22%)
    sev_scores = []
    for bid in bug_ids:
        gt = gt_map[bid]
        predicted = state.classifications.get(bid, "")
        sev_scores.append(_severity_adjacent(predicted, gt.severity) if predicted else 0.0)
    components["severity"] = (sum(sev_scores) / n) * 0.22

    # Team accuracy (18%) — with adjacency
    team_scores = []
    for bid in bug_ids:
        assigned = state.assignments.get(bid, "")
        team_scores.append(_team_score(assigned, gt_map[bid].team))
    components["team"] = (sum(team_scores) / n) * 0.18

    # Duplicate detection (18%)
    expected_dups = {bid: gt.is_duplicate_of for bid, gt in gt_map.items() if gt.is_duplicate_of}
    if expected_dups:
        dup_hits = sum(
            1 for dup_id, orig_id in expected_dups.items()
            if state.duplicates.get(dup_id) == orig_id
        )
        false_dup_count = sum(1 for bid in state.duplicates if bid not in expected_dups)
        dup_score = (dup_hits / len(expected_dups)) - false_dup_count * 0.15
    else:
        dup_score = 1.0
    components["duplicate_detection"] = _clamp(dup_score) * 0.18

    # SLA-critical escalations (18%)
    sla_critical_ids = [bid for bid, gt in gt_map.items() if gt.sla_critical and gt.should_escalate]
    if sla_critical_ids:
        escalated_count = 0
        for sla_id in sla_critical_ids:
            if sla_id in state.escalations:
                escalated_count += 1
            else:
                gt = gt_map[sla_id]
                if gt.is_duplicate_of and gt.is_duplicate_of in state.escalations:
                    escalated_count += 1
        sla_score = escalated_count / len(sla_critical_ids)
        low_sev_escalated = sum(
            1 for bid in state.escalations
            if gt_map.get(bid) and gt_map[bid].severity in ("low", "medium")
            and not gt_map[bid].should_escalate
        )
        sla_score -= low_sev_escalated * 0.10
    else:
        sla_score = 1.0
    components["sla_escalations"] = _clamp(sla_score) * 0.18

    # Info requests (8%)
    info_needed = [bid for bid, gt in gt_map.items() if gt.needs_info]
    if info_needed:
        info_hits = sum(1 for bid in info_needed if bid in state.info_requests)
        false_info = sum(1 for bid in state.info_requests if bid not in info_needed)
        info_score = (info_hits / len(info_needed)) - false_info * 0.10
    else:
        info_score = 1.0
    components["info_requests"] = _clamp(info_score) * 0.08

    # Efficiency (8%)
    if state.submitted_bugs and state.step_number > 0:
        eff = _efficiency_factor(state.step_number, scenario.max_steps, n)
    else:
        eff = 0.0
    components["efficiency"] = eff * 0.08

    # Completion rate (8%)
    completion = len(state.submitted_bugs) / n if n > 0 else 0.0
    components["completion"] = completion * 0.08

    total = _clamp(sum(components.values()))
    return {"score": round(total, 4), "components": {k: round(v, 4) for k, v in components.items()}}


# ---------------------------------------------------------------------------
# Episode-level grader: Task 4 — adversarial-triage (EXPERT)
# ---------------------------------------------------------------------------

def grade_adversarial_triage(state: "BugTriageState", scenario: "TaskScenario") -> Dict:
    """
    Weights (v2 — rebalanced):
      spam detection            18%
      severity accuracy          17%  (non-spam bugs only)
      team accuracy              13%  (with adjacency partial credit)
      duplicate detection        13%
      SLA-critical escalations   13%
      info requests               8%
      root-cause resolution       8%  ← raised from 2% to match feature prominence
      efficiency                  5%
      completion                  5%
    """
    gt_map = scenario.ground_truth
    bug_ids = [b.id for b in scenario.bug_reports]
    real_ids = [bid for bid in bug_ids if not gt_map[bid].is_spam]
    spam_ids = [bid for bid in bug_ids if gt_map[bid].is_spam]
    n_real = len(real_ids)
    components: Dict[str, float] = {}

    # Spam detection (18%)
    flagged = set(getattr(state, 'flagged_spam', []))
    correct_flags = sum(1 for sid in spam_ids if sid in flagged)
    false_flags = sum(1 for bid in flagged if bid not in spam_ids)
    missed_spam = sum(1 for sid in spam_ids if sid in state.submitted_bugs and sid not in flagged)
    if spam_ids:
        spam_score = (correct_flags / len(spam_ids)) - false_flags * 0.20 - missed_spam * 0.10
    else:
        spam_score = 1.0
    components["spam_detection"] = _clamp(spam_score) * 0.18

    # Severity accuracy (17%) — only real bugs, with adjacency
    sev_scores = []
    for bid in real_ids:
        gt = gt_map[bid]
        predicted = state.classifications.get(bid, "")
        sev_scores.append(_severity_adjacent(predicted, gt.severity) if predicted else 0.0)
    components["severity"] = (sum(sev_scores) / n_real) * 0.17 if n_real else 0.0

    # Team accuracy (13%) — only real bugs, with adjacency partial credit
    team_scores = []
    for bid in real_ids:
        assigned = state.assignments.get(bid, "")
        team_scores.append(_team_score(assigned, gt_map[bid].team))
    components["team"] = (sum(team_scores) / n_real) * 0.13 if n_real else 0.0

    # Duplicate detection (13%)
    expected_dups = {bid: gt.is_duplicate_of for bid, gt in gt_map.items() if gt.is_duplicate_of}
    if expected_dups:
        dup_hits = sum(1 for dup_id, orig_id in expected_dups.items() if state.duplicates.get(dup_id) == orig_id)
        false_dup_count = sum(1 for bid in state.duplicates if bid not in expected_dups)
        dup_score = (dup_hits / len(expected_dups)) - false_dup_count * 0.15
    else:
        dup_score = 1.0
    components["duplicate_detection"] = _clamp(dup_score) * 0.13

    # SLA escalations (13%)
    sla_critical_ids = [bid for bid, gt in gt_map.items() if gt.sla_critical and gt.should_escalate]
    if sla_critical_ids:
        esc_count = sum(1 for sid in sla_critical_ids if sid in state.escalations)
        sla_score = esc_count / len(sla_critical_ids)
        low_sev_esc = sum(1 for bid in state.escalations
                          if gt_map.get(bid) and gt_map[bid].severity in ("low", "medium") and not gt_map[bid].should_escalate)
        sla_score -= low_sev_esc * 0.10
    else:
        sla_score = 1.0
    components["sla_escalations"] = _clamp(sla_score) * 0.13

    # Info requests (8%)
    info_needed = [bid for bid, gt in gt_map.items() if gt.needs_info]
    if info_needed:
        info_hits = sum(1 for bid in info_needed if bid in state.info_requests)
        false_info = sum(1 for bid in state.info_requests if bid not in info_needed)
        info_score = (info_hits / len(info_needed)) - false_info * 0.10
    else:
        info_score = 1.0
    components["info_requests"] = _clamp(info_score) * 0.08

    # Root-cause resolution (8%) — raised from 2% to reward causal reasoning
    rc_bugs = {gt.root_cause_chain for bid, gt in gt_map.items()
               if getattr(gt, 'root_cause_chain', None)}
    if rc_bugs:
        resolved_before = 0
        for rc_id in rc_bugs:
            if rc_id in state.submitted_bugs:
                rc_gt = gt_map.get(rc_id)
                if rc_gt and state.classifications.get(rc_id) == rc_gt.severity:
                    # Check if root cause was submitted before ALL downstream bugs
                    downstream_ids = [
                        bid for bid, gt in gt_map.items()
                        if getattr(gt, 'root_cause_chain', None) == rc_id
                    ]
                    # Full credit if root cause submitted before any downstream
                    downstream_submitted = [d for d in downstream_ids if d in state.submitted_bugs]
                    if len(downstream_submitted) < len(downstream_ids):
                        resolved_before += 1  # Root cause resolved first — full credit
                    else:
                        resolved_before += 0.5  # Root cause resolved but after downstream — partial
        rc_score = resolved_before / len(rc_bugs)
    else:
        rc_score = 1.0
    components["root_cause_resolution"] = _clamp(rc_score) * 0.08

    # Efficiency (5%)
    total_bugs = len(scenario.bug_reports)
    if state.submitted_bugs and state.step_number > 0:
        eff = _efficiency_factor(state.step_number, scenario.max_steps, total_bugs)
    else:
        eff = 0.0
    components["efficiency"] = eff * 0.05

    # Completion (5%) — fraction of real bugs submitted + spam flagged
    accounted = len(state.submitted_bugs) + len(getattr(state, 'flagged_spam', []))
    completion = accounted / len(scenario.bug_reports) if scenario.bug_reports else 0.0
    components["completion"] = _clamp(completion) * 0.05

    total = _clamp(sum(components.values()))
    return {"score": round(total, 4), "components": {k: round(v, 4) for k, v in components.items()}}


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

GRADERS = {
    "single-triage": grade_single_triage,
    "batch-triage":  grade_batch_triage,
    "sla-crisis":    grade_sla_crisis,
    "adversarial-triage": grade_adversarial_triage,
}


def grade_episode(task_id: str, state: "BugTriageState", scenario: "TaskScenario") -> Dict:
    """
    Entry point for episode grading. Returns dict with 'score' and 'components'.
    Score is strictly in (0.0, 1.0) — open interval, as required by the OpenEnv
    validator (scores of exactly 0.0 or 1.0 are rejected).
    """
    grader = GRADERS.get(task_id)
    if grader is None:
        raise ValueError(f"No grader registered for task '{task_id}'")
    result = grader(state, scenario)
    # Clamp to open interval (0, 1) — validator rejects exactly 0.0 and 1.0
    raw = result["score"]
    result["score"] = round(max(0.01, min(0.99, raw)), 4)
    return result
