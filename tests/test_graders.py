"""
BugTriage OpenEnv — Grader Correctness Tests
Exhaustive tests of scoring logic and edge cases for all three graders.
"""
from __future__ import annotations

import pytest

from app.env import BugTriageEnv
from app.graders import compute_step_reward, grade_episode, _severity_adjacent, _efficiency_factor
from app.models import ActionType, Severity, Team, TriageAction
from app.scenarios import SCENARIOS


# ---------------------------------------------------------------------------
# Utility function tests
# ---------------------------------------------------------------------------

class TestSeverityAdjacent:
    def test_identical_scores_1(self):
        assert _severity_adjacent("critical", "critical") == 1.0
        assert _severity_adjacent("low", "low") == 1.0

    def test_adjacent_scores_half(self):
        assert _severity_adjacent("high", "critical") == 0.5
        assert _severity_adjacent("critical", "high") == 0.5
        assert _severity_adjacent("medium", "high") == 0.5
        assert _severity_adjacent("low", "medium") == 0.5

    def test_two_steps_scores_zero(self):
        assert _severity_adjacent("low", "high") == 0.0
        assert _severity_adjacent("critical", "medium") == 0.0

    def test_three_steps_scores_zero(self):
        assert _severity_adjacent("low", "critical") == 0.0

    def test_invalid_value_scores_zero(self):
        assert _severity_adjacent("unknown", "critical") == 0.0
        assert _severity_adjacent("critical", "") == 0.0


class TestEfficiencyFactor:
    def test_minimum_steps_gives_1(self):
        # 1 bug, max_steps=5 → min=3 steps (classify+assign+submit)
        assert _efficiency_factor(3, 5, 1) == 1.0

    def test_max_steps_gives_0_85(self):
        assert abs(_efficiency_factor(5, 5, 1) - 0.85) < 0.01

    def test_midpoint_gives_between(self):
        v = _efficiency_factor(4, 5, 1)
        assert 0.85 < v < 1.0

    def test_clamped_to_min(self):
        v = _efficiency_factor(100, 5, 1)
        assert v >= 0.85


# ---------------------------------------------------------------------------
# Step reward computation
# ---------------------------------------------------------------------------

class TestStepRewardComputation:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.scenario = SCENARIOS["single-triage"]
        self.env = BugTriageEnv("single-triage")
        self.env.reset()

    def _state(self):
        return self.env.state()

    def test_correct_classify_gives_positive(self):
        reward, msg = compute_step_reward(
            "classify", "PAY-001", self.scenario, self._state(), {"severity": "critical"}
        )
        assert reward == pytest.approx(0.10)

    def test_adjacent_classify_gives_partial(self):
        reward, msg = compute_step_reward(
            "classify", "PAY-001", self.scenario, self._state(), {"severity": "high"}
        )
        assert reward == pytest.approx(0.04)

    def test_wrong_classify_gives_negative(self):
        reward, msg = compute_step_reward(
            "classify", "PAY-001", self.scenario, self._state(), {"severity": "low"}
        )
        assert reward < 0

    def test_correct_assign_gives_positive(self):
        reward, msg = compute_step_reward(
            "assign", "PAY-001", self.scenario, self._state(), {"assigned_team": "backend"}
        )
        assert reward == pytest.approx(0.08)

    def test_wrong_assign_gives_negative(self):
        reward, msg = compute_step_reward(
            "assign", "PAY-001", self.scenario, self._state(), {"assigned_team": "qa"}
        )
        assert reward < 0

    def test_unnecessary_info_gives_negative(self):
        reward, msg = compute_step_reward(
            "request_info", "PAY-001", self.scenario, self._state(), {}
        )
        assert reward < 0  # PAY-001 doesn't need info

    def test_unknown_bug_gives_negative(self):
        reward, msg = compute_step_reward(
            "classify", "FAKE-000", self.scenario, self._state(), {"severity": "critical"}
        )
        assert reward < 0


# ---------------------------------------------------------------------------
# Task 1 episode grader — single-triage
# ---------------------------------------------------------------------------

class TestGraderTask1:
    def _perfect_env(self) -> BugTriageEnv:
        env = BugTriageEnv("single-triage")
        env.reset()
        env.step(TriageAction(action_type=ActionType.CLASSIFY, bug_id="PAY-001", severity=Severity.CRITICAL))
        env.step(TriageAction(action_type=ActionType.ASSIGN,   bug_id="PAY-001", assigned_team=Team.BACKEND))
        env.step(TriageAction(action_type=ActionType.ESCALATE, bug_id="PAY-001", escalation_reason="P0"))
        env.step(TriageAction(action_type=ActionType.SUBMIT,   bug_id="PAY-001"))
        return env

    def test_perfect_triage_score_above_threshold(self):
        env = self._perfect_env()
        result = env.grade()
        assert result["score"] >= SCENARIOS["single-triage"].reward_threshold

    def test_perfect_triage_full_severity_component(self):
        env = self._perfect_env()
        result = env.grade()
        assert result["components"]["severity"] == pytest.approx(0.40)

    def test_perfect_triage_full_team_component(self):
        env = self._perfect_env()
        result = env.grade()
        assert result["components"]["team"] == pytest.approx(0.30)

    def test_wrong_severity_reduces_score(self):
        env = BugTriageEnv("single-triage")
        env.reset()
        env.step(TriageAction(action_type=ActionType.CLASSIFY, bug_id="PAY-001", severity=Severity.LOW))
        env.step(TriageAction(action_type=ActionType.ASSIGN,   bug_id="PAY-001", assigned_team=Team.BACKEND))
        env.step(TriageAction(action_type=ActionType.SUBMIT,   bug_id="PAY-001"))
        result = env.grade()
        assert result["components"]["severity"] == pytest.approx(0.0)

    def test_wrong_team_reduces_score(self):
        env = BugTriageEnv("single-triage")
        env.reset()
        env.step(TriageAction(action_type=ActionType.CLASSIFY, bug_id="PAY-001", severity=Severity.CRITICAL))
        env.step(TriageAction(action_type=ActionType.ASSIGN,   bug_id="PAY-001", assigned_team=Team.QA))
        env.step(TriageAction(action_type=ActionType.SUBMIT,   bug_id="PAY-001"))
        result = env.grade()
        assert result["components"]["team"] == pytest.approx(0.0)

    def test_unnecessary_info_request_penalises_component(self):
        env = BugTriageEnv("single-triage")
        env.reset()
        env.step(TriageAction(action_type=ActionType.CLASSIFY,    bug_id="PAY-001", severity=Severity.CRITICAL))
        env.step(TriageAction(action_type=ActionType.ASSIGN,      bug_id="PAY-001", assigned_team=Team.BACKEND))
        env.step(TriageAction(action_type=ActionType.REQUEST_INFO, bug_id="PAY-001", info_requested=["steps"]))
        env.step(TriageAction(action_type=ActionType.SUBMIT,      bug_id="PAY-001"))
        result = env.grade()
        assert result["components"]["no_wasted_info"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Task 2 episode grader — batch-triage
# ---------------------------------------------------------------------------

class TestGraderTask2:
    def _perfect_env(self) -> BugTriageEnv:
        env = BugTriageEnv("batch-triage")
        env.reset()
        gt = SCENARIOS["batch-triage"].ground_truth
        for bug in SCENARIOS["batch-triage"].bug_reports:
            bid = bug.id
            g = gt[bid]
            if g.is_duplicate_of:
                env.step(TriageAction(action_type=ActionType.MARK_DUPLICATE, bug_id=bid, duplicate_of=g.is_duplicate_of))
            if g.needs_info:
                env.step(TriageAction(action_type=ActionType.REQUEST_INFO, bug_id=bid, info_requested=["steps_to_reproduce", "device_model"]))
            env.step(TriageAction(action_type=ActionType.CLASSIFY, bug_id=bid, severity=Severity(g.severity)))
            env.step(TriageAction(action_type=ActionType.ASSIGN,   bug_id=bid, assigned_team=Team(g.team)))
            if g.should_escalate:
                env.step(TriageAction(action_type=ActionType.ESCALATE, bug_id=bid, escalation_reason="escalation required"))
            env.step(TriageAction(action_type=ActionType.SUBMIT, bug_id=bid))
        return env

    def test_perfect_batch_above_threshold(self):
        env = self._perfect_env()
        result = env.grade()
        assert result["score"] >= SCENARIOS["batch-triage"].reward_threshold

    def test_duplicate_detected_gives_full_component(self):
        env = self._perfect_env()
        result = env.grade()
        assert result["components"]["duplicate_detection"] == pytest.approx(0.20)

    def test_security_escalation_gives_full_component(self):
        env = self._perfect_env()
        result = env.grade()
        assert result["components"]["security_escalation"] == pytest.approx(0.10)

    def test_info_request_component_for_bug005(self):
        env = self._perfect_env()
        result = env.grade()
        assert result["components"]["info_request"] == pytest.approx(0.10)

    def test_false_duplicate_penalises_score(self):
        env = BugTriageEnv("batch-triage")
        env.reset()
        # Incorrectly mark BUG-001 as duplicate of BUG-002 (wrong)
        env.step(TriageAction(action_type=ActionType.MARK_DUPLICATE, bug_id="BUG-001", duplicate_of="BUG-002"))
        result = env.grade()
        # Penalty is absorbed into the duplicate_detection component (clamped to 0.0 floor).
        # No bugs submitted yet so efficiency=0 too — overall score must be 0.
        assert result["components"]["duplicate_detection"] == 0.0
        assert result["score"] == 0.0


# ---------------------------------------------------------------------------
# Task 3 episode grader — sla-crisis
# ---------------------------------------------------------------------------

class TestGraderTask3:
    def test_empty_sla_crisis_score_zero(self):
        env = BugTriageEnv("sla-crisis")
        env.reset()
        result = env.grade()
        assert result["score"] == 0.0

    def test_correct_sla_escalations_give_component(self):
        env = BugTriageEnv("sla-crisis")
        env.reset()
        # Escalate all 5 SLA-critical bugs
        for bid in ["CRI-001", "CRI-004", "CRI-008", "CRI-011", "CRI-015"]:
            env.step(TriageAction(action_type=ActionType.ESCALATE, bug_id=bid, escalation_reason="SLA breach"))
        result = env.grade()
        assert result["components"]["sla_escalations"] > 0

    def test_correct_duplicate_pair_a(self):
        env = BugTriageEnv("sla-crisis")
        env.reset()
        env.step(TriageAction(action_type=ActionType.MARK_DUPLICATE, bug_id="CRI-009", duplicate_of="CRI-003"))
        result = env.grade()
        assert result["components"]["duplicate_detection"] > 0

    def test_correct_duplicate_pair_b(self):
        env = BugTriageEnv("sla-crisis")
        env.reset()
        env.step(TriageAction(action_type=ActionType.MARK_DUPLICATE, bug_id="CRI-012", duplicate_of="CRI-007"))
        result = env.grade()
        assert result["components"]["duplicate_detection"] > 0

    def test_correct_duplicate_pair_c(self):
        env = BugTriageEnv("sla-crisis")
        env.reset()
        env.step(TriageAction(action_type=ActionType.MARK_DUPLICATE, bug_id="CRI-011", duplicate_of="CRI-004"))
        result = env.grade()
        assert result["components"]["duplicate_detection"] > 0

    def test_wrong_duplicate_pair_penalised(self):
        env = BugTriageEnv("sla-crisis")
        env.reset()
        # Swap the direction — mark original as dup of the later one (wrong)
        env.step(TriageAction(action_type=ActionType.MARK_DUPLICATE, bug_id="CRI-003", duplicate_of="CRI-009"))
        result = env.grade()
        # CRI-003 is not a duplicate → false positive penalty
        assert result["components"]["duplicate_detection"] <= 0

    def test_info_request_for_incomplete_bugs(self):
        env = BugTriageEnv("sla-crisis")
        env.reset()
        env.step(TriageAction(action_type=ActionType.REQUEST_INFO, bug_id="CRI-005", info_requested=["steps_to_reproduce"]))
        env.step(TriageAction(action_type=ActionType.REQUEST_INFO, bug_id="CRI-014", info_requested=["device_model", "os_version"]))
        result = env.grade()
        assert result["components"]["info_requests"] > 0

    def test_all_component_keys_present(self):
        env = BugTriageEnv("sla-crisis")
        env.reset()
        result = env.grade()
        required = {"severity", "team", "duplicate_detection", "sla_escalations", "info_requests", "efficiency"}
        assert required.issubset(result["components"].keys())

    def test_score_bounds_respected_under_adversarial_actions(self):
        """Even if the agent makes all wrong decisions, score stays in [0, 1]."""
        env = BugTriageEnv("sla-crisis")
        env.reset()
        for bid in ["CRI-001", "CRI-002", "CRI-003"]:
            env.step(TriageAction(action_type=ActionType.CLASSIFY, bug_id=bid, severity=Severity.LOW))
            env.step(TriageAction(action_type=ActionType.ASSIGN,   bug_id=bid, assigned_team=Team.QA))
            env.step(TriageAction(action_type=ActionType.SUBMIT,   bug_id=bid))
            # Mark non-duplicates as duplicates
            if bid == "CRI-003":
                env.step(TriageAction(action_type=ActionType.MARK_DUPLICATE, bug_id="CRI-001", duplicate_of="CRI-002"))
        result = env.grade()
        assert 0.0 <= result["score"] <= 1.0
