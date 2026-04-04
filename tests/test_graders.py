"""BugTriage OpenEnv — Grader correctness tests."""
import pytest
from app.env import BugTriageEnv
from app.graders import compute_step_reward, _severity_adjacent, _efficiency_factor
from app.models import ActionType, Severity, Team, TriageAction
from app.scenarios import SCENARIOS


class TestSeverityAdjacent:
    def test_exact(self):
        assert _severity_adjacent("critical", "critical") == 1.0

    def test_adjacent(self):
        assert _severity_adjacent("high", "critical") == 0.4

    def test_two_off(self):
        assert _severity_adjacent("low", "high") == 0.1

    def test_three_off(self):
        assert _severity_adjacent("low", "critical") == 0.0

    def test_invalid(self):
        assert _severity_adjacent("unknown", "critical") == 0.0


class TestEfficiency:
    def test_min_steps(self):
        assert _efficiency_factor(3, 5, 1) == 1.0

    def test_max_steps(self):
        assert abs(_efficiency_factor(5, 5, 1) - 0.85) < 0.01


class TestStepRewards:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.scenario = SCENARIOS["single-triage"]
        self.env = BugTriageEnv("single-triage")
        self.env.reset()

    def test_correct_classify(self):
        r, _ = compute_step_reward("classify", "PAY-001", self.scenario, self.env.state(), {"severity": "critical"})
        assert r == pytest.approx(0.15)

    def test_adjacent_classify(self):
        r, _ = compute_step_reward("classify", "PAY-001", self.scenario, self.env.state(), {"severity": "high"})
        assert r == pytest.approx(0.06)

    def test_wrong_classify(self):
        r, _ = compute_step_reward("classify", "PAY-001", self.scenario, self.env.state(), {"severity": "low"})
        assert r < 0

    def test_correct_assign(self):
        r, _ = compute_step_reward("assign", "PAY-001", self.scenario, self.env.state(), {"assigned_team": "backend"})
        assert r == pytest.approx(0.12)

    def test_wrong_assign(self):
        r, _ = compute_step_reward("assign", "PAY-001", self.scenario, self.env.state(), {"assigned_team": "qa"})
        assert r < 0

    def test_correct_escalate(self):
        r, _ = compute_step_reward("escalate", "PAY-001", self.scenario, self.env.state(), {})
        assert r == pytest.approx(0.12)

    def test_correct_duplicate(self):
        scenario = SCENARIOS["batch-triage"]
        env = BugTriageEnv("batch-triage")
        env.reset()
        r, _ = compute_step_reward("mark_duplicate", "BUG-003", scenario, env.state(), {"duplicate_of": "BUG-006"})
        assert r == pytest.approx(0.18)

    def test_false_duplicate(self):
        scenario = SCENARIOS["batch-triage"]
        env = BugTriageEnv("batch-triage")
        env.reset()
        r, _ = compute_step_reward("mark_duplicate", "BUG-001", scenario, env.state(), {"duplicate_of": "BUG-002"})
        assert r == pytest.approx(-0.12)

    def test_needed_info_request(self):
        scenario = SCENARIOS["batch-triage"]
        env = BugTriageEnv("batch-triage")
        env.reset()
        r, _ = compute_step_reward("request_info", "BUG-005", scenario, env.state(), {"info_requested": ["steps"]})
        assert r == pytest.approx(0.10)


class TestGraderTask1:
    def _perfect(self):
        env = BugTriageEnv("single-triage")
        env.reset()
        env.step(TriageAction(action_type=ActionType.CLASSIFY, bug_id="PAY-001", severity=Severity.CRITICAL))
        env.step(TriageAction(action_type=ActionType.ASSIGN, bug_id="PAY-001", assigned_team=Team.BACKEND))
        env.step(TriageAction(action_type=ActionType.ESCALATE, bug_id="PAY-001", escalation_reason="P0"))
        env.step(TriageAction(action_type=ActionType.SUBMIT, bug_id="PAY-001"))
        return env

    def test_perfect_above_threshold(self):
        assert self._perfect().grade()["score"] >= 0.80

    def test_severity_component(self):
        assert self._perfect().grade()["components"]["severity"] == pytest.approx(0.40)

    def test_team_component(self):
        assert self._perfect().grade()["components"]["team"] == pytest.approx(0.30)


class TestGraderTask2:
    def _perfect(self):
        env = BugTriageEnv("batch-triage")
        env.reset()
        gt = SCENARIOS["batch-triage"].ground_truth
        for bug in SCENARIOS["batch-triage"].bug_reports:
            bid = bug.id
            g = gt[bid]
            if g.is_duplicate_of:
                env.step(TriageAction(action_type=ActionType.MARK_DUPLICATE, bug_id=bid, duplicate_of=g.is_duplicate_of))
            if g.needs_info:
                env.step(TriageAction(action_type=ActionType.REQUEST_INFO, bug_id=bid, info_requested=["steps_to_reproduce"]))
            env.step(TriageAction(action_type=ActionType.CLASSIFY, bug_id=bid, severity=Severity(g.severity)))
            env.step(TriageAction(action_type=ActionType.ASSIGN, bug_id=bid, assigned_team=Team(g.team)))
            if g.should_escalate:
                env.step(TriageAction(action_type=ActionType.ESCALATE, bug_id=bid, escalation_reason="required"))
            env.step(TriageAction(action_type=ActionType.SUBMIT, bug_id=bid))
        return env

    def test_perfect_above_threshold(self):
        assert self._perfect().grade()["score"] >= 0.65

    def test_duplicate_component(self):
        assert self._perfect().grade()["components"]["duplicate_detection"] == pytest.approx(0.20)

    def test_false_duplicate_zero(self):
        env = BugTriageEnv("batch-triage")
        env.reset()
        env.step(TriageAction(action_type=ActionType.MARK_DUPLICATE, bug_id="BUG-001", duplicate_of="BUG-002"))
        assert env.grade()["components"]["duplicate_detection"] == 0.0


class TestGraderTask3:
    def test_empty_zero(self):
        env = BugTriageEnv("sla-crisis")
        env.reset()
        assert env.grade()["score"] == 0.0

    def test_escalations_contribute(self):
        env = BugTriageEnv("sla-crisis")
        env.reset()
        for bid in ["CRI-001", "CRI-004", "CRI-008", "CRI-015"]:
            env.step(TriageAction(action_type=ActionType.ESCALATE, bug_id=bid, escalation_reason="SLA"))
        assert env.grade()["components"]["sla_escalations"] > 0

    def test_duplicate_pairs(self):
        env = BugTriageEnv("sla-crisis")
        env.reset()
        env.step(TriageAction(action_type=ActionType.MARK_DUPLICATE, bug_id="CRI-009", duplicate_of="CRI-003"))
        env.step(TriageAction(action_type=ActionType.MARK_DUPLICATE, bug_id="CRI-011", duplicate_of="CRI-004"))
        env.step(TriageAction(action_type=ActionType.MARK_DUPLICATE, bug_id="CRI-012", duplicate_of="CRI-007"))
        assert env.grade()["components"]["duplicate_detection"] == pytest.approx(0.20)

    def test_all_components_present(self):
        env = BugTriageEnv("sla-crisis")
        env.reset()
        required = {"severity", "team", "duplicate_detection", "sla_escalations", "info_requests", "efficiency"}
        assert required.issubset(env.grade()["components"].keys())

    def test_score_bounds(self):
        env = BugTriageEnv("sla-crisis")
        env.reset()
        for bid in ["CRI-001", "CRI-002"]:
            env.step(TriageAction(action_type=ActionType.CLASSIFY, bug_id=bid, severity=Severity.LOW))
            env.step(TriageAction(action_type=ActionType.ASSIGN, bug_id=bid, assigned_team=Team.QA))
            env.step(TriageAction(action_type=ActionType.SUBMIT, bug_id=bid))
        assert 0.0 <= env.grade()["score"] <= 1.0
