"""BugTriage OpenEnv — Grader correctness tests + generator tests."""
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


# ═══════════════════════════════════════════════════════════════════════════
# Dynamic Scenario Generator Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestGeneratorBasic:
    """Test that the generator produces valid scenarios."""

    def test_generate_easy(self):
        from app.generator import generate_scenario
        s = generate_scenario("single-triage", seed=42)
        assert len(s.bug_reports) == 1
        assert len(s.ground_truth) == 1
        assert s.difficulty == "easy"
        assert s.max_steps == 5

    def test_generate_medium(self):
        from app.generator import generate_scenario
        s = generate_scenario("batch-triage", seed=42)
        assert len(s.bug_reports) == 8
        assert len(s.ground_truth) == 8
        assert s.difficulty == "medium"

    def test_generate_hard(self):
        from app.generator import generate_scenario
        s = generate_scenario("sla-crisis", seed=42)
        assert len(s.bug_reports) == 15
        assert len(s.ground_truth) == 15
        assert s.difficulty == "hard"

    def test_deterministic(self):
        from app.generator import generate_scenario
        s1 = generate_scenario("batch-triage", seed=123)
        s2 = generate_scenario("batch-triage", seed=123)
        assert [b.id for b in s1.bug_reports] == [b.id for b in s2.bug_reports]
        assert [b.title for b in s1.bug_reports] == [b.title for b in s2.bug_reports]

    def test_different_seeds_different_bugs(self):
        from app.generator import generate_scenario
        s1 = generate_scenario("batch-triage", seed=1)
        s2 = generate_scenario("batch-triage", seed=999)
        titles1 = {b.title for b in s1.bug_reports}
        titles2 = {b.title for b in s2.bug_reports}
        assert titles1 != titles2  # different seeds → different bugs

    def test_unique_bug_ids(self):
        from app.generator import generate_scenario
        for task in ["single-triage", "batch-triage", "sla-crisis"]:
            s = generate_scenario(task, seed=77)
            ids = [b.id for b in s.bug_reports]
            assert len(ids) == len(set(ids)), f"Duplicate IDs in {task}"

    def test_ground_truth_matches_bugs(self):
        from app.generator import generate_scenario
        for task in ["single-triage", "batch-triage", "sla-crisis"]:
            s = generate_scenario(task, seed=55)
            bug_ids = {b.id for b in s.bug_reports}
            gt_ids = set(s.ground_truth.keys())
            assert bug_ids == gt_ids, f"Ground truth IDs don't match bug IDs in {task}"


class TestGeneratorStructure:
    """Test that generated scenarios have correct structural properties."""

    def test_easy_has_critical_bug(self):
        from app.generator import generate_scenario
        s = generate_scenario("single-triage", seed=42)
        gt = list(s.ground_truth.values())[0]
        assert gt.severity == "critical"
        assert gt.should_escalate is True

    def test_medium_has_duplicate_pair(self):
        from app.generator import generate_scenario
        s = generate_scenario("batch-triage", seed=42)
        dups = {bid: gt.is_duplicate_of for bid, gt in s.ground_truth.items() if gt.is_duplicate_of}
        assert len(dups) >= 1, "Medium task must have at least 1 duplicate pair"

    def test_medium_has_info_incomplete(self):
        from app.generator import generate_scenario
        s = generate_scenario("batch-triage", seed=42)
        needs_info = [bid for bid, gt in s.ground_truth.items() if gt.needs_info]
        assert len(needs_info) >= 1, "Medium task must have at least 1 info-incomplete bug"

    def test_medium_has_escalation(self):
        from app.generator import generate_scenario
        s = generate_scenario("batch-triage", seed=42)
        should_esc = [bid for bid, gt in s.ground_truth.items() if gt.should_escalate]
        assert len(should_esc) >= 1, "Medium task must have at least 1 escalation-worthy bug"

    def test_hard_has_three_dup_pairs(self):
        from app.generator import generate_scenario
        s = generate_scenario("sla-crisis", seed=42)
        dups = {bid: gt.is_duplicate_of for bid, gt in s.ground_truth.items() if gt.is_duplicate_of}
        assert len(dups) >= 3, f"Hard task must have 3+ duplicate pairs, got {len(dups)}"

    def test_hard_has_sla_critical(self):
        from app.generator import generate_scenario
        s = generate_scenario("sla-crisis", seed=42)
        sla = [bid for bid, gt in s.ground_truth.items() if gt.sla_critical]
        assert len(sla) >= 3, f"Hard task must have 3+ SLA-critical bugs, got {len(sla)}"


class TestGeneratorGrading:
    """Test that generated scenarios can be graded correctly."""

    def test_generated_easy_gradeable(self):
        env = BugTriageEnv("single-triage")
        result = env.reset(seed=42)
        bid = result.observation.bug_reports[0].id
        env.step(TriageAction(action_type=ActionType.CLASSIFY, bug_id=bid, severity=Severity.CRITICAL))
        env.step(TriageAction(action_type=ActionType.ASSIGN, bug_id=bid, assigned_team=Team.BACKEND))
        env.step(TriageAction(action_type=ActionType.SUBMIT, bug_id=bid))
        grade = env.grade()
        assert 0.0 <= grade["score"] <= 1.0
        assert "severity" in grade["components"]

    def test_generated_medium_gradeable(self):
        env = BugTriageEnv("batch-triage")
        result = env.reset(seed=42)
        # Just classify and submit a few bugs to verify grading works
        for bug in result.observation.bug_reports[:3]:
            env.step(TriageAction(action_type=ActionType.CLASSIFY, bug_id=bug.id, severity=Severity.MEDIUM))
            env.step(TriageAction(action_type=ActionType.ASSIGN, bug_id=bug.id, assigned_team=Team.BACKEND))
            env.step(TriageAction(action_type=ActionType.SUBMIT, bug_id=bug.id))
        grade = env.grade()
        assert 0.0 <= grade["score"] <= 1.0
        assert "duplicate_detection" in grade["components"]

    def test_generated_hard_gradeable(self):
        env = BugTriageEnv("sla-crisis")
        env.reset(seed=42)
        grade = env.grade()
        assert grade["score"] == 0.0  # no actions taken yet → zero score
        assert "sla_escalations" in grade["components"]

    def test_generated_perfect_easy_high_score(self):
        """Generated easy task with correct actions should score high."""
        from app.generator import generate_scenario
        s = generate_scenario("single-triage", seed=42)
        env = BugTriageEnv("single-triage")
        env.reset(seed=42)
        bid = s.bug_reports[0].id
        gt = s.ground_truth[bid]
        env.step(TriageAction(action_type=ActionType.CLASSIFY, bug_id=bid, severity=Severity(gt.severity)))
        env.step(TriageAction(action_type=ActionType.ASSIGN, bug_id=bid, assigned_team=Team(gt.team)))
        if gt.should_escalate:
            env.step(TriageAction(action_type=ActionType.ESCALATE, bug_id=bid, escalation_reason="SLA"))
        env.step(TriageAction(action_type=ActionType.SUBMIT, bug_id=bid))
        assert env.grade()["score"] >= 0.80

    def test_static_scenarios_unchanged(self):
        """Verify static scenarios still produce same results as before."""
        env = BugTriageEnv("single-triage")
        env.reset()  # no seed → static scenario
        env.step(TriageAction(action_type=ActionType.CLASSIFY, bug_id="PAY-001", severity=Severity.CRITICAL))
        env.step(TriageAction(action_type=ActionType.ASSIGN, bug_id="PAY-001", assigned_team=Team.BACKEND))
        env.step(TriageAction(action_type=ActionType.ESCALATE, bug_id="PAY-001", escalation_reason="P0"))
        env.step(TriageAction(action_type=ActionType.SUBMIT, bug_id="PAY-001"))
        assert env.grade()["score"] >= 0.80
        assert env.grade()["components"]["severity"] == pytest.approx(0.40)
        assert env.grade()["components"]["team"] == pytest.approx(0.30)
