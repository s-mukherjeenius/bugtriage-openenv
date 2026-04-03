"""
BugTriage OpenEnv — Test Suite
Tests environment logic, graders, and OpenEnv spec compliance.
"""
from __future__ import annotations

import pytest
from app.env import BugTriageEnv
from app.graders import grade_episode
from app.models import ActionType, Severity, Team, TriageAction
from app.scenarios import SCENARIOS


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def env_easy():
    env = BugTriageEnv("single-triage")
    env.reset()
    return env


@pytest.fixture
def env_medium():
    env = BugTriageEnv("batch-triage")
    env.reset()
    return env


@pytest.fixture
def env_hard():
    env = BugTriageEnv("sla-crisis")
    env.reset()
    return env


# ===========================================================================
# 1. Environment initialisation
# ===========================================================================

class TestReset:
    def test_reset_returns_reset_result(self):
        env = BugTriageEnv("single-triage")
        result = env.reset()
        assert result.done is False
        assert result.observation is not None

    def test_reset_observation_has_bugs(self):
        env = BugTriageEnv("single-triage")
        result = env.reset()
        obs = result.observation
        assert len(obs.bug_reports) == 1
        assert obs.bug_reports[0].id == "PAY-001"

    def test_reset_clears_state(self):
        env = BugTriageEnv("batch-triage")
        env.reset()
        # Take an action
        env.step(TriageAction(action_type=ActionType.CLASSIFY, bug_id="BUG-001", severity=Severity.LOW))
        # Reset again
        env.reset()
        st = env.state()
        assert len(st.classifications) == 0
        assert len(st.submitted_bugs) == 0
        assert st.step_number == 0

    def test_all_tasks_reset(self):
        for task_id in SCENARIOS:
            env = BugTriageEnv(task_id)
            result = env.reset()
            assert result.done is False
            assert len(result.observation.bug_reports) > 0

    def test_invalid_task_raises(self):
        with pytest.raises(ValueError, match="Unknown task"):
            BugTriageEnv("nonexistent-task")


# ===========================================================================
# 2. Step mechanics
# ===========================================================================

class TestStep:
    def test_step_returns_step_result(self, env_easy):
        result = env_easy.step(
            TriageAction(action_type=ActionType.CLASSIFY, bug_id="PAY-001", severity=Severity.CRITICAL)
        )
        assert result.reward is not None
        assert result.done is not None
        assert result.observation is not None

    def test_step_increments_counter(self, env_easy):
        env_easy.step(TriageAction(action_type=ActionType.CLASSIFY, bug_id="PAY-001", severity=Severity.CRITICAL))
        st = env_easy.state()
        assert st.step_number == 1

    def test_step_before_reset_raises(self):
        env = BugTriageEnv("single-triage")
        with pytest.raises(RuntimeError, match="reset"):
            env.step(TriageAction(action_type=ActionType.CLASSIFY, bug_id="PAY-001", severity=Severity.CRITICAL))

    def test_done_after_max_steps(self, env_easy):
        # Easy task has max 5 steps
        for i in range(5):
            result = env_easy.step(
                TriageAction(action_type=ActionType.CLASSIFY, bug_id="PAY-001", severity=Severity.LOW)
            )
        assert result.done is True

    def test_done_when_all_submitted(self, env_easy):
        # classify → assign → submit → done
        env_easy.step(TriageAction(action_type=ActionType.CLASSIFY, bug_id="PAY-001", severity=Severity.CRITICAL))
        env_easy.step(TriageAction(action_type=ActionType.ASSIGN, bug_id="PAY-001", assigned_team=Team.BACKEND))
        result = env_easy.step(TriageAction(action_type=ActionType.SUBMIT, bug_id="PAY-001"))
        assert result.done is True

    def test_invalid_bug_id_penalised(self, env_easy):
        result = env_easy.step(
            TriageAction(action_type=ActionType.CLASSIFY, bug_id="FAKE-999", severity=Severity.CRITICAL)
        )
        assert result.reward < 0

    def test_unnecessary_info_request_penalised(self, env_easy):
        # PAY-001 has complete info — requesting info should be penalised
        result = env_easy.step(
            TriageAction(action_type=ActionType.REQUEST_INFO, bug_id="PAY-001", info_requested=["steps"])
        )
        assert result.reward < 0

    def test_step_after_done_raises(self, env_easy):
        # Submit to finish
        env_easy.step(TriageAction(action_type=ActionType.CLASSIFY, bug_id="PAY-001", severity=Severity.CRITICAL))
        env_easy.step(TriageAction(action_type=ActionType.ASSIGN, bug_id="PAY-001", assigned_team=Team.BACKEND))
        env_easy.step(TriageAction(action_type=ActionType.SUBMIT, bug_id="PAY-001"))
        with pytest.raises(RuntimeError, match="done"):
            env_easy.step(TriageAction(action_type=ActionType.CLASSIFY, bug_id="PAY-001", severity=Severity.CRITICAL))


# ===========================================================================
# 3. Reward signals
# ===========================================================================

class TestRewards:
    def test_correct_classify_positive_reward(self, env_easy):
        result = env_easy.step(
            TriageAction(action_type=ActionType.CLASSIFY, bug_id="PAY-001", severity=Severity.CRITICAL)
        )
        assert result.reward > 0

    def test_wrong_classify_negative_reward(self, env_easy):
        result = env_easy.step(
            TriageAction(action_type=ActionType.CLASSIFY, bug_id="PAY-001", severity=Severity.LOW)
        )
        assert result.reward < 0

    def test_adjacent_classify_partial_reward(self, env_easy):
        # high is adjacent to critical → partial reward
        result = env_easy.step(
            TriageAction(action_type=ActionType.CLASSIFY, bug_id="PAY-001", severity=Severity.HIGH)
        )
        assert 0 < result.reward < 0.10  # partial, not full

    def test_correct_assign_positive_reward(self, env_easy):
        env_easy.step(TriageAction(action_type=ActionType.CLASSIFY, bug_id="PAY-001", severity=Severity.CRITICAL))
        result = env_easy.step(
            TriageAction(action_type=ActionType.ASSIGN, bug_id="PAY-001", assigned_team=Team.BACKEND)
        )
        assert result.reward > 0

    def test_wrong_assign_negative_reward(self, env_easy):
        env_easy.step(TriageAction(action_type=ActionType.CLASSIFY, bug_id="PAY-001", severity=Severity.CRITICAL))
        result = env_easy.step(
            TriageAction(action_type=ActionType.ASSIGN, bug_id="PAY-001", assigned_team=Team.QA)
        )
        assert result.reward < 0

    def test_correct_duplicate_positive_reward(self, env_medium):
        result = env_medium.step(
            TriageAction(action_type=ActionType.MARK_DUPLICATE, bug_id="BUG-003", duplicate_of="BUG-006")
        )
        assert result.reward > 0.10

    def test_false_duplicate_negative_reward(self, env_medium):
        result = env_medium.step(
            TriageAction(action_type=ActionType.MARK_DUPLICATE, bug_id="BUG-001", duplicate_of="BUG-002")
        )
        assert result.reward < 0

    def test_warranted_escalation_positive_reward(self, env_medium):
        result = env_medium.step(
            TriageAction(action_type=ActionType.ESCALATE, bug_id="BUG-007", escalation_reason="Security issue")
        )
        assert result.reward > 0

    def test_info_request_for_incomplete_positive(self, env_medium):
        result = env_medium.step(
            TriageAction(
                action_type=ActionType.REQUEST_INFO,
                bug_id="BUG-005",
                info_requested=["steps_to_reproduce", "device_model"]
            )
        )
        assert result.reward > 0

    def test_cumulative_reward_tracked(self, env_easy):
        env_easy.step(TriageAction(action_type=ActionType.CLASSIFY, bug_id="PAY-001", severity=Severity.CRITICAL))
        st = env_easy.state()
        assert st.total_reward > 0


# ===========================================================================
# 4. State management
# ===========================================================================

class TestState:
    def test_state_records_classifications(self, env_easy):
        env_easy.step(TriageAction(action_type=ActionType.CLASSIFY, bug_id="PAY-001", severity=Severity.CRITICAL))
        st = env_easy.state()
        assert st.classifications.get("PAY-001") == "critical"

    def test_state_records_assignments(self, env_easy):
        env_easy.step(TriageAction(action_type=ActionType.ASSIGN, bug_id="PAY-001", assigned_team=Team.BACKEND))
        st = env_easy.state()
        assert st.assignments.get("PAY-001") == "backend"

    def test_state_records_duplicates(self, env_medium):
        env_medium.step(
            TriageAction(action_type=ActionType.MARK_DUPLICATE, bug_id="BUG-003", duplicate_of="BUG-006")
        )
        st = env_medium.state()
        assert st.duplicates.get("BUG-003") == "BUG-006"

    def test_state_records_escalations(self, env_medium):
        env_medium.step(
            TriageAction(action_type=ActionType.ESCALATE, bug_id="BUG-007", escalation_reason="Security")
        )
        st = env_medium.state()
        assert "BUG-007" in st.escalations

    def test_state_records_submitted(self, env_easy):
        env_easy.step(TriageAction(action_type=ActionType.CLASSIFY, bug_id="PAY-001", severity=Severity.CRITICAL))
        env_easy.step(TriageAction(action_type=ActionType.ASSIGN, bug_id="PAY-001", assigned_team=Team.BACKEND))
        env_easy.step(TriageAction(action_type=ActionType.SUBMIT, bug_id="PAY-001"))
        st = env_easy.state()
        assert "PAY-001" in st.submitted_bugs

    def test_observation_unprocessed_updated(self, env_easy):
        env_easy.step(TriageAction(action_type=ActionType.CLASSIFY, bug_id="PAY-001", severity=Severity.CRITICAL))
        env_easy.step(TriageAction(action_type=ActionType.ASSIGN, bug_id="PAY-001", assigned_team=Team.BACKEND))
        result = env_easy.step(TriageAction(action_type=ActionType.SUBMIT, bug_id="PAY-001"))
        obs = result.observation
        assert "PAY-001" not in obs.unprocessed_bug_ids
        assert "PAY-001" in obs.submitted_bug_ids


# ===========================================================================
# 5. Episode-level graders
# ===========================================================================

class TestGraders:
    def test_grade_returns_score_in_range(self, env_easy):
        # Perfect triage
        env_easy.step(TriageAction(action_type=ActionType.CLASSIFY, bug_id="PAY-001", severity=Severity.CRITICAL))
        env_easy.step(TriageAction(action_type=ActionType.ASSIGN, bug_id="PAY-001", assigned_team=Team.BACKEND))
        env_easy.step(TriageAction(action_type=ActionType.SUBMIT, bug_id="PAY-001"))
        result = env_easy.grade()
        assert 0.0 <= result["score"] <= 1.0

    def test_perfect_easy_triage_high_score(self, env_easy):
        env_easy.step(TriageAction(action_type=ActionType.CLASSIFY, bug_id="PAY-001", severity=Severity.CRITICAL))
        env_easy.step(TriageAction(action_type=ActionType.ASSIGN, bug_id="PAY-001", assigned_team=Team.BACKEND))
        env_easy.step(TriageAction(action_type=ActionType.ESCALATE, bug_id="PAY-001", escalation_reason="P0"))
        env_easy.step(TriageAction(action_type=ActionType.SUBMIT, bug_id="PAY-001"))
        result = env_easy.grade()
        # Perfect triage: severity(0.40) + team(0.30) + no_info(0.15) + escalation(0.10) + efficiency(0.05) = 1.0
        assert result["score"] >= 0.85

    def test_empty_triage_zero_score(self, env_easy):
        # Nothing done
        result = env_easy.grade()
        assert result["score"] == 0.0

    def test_wrong_triage_low_score(self, env_easy):
        env_easy.step(TriageAction(action_type=ActionType.CLASSIFY, bug_id="PAY-001", severity=Severity.LOW))
        env_easy.step(TriageAction(action_type=ActionType.ASSIGN, bug_id="PAY-001", assigned_team=Team.QA))
        env_easy.step(TriageAction(action_type=ActionType.SUBMIT, bug_id="PAY-001"))
        result = env_easy.grade()
        # Score should be low: only no_wasted_info(0.15) + efficiency(0.05) can accrue;
        # severity and team are both wrong so those 70% of weight score 0.
        assert result["score"] < 0.25

    def test_grade_has_components(self, env_easy):
        env_easy.step(TriageAction(action_type=ActionType.CLASSIFY, bug_id="PAY-001", severity=Severity.CRITICAL))
        result = env_easy.grade()
        assert "components" in result
        assert len(result["components"]) > 0

    def test_grade_deterministic(self, env_easy):
        env_easy.step(TriageAction(action_type=ActionType.CLASSIFY, bug_id="PAY-001", severity=Severity.CRITICAL))
        env_easy.step(TriageAction(action_type=ActionType.ASSIGN, bug_id="PAY-001", assigned_team=Team.BACKEND))
        r1 = env_easy.grade()
        r2 = env_easy.grade()
        assert r1["score"] == r2["score"]

    def test_medium_perfect_duplicate_detection(self, env_medium):
        # Only mark the duplicate — partial grade but duplicate component should be full
        env_medium.step(
            TriageAction(action_type=ActionType.MARK_DUPLICATE, bug_id="BUG-003", duplicate_of="BUG-006")
        )
        result = env_medium.grade()
        assert result["components"]["duplicate_detection"] > 0

    def test_all_graders_callable(self):
        for task_id in SCENARIOS:
            env = BugTriageEnv(task_id)
            env.reset()
            result = env.grade()
            assert "score" in result
            assert "components" in result
            assert 0.0 <= result["score"] <= 1.0


# ===========================================================================
# 6. Scenario data integrity
# ===========================================================================

class TestScenarios:
    def test_all_scenarios_have_ground_truth(self):
        for task_id, scenario in SCENARIOS.items():
            for bug in scenario.bug_reports:
                assert bug.id in scenario.ground_truth, (
                    f"Bug {bug.id} in task '{task_id}' missing ground truth"
                )

    def test_duplicate_pointers_are_consistent(self):
        """Each bug marked as duplicate must point to a bug that exists in the scenario."""
        for task_id, scenario in SCENARIOS.items():
            bug_ids = {b.id for b in scenario.bug_reports}
            for bug_id, gt in scenario.ground_truth.items():
                if gt.is_duplicate_of:
                    assert gt.is_duplicate_of in bug_ids, (
                        f"{task_id}: {bug_id} points to non-existent original {gt.is_duplicate_of}"
                    )

    def test_task1_has_one_bug(self):
        assert len(SCENARIOS["single-triage"].bug_reports) == 1

    def test_task2_has_eight_bugs(self):
        assert len(SCENARIOS["batch-triage"].bug_reports) == 8

    def test_task3_has_fifteen_bugs(self):
        assert len(SCENARIOS["sla-crisis"].bug_reports) == 15

    def test_difficulties_correct(self):
        assert SCENARIOS["single-triage"].difficulty == "easy"
        assert SCENARIOS["batch-triage"].difficulty == "medium"
        assert SCENARIOS["sla-crisis"].difficulty == "hard"

    def test_max_steps_correct(self):
        assert SCENARIOS["single-triage"].max_steps == 5
        assert SCENARIOS["batch-triage"].max_steps == 32
        assert SCENARIOS["sla-crisis"].max_steps == 50

    def test_all_bug_ids_unique_within_task(self):
        for task_id, scenario in SCENARIOS.items():
            ids = [b.id for b in scenario.bug_reports]
            assert len(ids) == len(set(ids)), f"Duplicate bug IDs in task '{task_id}'"


# ===========================================================================
# 7. OpenEnv spec compliance
# ===========================================================================

class TestOpenEnvCompliance:
    def test_reset_returns_observation_and_done(self):
        env = BugTriageEnv("single-triage")
        result = env.reset()
        # Must have observation and done (OpenEnv spec)
        assert hasattr(result, "observation")
        assert hasattr(result, "done")
        assert isinstance(result.done, bool)

    def test_step_returns_observation_reward_done_info(self):
        env = BugTriageEnv("single-triage")
        env.reset()
        result = env.step(
            TriageAction(action_type=ActionType.CLASSIFY, bug_id="PAY-001", severity=Severity.CRITICAL)
        )
        assert hasattr(result, "observation")
        assert hasattr(result, "reward")
        assert hasattr(result, "done")
        assert hasattr(result, "info")
        assert isinstance(result.reward, float)
        assert isinstance(result.done, bool)
        assert isinstance(result.info, dict)

    def test_state_returns_bug_triage_state(self):
        env = BugTriageEnv("single-triage")
        env.reset()
        st = env.state()
        assert hasattr(st, "task_name")
        assert hasattr(st, "step_number")
        assert hasattr(st, "done")

    def test_reward_within_bounds(self, env_easy):
        """Step reward must be in [-0.15, 0.20] per spec."""
        for _ in range(5):
            result = env_easy.step(
                TriageAction(action_type=ActionType.CLASSIFY, bug_id="PAY-001", severity=Severity.LOW)
            )
            assert -0.15 <= result.reward <= 0.20
            if result.done:
                break

    def test_episode_score_in_0_1(self, env_easy):
        env_easy.step(TriageAction(action_type=ActionType.CLASSIFY, bug_id="PAY-001", severity=Severity.CRITICAL))
        env_easy.step(TriageAction(action_type=ActionType.ASSIGN, bug_id="PAY-001", assigned_team=Team.BACKEND))
        env_easy.step(TriageAction(action_type=ActionType.SUBMIT, bug_id="PAY-001"))
        result = env_easy.grade()
        assert 0.0 <= result["score"] <= 1.0

    def test_step_after_done_raises(self, env_easy):
        """OpenEnv spec: stepping into a done episode must raise."""
        env_easy.step(TriageAction(action_type=ActionType.CLASSIFY, bug_id="PAY-001", severity=Severity.CRITICAL))
        env_easy.step(TriageAction(action_type=ActionType.ASSIGN, bug_id="PAY-001", assigned_team=Team.BACKEND))
        env_easy.step(TriageAction(action_type=ActionType.SUBMIT, bug_id="PAY-001"))
        with pytest.raises(RuntimeError):
            env_easy.step(TriageAction(action_type=ActionType.SUBMIT, bug_id="PAY-001"))
