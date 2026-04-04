"""BugTriage OpenEnv — Environment logic + OpenEnv spec compliance tests."""
import pytest
from app.env import BugTriageEnv
from app.models import ActionType, Severity, Team, TriageAction
from app.scenarios import SCENARIOS


# Fixtures: env_easy, env_medium, env_hard — defined in conftest.py


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
        env.step(TriageAction(action_type=ActionType.CLASSIFY, bug_id="BUG-001", severity=Severity.LOW))
        env.reset()
        st = env.state()
        assert len(st.classifications) == 0
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
        assert env_easy.state().step_number == 1

    def test_step_before_reset_raises(self):
        env = BugTriageEnv("single-triage")
        with pytest.raises(RuntimeError, match="reset"):
            env.step(TriageAction(action_type=ActionType.CLASSIFY, bug_id="PAY-001", severity=Severity.CRITICAL))

    def test_done_after_max_steps(self, env_easy):
        for _ in range(5):
            result = env_easy.step(
                TriageAction(action_type=ActionType.CLASSIFY, bug_id="PAY-001", severity=Severity.LOW)
            )
        assert result.done is True

    def test_done_when_all_submitted(self, env_easy):
        env_easy.step(TriageAction(action_type=ActionType.CLASSIFY, bug_id="PAY-001", severity=Severity.CRITICAL))
        env_easy.step(TriageAction(action_type=ActionType.ASSIGN, bug_id="PAY-001", assigned_team=Team.BACKEND))
        result = env_easy.step(TriageAction(action_type=ActionType.SUBMIT, bug_id="PAY-001"))
        assert result.done is True

    def test_invalid_bug_id_penalised(self, env_easy):
        result = env_easy.step(
            TriageAction(action_type=ActionType.CLASSIFY, bug_id="FAKE-999", severity=Severity.CRITICAL)
        )
        assert result.reward < 0

    def test_step_after_done_raises(self, env_easy):
        env_easy.step(TriageAction(action_type=ActionType.CLASSIFY, bug_id="PAY-001", severity=Severity.CRITICAL))
        env_easy.step(TriageAction(action_type=ActionType.ASSIGN, bug_id="PAY-001", assigned_team=Team.BACKEND))
        env_easy.step(TriageAction(action_type=ActionType.SUBMIT, bug_id="PAY-001"))
        with pytest.raises(RuntimeError, match="done"):
            env_easy.step(TriageAction(action_type=ActionType.CLASSIFY, bug_id="PAY-001", severity=Severity.CRITICAL))


class TestRewards:
    def test_correct_classify_positive(self, env_easy):
        result = env_easy.step(
            TriageAction(action_type=ActionType.CLASSIFY, bug_id="PAY-001", severity=Severity.CRITICAL)
        )
        assert result.reward > 0

    def test_wrong_classify_negative(self, env_easy):
        result = env_easy.step(
            TriageAction(action_type=ActionType.CLASSIFY, bug_id="PAY-001", severity=Severity.LOW)
        )
        assert result.reward < 0

    def test_adjacent_classify_partial(self, env_easy):
        result = env_easy.step(
            TriageAction(action_type=ActionType.CLASSIFY, bug_id="PAY-001", severity=Severity.HIGH)
        )
        assert 0 < result.reward < 0.15

    def test_correct_assign_positive(self, env_easy):
        env_easy.step(TriageAction(action_type=ActionType.CLASSIFY, bug_id="PAY-001", severity=Severity.CRITICAL))
        result = env_easy.step(
            TriageAction(action_type=ActionType.ASSIGN, bug_id="PAY-001", assigned_team=Team.BACKEND)
        )
        assert result.reward > 0

    def test_wrong_assign_negative(self, env_easy):
        env_easy.step(TriageAction(action_type=ActionType.CLASSIFY, bug_id="PAY-001", severity=Severity.CRITICAL))
        result = env_easy.step(
            TriageAction(action_type=ActionType.ASSIGN, bug_id="PAY-001", assigned_team=Team.QA)
        )
        assert result.reward < 0

    def test_correct_duplicate_positive(self, env_medium):
        result = env_medium.step(
            TriageAction(action_type=ActionType.MARK_DUPLICATE, bug_id="BUG-003", duplicate_of="BUG-006")
        )
        assert result.reward > 0.10

    def test_false_duplicate_negative(self, env_medium):
        result = env_medium.step(
            TriageAction(action_type=ActionType.MARK_DUPLICATE, bug_id="BUG-001", duplicate_of="BUG-002")
        )
        assert result.reward < 0

    def test_warranted_escalation_positive(self, env_medium):
        result = env_medium.step(
            TriageAction(action_type=ActionType.ESCALATE, bug_id="BUG-007", escalation_reason="Security")
        )
        assert result.reward > 0

    def test_info_request_for_incomplete_positive(self, env_medium):
        result = env_medium.step(
            TriageAction(action_type=ActionType.REQUEST_INFO, bug_id="BUG-005",
                         info_requested=["steps_to_reproduce", "device_model"])
        )
        assert result.reward > 0

    def test_cumulative_reward_tracked(self, env_easy):
        env_easy.step(TriageAction(action_type=ActionType.CLASSIFY, bug_id="PAY-001", severity=Severity.CRITICAL))
        assert env_easy.state().total_reward > 0


class TestState:
    def test_classifications_recorded(self, env_easy):
        env_easy.step(TriageAction(action_type=ActionType.CLASSIFY, bug_id="PAY-001", severity=Severity.CRITICAL))
        assert env_easy.state().classifications.get("PAY-001") == "critical"

    def test_assignments_recorded(self, env_easy):
        env_easy.step(TriageAction(action_type=ActionType.ASSIGN, bug_id="PAY-001", assigned_team=Team.BACKEND))
        assert env_easy.state().assignments.get("PAY-001") == "backend"

    def test_duplicates_recorded(self, env_medium):
        env_medium.step(TriageAction(action_type=ActionType.MARK_DUPLICATE, bug_id="BUG-003", duplicate_of="BUG-006"))
        assert env_medium.state().duplicates.get("BUG-003") == "BUG-006"

    def test_escalations_recorded(self, env_medium):
        env_medium.step(TriageAction(action_type=ActionType.ESCALATE, bug_id="BUG-007", escalation_reason="Security"))
        assert "BUG-007" in env_medium.state().escalations

    def test_submitted_recorded(self, env_easy):
        env_easy.step(TriageAction(action_type=ActionType.CLASSIFY, bug_id="PAY-001", severity=Severity.CRITICAL))
        env_easy.step(TriageAction(action_type=ActionType.ASSIGN, bug_id="PAY-001", assigned_team=Team.BACKEND))
        env_easy.step(TriageAction(action_type=ActionType.SUBMIT, bug_id="PAY-001"))
        assert "PAY-001" in env_easy.state().submitted_bugs

    def test_observation_unprocessed_updated(self, env_easy):
        env_easy.step(TriageAction(action_type=ActionType.CLASSIFY, bug_id="PAY-001", severity=Severity.CRITICAL))
        env_easy.step(TriageAction(action_type=ActionType.ASSIGN, bug_id="PAY-001", assigned_team=Team.BACKEND))
        result = env_easy.step(TriageAction(action_type=ActionType.SUBMIT, bug_id="PAY-001"))
        assert "PAY-001" not in result.observation.unprocessed_bug_ids
        assert "PAY-001" in result.observation.submitted_bug_ids


class TestGraders:
    def test_perfect_easy_high_score(self, env_easy):
        env_easy.step(TriageAction(action_type=ActionType.CLASSIFY, bug_id="PAY-001", severity=Severity.CRITICAL))
        env_easy.step(TriageAction(action_type=ActionType.ASSIGN, bug_id="PAY-001", assigned_team=Team.BACKEND))
        env_easy.step(TriageAction(action_type=ActionType.ESCALATE, bug_id="PAY-001", escalation_reason="P0"))
        env_easy.step(TriageAction(action_type=ActionType.SUBMIT, bug_id="PAY-001"))
        assert env_easy.grade()["score"] >= 0.85

    def test_empty_triage_zero(self, env_easy):
        assert env_easy.grade()["score"] == 0.0

    def test_wrong_triage_low(self, env_easy):
        env_easy.step(TriageAction(action_type=ActionType.CLASSIFY, bug_id="PAY-001", severity=Severity.LOW))
        env_easy.step(TriageAction(action_type=ActionType.ASSIGN, bug_id="PAY-001", assigned_team=Team.QA))
        env_easy.step(TriageAction(action_type=ActionType.SUBMIT, bug_id="PAY-001"))
        assert env_easy.grade()["score"] < 0.25

    def test_grade_deterministic(self, env_easy):
        env_easy.step(TriageAction(action_type=ActionType.CLASSIFY, bug_id="PAY-001", severity=Severity.CRITICAL))
        env_easy.step(TriageAction(action_type=ActionType.ASSIGN, bug_id="PAY-001", assigned_team=Team.BACKEND))
        assert env_easy.grade()["score"] == env_easy.grade()["score"]

    def test_all_graders_callable(self):
        for task_id in SCENARIOS:
            env = BugTriageEnv(task_id)
            env.reset()
            result = env.grade()
            assert 0.0 <= result["score"] <= 1.0
            assert "components" in result


class TestScenarios:
    def test_all_scenarios_have_ground_truth(self):
        for task_id, scenario in SCENARIOS.items():
            for bug in scenario.bug_reports:
                assert bug.id in scenario.ground_truth

    def test_duplicate_pointers_consistent(self):
        for task_id, scenario in SCENARIOS.items():
            bug_ids = {b.id for b in scenario.bug_reports}
            for bug_id, gt in scenario.ground_truth.items():
                if gt.is_duplicate_of:
                    assert gt.is_duplicate_of in bug_ids

    def test_task_sizes(self):
        assert len(SCENARIOS["single-triage"].bug_reports) == 1
        assert len(SCENARIOS["batch-triage"].bug_reports) == 8
        assert len(SCENARIOS["sla-crisis"].bug_reports) == 15

    def test_difficulties(self):
        assert SCENARIOS["single-triage"].difficulty == "easy"
        assert SCENARIOS["batch-triage"].difficulty == "medium"
        assert SCENARIOS["sla-crisis"].difficulty == "hard"

    def test_unique_bug_ids(self):
        for task_id, scenario in SCENARIOS.items():
            ids = [b.id for b in scenario.bug_reports]
            assert len(ids) == len(set(ids))


class TestOpenEnvCompliance:
    def test_reset_contract(self):
        env = BugTriageEnv("single-triage")
        result = env.reset()
        assert hasattr(result, "observation")
        assert hasattr(result, "done")
        assert isinstance(result.done, bool)

    def test_step_contract(self):
        env = BugTriageEnv("single-triage")
        env.reset()
        result = env.step(
            TriageAction(action_type=ActionType.CLASSIFY, bug_id="PAY-001", severity=Severity.CRITICAL)
        )
        assert isinstance(result.reward, float)
        assert isinstance(result.done, bool)
        assert isinstance(result.info, dict)

    def test_state_contract(self):
        env = BugTriageEnv("single-triage")
        env.reset()
        st = env.state()
        assert hasattr(st, "task_name")
        assert hasattr(st, "step_number")
        assert hasattr(st, "done")

    def test_reward_within_bounds(self, env_easy):
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
        assert 0.0 <= env_easy.grade()["score"] <= 1.0
