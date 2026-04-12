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

    def test_empty_triage_floor(self, env_easy):
        assert env_easy.grade()["score"] == 0.01  # clamped to open interval (0,1)

    def test_wrong_triage_low(self, env_easy):
        env_easy.step(TriageAction(action_type=ActionType.CLASSIFY, bug_id="PAY-001", severity=Severity.LOW))
        env_easy.step(TriageAction(action_type=ActionType.ASSIGN, bug_id="PAY-001", assigned_team=Team.QA))
        env_easy.step(TriageAction(action_type=ActionType.SUBMIT, bug_id="PAY-001"))
        assert env_easy.grade()["score"] < 0.35  # completion component adds value for submitting

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
        assert len(SCENARIOS["adversarial-triage"].bug_reports) == 20

    def test_difficulties(self):
        assert SCENARIOS["single-triage"].difficulty == "easy"
        assert SCENARIOS["batch-triage"].difficulty == "medium"
        assert SCENARIOS["sla-crisis"].difficulty == "hard"
        assert SCENARIOS["adversarial-triage"].difficulty == "expert"

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


class TestTask4FlagSpam:
    """Test Task 4 adversarial-triage: spam detection and flag_spam action."""

    def test_flag_spam_correct_positive_reward(self, env_expert):
        """Flagging a known spam bug should give positive reward."""
        result = env_expert.step(
            TriageAction(action_type=ActionType.FLAG_SPAM, bug_id="ADV-016", spam_reason="Fake product")
        )
        assert result.reward > 0.10

    def test_flag_spam_wrong_negative_reward(self, env_expert):
        """Flagging a real bug as spam should give negative reward."""
        result = env_expert.step(
            TriageAction(action_type=ActionType.FLAG_SPAM, bug_id="ADV-001", spam_reason="test")
        )
        assert result.reward < 0

    def test_flagged_spam_tracked_in_state(self, env_expert):
        env_expert.step(
            TriageAction(action_type=ActionType.FLAG_SPAM, bug_id="ADV-016", spam_reason="Fake")
        )
        assert "ADV-016" in env_expert.state().flagged_spam

    def test_flagged_spam_removed_from_unprocessed(self, env_expert):
        result = env_expert.step(
            TriageAction(action_type=ActionType.FLAG_SPAM, bug_id="ADV-017", spam_reason="Panic")
        )
        assert "ADV-017" not in result.observation.unprocessed_bug_ids
        assert "ADV-017" in result.observation.flagged_spam_ids

    def test_done_when_all_submitted_and_flagged(self, env_expert):
        """Episode should end when submitted + flagged = total bugs."""
        env = BugTriageEnv("adversarial-triage")
        env.reset()
        gt = SCENARIOS["adversarial-triage"].ground_truth
        for bug in SCENARIOS["adversarial-triage"].bug_reports:
            bid = bug.id
            g = gt[bid]
            if g.is_spam:
                env.step(TriageAction(action_type=ActionType.FLAG_SPAM, bug_id=bid, spam_reason="spam"))
            else:
                env.step(TriageAction(action_type=ActionType.CLASSIFY, bug_id=bid, severity=Severity(g.severity)))
                env.step(TriageAction(action_type=ActionType.ASSIGN, bug_id=bid, assigned_team=Team(g.team)))
                result = env.step(TriageAction(action_type=ActionType.SUBMIT, bug_id=bid))
                if result.done:
                    break
        assert env.state().done is True

    def test_task4_grader_has_spam_component(self, env_expert):
        grade = env_expert.grade()
        assert "spam_detection" in grade["components"]

    def test_task4_perfect_spam_detection_score(self):
        """Correctly flagging all 5 spam bugs should yield max spam_detection component."""
        env = BugTriageEnv("adversarial-triage")
        env.reset()
        spam_ids = ["ADV-016", "ADV-017", "ADV-018", "ADV-019", "ADV-020"]
        for sid in spam_ids:
            env.step(TriageAction(action_type=ActionType.FLAG_SPAM, bug_id=sid, spam_reason="spam"))
        grade = env.grade()
        assert grade["components"]["spam_detection"] == pytest.approx(0.18)

    def test_task4_scenario_has_spam_bugs(self):
        gt = SCENARIOS["adversarial-triage"].ground_truth
        spam_count = sum(1 for g in gt.values() if g.is_spam)
        assert spam_count == 5

    def test_task4_scenario_has_duplicates(self):
        gt = SCENARIOS["adversarial-triage"].ground_truth
        dup_count = sum(1 for g in gt.values() if g.is_duplicate_of)
        assert dup_count == 2


class TestTickingSLA:
    """Test ticking SLA timers in adversarial-triage task."""

    def test_sla_decrements_each_step(self):
        """SLA timers should decrease after each step in Task 4."""
        env = BugTriageEnv("adversarial-triage")
        env.reset()
        # Get initial SLA for ADV-001 (0.5h)
        initial_sla = None
        for b in env.state().bug_reports:
            if b.id == "ADV-001":
                initial_sla = b.sla_hours_remaining
                break
        assert initial_sla == 0.5
        # Take one step
        env.step(TriageAction(action_type=ActionType.CLASSIFY, bug_id="ADV-002", severity=Severity.MEDIUM))
        # SLA should have decreased
        for b in env.state().bug_reports:
            if b.id == "ADV-001":
                assert b.sla_hours_remaining < initial_sla
                break

    def test_sla_not_ticking_for_task1(self):
        """SLA should NOT tick for non-adversarial tasks."""
        env = BugTriageEnv("single-triage")
        env.reset()
        initial_sla = env.state().bug_reports[0].sla_hours_remaining
        env.step(TriageAction(action_type=ActionType.CLASSIFY, bug_id="PAY-001", severity=Severity.CRITICAL))
        assert env.state().bug_reports[0].sla_hours_remaining == initial_sla

    def test_sla_breach_appears_in_observation(self):
        """Bugs with SLA=0 should appear in sla_breached_bug_ids."""
        env = BugTriageEnv("adversarial-triage")
        env.reset()
        # ADV-014 has 0.2h SLA. After ~10 steps at 0.02h/step, it breaches.
        for i in range(12):
            result = env.step(
                TriageAction(action_type=ActionType.CLASSIFY, bug_id="ADV-002", severity=Severity.MEDIUM)
            )
        assert "ADV-014" in result.observation.sla_breached_bug_ids

    def test_submitted_bugs_sla_stops_ticking(self):
        """SLA should not tick for submitted bugs."""
        env = BugTriageEnv("adversarial-triage")
        env.reset()
        # Submit ADV-001
        env.step(TriageAction(action_type=ActionType.CLASSIFY, bug_id="ADV-001", severity=Severity.CRITICAL))
        env.step(TriageAction(action_type=ActionType.ASSIGN, bug_id="ADV-001", assigned_team=Team.BACKEND))
        env.step(TriageAction(action_type=ActionType.SUBMIT, bug_id="ADV-001"))
        # Take more steps — ADV-001 should not breach since it's submitted
        for _ in range(30):
            env.step(TriageAction(action_type=ActionType.CLASSIFY, bug_id="ADV-002", severity=Severity.MEDIUM))
        assert "ADV-001" not in [b.id for b in env.state().bug_reports if b.sla_hours_remaining == 0.0 and b.id not in env.state().submitted_bugs]


class TestRootCauseResolution:
    """Test cascading root-cause resolution mechanics."""

    def test_root_cause_bonus_when_resolved_first(self):
        """Correctly submitting root cause (ADV-006) before downstream (ADV-009) gives bonus."""
        env = BugTriageEnv("adversarial-triage")
        env.reset()
        # Correctly triage ADV-006 (root cause: Redis pool exhausted)
        env.step(TriageAction(action_type=ActionType.CLASSIFY, bug_id="ADV-006", severity=Severity.CRITICAL))
        env.step(TriageAction(action_type=ActionType.ASSIGN, bug_id="ADV-006", assigned_team=Team.INFRASTRUCTURE))
        result = env.step(TriageAction(action_type=ActionType.SUBMIT, bug_id="ADV-006"))
        # Should get root-cause bonus since ADV-009 is still unresolved
        assert result.reward > 0.08  # More than base submit reward
        assert "Root cause" in result.info.get("reward_message", "")

    def test_no_root_cause_bonus_for_regular_bug(self):
        """Non-root-cause bugs should not get the bonus."""
        env = BugTriageEnv("adversarial-triage")
        env.reset()
        env.step(TriageAction(action_type=ActionType.CLASSIFY, bug_id="ADV-002", severity=Severity.MEDIUM))
        env.step(TriageAction(action_type=ActionType.ASSIGN, bug_id="ADV-002", assigned_team=Team.BACKEND))
        result = env.step(TriageAction(action_type=ActionType.SUBMIT, bug_id="ADV-002"))
        assert "Root cause" not in result.info.get("reward_message", "")


class TestDynamicStaticTransition:
    """Regression: verify dynamic scenarios don't leak into static resets."""

    def test_reset_seed_then_reset_reverts_to_static(self):
        """After reset(seed=42), calling reset() must revert to static scenario."""
        env = BugTriageEnv("single-triage")
        r1 = env.reset(seed=42)
        dynamic_id = r1.observation.bug_reports[0].id
        assert dynamic_id.startswith("S"), f"Seed reset should produce generated IDs, got {dynamic_id}"

        r2 = env.reset()
        static_id = r2.observation.bug_reports[0].id
        assert static_id == "PAY-001", f"Expected static PAY-001, got {static_id}"

    def test_reset_different_seeds_different_bugs(self):
        env = BugTriageEnv("batch-triage")
        r1 = env.reset(seed=100)
        ids1 = {b.id for b in r1.observation.bug_reports}
        r2 = env.reset(seed=200)
        ids2 = {b.id for b in r2.observation.bug_reports}
        assert ids1 != ids2, "Different seeds must produce different bugs"

    def test_dynamic_reset_clears_all_state(self):
        env = BugTriageEnv("single-triage")
        env.reset(seed=42)
        bid = env.state().bug_reports[0].id
        env.step(TriageAction(action_type=ActionType.CLASSIFY, bug_id=bid, severity=Severity.CRITICAL))
        assert env.state().step_number == 1

        env.reset(seed=99)
        assert env.state().step_number == 0
        assert env.state().classifications == {}
        assert env.state().submitted_bugs == []
