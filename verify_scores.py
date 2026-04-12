"""Verify that no task score is exactly 0.0 or 1.0 — open interval (0, 1)."""
from app.env import BugTriageEnv
from app.models import TriageAction, ActionType, Severity, Team

TASKS = ["single-triage", "batch-triage", "sla-crisis", "adversarial-triage"]

print("=== Verifying scores are in open interval (0, 1) ===\n")

# Test 1: Empty episodes (no actions) — should NOT be 0.0
for task in TASKS:
    env = BugTriageEnv(task)
    env.reset()
    score = env.grade()["score"]
    assert 0 < score < 1, f"FAIL: {task} empty score = {score} (must be strictly (0,1))"
    print(f"  {task} (empty):   {score} OK")

# Test 2: Perfect single-triage — should NOT be 1.0
env = BugTriageEnv("single-triage")
env.reset()
env.step(TriageAction(action_type=ActionType.CLASSIFY, bug_id="PAY-001", severity=Severity.CRITICAL))
env.step(TriageAction(action_type=ActionType.ASSIGN, bug_id="PAY-001", assigned_team=Team.BACKEND))
env.step(TriageAction(action_type=ActionType.ESCALATE, bug_id="PAY-001", escalation_reason="SLA breach"))
env.step(TriageAction(action_type=ActionType.SUBMIT, bug_id="PAY-001"))
score = env.grade()["score"]
assert 0 < score < 1, f"FAIL: single-triage perfect score = {score}"
print(f"  single-triage (perfect): {score} OK")

# Test 3: Dynamic scenarios
for task in TASKS:
    for seed in [1, 42, 100]:
        env = BugTriageEnv(task)
        env.reset(seed=seed)
        score = env.grade()["score"]
        assert 0 < score < 1, f"FAIL: {task} seed={seed} score = {score}"
    print(f"  {task} (seeds 1,42,100): all OK")

print("\n=== ALL SCORES IN OPEN INTERVAL (0, 1) — VALIDATOR COMPLIANT ===")
