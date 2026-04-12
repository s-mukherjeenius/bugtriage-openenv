"""Quick smoke test for Option A hidden_details reveal mechanic."""
from app.env import BugTriageEnv
from app.models import TriageAction, ActionType

# Test batch-triage: BUG-005 has hidden_details
env = BugTriageEnv("batch-triage")
result = env.reset()
obs = result.observation

# Verify BUG-005 starts with no steps_to_reproduce
bug005 = [b for b in obs.bug_reports if b.id == "BUG-005"][0]
assert bug005.steps_to_reproduce is None, "BUG-005 should start without steps"
assert bug005.environment_info is None, "BUG-005 should start without env info"
assert bug005.hidden_details is not None, "BUG-005 should have hidden_details"
assert obs.info_revealed_bug_ids == [], "No bugs revealed at start"
print("PASS: BUG-005 starts incomplete with hidden_details present")

# Call request_info on BUG-005
action = TriageAction(
    action_type=ActionType.REQUEST_INFO,
    bug_id="BUG-005",
    info_requested=["steps_to_reproduce", "environment_info"],
)
step_result = env.step(action)
obs2 = step_result.observation

# Verify hidden_details were revealed
bug005_after = [b for b in obs2.bug_reports if b.id == "BUG-005"][0]
assert bug005_after.hidden_details is None, "hidden_details should be consumed"
assert bug005_after.steps_to_reproduce is not None, "steps should be populated"
assert bug005_after.environment_info is not None, "env_info should be populated"
assert "[REPORTER RESPONSE" in bug005_after.description, "Description should contain revealed text"
assert "BUG-005" in obs2.info_revealed_bug_ids, "BUG-005 should be in revealed list"
print("PASS: request_info revealed hidden_details into BUG-005")
print(f"  steps_to_reproduce: {bug005_after.steps_to_reproduce[:80]}...")
print(f"  environment_info: {bug005_after.environment_info}")
print(f"  info_revealed_bug_ids: {obs2.info_revealed_bug_ids}")

# Test that re-requesting doesn't re-reveal
action2 = TriageAction(
    action_type=ActionType.REQUEST_INFO,
    bug_id="BUG-005",
    info_requested=["more_info"],
)
step_result2 = env.step(action2)
obs3 = step_result2.observation
assert "BUG-005" not in obs3.info_revealed_bug_ids, "BUG-005 should NOT be re-revealed"
print("PASS: Second request_info does not re-reveal (consumed)")

# Test dynamic generation with seed
env2 = BugTriageEnv("batch-triage")
result2 = env2.reset(seed=42)
needs_info_bugs = [b for b in result2.observation.bug_reports if b.hidden_details is not None]
print(f"\nDynamic scenario (seed=42): {len(needs_info_bugs)} bugs with hidden_details")
for b in needs_info_bugs:
    print(f"  {b.id}: hidden_details length={len(b.hidden_details)}")

# Test sla-crisis: CRI-005 and CRI-014
env3 = BugTriageEnv("sla-crisis")
result3 = env3.reset()
cri005 = [b for b in result3.observation.bug_reports if b.id == "CRI-005"][0]
cri014 = [b for b in result3.observation.bug_reports if b.id == "CRI-014"][0]
assert cri005.hidden_details is not None, "CRI-005 should have hidden_details"
assert cri014.hidden_details is not None, "CRI-014 should have hidden_details"
print("\nPASS: CRI-005 and CRI-014 have hidden_details in sla-crisis")

# Test adversarial-triage: ADV-005 and ADV-013
env4 = BugTriageEnv("adversarial-triage")
result4 = env4.reset()
adv005 = [b for b in result4.observation.bug_reports if b.id == "ADV-005"][0]
adv013 = [b for b in result4.observation.bug_reports if b.id == "ADV-013"][0]
assert adv005.hidden_details is not None, "ADV-005 should have hidden_details"
assert adv013.hidden_details is not None, "ADV-013 should have hidden_details"
print("PASS: ADV-005 and ADV-013 have hidden_details in adversarial-triage")

print("\n=== ALL SMOKE TESTS PASSED ===")
