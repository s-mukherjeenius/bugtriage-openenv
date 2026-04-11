"""BugTriage OpenEnv — HTTP API tests (validates pre-submission checklist)."""
from fastapi.testclient import TestClient
from app.server import app

client = TestClient(app)


class TestHealth:
    def test_returns_200(self):
        assert client.get("/health").status_code == 200

    def test_status_healthy(self):
        assert client.get("/health").json()["status"] == "healthy"


class TestTasks:
    def test_returns_four(self):
        tasks = client.get("/tasks").json()
        assert len(tasks) == 4

    def test_difficulty_range(self):
        difficulties = {t["difficulty"] for t in client.get("/tasks").json()}
        assert difficulties == {"easy", "medium", "hard", "expert"}


class TestReset:
    def test_empty_body_200(self):
        """Pre-submission validator sends POST /reset with {} — must return 200."""
        assert client.post("/reset", json={}).status_code == 200

    def test_with_task_200(self):
        assert client.post("/reset", json={"task": "single-triage"}).status_code == 200

    def test_returns_observation(self):
        data = client.post("/reset", json={"task": "single-triage"}).json()
        assert "observation" in data
        assert "done" in data

    def test_bug_count_per_task(self):
        assert len(client.post("/reset", json={"task": "single-triage"}).json()["observation"]["bug_reports"]) == 1
        assert len(client.post("/reset", json={"task": "batch-triage"}).json()["observation"]["bug_reports"]) == 8
        assert len(client.post("/reset", json={"task": "sla-crisis"}).json()["observation"]["bug_reports"]) == 15
        assert len(client.post("/reset", json={"task": "adversarial-triage"}).json()["observation"]["bug_reports"]) == 20

    def test_invalid_task_422(self):
        assert client.post("/reset", json={"task": "nonexistent"}).status_code == 422

    def test_clears_state(self):
        client.post("/reset", json={"task": "single-triage"})
        client.post("/step", json={"action_type": "classify", "bug_id": "PAY-001", "severity": "critical"})
        obs = client.post("/reset", json={"task": "single-triage"}).json()["observation"]
        assert obs["step_number"] == 0
        assert obs["current_classifications"] == {}


class TestDynamicScenarios:
    """Tests for seed-based dynamic scenario generation via the HTTP API."""

    def test_seed_reset_200(self):
        """Reset with seed should return 200."""
        r = client.post("/reset", json={"task": "batch-triage", "seed": 42})
        assert r.status_code == 200

    def test_seed_returns_correct_bug_count(self):
        """Dynamic scenarios must have the same bug count as static ones."""
        obs = client.post("/reset", json={"task": "single-triage", "seed": 42}).json()["observation"]
        assert len(obs["bug_reports"]) == 1
        obs = client.post("/reset", json={"task": "batch-triage", "seed": 42}).json()["observation"]
        assert len(obs["bug_reports"]) == 8
        obs = client.post("/reset", json={"task": "sla-crisis", "seed": 42}).json()["observation"]
        assert len(obs["bug_reports"]) == 15

    def test_seed_deterministic(self):
        """Same seed + same task → identical bug IDs and titles."""
        obs1 = client.post("/reset", json={"task": "batch-triage", "seed": 123}).json()["observation"]
        obs2 = client.post("/reset", json={"task": "batch-triage", "seed": 123}).json()["observation"]
        ids1 = [b["id"] for b in obs1["bug_reports"]]
        ids2 = [b["id"] for b in obs2["bug_reports"]]
        assert ids1 == ids2

    def test_different_seeds_different_bugs(self):
        """Different seeds → different bug IDs."""
        obs1 = client.post("/reset", json={"task": "batch-triage", "seed": 1}).json()["observation"]
        obs2 = client.post("/reset", json={"task": "batch-triage", "seed": 999}).json()["observation"]
        ids1 = {b["id"] for b in obs1["bug_reports"]}
        ids2 = {b["id"] for b in obs2["bug_reports"]}
        assert ids1 != ids2

    def test_seed_bugs_differ_from_static(self):
        """Seeded bugs should have different IDs than static scenario bugs."""
        static = client.post("/reset", json={"task": "batch-triage"}).json()["observation"]
        dynamic = client.post("/reset", json={"task": "batch-triage", "seed": 42}).json()["observation"]
        static_ids = {b["id"] for b in static["bug_reports"]}
        dynamic_ids = {b["id"] for b in dynamic["bug_reports"]}
        assert static_ids != dynamic_ids

    def test_seed_null_gives_static(self):
        """seed=null or no seed → static scenario with original bug IDs."""
        obs = client.post("/reset", json={"task": "single-triage", "seed": None}).json()["observation"]
        assert obs["bug_reports"][0]["id"] == "PAY-001"

    def test_dynamic_step_and_grade(self):
        """Dynamic scenarios can be stepped and graded end-to-end."""
        obs = client.post("/reset", json={"task": "single-triage", "seed": 77}).json()["observation"]
        bug_id = obs["bug_reports"][0]["id"]
        assert bug_id.startswith("S")  # generated IDs start with S prefix

        r = client.post("/step", json={"action_type": "classify", "bug_id": bug_id, "severity": "critical"})
        assert r.status_code == 200
        assert isinstance(r.json()["reward"], float)

        r = client.post("/step", json={"action_type": "assign", "bug_id": bug_id, "assigned_team": "backend"})
        assert r.status_code == 200

        r = client.post("/step", json={"action_type": "submit", "bug_id": bug_id})
        assert r.json()["done"] is True

        grade = client.post("/grade").json()
        assert 0.0 <= grade["score"] <= 1.0
        assert "severity" in grade["components"]

    def test_info_field_reports_generated(self):
        """Reset info should indicate whether scenario was generated."""
        static = client.post("/reset", json={"task": "single-triage"}).json()
        assert static["info"].get("generated") is False or static["info"].get("seed") is None

        dynamic = client.post("/reset", json={"task": "single-triage", "seed": 42}).json()
        assert dynamic["info"].get("generated") is True
        assert dynamic["info"].get("seed") == 42


class TestStep:
    def setup_method(self):
        client.post("/reset", json={"task": "single-triage"})

    def test_classify_200(self):
        r = client.post("/step", json={"action_type": "classify", "bug_id": "PAY-001", "severity": "critical"})
        assert r.status_code == 200

    def test_response_fields(self):
        data = client.post("/step", json={"action_type": "classify", "bug_id": "PAY-001", "severity": "critical"}).json()
        assert "observation" in data
        assert isinstance(data["reward"], float)
        assert isinstance(data["done"], bool)

    def test_full_triage_done(self):
        client.post("/step", json={"action_type": "classify", "bug_id": "PAY-001", "severity": "critical"})
        client.post("/step", json={"action_type": "assign", "bug_id": "PAY-001", "assigned_team": "backend"})
        r = client.post("/step", json={"action_type": "submit", "bug_id": "PAY-001"})
        assert r.json()["done"] is True
        assert 0.0 <= r.json()["info"]["final_score"] <= 1.0

    def test_invalid_action_422(self):
        assert client.post("/step", json={"action_type": "teleport", "bug_id": "PAY-001"}).status_code == 422


class TestState:
    def setup_method(self):
        client.post("/reset", json={"task": "single-triage"})

    def test_returns_200(self):
        assert client.get("/state").status_code == 200

    def test_reflects_actions(self):
        client.post("/step", json={"action_type": "classify", "bug_id": "PAY-001", "severity": "high"})
        assert client.get("/state").json()["classifications"]["PAY-001"] == "high"


class TestGrade:
    def setup_method(self):
        client.post("/reset", json={"task": "single-triage"})

    def test_returns_score(self):
        data = client.post("/grade").json()
        assert "score" in data
        assert 0.0 <= data["score"] <= 1.0

    def test_improves_with_correct_actions(self):
        before = client.post("/grade").json()["score"]
        client.post("/step", json={"action_type": "classify", "bug_id": "PAY-001", "severity": "critical"})
        client.post("/step", json={"action_type": "assign", "bug_id": "PAY-001", "assigned_team": "backend"})
        after = client.post("/grade").json()["score"]
        assert after > before


class TestTask4API:
    """Tests for Task 4 adversarial-triage via HTTP API."""

    def test_reset_task4(self):
        r = client.post("/reset", json={"task": "adversarial-triage"})
        assert r.status_code == 200
        assert len(r.json()["observation"]["bug_reports"]) == 20

    def test_flag_spam_action(self):
        client.post("/reset", json={"task": "adversarial-triage"})
        r = client.post("/step", json={"action_type": "flag_spam", "bug_id": "ADV-016", "spam_reason": "fake"})
        assert r.status_code == 200
        assert r.json()["reward"] > 0

    def test_flagged_spam_in_observation(self):
        client.post("/reset", json={"task": "adversarial-triage"})
        r = client.post("/step", json={"action_type": "flag_spam", "bug_id": "ADV-017", "spam_reason": "panic"})
        obs = r.json()["observation"]
        assert "ADV-017" in obs["flagged_spam_ids"]
        assert "ADV-017" not in obs["unprocessed_bug_ids"]

    def test_task4_grade_has_spam_component(self):
        client.post("/reset", json={"task": "adversarial-triage"})
        grade = client.post("/grade").json()
        assert "spam_detection" in grade["components"]

    def test_task4_dynamic_seed(self):
        r = client.post("/reset", json={"task": "adversarial-triage", "seed": 42})
        assert r.status_code == 200
        assert len(r.json()["observation"]["bug_reports"]) == 20
        assert r.json()["info"]["generated"] is True
