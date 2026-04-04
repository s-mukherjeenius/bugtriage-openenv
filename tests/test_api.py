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
    def test_returns_three(self):
        tasks = client.get("/tasks").json()
        assert len(tasks) == 3

    def test_difficulty_range(self):
        difficulties = {t["difficulty"] for t in client.get("/tasks").json()}
        assert difficulties == {"easy", "medium", "hard"}


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

    def test_invalid_task_422(self):
        assert client.post("/reset", json={"task": "nonexistent"}).status_code == 422

    def test_clears_state(self):
        client.post("/reset", json={"task": "single-triage"})
        client.post("/step", json={"action_type": "classify", "bug_id": "PAY-001", "severity": "critical"})
        obs = client.post("/reset", json={"task": "single-triage"}).json()["observation"]
        assert obs["step_number"] == 0
        assert obs["current_classifications"] == {}


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
