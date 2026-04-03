"""
BugTriage OpenEnv — HTTP API Tests
Tests the FastAPI server endpoints directly via TestClient.
These validate OpenEnv spec HTTP compliance.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.server import app

client = TestClient(app)


class TestHealthEndpoint:
    def test_health_returns_200(self):
        r = client.get("/health")
        assert r.status_code == 200

    def test_health_has_status_ok(self):
        r = client.get("/health")
        data = r.json()
        assert data["status"] == "healthy"

    def test_health_has_version(self):
        r = client.get("/health")
        data = r.json()
        assert "version" in data


class TestTasksEndpoint:
    def test_tasks_returns_200(self):
        r = client.get("/tasks")
        assert r.status_code == 200

    def test_tasks_returns_three_tasks(self):
        r = client.get("/tasks")
        tasks = r.json()
        assert len(tasks) == 3

    def test_tasks_have_required_fields(self):
        r = client.get("/tasks")
        for task in r.json():
            assert "id" in task
            assert "name" in task
            assert "difficulty" in task
            assert "max_steps" in task
            assert "reward_threshold" in task

    def test_tasks_difficulty_range(self):
        r = client.get("/tasks")
        difficulties = {t["difficulty"] for t in r.json()}
        assert "easy" in difficulties
        assert "medium" in difficulties
        assert "hard" in difficulties


class TestResetEndpoint:
    def test_reset_empty_body_returns_200(self):
        """Validator script sends empty JSON body — must work."""
        r = client.post("/reset", json={})
        assert r.status_code == 200

    def test_reset_with_task_returns_200(self):
        r = client.post("/reset", json={"task": "single-triage"})
        assert r.status_code == 200

    def test_reset_returns_observation(self):
        r = client.post("/reset", json={"task": "single-triage"})
        data = r.json()
        assert "observation" in data
        assert "done" in data

    def test_reset_observation_has_bug_reports(self):
        r = client.post("/reset", json={"task": "single-triage"})
        obs = r.json()["observation"]
        assert len(obs["bug_reports"]) == 1

    def test_reset_batch_has_eight_bugs(self):
        r = client.post("/reset", json={"task": "batch-triage"})
        obs = r.json()["observation"]
        assert len(obs["bug_reports"]) == 8

    def test_reset_sla_crisis_has_fifteen_bugs(self):
        r = client.post("/reset", json={"task": "sla-crisis"})
        obs = r.json()["observation"]
        assert len(obs["bug_reports"]) == 15

    def test_reset_invalid_task_returns_422(self):
        r = client.post("/reset", json={"task": "nonexistent"})
        assert r.status_code == 422

    def test_reset_clears_previous_state(self):
        # Run some steps then reset
        client.post("/reset", json={"task": "single-triage"})
        client.post("/step", json={"action_type": "classify", "bug_id": "PAY-001", "severity": "critical"})
        r = client.post("/reset", json={"task": "single-triage"})
        obs = r.json()["observation"]
        assert obs["step_number"] == 0
        assert len(obs["submitted_bug_ids"]) == 0
        assert obs["current_classifications"] == {}


class TestStepEndpoint:
    def setup_method(self):
        client.post("/reset", json={"task": "single-triage"})

    def test_step_classify_returns_200(self):
        r = client.post("/step", json={
            "action_type": "classify",
            "bug_id": "PAY-001",
            "severity": "critical"
        })
        assert r.status_code == 200

    def test_step_response_has_all_fields(self):
        r = client.post("/step", json={
            "action_type": "classify",
            "bug_id": "PAY-001",
            "severity": "critical"
        })
        data = r.json()
        assert "observation" in data
        assert "reward" in data
        assert "done" in data
        assert "info" in data

    def test_step_reward_is_float(self):
        r = client.post("/step", json={
            "action_type": "classify",
            "bug_id": "PAY-001",
            "severity": "critical"
        })
        assert isinstance(r.json()["reward"], float)

    def test_step_done_is_bool(self):
        r = client.post("/step", json={
            "action_type": "classify",
            "bug_id": "PAY-001",
            "severity": "critical"
        })
        assert isinstance(r.json()["done"], bool)

    def test_step_full_triage_reaches_done(self):
        client.post("/step", json={"action_type": "classify",  "bug_id": "PAY-001", "severity": "critical"})
        client.post("/step", json={"action_type": "assign",    "bug_id": "PAY-001", "assigned_team": "backend"})
        r = client.post("/step", json={"action_type": "submit", "bug_id": "PAY-001"})
        assert r.json()["done"] is True

    def test_step_includes_final_score_on_done(self):
        client.post("/step", json={"action_type": "classify",  "bug_id": "PAY-001", "severity": "critical"})
        client.post("/step", json={"action_type": "assign",    "bug_id": "PAY-001", "assigned_team": "backend"})
        r = client.post("/step", json={"action_type": "submit", "bug_id": "PAY-001"})
        info = r.json()["info"]
        assert "final_score" in info
        assert 0.0 <= info["final_score"] <= 1.0

    def test_step_invalid_action_type_returns_422(self):
        r = client.post("/step", json={
            "action_type": "teleport",
            "bug_id": "PAY-001"
        })
        assert r.status_code == 422

    def test_step_missing_bug_id_returns_422(self):
        r = client.post("/step", json={"action_type": "classify", "severity": "critical"})
        assert r.status_code == 422


class TestStateEndpoint:
    def setup_method(self):
        client.post("/reset", json={"task": "single-triage"})

    def test_state_returns_200(self):
        r = client.get("/state")
        assert r.status_code == 200

    def test_state_has_required_fields(self):
        r = client.get("/state")
        data = r.json()
        assert "task_name" in data
        assert "step_number" in data
        assert "done" in data
        assert "classifications" in data
        assert "assignments" in data

    def test_state_reflects_actions(self):
        client.post("/step", json={
            "action_type": "classify", "bug_id": "PAY-001", "severity": "high"
        })
        r = client.get("/state")
        assert r.json()["classifications"].get("PAY-001") == "high"


class TestGradeEndpoint:
    def setup_method(self):
        client.post("/reset", json={"task": "single-triage"})

    def test_grade_returns_200(self):
        r = client.post("/grade")
        assert r.status_code == 200

    def test_grade_has_score(self):
        r = client.post("/grade")
        data = r.json()
        assert "score" in data
        assert 0.0 <= data["score"] <= 1.0

    def test_grade_has_components(self):
        r = client.post("/grade")
        assert "components" in r.json()

    def test_grade_improves_with_correct_actions(self):
        r0 = client.post("/grade")
        score_before = r0.json()["score"]

        client.post("/step", json={"action_type": "classify", "bug_id": "PAY-001", "severity": "critical"})
        client.post("/step", json={"action_type": "assign",   "bug_id": "PAY-001", "assigned_team": "backend"})

        r1 = client.post("/grade")
        score_after = r1.json()["score"]
        assert score_after > score_before


class TestRootEndpoint:
    def test_root_returns_200(self):
        r = client.get("/")
        assert r.status_code == 200

    def test_root_lists_endpoints(self):
        r = client.get("/", follow_redirects=False)
        assert r.status_code in (200, 307, 308)
