"""
Shared pytest fixtures for BugTriage OpenEnv test suite.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.env import BugTriageEnv
from app.server import app


# ---------------------------------------------------------------------------
# Environment fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="function")
def env_easy():
    env = BugTriageEnv("single-triage")
    env.reset()
    return env


@pytest.fixture(scope="function")
def env_medium():
    env = BugTriageEnv("batch-triage")
    env.reset()
    return env


@pytest.fixture(scope="function")
def env_hard():
    env = BugTriageEnv("sla-crisis")
    env.reset()
    return env


# ---------------------------------------------------------------------------
# HTTP test client fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def http_client():
    """FastAPI TestClient — exercises actual HTTP endpoints."""
    with TestClient(app) as client:
        yield client
