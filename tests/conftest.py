"""Shared pytest fixtures for BugTriage OpenEnv test suite."""
import pytest
from app.env import BugTriageEnv


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
