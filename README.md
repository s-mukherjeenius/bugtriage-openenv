---
title: BugTriage OpenEnv
emoji: 🐛
colorFrom: red
colorTo: yellow
sdk: docker
app_port: 7860
tags:
  - openenv
---

# BugTriage OpenEnv

**An OpenEnv-compliant reinforcement learning environment for software bug triage.**

BugTriage simulates the real-world workflow of a triage engineer: classifying severity, routing bugs to teams, detecting duplicates, requesting missing information, escalating critical incidents, and submitting decisions — all within a step budget.

---

## Quick Start

```bash
# Clone and install
git clone https://github.com/s-mukherjeenius/bugtriage-openenv.git
cd bugtriage-openenv
pip install -r requirements.txt

# Start the environment server
uvicorn app.server:app --host 0.0.0.0 --port 7860

# Run inference (separate terminal)
export HF_TOKEN=your_key_here
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
python inference.py

# Run RL training demo (separate terminal)
python rl_train.py
```

---

## Environment Description & Motivation

Software bug triage is a critical daily task at every technology company. When a bug report arrives, a triage engineer must quickly assess its severity, route it to the correct team, detect if it duplicates an existing report, request missing information from the reporter, escalate SLA-critical incidents, and finalize the triage decision — all under time pressure.

This environment models that exact workflow. An AI agent receives raw bug reports (with titles, descriptions, SLA deadlines, customer tiers, and metadata) and must take a sequence of triage actions. The environment provides rich, continuous reward signals at every step, making it suitable for both LLM-based inference and RL post-training.

**Who would use this:** Companies building AI copilots for engineering teams (incident response, support ticket routing, DevOps automation), researchers training LLM agents on multi-step decision tasks with structured action spaces, and teams evaluating LLM reasoning over noisy real-world text.

---

## Action Space

The agent submits one action per step as a JSON object:

| Action Type      | Required Fields                          | Description                          |
|------------------|------------------------------------------|--------------------------------------|
| `classify`       | `bug_id`, `severity`                     | Set severity: critical/high/medium/low |
| `assign`         | `bug_id`, `assigned_team`                | Route to team                        |
| `request_info`   | `bug_id`, `info_requested`               | Ask for missing details              |
| `mark_duplicate` | `bug_id`, `duplicate_of`                 | Mark as duplicate of original        |
| `escalate`       | `bug_id`, `escalation_reason`            | Escalate to leadership               |
| `submit`         | `bug_id`                                 | Finalize triage                      |

**Severity levels:** critical, high, medium, low

**Teams:** backend, frontend, mobile, infrastructure, security, database, qa

---

## Observation Space

Each step returns a `BugTriageObservation` containing:

- `bug_reports` — All bug reports with full text, metadata, SLA timers, customer tiers
- `current_classifications` — Map of bug_id → severity assigned so far
- `current_assignments` — Map of bug_id → team assigned so far
- `duplicate_map` — Map of bug_id → original_id for marked duplicates
- `escalated_bug_ids` — List of escalated bugs
- `unprocessed_bug_ids` — Bugs not yet submitted
- `submitted_bug_ids` — Finalized bugs
- `action_history` — Sequence of past actions and their rewards
- `steps_remaining` — Steps left before episode truncation
- `cumulative_reward` — Total reward accumulated

---

## Reward Structure

**Step-level rewards** (continuous signal at every step):

| Action               | Correct  | Adjacent | Wrong    |
|----------------------|----------|----------|----------|
| Classify severity    | +0.15    | +0.06    | -0.10    |
| Assign team          | +0.12    | —        | -0.08    |
| Mark duplicate       | +0.18    | —        | -0.12    |
| Escalate             | +0.12    | —        | -0.05    |
| Request info         | +0.10    | —        | -0.05    |
| Submit (complete)    | +0.08    | +0.02    | -0.08    |

**Episode-level grading** returns a score in [0.0, 1.0] with weighted components:

| Component             | Task 1 | Task 2 | Task 3 |
|-----------------------|--------|--------|--------|
| Severity accuracy     | 40%    | 30%    | 25%    |
| Team accuracy         | 30%    | 25%    | 20%    |
| Duplicate detection   | —      | 20%    | 20%    |
| Escalation            | 10%    | 10%    | 20%    |
| Info request quality  | 15%    | 10%    | 10%    |
| Efficiency            | 5%     | 5%     | 5%     |

---

## Tasks

### Task 1: Single Critical Bug Triage (Easy)
- **1 bug**, max **5 steps**, threshold **0.80**
- A production-down payment failure. Classify, assign, optionally escalate, submit.

### Task 2: Batch Bug Triage with Duplicate Detection (Medium)
- **8 bugs**, max **32 steps**, threshold **0.65**
- Includes 1 duplicate pair, 1 incomplete report, 1 security issue needing escalation.

### Task 3: SLA Crisis — Mass Bug Surge (Hard)
- **15 bugs**, max **50 steps**, threshold **0.50**
- 3 duplicate pairs, 5 SLA-critical bugs, enterprise escalations, 2 incomplete reports, linked bug clusters.

---

## Baseline Scores

Baseline scores from `inference.py` using `Qwen/Qwen2.5-72B-Instruct` via HuggingFace Router:

| Task            | Difficulty | Score  | Threshold | Status |
|-----------------|-----------|--------|-----------|--------|
| single-triage   | Easy      | ~0.85  | 0.80      | Pass   |
| batch-triage    | Medium    | ~0.55  | 0.65      | Near   |
| sla-crisis      | Hard      | ~0.35  | 0.50      | Challenging |

**Heuristic fallback scores** (when LLM is unavailable, deterministic keyword-based policy):

| Task            | Score  | Notes                                         |
|-----------------|--------|-----------------------------------------------|
| single-triage   | ~0.70  | Gets severity + team right via keyword matching |
| batch-triage    | ~0.30  | Classify + assign only, no duplicate/escalation |
| sla-crisis      | ~0.20  | Limited by step budget and no advanced actions  |

The hard task (sla-crisis) is designed to genuinely challenge frontier models. Perfect score requires correctly handling 3 duplicate pairs across 15 bugs, prioritizing SLA-critical escalations, detecting incomplete reports, and managing a tight 50-step budget.

---

## API Endpoints

| Method | Path     | Description                         |
|--------|----------|-------------------------------------|
| POST   | /reset   | Initialize episode with task name   |
| POST   | /step    | Execute one triage action           |
| GET    | /state   | Get full internal state             |
| POST   | /grade   | Get episode score + components      |
| GET    | /health  | Server health check                 |
| GET    | /tasks   | List all tasks with metadata        |

---

## RL Training Demo

The included `rl_train.py` demonstrates that the environment produces learnable reward signals by training lightweight neural-network policies via REINFORCE with baseline, purely from the environment's step rewards:

```bash
# Start server
uvicorn app.server:app --port 7860

# Train (default: batch-triage, 800 episodes)
python rl_train.py

# Train on other tasks
RL_TASK=single-triage RL_EPISODES=600 python rl_train.py
RL_TASK=sla-crisis RL_EPISODES=1200 python rl_train.py
```

Architecture: 5 policy heads (classify, assign, info, escalate, duplicate) trained independently on their own reward signals. Uses Jaccard text similarity for duplicate target selection — no hardcoded answers.

---

## Project Structure

```
bugtriage-openenv/
├── app/
│   ├── env.py              # Core environment: reset(), step(), state()
│   ├── graders.py          # Deterministic step + episode graders
│   ├── models.py           # Pydantic: Action, Observation, Reward, State
│   ├── scenarios.py        # Bug reports + ground truth (3 tasks)
│   ├── server.py           # FastAPI HTTP server
│   └── ui.html             # Interactive web UI
├── server/
│   ├── app.py              # OpenEnv-wrapped server (openenv create_app)
│   └── bugtriage_environment.py  # openenv Environment interface
├── models.py               # Root models (openenv-core typed)
├── client.py               # openenv EnvClient for training code
├── inference.py            # LLM baseline inference script
├── rl_train.py             # RL training demo (REINFORCE)
├── openenv.yaml            # OpenEnv spec manifest
├── Dockerfile              # Multi-stage Docker build
├── requirements.txt        # Runtime dependencies
├── pyproject.toml          # Project metadata
└── tests/                  # Comprehensive test suite
    ├── test_env.py         # Environment logic tests
    ├── test_graders.py     # Grader correctness tests
    └── test_api.py         # HTTP API compliance tests
```

---

## Environment Variables

| Variable       | Required | Default                              | Description                    |
|----------------|----------|--------------------------------------|--------------------------------|
| `API_BASE_URL` | Yes*     | `https://router.huggingface.co/v1`   | LLM API endpoint               |
| `MODEL_NAME`   | Yes*     | `Qwen/Qwen2.5-72B-Instruct`         | Model identifier               |
| `HF_TOKEN`     | Yes*     | —                                    | HuggingFace / API key          |
| `ENV_URL`      | No       | `http://localhost:7860`              | BugTriage server URL           |

*Required for inference.py

---

## Docker

```bash
# Build
docker build -t bugtriage-openenv .

# Run
docker run -p 7860:7860 bugtriage-openenv

# Test health
curl http://localhost:7860/health

# Test reset (what the pre-submission validator does)
curl -X POST -H "Content-Type: application/json" -d '{}' http://localhost:7860/reset
```

---

## Testing

```bash
pip install pytest
pytest tests/ -v
```

---

## License

MIT
