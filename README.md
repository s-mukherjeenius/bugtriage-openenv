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

# Run RL training demo (separate terminal — interactive menu)
python rl_train.py
```

---

## Environment Description & Motivation

Software bug triage is a critical daily task at every technology company. When a bug report arrives, a triage engineer must quickly assess its severity, route it to the correct team, detect if it duplicates an existing report, request missing information from the reporter, escalate SLA-critical incidents, and finalize the triage decision — all under time pressure.

This environment models that exact workflow. An AI agent receives raw bug reports (with titles, descriptions, SLA deadlines, customer tiers, and metadata) and must take a sequence of triage actions. The environment provides rich, continuous reward signals at every step, making it suitable for both LLM-based inference and RL post-training.

**Who would use this:** Companies building AI copilots for engineering teams (incident response, support ticket routing, DevOps automation), researchers training LLM agents on multi-step decision tasks with structured action spaces, and teams evaluating LLM reasoning over noisy real-world text.

---

## Dynamic Scenario Generation

BugTriage supports **seed-based dynamic scenario generation** for unlimited training variety. Instead of training on the same 3 static scenarios, agents can face fresh, structurally valid bug sets on every episode.

### How it works

Pass a `seed` parameter to `/reset` to generate a new scenario:

```bash
# Static scenario (default — same bugs every time, used for evaluation)
curl -X POST -d '{"task": "batch-triage"}' http://localhost:7860/reset

# Dynamic scenario (seed=42 — fresh bugs, deterministic for that seed)
curl -X POST -d '{"task": "batch-triage", "seed": 42}' http://localhost:7860/reset

# Different seed = different bugs
curl -X POST -d '{"task": "batch-triage", "seed": 999}' http://localhost:7860/reset
```

Same seed + same task always produces identical output (fully deterministic via `random.Random(seed)`).

### What gets generated

The generator (`app/generator.py`) draws from a pool of **30+ realistic bug templates** across 6 engineering teams (backend, frontend, security, database, infrastructure, mobile) and assembles structurally valid scenarios:

| Task | Bugs | Structure |
|------|------|-----------|
| Easy | 1 | 1 critical bug, enterprise tier, SLA-critical, needs escalation |
| Medium | 8 | 1 security escalation + 1 info-incomplete + 1 duplicate pair + 4 varied |
| Hard | 15 | 3 duplicate pairs + 2 info-incomplete + 5 SLA-critical + varied filler |

Duplicate pairs use semantically similar but differently worded variant descriptions — the agent must recognize the same root cause described by different reporters.

### Using in the Web UI

The interactive UI at `http://localhost:7860/ui` has a **Static / Dynamic** toggle in the left panel. Select Dynamic, enter a seed (or click 🎲 for random), and click Start to play with generated scenarios.

### Using in RL training

`rl_train.py` uses dynamic scenarios by default, generating a unique seed per episode so the agent trains on varied bug distributions:

```bash
python rl_train.py
# Interactive menu asks:
#   [1] single-triage  [2] batch-triage  [3] sla-crisis
#   Use dynamic scenarios? (y/N)
#   Number of episodes?
```

### Python API

```python
from app.generator import generate_scenario

# Generate a medium scenario with seed 42
scenario = generate_scenario("batch-triage", seed=42)
print(len(scenario.bug_reports))  # 8
print(scenario.ground_truth)       # {bug_id: BugGroundTruth, ...}

# Use via the environment
from app.env import BugTriageEnv
env = BugTriageEnv("batch-triage")
result = env.reset(seed=42)  # dynamic bugs
result = env.reset()          # static bugs (default)
```

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

The hard task (sla-crisis) is designed to genuinely challenge frontier models. Perfect score requires correctly handling 3 duplicate pairs across 15 bugs, prioritizing SLA-critical escalations, detecting incomplete reports, and managing a tight 50-step budget.

---

## API Endpoints

| Method | Path     | Description                         |
|--------|----------|-------------------------------------|
| POST   | /reset   | Initialize episode (`{task, seed?}`) |
| POST   | /step    | Execute one triage action           |
| GET    | /state   | Get full internal state             |
| POST   | /grade   | Get episode score + components      |
| GET    | /health  | Server health check                 |
| GET    | /tasks   | List all tasks with metadata        |

---

## RL Training Demo

The included `rl_train.py` demonstrates that the environment produces learnable reward signals. It features an interactive startup menu:

```bash
# Start server
uvicorn app.server:app --port 7860

# Launch training with interactive menu
python rl_train.py
#   Select a task to train on:
#   [1] single-triage  (Easy  — 1 bug,  5 steps)
#   [2] batch-triage   (Medium — 8 bugs, 32 steps)
#   [3] sla-crisis     (Hard  — 15 bugs, 50 steps)
#   Use dynamic scenarios? (y/N): y
#   Number of episodes? (default=800): 1000

# Or use environment variables for non-interactive mode
RL_TASK=sla-crisis RL_DYNAMIC=true RL_EPISODES=1200 python rl_train.py
```

Architecture: 5 policy heads (classify, assign, info, escalate, duplicate) trained independently via REINFORCE with baseline. Uses Jaccard text similarity for duplicate target selection — no hardcoded answers. Dynamic mode generates a unique scenario per episode for maximum training diversity.

---

## Project Structure

```
bugtriage-openenv/
├── app/
│   ├── env.py              # Core environment: reset(seed=None), step(), grade()
│   ├── generator.py        # Dynamic scenario generator (30+ bug templates)
│   ├── graders.py          # Generic step + episode graders (no hardcoded IDs)
│   ├── models.py           # Pydantic: Action, Observation, Reward, State
│   ├── scenarios.py        # Static bug reports + ground truth (3 tasks)
│   ├── server.py           # FastAPI HTTP server (supports seed in /reset)
│   └── ui.html             # Interactive web UI (Static/Dynamic toggle)
├── server/
│   ├── app.py              # OpenEnv-wrapped server (seed passthrough)
│   └── bugtriage_environment.py  # openenv Environment (seed support)
├── models.py               # Root models (openenv-core typed)
├── client.py               # openenv EnvClient for training code
├── inference.py            # LLM baseline inference script
├── rl_train.py             # RL training (interactive menu + dynamic seeds)
├── openenv.yaml            # OpenEnv spec manifest
├── Dockerfile              # Multi-stage Docker build
├── requirements.txt        # Runtime dependencies
├── pyproject.toml          # Project metadata
└── tests/                  # Comprehensive test suite (28 seed tests)
    ├── test_env.py         # Environment logic tests
    ├── test_graders.py     # Grader + generator tests (20 generator tests)
    └── test_api.py         # HTTP API + dynamic scenario tests
```

---

## Environment Variables

| Variable       | Required | Default                              | Description                    |
|----------------|----------|--------------------------------------|--------------------------------|
| `API_BASE_URL` | Yes*     | `https://router.huggingface.co/v1`   | LLM API endpoint               |
| `MODEL_NAME`   | Yes*     | `Qwen/Qwen2.5-72B-Instruct`         | Model identifier               |
| `HF_TOKEN`     | Yes*     | —                                    | HuggingFace / API key          |
| `ENV_URL`      | No       | `http://localhost:7860`              | BugTriage server URL           |
| `RL_TASK`      | No       | `batch-triage`                       | Task for RL training           |
| `RL_DYNAMIC`   | No       | `true`                               | Use dynamic scenarios in RL    |
| `RL_EPISODES`  | No       | `800`                                | Number of training episodes    |

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
# Runs 114+ tests including 28 dynamic scenario generation tests
```

---

## License

MIT
