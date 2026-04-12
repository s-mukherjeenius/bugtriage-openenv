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

# 🐛 BugTriage OpenEnv

**An OpenEnv-compliant reinforcement learning environment for software bug triage — featuring adversarial spam detection, cascading failure chains, and dynamic scenario generation.**

BugTriage simulates the real-world workflow of a triage engineer: classifying severity, routing bugs to teams, detecting duplicates, requesting missing information, escalating critical incidents, flagging spam reports, and submitting decisions — all within a step budget. It's the only OpenEnv environment that tests an agent's ability to **distinguish signal from noise** in a realistic software engineering context.

---

## Why BugTriage?

Every engineering team triages bugs daily. But real triage isn't just classification — it's a multi-step decision process under time pressure with noisy, incomplete, and sometimes **adversarial** inputs. BugTriage captures this complexity:

- **7 distinct action types** — the richest action space of any OpenEnv environment
- **4 difficulty tiers** — from single-bug warmup to 20-bug adversarial gauntlet
- **Continuous reward signal** at every step — not just end-of-episode scoring
- **Dynamic scenario generation** — unlimited training variety via seed-based generation
- **Adversarial robustness testing** — agents must detect spam, pranks, and self-resolved tickets
- **Root-cause chain reasoning** — cascading failures that require connecting upstream incidents

**Who would use this:** Teams building AI copilots for engineering (incident response, support ticket routing, DevOps automation), researchers training LLM agents on multi-step structured-action decision tasks, and teams evaluating LLM reasoning over noisy real-world text.

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

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        BugTriage OpenEnv                         │
├─────────────┬───────────────────────┬───────────────────────────┤
│  Inference  │   RL Training         │   Interactive UI          │
│  (LLM)      │   (REINFORCE)         │   (Browser)              │
│  inference.py│   rl_train.py        │   /ui endpoint           │
├─────────────┴───────────────────────┴───────────────────────────┤
│                    HTTP API Layer                                │
│  POST /reset  POST /step  GET /state  POST /grade  GET /tasks   │
├─────────────────────────────────────────────────────────────────┤
│                    Environment Core                              │
│  ┌──────────┐ ┌──────────┐ ┌───────────┐ ┌───────────────────┐ │
│  │ Scenarios│ │ Graders  │ │ Generator │ │ Models (Pydantic) │ │
│  │ 4 static │ │ 4 graders│ │ 4 dynamic │ │ Action/Obs/State  │ │
│  │ tasks    │ │ + step   │ │ generators│ │ + Reward          │ │
│  └──────────┘ └──────────┘ └───────────┘ └───────────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│  OpenEnv Compatibility Layer (server/bugtriage_environment.py)  │
│  Implements Environment interface for openenv-core integration  │
└─────────────────────────────────────────────────────────────────┘
```

**Key design decisions:**

1. **Dual server pattern** — `app/server.py` is the standalone FastAPI server with concurrent session support (up to 64 parallel evaluations); `server/app.py` wraps it for openenv-core's WebSocket protocol. Both share the same environment logic via thin adapter.
2. **Generic graders** — All graders derive expected answers from `scenario.ground_truth`, not hardcoded bug IDs. Static and dynamically-generated scenarios use the same grading code. Team accuracy uses **adjacency partial credit** (e.g., assigning to `infrastructure` when `backend` is correct earns 0.3x instead of 0x).
3. **Seed-based determinism** — `random.Random(seed)` per scenario, fully reproducible. Same seed = same bugs, same ground truth.
4. **6 policy heads for RL** — Classify, Assign, Info, Escalate, Duplicate, and Spam each have independent REINFORCE networks with separate baselines.
5. **Production-grade bug data** — Generated bugs include realistic stack traces, structured log snippets, and quantified impact metrics (affected users, error rates, revenue impact) for data-driven severity classification.

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

### Task 4: Adversarial Triage — Spam Detection + Cascading Failures (Expert)
- **20 bugs** (15 real + 5 spam/fake), max **65 steps**, threshold **0.45**
- **Ticking SLA timers** — every step the agent takes costs 0.02h of SLA across ALL active bugs. Spend 3 steps triaging a cosmetic tooltip? That critical SQL injection just got closer to breach. This forces genuine strategic prioritization, not just sequential processing.
- **Cascading root-cause resolution** — ADV-006 (Redis pool exhaustion) is the root cause of ADV-009 (stale search cache). When the agent correctly resolves the root cause BEFORE its downstream symptom, it earns a +0.12 bonus. The optimal strategy requires understanding incident dependency graphs.
- **Adversarial spam detection** — 5 fake reports (quantum paradoxes, admitted pranks, sentient AI, self-resolved tickets) must be flagged to save steps. Correctly flagging earns +0.20; falsely flagging a real bug costs -0.15.
- **SLA breach penalties** — when a bug's SLA timer hits zero, the agent takes a -0.03 penalty per breach. Multiple breaches cascade rapidly.

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
| `flag_spam`      | `bug_id`, `spam_reason`                  | Flag as spam/fake (Task 4)           |
| `submit`         | `bug_id`                                 | Finalize triage                      |

**Severity levels:** critical, high, medium, low

**Teams:** backend, frontend, mobile, infrastructure, security, database, qa

---

## Observation Space

Each step returns a `BugTriageObservation` containing:

- `bug_reports` — All bug reports with full text, metadata, SLA timers, customer tiers, stack traces, log snippets, and quantified impact metrics
- `current_classifications` — Map of bug_id → severity assigned so far
- `current_assignments` — Map of bug_id → team assigned so far
- `duplicate_map` — Map of bug_id → original_id for marked duplicates
- `escalated_bug_ids` — List of escalated bugs
- `flagged_spam_ids` — Bug IDs flagged as spam (Task 4)
- `unprocessed_bug_ids` — Bugs not yet submitted or flagged
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
| Assign team          | +0.12    | +0.04    | -0.08    |
| Mark duplicate       | +0.18    | —        | -0.12    |
| Escalate             | +0.12    | —        | -0.05    |
| Request info         | +0.10    | —        | -0.05    |
| Flag spam (correct)  | +0.20    | —        | -0.15    |
| Submit (complete)    | +0.08    | +0.02    | -0.08    |
| Root cause resolved  | +0.12    | +0.05    | —        |
| SLA breach (per bug) | —        | —        | -0.03    |

**Episode-level grading** returns a score in [0.0, 1.0] with weighted components:

| Component             | Task 1 | Task 2 | Task 3 | Task 4 |
|-----------------------|--------|--------|--------|--------|
| Spam detection        | —      | —      | —      | 18%    |
| Severity accuracy     | 35%    | 25%    | 22%    | 17%    |
| Team accuracy         | 25%    | 20%    | 18%    | 13%    |
| Duplicate detection   | —      | 20%    | 18%    | 13%    |
| SLA/Security escalation| 10%   | 10%    | 18%    | 13%    |
| Info request quality  | 12%    | 10%    | 8%     | 8%     |
| Root-cause resolution | —      | —      | —      | 8%     |
| Efficiency            | 8%     | 8%     | 8%     | 5%     |
| Completion            | 10%    | 7%     | 8%     | 5%     |

---

## Dynamic Scenario Generation

BugTriage supports **seed-based dynamic scenario generation** for unlimited training variety across all 4 tasks.

### How it works

Pass a `seed` parameter to `/reset` to generate a new scenario:

```bash
# Static scenario (default — same bugs every time, used for evaluation)
curl -X POST -d '{"task": "batch-triage"}' http://localhost:7860/reset

# Dynamic scenario (seed=42 — fresh bugs, deterministic for that seed)
curl -X POST -d '{"task": "batch-triage", "seed": 42}' http://localhost:7860/reset

# Adversarial task with dynamic generation
curl -X POST -d '{"task": "adversarial-triage", "seed": 42}' http://localhost:7860/reset
```

Same seed + same task always produces identical output (fully deterministic via `random.Random(seed)`).

### What gets generated

The generator (`app/generator.py`) draws from pools of **30+ realistic bug templates** and **7 spam templates** across 6 engineering teams and assembles structurally valid scenarios:

| Task     | Bugs | Structure |
|----------|------|-----------|
| Easy     | 1    | 1 critical bug, enterprise tier, SLA-critical, needs escalation |
| Medium   | 8    | 1 security escalation + 1 info-incomplete + 1 duplicate pair + 4 varied |
| Hard     | 15   | 3 duplicate pairs + 2 info-incomplete + 5 SLA-critical + varied filler |
| Expert   | 20   | 5 spam + 2 duplicate pairs + 2 info-incomplete + 4 SLA-critical + varied real bugs |

### Using in the Web UI

The interactive UI at `http://localhost:7860/ui` has a **Static / Dynamic** toggle. Select Dynamic, enter a seed (or click 🎲 for random), and start playing.

### Python API

```python
from app.generator import generate_scenario

# Generate an expert scenario with seed 42
scenario = generate_scenario("adversarial-triage", seed=42)
print(len(scenario.bug_reports))  # 20
spam_count = sum(1 for g in scenario.ground_truth.values() if g.is_spam)
print(f"Spam bugs: {spam_count}")  # 5
```

---

## Baseline Scores

Baseline scores from `inference.py` using `Qwen/Qwen2.5-72B-Instruct` via HuggingFace Router:

| Task                | Difficulty | Score  | Threshold | Status |
|---------------------|-----------|--------|-----------|--------|
| single-triage       | Easy      | ~0.85  | 0.80      | Pass   |
| batch-triage        | Medium    | ~0.55  | 0.65      | Near   |
| sla-crisis          | Hard      | ~0.35  | 0.50      | Challenging |
| adversarial-triage  | Expert    | ~0.30  | 0.45      | Very Challenging |

The expert task (adversarial-triage) is designed to genuinely challenge frontier models. Perfect score requires correctly identifying 5 spam reports, handling 2 duplicate pairs across 20 bugs, prioritizing SLA-critical escalations, detecting incomplete reports, recognizing root-cause chains, and managing a 65-step budget efficiently.

---

## API Endpoints

| Method | Path                    | Description                                     |
|--------|-------------------------|-------------------------------------------------|
| POST   | /reset                  | Initialize episode (`{task, seed?, session_id?}`) |
| POST   | /step                   | Execute one triage action (`?session_id=...`)   |
| GET    | /state                  | Get full internal state (`?session_id=...`)     |
| POST   | /grade                  | Get episode score + components (`?session_id=...`) |
| GET    | /health                 | Server health check                              |
| GET    | /tasks                  | List all tasks with metadata                     |
| GET    | /sessions               | List active sessions                             |
| DELETE | /sessions/{session_id}  | Delete a session                                 |

---

## RL Training Demo

The included `rl_train.py` demonstrates that the environment produces learnable reward signals. It features 6 independent policy heads and an interactive startup menu:

```bash
# Start server
uvicorn app.server:app --port 7860

# Launch training with interactive menu
python rl_train.py
#   Select a task to train on:
#   [1] single-triage       (Easy   — 1 bug,  5 steps)
#   [2] batch-triage        (Medium — 8 bugs, 32 steps)
#   [3] sla-crisis          (Hard   — 15 bugs, 50 steps)
#   [4] adversarial-triage  (Expert — 20 bugs, 65 steps)
#   Use dynamic scenarios? (y/N): y
#   Number of episodes?

# Or use environment variables for non-interactive mode
RL_TASK=adversarial-triage RL_DYNAMIC=true RL_EPISODES=1200 python rl_train.py
```

Architecture: 6 policy heads (classify, assign, info, escalate, duplicate, **spam**) trained independently via REINFORCE with baseline. For adversarial-triage, the spam head runs first (Phase 0), filtering out fakes before the classify/assign/escalate phases operate on real bugs only.

---

## Project Structure

```
bugtriage-openenv/
├── app/
│   ├── env.py              # Core environment: reset(seed=None), step(), grade()
│   ├── generator.py        # Dynamic scenario generator (30+ bug + 7 spam templates)
│   ├── graders.py          # Generic step + episode graders (4 tasks, no hardcoded IDs)
│   ├── models.py           # Pydantic: Action (7 types), Observation, Reward, State
│   ├── scenarios.py        # Static bug reports + ground truth (4 tasks, 44 bugs)
│   ├── server.py           # FastAPI HTTP server (concurrent sessions, seed support)
│   └── ui.html             # Interactive web UI (Static/Dynamic toggle, 4 tasks)
├── server/
│   ├── app.py              # OpenEnv-wrapped server (seed passthrough)
│   └── bugtriage_environment.py  # Thin adapter wrapping app/env.py for openenv-core
├── models.py               # Root models (openenv-core typed, FLAG_SPAM action)
├── client.py               # openenv EnvClient for training code
├── inference.py            # LLM baseline inference (4 tasks + spam heuristic)
├── rl_train.py             # RL training (6 policy heads, spam phase, interactive)
├── openenv.yaml            # OpenEnv spec manifest (4 tasks)
├── Dockerfile              # Multi-stage Docker build
├── requirements.txt        # Runtime dependencies
├── pyproject.toml          # Project metadata
└── tests/                  # Comprehensive test suite (156 tests)
    ├── test_env.py         # Environment logic + Task 4 spam tests
    ├── test_graders.py     # Grader + generator tests (expert generator tests)
    └── test_api.py         # HTTP API + Task 4 endpoint tests
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
# Runs 156 tests including Task 4 spam detection, adversarial grading,
# and dynamic expert scenario generation tests
```

---

## License

MIT
