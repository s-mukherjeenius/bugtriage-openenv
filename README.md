---
title: BugTriage OpenEnv
emoji: 🐛
colorFrom: red
colorTo: yellow
sdk: docker
pinned: true
tags:
  - openenv
  - rl-environment
  - agent-benchmark
  - software-engineering
  - triage
---

# 🐛 BugTriage OpenEnv

> **An OpenEnv-compliant environment for training and evaluating AI agents on real-world software bug triage and priority queue management.**

[![OpenEnv](https://img.shields.io/badge/OpenEnv-1.0.0-blue)](https://github.com/openenv)
[![License](https://img.shields.io/badge/License-Apache_2.0-green)](LICENSE)
[![HF Space](https://img.shields.io/badge/HuggingFace-Space-yellow)](https://huggingface.co/spaces/bugtriage/bugtriage-openenv)

---

## 🎯 Motivation

Every software company — from two-person startups to hyperscalers — runs a bug triage process. Real triage engineers spend hours each day:

- Reading incoming reports and classifying their severity
- Routing bugs to the correct engineering team
- Detecting duplicate submissions hidden in a flood of tickets
- Requesting missing reproduction steps from users
- Escalating critical security or SLA-breaching issues to leadership
- Managing a priority queue under time pressure

This environment simulates that exact workflow at three difficulty levels, providing a **meaningful, real-world benchmark** for language-model agents that no existing OpenEnv environment covers.

---

## 🏗️ Environment Design

### Architecture

```
POST /reset  →  Initial Observation (all bug reports, instructions)
POST /step   →  TriageAction  →  Updated Observation + Step Reward + Done flag
GET  /state  →  Full Internal State (for debugging)
POST /grade  →  Final Score [0.0, 1.0] + Component Breakdown
GET  /tasks  →  Task metadata list
GET  /health →  Server health
```

The environment is a **FastAPI server** running on port 7860 (HuggingFace Spaces default), packaged as a Docker container.

### Observation Space

| Field | Type | Description |
|---|---|---|
| `step_number` | int | Current step (0 before first action) |
| `task_name` | str | Active task ID |
| `bug_reports` | List[BugReport] | All bug reports in the episode |
| `action_history` | List[dict] | Last 20 actions + rewards + messages |
| `unprocessed_bug_ids` | List[str] | Bugs not yet submitted |
| `submitted_bug_ids` | List[str] | Finalized bugs |
| `current_classifications` | Dict[str,str] | bug_id → severity |
| `current_assignments` | Dict[str,str] | bug_id → team |
| `duplicate_map` | Dict[str,str] | bug_id → original_id |
| `escalated_bug_ids` | List[str] | Escalated bug IDs |
| `steps_remaining` | int | Steps left before truncation |
| `cumulative_reward` | float | Total reward accumulated |

Each `BugReport` contains: `id`, `title`, `description`, `reporter`, `timestamp`, `product`, `version`, `steps_to_reproduce`, `expected_behavior`, `actual_behavior`, `environment_info`, `customer_tier`, `sla_hours_remaining`, `linked_bug_ids`.

### Action Space

One action per step. `action_type` and `bug_id` always required:

```json
// Classify severity
{"action_type": "classify", "bug_id": "BUG-007", "severity": "critical"}

// Assign to team  
{"action_type": "assign", "bug_id": "BUG-007", "assigned_team": "security"}

// Request missing information
{"action_type": "request_info", "bug_id": "BUG-005", "info_requested": ["steps_to_reproduce", "device_model", "os_version"]}

// Mark as duplicate
{"action_type": "mark_duplicate", "bug_id": "BUG-003", "duplicate_of": "BUG-006"}

// Escalate to leadership
{"action_type": "escalate", "bug_id": "BUG-007", "escalation_reason": "Security vulnerability allowing rate limit bypass"}

// Submit (finalize triage)
{"action_type": "submit", "bug_id": "BUG-007"}
```

### Reward Function

The reward is **continuous** — agents receive signal on every step:

| Action | Condition | Reward |
|---|---|---|
| `classify` | Correct severity | +0.10 |
| `classify` | Adjacent severity (off by 1) | +0.04 |
| `classify` | Wrong severity | −0.05 |
| `assign` | Correct team | +0.08 |
| `assign` | Wrong team | −0.05 |
| `request_info` | Info genuinely missing | +0.06 |
| `request_info` | Info not needed (wasted step) | −0.04 |
| `mark_duplicate` | Correct duplicate pair | +0.12 |
| `mark_duplicate` | Wrong original | −0.05 |
| `mark_duplicate` | False positive | −0.08 |
| `escalate` | Warranted escalation | +0.08 |
| `escalate` | Unwarranted | −0.03 |
| `submit` | After classify + assign | +0.05 |
| `submit` | Missing classify or assign | +0.01 |
| `submit` | Neither classified nor assigned | −0.05 |

**Episode-level grading** (0.0–1.0) uses weighted multi-dimensional scoring against hardcoded ground truth. See [Grading](#grading) below.

---

## 📋 Tasks

### Task 1: `single-triage` (Easy)

**Difficulty**: ⭐☆☆  
**Max Steps**: 5  
**Reward Threshold**: 0.80  
**Bugs**: 1

A production-down payment processing failure (`PAY-001`) has been reported. Every checkout is returning HTTP 500. The report contains all necessary information — no info request is needed.

The agent must:
1. Classify severity as `critical`
2. Assign to `backend` team
3. (Optionally) Escalate — rewarded
4. Submit

**Why it's easy**: Single bug, complete information, obvious severity. Tests basic action sequencing.

**Baseline expected score**: ~0.85–0.95

---

### Task 2: `batch-triage` (Medium)

**Difficulty**: ⭐⭐☆  
**Max Steps**: 32  
**Reward Threshold**: 0.65  
**Bugs**: 8

Eight bugs spanning UI issues, file upload crashes, database query timeouts, a security rate-limit bypass, a CSV export data leak, and one ambiguous report missing all reproduction details. Hidden within the set: `BUG-003` and `BUG-006` describe the **same dark-mode contrast issue** — the agent must identify `BUG-003` as a duplicate of the earlier-submitted `BUG-006`.

The agent must:
- Correctly classify all 8 severities (low / high / medium / high / medium / medium / critical / high)
- Route each to the correct team
- Mark `BUG-003` as duplicate of `BUG-006`
- Request info for `BUG-005` (missing steps + environment)
- Escalate `BUG-007` (security vulnerability)
- Submit all non-duplicate bugs

**Why it's medium**: Requires reading comprehension to detect subtle duplicate, recognising when info is genuinely missing, and correctly routing a security issue.

**Baseline expected score**: ~0.50–0.65

---

### Task 3: `sla-crisis` (Hard)

**Difficulty**: ⭐⭐⭐  
**Max Steps**: 50  
**Reward Threshold**: 0.50  
**Bugs**: 15

A critical incident window produces 15 simultaneous reports including:
- **3 duplicate pairs** (subtle — same root cause, different descriptions)
- **5 SLA-critical bugs** (under 2 hours remaining)
- **5 enterprise customer bugs** requiring escalation
- **2 info-incomplete bugs**
- **Linked bug clusters** (auth service failures, DB primary down)

The agent must manage all dimensions correctly under a tight 50-step budget. Processing 15 bugs perfectly requires ~3 steps/bug minimum = 45 steps, leaving almost no margin for mistakes.

**Why it's hard**: Information overload, overlapping priorities, subtle duplicates (e.g., `CRI-003` and `CRI-009` both describe Firefox logo rendering via different reporters), and SLA urgency that doesn't always correlate with severity.

**Frontier model challenge**: Even GPT-4-class models score below 0.50 on this task due to the cross-referencing required for duplicate detection, the tight step budget (only 5 spare steps for a perfect run), and the need to simultaneously manage 5 different priority dimensions.

**Baseline expected score**: ~0.25–0.45

---

## 📊 Grading

Each task uses a **deterministic, multi-dimensional grader** with weighted components:

### Task 1 — `single-triage`
| Component | Weight |
|---|---|
| Severity correct | 40% |
| Team correct | 30% |
| No unnecessary info request | 15% |
| Escalation when warranted | 10% |
| Efficiency (steps used) | 5% |

### Task 2 — `batch-triage`
| Component | Weight |
|---|---|
| Severity accuracy (all 8, partial credit for adjacent) | 30% |
| Team assignment accuracy (all 8) | 25% |
| Duplicate detection (BUG-003 → BUG-006) | 20% |
| Info request for BUG-005 | 10% |
| Security escalation for BUG-007 | 10% |
| Efficiency | 5% |

### Task 3 — `sla-crisis`
| Component | Weight |
|---|---|
| Severity accuracy (all 15, partial credit) | 25% |
| Team assignment accuracy (all 15) | 20% |
| Duplicate detection (3 correct pairs) | 20% |
| SLA-critical escalations (5 bugs) | 20% |
| Info requests for incomplete bugs | 10% |
| Efficiency | 5% |

Graders apply **partial credit** for adjacent severity levels (e.g., classifying `critical` as `high` → 0.5 severity credit). False positives (unnecessary escalations, wrong duplicate claims) incur penalties.

---

## 🚀 Setup & Usage

### Prerequisites

- Python 3.10, 3.11, or 3.12
- Docker
- OpenAI-compatible LLM API access

### Local Development

```bash
git clone https://huggingface.co/spaces/bugtriage/bugtriage-openenv
cd bugtriage-openenv

pip install -r requirements.txt

# Start the server
uvicorn app.server:app --host 0.0.0.0 --port 7860 --reload

# Verify
curl http://localhost:7860/health
curl -X POST http://localhost:7860/reset -H "Content-Type: application/json" -d '{"task":"single-triage"}'

# Open interactive UI
# http://localhost:7860/ui
```

### Docker

```bash
docker build -t bugtriage-openenv .
docker run -p 7860:7860 bugtriage-openenv

# Verify
curl http://localhost:7860/health
```

### Running Inference

```bash
export HF_TOKEN=your_hf_token
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
export ENV_URL=http://localhost:7860

python inference.py
```

### Interactive API (Swagger UI)

Visit `http://localhost:7860/docs` for an interactive API explorer.

### Python Usage Example

```python
import requests

BASE = "http://localhost:7860"

# Reset to the medium task
reset = requests.post(f"{BASE}/reset", json={"task": "batch-triage"}).json()
obs = reset["observation"]
print(f"Task: {obs['task_name']}, Bugs: {len(obs['bug_reports'])}")

# Take a triage action
action = {
    "action_type": "classify",
    "bug_id": "BUG-007",
    "severity": "critical"
}
result = requests.post(f"{BASE}/step", json=action).json()
print(f"Reward: {result['reward']}, Done: {result['done']}")

# Check full state
state = requests.get(f"{BASE}/state").json()

# Get final score (call anytime)
grade = requests.post(f"{BASE}/grade").json()
print(f"Score: {grade['score']}, Components: {grade['components']}")
```

---

## 🤖 RL Training Integration

BugTriage OpenEnv is designed to plug directly into RL training frameworks.

### With TRL (GRPO)

```python
import asyncio
from bugtriage_openenv import BugTriageEnv, BugTriageAction

async def rollout(prompt, trainer):
    """Custom rollout function for TRL GRPOTrainer."""
    async with BugTriageEnv(base_url="ws://localhost:7860") as env:
        result = await env.reset()
        total_reward = 0.0
        for _ in range(32):  # max steps for batch-triage
            # Agent generates an action from the observation
            action_json = trainer.generate(str(result.observation))
            action = BugTriageAction.model_validate_json(action_json)
            result = await env.step(action)
            total_reward += result.reward
            if result.done:
                break
    return total_reward
```

### With any OpenEnv-compatible framework

```bash
# Install the environment client
pip install "git+https://huggingface.co/spaces/YOUR_USERNAME/bugtriage-openenv"

# Use in training code
from bugtriage_openenv import BugTriageEnv, BugTriageAction, BugTriageObservation
```

---

## 📈 Baseline Scores

Run against `qwen2.5-coder:7b` via local Ollama (reproducible):

| Task | Difficulty | Score | Success |
|---|---|---|---|
| `single-triage` | Easy | 0.900 | ✅ |
| `batch-triage` | Medium | 0.509 | ✅ |
| `sla-crisis` | Hard | 0.286 | ❌ |
| **Mean** | | **0.565** | |

*Scores vary ±0.05 across runs due to temperature sampling. Run `python inference.py` to reproduce.*

---

## 🧪 Tests

```bash
python -m pytest tests/ -v
```

---

## 📁 Project Structure

```
bugtriage-openenv/
├── Dockerfile              # HF Spaces + openenv-base image
├── README.md               # This file
├── openenv.yaml            # OpenEnv spec (spec_version: 1)
├── pyproject.toml          # Package config + [project.scripts] server entry
├── requirements.txt        # Python dependencies
├── uv.lock                 # Locked dependency versions
├── inference.py            # Baseline inference script
├── models.py               # Root models: BugTriageAction, BugTriageObservation, BugTriageState
├── client.py               # BugTriageEnv(EnvClient) — typed client
├── __init__.py             # Package exports
├── app/
│   ├── models.py           # Internal Pydantic models (BugReport, etc.)
│   ├── scenarios.py        # Bug report data + ground truth (hidden from agents)
│   ├── graders.py          # Deterministic step + episode graders
│   ├── env.py              # BugTriageEnv core logic
│   ├── server.py           # Legacy FastAPI server (used by tests)
│   └── ui.html             # Interactive web UI
├── server/
│   ├── app.py              # OpenEnv create_app() entry point
│   └── bugtriage_environment.py  # BugTriageEnvironment(Environment)
└── tests/
    ├── test_env.py         # Unit + integration tests (119 tests)
    ├── test_api.py         # HTTP endpoint tests
    └── test_graders.py     # Grader correctness tests
```

---

## 🔧 Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `HF_TOKEN` | Yes | — | HuggingFace / LLM API key (read first) |
| `OPENAI_API_KEY` | No | — | OpenAI-compatible API key (fallback to HF_TOKEN) |
| `API_BASE_URL` | No | `https://router.huggingface.co/v1` | LLM API endpoint |
| `MODEL_NAME` | No | `Qwen/Qwen2.5-72B-Instruct` | Model identifier |
| `ENV_URL` | No | `http://localhost:7860` | BugTriage server URL for inference script |

---

## 📜 License

Apache 2.0 — see [LICENSE](LICENSE).
