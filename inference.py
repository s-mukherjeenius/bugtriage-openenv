"""
BugTriage OpenEnv — Inference Script
=====================================
Runs a language model against all four BugTriage tasks using the OpenAI client.
Outputs strictly-formatted [START] / [STEP] / [END] log lines for evaluation.

Mandatory environment variables:
  API_BASE_URL   The API endpoint for the LLM.
  MODEL_NAME     The model identifier to use for inference.
  HF_TOKEN       Your Hugging Face / API key.

Optional:
  ENV_URL        BugTriage server base URL (default: http://localhost:7860)

STDOUT FORMAT (must match exactly — any deviation = incorrect scoring):
  [START] task=<task_name> env=<benchmark> model=<model_name>
  [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> rewards=<r1,r2,...,rn>

Rules:
  - One [START] line at episode begin.
  - One [STEP] line per step, immediately after env.step() returns.
  - One [END] line always emitted (even on exception) via finally block.
  - reward and rewards formatted to 2 decimal places.
  - done and success are lowercase booleans: true or false.
  - error is the raw error string, or null if none.
  - All fields on a single line with no newlines within a line.
"""
from __future__ import annotations

import json
import os
import re
import sys
import textwrap
import time
from typing import Any, Dict, List, Optional, Tuple

import requests
from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration — read from environment (mandatory per spec)
# ---------------------------------------------------------------------------

API_BASE_URL: str = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME: str   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
API_KEY: str      = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")
if not API_KEY:
    raise ValueError("HF_TOKEN environment variable is required")
ENV_URL: str      = os.getenv("ENV_URL", "http://localhost:7860").rstrip("/")

BENCHMARK: str         = "bugtriage-openenv"
TEMPERATURE: float     = 0.2
MAX_TOKENS: int        = 300
SUCCESS_THRESHOLD: float = 0.50
MAX_LLM_RETRIES: int  = 2

TASKS: List[str] = ["single-triage", "batch-triage", "sla-crisis", "adversarial-triage"]

MAX_STEPS: Dict[str, int] = {
    "single-triage": 5,
    "batch-triage":  32,
    "sla-crisis":    50,
    "adversarial-triage": 65,
}

# ---------------------------------------------------------------------------
# Spam detection keywords for heuristic fallback (Task 4)
# ---------------------------------------------------------------------------

_SPAM_KEYWORDS = [
    "quantum", "paradox", "sentient", "sentience", "timeline", "dimension",
    "april fools", "just kidding", "lol", "prank", "made this up",
    "never mind", "already resolved", "false alarm", "you can close",
    "hacked!!", "hackers everywhere", "consciousness", "superposition",
    "wet_circuits", "qt-paradox", "sentience_achieved", "hacked-123",
]

_SPAM_PRODUCTS = [
    "quantum module", "ai chatbot",
]


# ---------------------------------------------------------------------------
# Mandatory log helpers — match sample inference.py EXACTLY
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    action_safe = action.replace("\n", " ").replace("\r", "")[:200]
    print(f"[STEP] step={step} action={action_safe} reward={reward:.2f} done={done_val} error={error_val}", flush=True)


def log_end(success: bool, steps: int, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}", flush=True)


# ---------------------------------------------------------------------------
# Environment HTTP client
# ---------------------------------------------------------------------------

class BugTriageClient:
    """Thin HTTP wrapper around the BugTriage FastAPI server."""

    def __init__(self, base_url: str, timeout: int = 60) -> None:
        self.base_url = base_url
        self.timeout  = timeout
        self.session  = requests.Session()

    def reset(self, task: str) -> Dict[str, Any]:
        r = self.session.post(
            f"{self.base_url}/reset",
            json={"task": task},
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()

    def step(self, action: Dict[str, Any]) -> Dict[str, Any]:
        r = self.session.post(
            f"{self.base_url}/step",
            json=action,
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()

    def grade(self) -> Dict[str, Any]:
        r = self.session.post(f"{self.base_url}/grade", timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def health(self) -> Dict[str, Any]:
        r = self.session.get(f"{self.base_url}/health", timeout=self.timeout)
        r.raise_for_status()
        return r.json()


# ---------------------------------------------------------------------------
# LLM prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent("""\
You are an expert software triage engineer processing bug reports.
Respond with ONLY a single valid JSON object — no markdown, no explanation.

Action formats (exactly one per response):
  {"action_type":"classify","bug_id":"ID","severity":"critical|high|medium|low"}
  {"action_type":"assign","bug_id":"ID","assigned_team":"backend|frontend|mobile|infrastructure|security|database|qa"}
  {"action_type":"request_info","bug_id":"ID","info_requested":["item1","item2"]}
  {"action_type":"mark_duplicate","bug_id":"ID","duplicate_of":"ORIGINAL-ID"}
  {"action_type":"escalate","bug_id":"ID","escalation_reason":"reason"}
  {"action_type":"flag_spam","bug_id":"ID","spam_reason":"reason"}
  {"action_type":"submit","bug_id":"ID"}

SEVERITY RULES:
  critical = production down OR security breach OR 100% users affected OR data loss
  high     = major feature broken, crash, OOM, data integrity issue, significant user impact
  medium   = notable bug with workaround, intermittent issues with partial info
  low      = cosmetic, typo, alignment, tooltip wrapping, minor UI issues

TEAM RULES:
  security       → auth vulnerabilities, token issues, rate limit bypass, session exploits, SQL injection
  backend        → API errors, payment processing, webhook failures, server crashes, billing, CSV/export bugs, GraphQL resolvers
  frontend       → CSS, UI layout, dark mode, contrast, button alignment, display issues, CORS
  mobile         → iOS/Android app crashes, mobile-specific bugs
  database       → SQL queries, indexing, DB performance, PostgreSQL issues
  infrastructure → server nodes, cluster outages, primary DB node down, deployment, Redis, CI/CD, connection pools
  qa             → test infrastructure

DUPLICATE RULES:
  - Two bugs are duplicates if they describe the SAME root cause
  - Mark the LATER-filed report (higher timestamp) as duplicate of the EARLIER one

ESCALATION RULES:
  - Escalate if: sla_hours_remaining < 2.0
  - Escalate if: enterprise customer + critical or high severity
  - Escalate if: security vulnerability with active exploit
  - Escalate if: compliance/regulatory risk (GDPR, etc.)

REQUEST_INFO RULES:
  - ONLY when BOTH steps_to_reproduce AND environment_info are completely absent/null

SPAM DETECTION RULES (for adversarial-triage task):
  - Flag as spam if: non-existent products, made-up error codes (QT-PARADOX, SENTIENCE_ACHIEVED, etc.)
  - Flag as spam if: physically impossible or fantastical descriptions (quantum paradoxes, AI sentience)
  - Flag as spam if: reporter explicitly admits it's a prank or joke
  - Flag as spam if: reporter says the issue is already resolved / "never mind"
  - Flag as spam if: ALL CAPS panic with zero technical substance (loading spinner = hacked)
  - DO NOT flag as spam: vague but genuine reports (those get request_info instead)

WORKFLOW per bug:
  For spam: flag_spam → done (do NOT classify/assign/submit spam)
  For real: classify → assign → (optional: escalate/request_info/mark_duplicate) → submit
  Always classify and assign BEFORE submitting.

PRIORITY ORDER: Flag spam first, then handle SLA-critical bugs, then the rest.

IMPORTANT (adversarial-triage only): SLA timers TICK DOWN with every step you take.
Every action you spend on a low-priority bug costs SLA time on critical ones.
Prioritize ruthlessly: flag spam fast, then handle the shortest SLA bugs FIRST.""")


def _build_user_prompt(obs: Dict[str, Any], step_num: int) -> str:
    """Build a concise prompt from the observation."""
    unprocessed     = obs.get("unprocessed_bug_ids", [])
    submitted       = obs.get("submitted_bug_ids", [])
    flagged_spam    = obs.get("flagged_spam_ids", [])
    sla_breached    = obs.get("sla_breached_bug_ids", [])
    steps_remaining = obs.get("steps_remaining", 0)
    classifications = obs.get("current_classifications", {})
    assignments     = obs.get("current_assignments", {})
    duplicates      = obs.get("duplicate_map", {})
    escalations     = obs.get("escalated_bug_ids", [])
    action_history  = obs.get("action_history", [])

    bug_summaries = []
    for bug in obs.get("bug_reports", []):
        bid = bug.get("id", "") if isinstance(bug, dict) else bug.id
        if bid not in unprocessed:
            continue

        classified = classifications.get(bid, "NOT SET")
        assigned   = assignments.get(bid,   "NOT SET")
        dup        = duplicates.get(bid, "")
        esc        = "YES" if bid in escalations else "no"

        if isinstance(bug, dict):
            title = bug.get("title", "")[:120]
            desc  = bug.get("description", "")[:250].replace("\n", " ")
            sla   = bug.get("sla_hours_remaining")
            tier  = bug.get("customer_tier", "unknown")
            product = bug.get("product", "")
            has_steps = bool(bug.get("steps_to_reproduce"))
            has_env   = bool(bug.get("environment_info"))
            timestamp = bug.get("timestamp", "")
            linked    = bug.get("linked_bug_ids", []) or []
        else:
            title = getattr(bug, "title", "")[:120]
            desc  = getattr(bug, "description", "")[:250].replace("\n", " ")
            sla   = getattr(bug, "sla_hours_remaining", None)
            tier  = getattr(bug, "customer_tier", "unknown")
            product = getattr(bug, "product", "")
            has_steps = bool(getattr(bug, "steps_to_reproduce", None))
            has_env   = bool(getattr(bug, "environment_info", None))
            timestamp = getattr(bug, "timestamp", "")
            linked    = getattr(bug, "linked_bug_ids", []) or []

        sla_str = f"{sla:.1f}h" if sla is not None else "none"
        missing = []
        if not has_steps: missing.append("steps_to_reproduce")
        if not has_env:   missing.append("environment_info")

        summary = f"\n  {bid}: severity={classified} team={assigned} sla={sla_str} tier={tier} esc={esc}"
        if dup:      summary += f" DUP_OF={dup}"
        if missing:  summary += f" MISSING=[{','.join(missing)}]"
        if linked:   summary += f" linked={linked}"
        summary += f"\n    Title: {title}"
        summary += f"\n    Desc: {desc}"
        summary += f"\n    Product: {product} | Time: {timestamp}"
        bug_summaries.append(summary)

    history_lines = [
        f"  Step {h.get('step',0)}: {h.get('action_type','')} {h.get('bug_id','')} → r={h.get('reward',0):+.2f} {h.get('message','')[:80]}"
        for h in action_history[-8:]
    ]

    return textwrap.dedent(f"""\
Step {step_num} | Steps remaining: {steps_remaining}
Unprocessed ({len(unprocessed)}): {', '.join(unprocessed)}
Submitted ({len(submitted)}): {', '.join(submitted) if submitted else 'none'}
Flagged spam ({len(flagged_spam)}): {', '.join(flagged_spam) if flagged_spam else 'none'}
{'SLA BREACHED: ' + ', '.join(sla_breached) if sla_breached else ''}

Bugs needing action:
{''.join(bug_summaries) if bug_summaries else '  All bugs processed.'}

Recent actions:
{chr(10).join(history_lines) if history_lines else '  None yet.'}

Output ONE JSON action now.""")


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Robustly extract a JSON object from LLM output."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    text_clean = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()
    try:
        return json.loads(text_clean)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None


def _is_spam_heuristic(bug_data: Dict[str, Any]) -> bool:
    """Keyword-based spam detection for heuristic fallback."""
    desc = (bug_data.get("description", "") + " " + bug_data.get("title", "")).lower()
    product = bug_data.get("product", "").lower()

    # Check spam products
    for sp in _SPAM_PRODUCTS:
        if sp in product:
            return True

    # Check spam keywords
    spam_hits = sum(1 for kw in _SPAM_KEYWORDS if kw in desc)
    if spam_hits >= 2:
        return True

    # ALL CAPS check: if >60% of alphabetic chars are uppercase and description is short on tech detail
    alpha_chars = [c for c in bug_data.get("description", "") if c.isalpha()]
    if len(alpha_chars) > 50:
        upper_ratio = sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars)
        if upper_ratio > 0.6 and spam_hits >= 1:
            return True

    return False


def _heuristic_action(obs: Dict[str, Any]) -> Dict[str, Any]:
    """Deterministic fallback when LLM fails."""
    unprocessed     = obs.get("unprocessed_bug_ids", [])
    classifications = obs.get("current_classifications", {})
    assignments     = obs.get("current_assignments", {})
    flagged_spam    = obs.get("flagged_spam_ids", [])

    if not unprocessed:
        return {"action_type": "submit", "bug_id": "unknown"}

    bug_id = unprocessed[0]
    bug_data = {}
    for b in obs.get("bug_reports", []):
        if isinstance(b, dict) and b.get("id") == bug_id:
            bug_data = b
            break

    # Check for spam first (Task 4)
    if bug_id not in flagged_spam and _is_spam_heuristic(bug_data):
        return {"action_type": "flag_spam", "bug_id": bug_id, "spam_reason": "Spam indicators detected"}

    if bug_id not in classifications:
        desc = (bug_data.get("description", "") + " " + bug_data.get("title", "")).lower()
        sla = bug_data.get("sla_hours_remaining")
        if any(w in desc for w in ["production down", "100%", "all transactions", "security breach", "p0"]):
            sev = "critical"
        elif any(w in desc for w in ["vulnerability", "exploit", "bypass"]) and sla and sla < 2:
            sev = "critical"
        elif any(w in desc for w in ["crash", "oom", "fails", "broken", "500", "down"]):
            sev = "high"
        elif any(w in desc for w in ["cosmetic", "tooltip", "alignment", "typo", "logo"]):
            sev = "low"
        else:
            sev = "medium"
        return {"action_type": "classify", "bug_id": bug_id, "severity": sev}

    if bug_id not in assignments:
        desc = (bug_data.get("description", "") + " " + bug_data.get("title", "")).lower()
        if any(w in desc for w in ["security", "auth", "token", "session", "vulnerability", "exploit"]):
            team = "security"
        elif any(w in desc for w in ["css", "ui", "button", "layout", "dark mode", "contrast", "logo", "tooltip"]):
            team = "frontend"
        elif any(w in desc for w in ["mobile", "ios", "android", "app crash"]):
            team = "mobile"
        elif any(w in desc for w in ["database", "sql", "query", "index", "postgres"]):
            team = "database"
        elif any(w in desc for w in ["cluster", "node", "infra"]):
            team = "infrastructure"
        else:
            team = "backend"
        return {"action_type": "assign", "bug_id": bug_id, "assigned_team": team}

    return {"action_type": "submit", "bug_id": bug_id}


def get_llm_action(client: OpenAI, obs: Dict[str, Any], step_num: int) -> Dict[str, Any]:
    """Call the LLM with retry, falling back to heuristic on failure."""
    user_prompt = _build_user_prompt(obs, step_num)

    for attempt in range(MAX_LLM_RETRIES + 1):
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_prompt},
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                stream=False,
            )
            raw    = (completion.choices[0].message.content or "").strip()
            parsed = _extract_json(raw)
            if parsed and "action_type" in parsed and "bug_id" in parsed:
                return parsed
            print(f"[DEBUG] Attempt {attempt+1}: could not parse: {raw[:200]}", file=sys.stderr)
        except Exception as exc:
            print(f"[DEBUG] Attempt {attempt+1} LLM error: {exc}", file=sys.stderr)
            if attempt < MAX_LLM_RETRIES:
                time.sleep(1)

    print(f"[DEBUG] Using heuristic fallback", file=sys.stderr)
    return _heuristic_action(obs)


# ---------------------------------------------------------------------------
# Single-task episode runner
# ---------------------------------------------------------------------------

def run_task(
    client: OpenAI,
    env_client: BugTriageClient,
    task_name: str,
) -> Tuple[bool, int, float, List[float]]:
    """
    Run one complete episode for `task_name`.
    Returns (success, steps_taken, final_score, rewards_per_step).
    [END] is ALWAYS emitted via finally block even on exception.
    """
    rewards: List[float]  = []
    steps_taken: int      = 0
    final_score: float    = 0.0
    success: bool         = False
    max_steps: int        = MAX_STEPS[task_name]

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        reset_result = env_client.reset(task_name)
        obs          = reset_result.get("observation", {})

        for step in range(1, max_steps + 1):
            if reset_result.get("done", False):
                break

            unprocessed = obs.get("unprocessed_bug_ids", [])
            if not unprocessed:
                break

            action_dict = get_llm_action(client, obs, step)

            if "action_type" not in action_dict:
                action_dict["action_type"] = "submit"
            if "bug_id" not in action_dict and unprocessed:
                action_dict["bug_id"] = unprocessed[0]

            action_str = json.dumps(action_dict, separators=(",", ":"))
            error_msg: Optional[str] = None

            try:
                step_result = env_client.step(action_dict)
                reward      = step_result.get("reward", 0.0)
                done        = step_result.get("done",   False)
                obs         = step_result.get("observation", obs)

                if done:
                    info = step_result.get("info", {})
                    final_score = info.get("final_score", 0.0)
                    if final_score == 0.0:
                        grade_result = env_client.grade()
                        final_score  = grade_result.get("score", 0.0)

            except requests.HTTPError as e:
                error_msg = str(e)[:100]
                reward    = 0.0
                done      = False

            rewards.append(reward)
            steps_taken = step
            log_step(step=step, action=action_str, reward=reward, done=done, error=error_msg)

            if done:
                break

        # Final grade if not already obtained
        if final_score == 0.0:
            try:
                grade_result = env_client.grade()
                final_score  = grade_result.get("score", 0.0)
            except Exception:
                pass

        # Clamp score to [0, 1]
        final_score = min(max(final_score, 0.0), 1.0)
        success = final_score >= SUCCESS_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Episode error for task '{task_name}': {exc}", file=sys.stderr)

    finally:
        # [END] ALWAYS emitted — even on exception (spec requirement)
        log_end(success=success, steps=steps_taken, rewards=rewards)

    return success, steps_taken, final_score, rewards


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    client     = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env_client = BugTriageClient(ENV_URL)

    # Verify server reachability
    try:
        health = env_client.health()
        print(f"[INFO] Server healthy: {health}", file=sys.stderr)
    except Exception as exc:
        print(f"[ERROR] Cannot reach BugTriage server at {ENV_URL}: {exc}", file=sys.stderr)
        sys.exit(1)

    all_scores: List[float] = []

    for task_name in TASKS:
        print(f"\n[INFO] ===== Task: {task_name} =====", file=sys.stderr)
        try:
            success, steps, score, rewards = run_task(client, env_client, task_name)
            all_scores.append(score)
            print(
                f"[INFO] {task_name}: score={score:.3f} success={success} steps={steps}",
                file=sys.stderr,
            )
        except Exception as exc:
            print(f"[ERROR] Task '{task_name}' outer crash: {exc}", file=sys.stderr)
            all_scores.append(0.0)

        time.sleep(1)

    mean_score = sum(all_scores) / len(all_scores) if all_scores else 0.0
    print(f"\n[INFO] === Final Results ===", file=sys.stderr)
    for task_name, score in zip(TASKS, all_scores):
        print(f"[INFO]   {task_name}: {score:.3f}", file=sys.stderr)
    print(f"[INFO]   Mean score: {mean_score:.3f}", file=sys.stderr)


if __name__ == "__main__":
    main()
