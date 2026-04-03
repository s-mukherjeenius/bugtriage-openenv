"""
BugTriage OpenEnv — Inference Script
=====================================
Runs a language model against all three BugTriage tasks using the OpenAI client.
Outputs strictly-formatted [START] / [STEP] / [END] log lines for evaluation.

Mandatory environment variables:
  API_BASE_URL   The API endpoint for the LLM.
  MODEL_NAME     The model identifier to use for inference.
  HF_TOKEN       Your HuggingFace / API key.

Optional:
  ENV_URL        BugTriage server base URL (default: http://localhost:7860)

Stdout format (exactly as specified — do NOT modify):
  [START] task=<task_name> env=<benchmark> model=<model_name>
  [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> rewards=<r1,r2,...,rn>

Rules:
  - One [START] line at episode begin.
  - One [STEP] line per step, immediately after env.step() returns.
  - One [END] line always emitted (even on exception) via finally block.
  - reward and rewards formatted to 2 decimal places.
  - done and success are lowercase booleans: true or false.
  - error is the error string, or null if none.
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
# Reads credentials from HF_TOKEN, OPENAI_API_KEY, or API_KEY (in priority order)
API_KEY: str      = (
    os.getenv("HF_TOKEN")
    or os.getenv("OPENAI_API_KEY")
    or os.getenv("API_KEY")
    or "hf-placeholder"
)
ENV_URL: str      = os.getenv("ENV_URL", "http://localhost:7860").rstrip("/")

BENCHMARK: str         = "bugtriage-openenv"
TEMPERATURE: float     = 0.2
MAX_TOKENS: int        = 256
SUCCESS_THRESHOLD: float = 0.50

TASKS: List[str] = ["single-triage", "batch-triage", "sla-crisis"]

# Must match scenario max_steps exactly
MAX_STEPS: Dict[str, int] = {
    "single-triage": 5,
    "batch-triage":  32,
    "sla-crisis":    50,
}


# ---------------------------------------------------------------------------
# Mandatory log helpers — field names and ordering must not be changed
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val  = str(done).lower()
    # Sanitise — spec requires single line, no embedded newlines
    action_safe = action.replace("\n", " ").replace("\r", "")[:200]
    print(
        f"[STEP] step={step} action={action_safe} reward={reward:.2f} "
        f"done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    # Format matches sample exactly: success= steps= rewards= (no score= field)
    print(
        f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}",
        flush=True,
    )


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

SYSTEM_PROMPT = textwrap.dedent("""
You are an expert software triage engineer. Your job is to process bug reports
by taking exactly ONE triage action per response.

Available action_types and their required fields:
  classify      → {"action_type": "classify",      "bug_id": "...", "severity": "critical|high|medium|low"}
  assign        → {"action_type": "assign",         "bug_id": "...", "assigned_team": "backend|frontend|mobile|infrastructure|security|database|qa"}
  request_info  → {"action_type": "request_info",   "bug_id": "...", "info_requested": ["item1", "item2"]}
  mark_duplicate→ {"action_type": "mark_duplicate", "bug_id": "...", "duplicate_of": "ORIGINAL-ID"}
  escalate      → {"action_type": "escalate",       "bug_id": "...", "escalation_reason": "brief reason"}
  submit        → {"action_type": "submit",          "bug_id": "..."}

Decision guidelines:
- Severity: critical=production down/security breach, high=major feature broken,
            medium=notable bug with workaround, low=cosmetic/minor.
- Teams: security→security; auth/payments/APIs→backend; CSS/UI→frontend;
         iOS/Android→mobile; DB queries→database; servers→infrastructure; tests→qa.
- Duplicates: mark the LATER-submitted report as duplicate of the EARLIER one.
- Request info ONLY when steps_to_reproduce AND environment_info are both absent.
- Escalate: critical security bugs + enterprise SLA <4h critical/high bugs.
- Submit after classify + assign (+ any optional actions) are complete.

CRITICAL: Respond with ONLY a valid JSON object. No markdown, no explanation.
Example: {"action_type": "classify", "bug_id": "BUG-007", "severity": "critical"}
""").strip()


def _build_user_prompt(obs: Dict[str, Any], step_num: int) -> str:
    unprocessed    = obs.get("unprocessed_bug_ids", [])
    submitted      = obs.get("submitted_bug_ids", [])
    steps_remaining= obs.get("steps_remaining", 0)
    classifications= obs.get("current_classifications", {})
    assignments    = obs.get("current_assignments", {})
    duplicates     = obs.get("duplicate_map", {})
    escalations    = obs.get("escalated_bug_ids", [])
    action_history = obs.get("action_history", [])

    bug_summaries = []
    for bug in obs.get("bug_reports", []):
        bid = bug["id"]
        if bid not in unprocessed:
            continue
        classified = classifications.get(bid, "NOT SET")
        assigned   = assignments.get(bid,   "NOT SET")
        dup        = duplicates.get(bid, "")
        esc        = "YES" if bid in escalations else "no"
        sla        = bug.get("sla_hours_remaining")
        sla_str    = f"{sla:.1f}h" if sla is not None else "none"
        tier       = bug.get("customer_tier", "unknown")

        summary = (
            f"  {bid}: [{classified}|{assigned}] sla={sla_str} tier={tier} esc={esc}"
        )
        if dup:
            summary += f" DUP_OF={dup}"
        if not bug.get("steps_to_reproduce"):
            summary += " [MISSING:steps]"
        if not bug.get("environment_info"):
            summary += " [MISSING:env]"
        bug_summaries.append(summary)
        bug_summaries.append(f"    Title: {bug['title'][:100]}")
        desc = bug.get("description", "")[:150].replace("\n", " ")
        bug_summaries.append(f"    Desc:  {desc}...")

    history_lines = [
        f"  Step {h['step']}: {h['action_type']} {h['bug_id']} → {h.get('message','')[:80]}"
        for h in action_history[-5:]
    ]

    return textwrap.dedent(f"""
        Step {step_num} | Steps remaining: {steps_remaining}
        Unprocessed ({len(unprocessed)}): {', '.join(unprocessed)}
        Submitted:   {', '.join(submitted) if submitted else 'none'}

        Bug status:
        {chr(10).join(bug_summaries) if bug_summaries else '  All bugs submitted.'}

        Recent actions:
        {chr(10).join(history_lines) if history_lines else '  None yet.'}

        Output ONE JSON action now.
    """).strip()


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


def get_llm_action(
    client: OpenAI,
    obs: Dict[str, Any],
    step_num: int,
    unprocessed: List[str],
) -> Dict[str, Any]:
    """Call the LLM and return a parsed action dict with deterministic fallback."""
    user_prompt = _build_user_prompt(obs, step_num)

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
        print(f"[DEBUG] Could not parse LLM output: {raw[:200]}", file=sys.stderr)
    except Exception as exc:
        print(f"[DEBUG] LLM call failed: {exc}", file=sys.stderr)

    # Deterministic heuristic fallback: classify → assign → submit
    if unprocessed:
        bug_id = unprocessed[0]
        if bug_id not in obs.get("current_classifications", {}):
            return {"action_type": "classify", "bug_id": bug_id, "severity": "medium"}
        elif bug_id not in obs.get("current_assignments", {}):
            return {"action_type": "assign", "bug_id": bug_id, "assigned_team": "backend"}
        else:
            return {"action_type": "submit", "bug_id": bug_id}

    return {"action_type": "submit", "bug_id": unprocessed[0] if unprocessed else "unknown"}


# ---------------------------------------------------------------------------
# Single-task episode runner
# [END] is guaranteed via finally — spec requirement
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

            action_dict = get_llm_action(client, obs, step, unprocessed)

            # Guard against malformed fallback
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
                    info        = step_result.get("info", {})
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

        # Final grade if episode ended without /done flag (step budget exhausted)
        if final_score == 0.0:
            try:
                grade_result = env_client.grade()
                final_score  = grade_result.get("score", 0.0)
            except Exception:
                pass

        success = final_score >= SUCCESS_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Episode error for task '{task_name}': {exc}", file=sys.stderr)

    finally:
        # Spec: [END] must always be emitted, even on exception
        log_end(success=success, steps=steps_taken, score=final_score, rewards=rewards)

    return success, steps_taken, final_score, rewards


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    client     = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env_client = BugTriageClient(ENV_URL)

    # Verify server reachability before starting
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
            # run_task's finally block already emitted [END] — just record the score
            print(f"[ERROR] Task '{task_name}' outer crash: {exc}", file=sys.stderr)
            all_scores.append(0.0)

        time.sleep(1)   # brief pause between tasks

    mean_score = sum(all_scores) / len(all_scores) if all_scores else 0.0
    print(f"\n[INFO] === Final Results ===", file=sys.stderr)
    for task_name, score in zip(TASKS, all_scores):
        print(f"[INFO]   {task_name}: {score:.3f}", file=sys.stderr)
    print(f"[INFO]   Mean score: {mean_score:.3f}", file=sys.stderr)


if __name__ == "__main__":
    main()
