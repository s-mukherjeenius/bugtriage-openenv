"""
BugTriage OpenEnv — Enterprise RL Training (Multi-Phase)
=========================================================
Trains lightweight neural-network policies via REINFORCE with baseline
across all BugTriage tasks (single-triage, batch-triage, sla-crisis).

Architecture: 5 policy heads
  1. classify_net  — severity selection (4 actions)
  2. assign_net    — team routing (7 actions)
  3. info_net      — binary: request info? (2 actions)
  4. esc_net       — binary: escalate? (2 actions)
  5. dup_net       — binary: mark duplicate? (2 actions)

Key design for batch/sla-crisis:
  Two-phase episode flow:
    Phase 1: Classify + Assign all bugs (builds full picture)
    Phase 2: Info/Escalate/Duplicate decisions (full context available)
    Phase 3: Submit all bugs

  Duplicate target selection uses text similarity (word overlap)
  between bug pairs — not hardcoded targets.

  All binary nets train on BOTH act and skip decisions using
  counterfactual rewards from the episode grader.
"""

from __future__ import annotations

import math
import os
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import requests

# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════

ENV_URL        = os.getenv("ENV_URL", "http://localhost:7860")
TASK           = os.getenv("RL_TASK", "batch-triage")
N_EPISODES     = int(os.getenv("RL_EPISODES", "800"))
LR             = float(os.getenv("RL_LR", "0.015"))
ENTROPY_BETA   = 0.01
BASELINE_DECAY = 0.95
EPS_START      = 0.90
EPS_END        = 0.03
EPS_DECAY_FRAC = 0.5
PRINT_EVERY    = 20
DYNAMIC_SEEDS  = os.getenv("RL_DYNAMIC", "true").lower() == "true"  # Use varied scenarios each episode

SEVERITIES = ["critical", "high", "medium", "low"]
TEAMS      = ["backend", "frontend", "mobile", "infrastructure",
              "security", "database", "qa"]

# Spam detection keywords for the spam policy head
_SPAM_KEYWORDS_RL = [
    "quantum", "paradox", "sentient", "sentience", "timeline", "dimension",
    "april fools", "just kidding", "lol", "prank", "made this up",
    "never mind", "already resolved", "false alarm", "you can close",
    "hacked!!", "hackers everywhere", "consciousness", "superposition",
    "wet_circuits", "qt-paradox", "sentience_achieved", "hacked-123",
]
_SPAM_PRODUCTS_RL = ["quantum module", "ai chatbot", "ml pipeline", "blockchain module"]


# ═══════════════════════════════════════════════════════════════════════════
# Policy Network — REINFORCE with baseline + entropy bonus
# ═══════════════════════════════════════════════════════════════════════════

class PolicyNet:
    def __init__(self, n_in: int, n_hidden: int, n_out: int, lr: float, name: str = "") -> None:
        self.name = name
        self.n_out = n_out
        self.lr = lr
        self.W1 = np.random.randn(n_in, n_hidden).astype(np.float64) * np.sqrt(2.0 / n_in)
        self.b1 = np.zeros(n_hidden, dtype=np.float64)
        self.W2 = np.random.randn(n_hidden, n_out).astype(np.float64) * np.sqrt(2.0 / n_hidden)
        self.b2 = np.zeros(n_out, dtype=np.float64)
        self._baseline = 0.0
        self._bl_count = 0
        self._h: Optional[np.ndarray] = None

    def probs(self, x: np.ndarray) -> np.ndarray:
        h = np.maximum(0, x @ self.W1 + self.b1)
        logits = h @ self.W2 + self.b2
        logits -= logits.max()
        e = np.exp(logits)
        self._h = h
        return e / e.sum()

    def select(self, x: np.ndarray, epsilon: float = 0.0) -> Tuple[int, np.ndarray]:
        p = self.probs(x)
        if np.random.rand() < epsilon:
            return np.random.randint(self.n_out), p
        return int(np.argmax(p)), p

    def update(self, x: np.ndarray, action_idx: int, reward: float) -> None:
        self._bl_count += 1
        alpha = max(1 - BASELINE_DECAY, 1.0 / self._bl_count)
        self._baseline = (1 - alpha) * self._baseline + alpha * reward
        advantage = reward - self._baseline
        p = self.probs(x)
        h = self._h
        d = p.copy()
        d[action_idx] -= 1.0
        d *= advantage
        d -= ENTROPY_BETA * p * (1.0 + np.log(np.maximum(p, 1e-10)))
        self.W2 -= self.lr * np.outer(h, d)
        self.b2 -= self.lr * d
        dh = (d @ self.W2.T) * (h > 0).astype(np.float64)
        self.W1 -= self.lr * np.outer(x, dh)
        self.b1 -= self.lr * dh

    @property
    def param_count(self) -> int:
        return self.W1.size + self.b1.size + self.W2.size + self.b2.size


# ═══════════════════════════════════════════════════════════════════════════
# Feature Encoding
# ═══════════════════════════════════════════════════════════════════════════

def _words(text: str) -> Set[str]:
    return set(re.findall(r'[a-z]+', text.lower()))

def _bug_text(bug: Dict) -> str:
    return (bug.get("title", "") + " " + bug.get("description", "")).lower()

def _sim(bug_a: Dict, bug_b: Dict) -> float:
    wa, wb = _words(_bug_text(bug_a)), _words(_bug_text(bug_b))
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)

def encode_bug(bug: Dict) -> np.ndarray:
    desc = _bug_text(bug)
    f = []
    f.append(1.0 if any(w in desc for w in ["crash", "down", "fail", "500", "broken", "error"]) else 0.0)
    f.append(1.0 if any(w in desc for w in ["critical", "production", "p0", "100%", "all user"]) else 0.0)
    f.append(1.0 if any(w in desc for w in ["security", "vulnerability", "exploit", "bypass"]) else 0.0)
    f.append(1.0 if any(w in desc for w in ["cosmetic", "minor", "typo", "alignment", "tooltip", "logo"]) else 0.0)
    f.append(1.0 if any(w in desc for w in ["slow", "performance", "timeout", "leak", "memory"]) else 0.0)
    f.append(1.0 if any(w in desc for w in ["css", "ui", "button", "layout", "dark mode", "contrast", "display"]) else 0.0)
    f.append(1.0 if any(w in desc for w in ["api", "server", "backend", "payment", "webhook", "csv", "export"]) else 0.0)
    f.append(1.0 if any(w in desc for w in ["mobile", "ios", "android", "app crash"]) else 0.0)
    f.append(1.0 if any(w in desc for w in ["database", "sql", "query", "index", "postgres", "node"]) else 0.0)
    f.append(1.0 if any(w in desc for w in ["auth", "token", "session", "jwt", "2fa", "login"]) else 0.0)
    sla = bug.get("sla_hours_remaining")
    f.append(1.0 if sla is not None and sla < 2.0 else 0.0)
    tier = bug.get("customer_tier", "free")
    f.append(1.0 if tier in ("enterprise", "business") else 0.0)
    f.append(1.0 if not bug.get("steps_to_reproduce") and not bug.get("environment_info") else 0.0)
    # Spam features (3 additional dimensions)
    product = bug.get("product", "").lower()
    spam_product = 1.0 if any(sp in product for sp in _SPAM_PRODUCTS_RL) else 0.0
    spam_keyword_count = sum(1 for kw in _SPAM_KEYWORDS_RL if kw in desc)
    spam_kw = min(1.0, spam_keyword_count / 3.0)  # normalize: 3+ keywords = 1.0
    alpha_chars = [c for c in bug.get("description", "") if c.isalpha()]
    caps_ratio = (sum(1 for c in alpha_chars if c.isupper()) / max(1, len(alpha_chars)))
    spam_caps = 1.0 if caps_ratio > 0.5 else 0.0
    f.extend([spam_product, spam_kw, spam_caps])
    return np.array(f, dtype=np.float64)

BUG_DIM = 16  # 13 original + 3 spam features

def encode_assign(bug: Dict, severity_idx: int) -> np.ndarray:
    base = encode_bug(bug)
    sev = np.zeros(len(SEVERITIES), dtype=np.float64)
    if 0 <= severity_idx < len(SEVERITIES):
        sev[severity_idx] = 1.0
    return np.concatenate([base, sev])

def encode_duplicate(bug: Dict, max_sim: float, avg_sim: float) -> np.ndarray:
    return np.concatenate([encode_bug(bug), [max_sim, avg_sim]])


def find_dup_target(bug: Dict, candidates: List[Dict]) -> Tuple[Optional[str], float, float]:
    if not candidates:
        return None, 0.0, 0.0
    sims = [(c.get("id", ""), _sim(bug, c)) for c in candidates]
    sims.sort(key=lambda x: -x[1])
    max_sim = sims[0][1]
    avg_sim = sum(s for _, s in sims) / len(sims)
    target_id = sims[0][0] if max_sim > 0.15 else None
    return target_id, max_sim, avg_sim


# ═══════════════════════════════════════════════════════════════════════════
# Two-Phase Episode Runner
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class EpisodeResult:
    score: float = 0.0
    total_reward: float = 0.0
    steps: int = 0
    components: Dict[str, float] = field(default_factory=dict)


def run_episode(
    session: requests.Session,
    nets: Dict[str, PolicyNet],
    task: str,
    epsilon: float,
    train: bool = True,
    ep_idx: int = 0,
) -> EpisodeResult:
    result = EpisodeResult()
    seed = (ep_idx * 7919 + hash(task)) % 99999 + 1 if DYNAMIC_SEEDS else None  # varied per episode
    reset_body = {"task": task}
    if seed is not None:
        reset_body["seed"] = seed
    reset_resp = session.post(f"{ENV_URL}/reset", json=reset_body, timeout=30).json()
    obs = reset_resp.get("observation", {})

    max_steps_map = {"single-triage": 5, "batch-triage": 32, "sla-crisis": 50, "adversarial-triage": 65}
    max_steps = max_steps_map.get(task, 65)

    bug_reports = obs.get("bug_reports", [])
    all_bug_ids = [b.get("id", "") for b in bug_reports]
    bug_by_id = {b.get("id", ""): b for b in bug_reports}

    step_count = 0
    done = False

    def do_step(action: Dict) -> float:
        """Execute step, update obs, return reward. Sets done flag."""
        nonlocal obs, step_count, done
        resp = session.post(f"{ENV_URL}/step", json=action, timeout=30).json()
        obs = resp.get("observation", obs)
        rew = resp.get("reward", 0.0)
        result.total_reward += rew
        step_count += 1
        done = resp.get("done", False)
        return rew

    # ── Phase 0 (adversarial only): Flag spam ─────────────────────────────
    if task == "adversarial-triage":
        for bug_id in all_bug_ids:
            if done or step_count >= max_steps - len(all_bug_ids) * 2:
                break
            bug = bug_by_id.get(bug_id, {})
            state_s = encode_bug(bug)
            ai_spam, _ = nets["spam"].select(state_s, epsilon if train else 0.0)
            if ai_spam == 1:
                rew = do_step({
                    "action_type": "flag_spam", "bug_id": bug_id,
                    "spam_reason": "Spam indicators detected"
                })
                if train:
                    nets["spam"].update(state_s, ai_spam, rew)
                if done:
                    break
            else:
                if train:
                    nets["spam"].update(state_s, ai_spam, 0.01)

    # Refresh unprocessed list after spam flagging
    flagged_spam = set(obs.get("flagged_spam_ids", []))
    real_bug_ids = [bid for bid in all_bug_ids if bid not in flagged_spam]

    # ── Phase 1: Classify + Assign all bugs ──────────────────────────────
    for bug_id in real_bug_ids:
        if done or step_count >= max_steps - len(real_bug_ids):
            break
        bug = bug_by_id.get(bug_id, {})

        # Classify
        state_c = encode_bug(bug)
        si, _ = nets["classify"].select(state_c, epsilon if train else 0.0)
        rew = do_step({"action_type": "classify", "bug_id": bug_id, "severity": SEVERITIES[si]})
        if train:
            nets["classify"].update(state_c, si, rew)
        if done:
            break

        # Assign
        state_a = encode_assign(bug, si)
        ti, _ = nets["assign"].select(state_a, epsilon if train else 0.0)
        rew = do_step({"action_type": "assign", "bug_id": bug_id, "assigned_team": TEAMS[ti]})
        if train:
            nets["assign"].update(state_a, ti, rew)

    # ── Phase 2: Info / Escalate / Duplicate (full context) ──────────────
    for bug_id in real_bug_ids:
        if done or step_count >= max_steps - len(all_bug_ids):
            break
        bug = bug_by_id.get(bug_id, {})

        # Info request decision
        state_i = encode_bug(bug)
        ai_info, _ = nets["info"].select(state_i, epsilon if train else 0.0)
        if ai_info == 1:
            rew = do_step({
                "action_type": "request_info", "bug_id": bug_id,
                "info_requested": ["steps_to_reproduce", "environment_info"]
            })
            if train:
                nets["info"].update(state_i, ai_info, rew)
            if done:
                break
        else:
            if train:
                nets["info"].update(state_i, ai_info, 0.02)

        # Escalation decision
        state_e = encode_bug(bug)
        ai_esc, _ = nets["escalate"].select(state_e, epsilon if train else 0.0)
        if ai_esc == 1:
            rew = do_step({
                "action_type": "escalate", "bug_id": bug_id,
                "escalation_reason": "SLA/severity/enterprise"
            })
            if train:
                nets["escalate"].update(state_e, ai_esc, rew)
            if done:
                break
        else:
            if train:
                nets["escalate"].update(state_e, ai_esc, 0.01)

        # Duplicate detection
        other_bugs = [bug_by_id[bid] for bid in all_bug_ids if bid != bug_id]
        target_id, max_sim, avg_sim = find_dup_target(bug, other_bugs)
        state_d = encode_duplicate(bug, max_sim, avg_sim)
        ai_dup, _ = nets["duplicate"].select(state_d, epsilon if train else 0.0)
        if ai_dup == 1 and target_id:
            rew = do_step({
                "action_type": "mark_duplicate", "bug_id": bug_id,
                "duplicate_of": target_id
            })
            if train:
                nets["duplicate"].update(state_d, ai_dup, rew)
            if done:
                break
        else:
            if train:
                nets["duplicate"].update(state_d, ai_dup, 0.01)

    # ── Phase 3: Submit all (real bugs only) ─────────────────────────────
    for bug_id in real_bug_ids:
        if done or step_count >= max_steps:
            break
        submitted = obs.get("submitted_bug_ids", [])
        if bug_id in submitted:
            continue
        do_step({"action_type": "submit", "bug_id": bug_id})

    # Final grade
    grade = session.post(f"{ENV_URL}/grade", timeout=30).json()
    result.score = grade.get("score", 0.0)
    result.steps = step_count
    result.components = grade.get("components", {})
    return result


# ═══════════════════════════════════════════════════════════════════════════
# Exploration schedule
# ═══════════════════════════════════════════════════════════════════════════

def epsilon_schedule(ep: int, total: int) -> float:
    decay_eps = total * EPS_DECAY_FRAC
    if ep >= decay_eps:
        return EPS_END
    t = ep / decay_eps
    return EPS_END + (EPS_START - EPS_END) * math.exp(-4.0 * t)


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    global TASK, DYNAMIC_SEEDS, N_EPISODES

    # ── Server health check ──────────────────────────────────
    try:
        h = requests.get(f"{ENV_URL}/health", timeout=10).json()
        print(f"\n✅ Server: {h['status']} v{h['version']}")
    except Exception as e:
        print(f"\n❌ Cannot reach server at {ENV_URL}: {e}")
        print(f"   Start: cd bugtriage-openenv && uvicorn app.server:app --port 7860")
        sys.exit(1)

    # ── Interactive task selection ────────────────────────────
    if not os.getenv("RL_TASK"):
        print("\n  Select a task to train on:")
        print("  [1] single-triage       (Easy   — 1 bug,  5 steps)")
        print("  [2] batch-triage        (Medium — 8 bugs, 32 steps)")
        print("  [3] sla-crisis          (Hard   — 15 bugs, 50 steps)")
        print("  [4] adversarial-triage  (Expert — 20 bugs, 65 steps)")
        try:
            choice = input("\n  Enter 1, 2, or 3 (default=2): ").strip()
            if choice == "1":
                TASK = "single-triage"
            elif choice == "3":
                TASK = "sla-crisis"
            elif choice == "4":
                TASK = "adversarial-triage"
            else:
                TASK = "batch-triage"
        except (EOFError, KeyboardInterrupt):
            pass
        print(f"  → Training on: {TASK}")

        # Ask about dynamic scenarios
        try:
            dyn = input("  Use dynamic scenarios? (y/N): ").strip().lower()
            if dyn in ("y", "yes"):
                DYNAMIC_SEEDS = True
                print("  → Dynamic mode: ON (different bugs each episode)")
            else:
                DYNAMIC_SEEDS = False
                print("  → Dynamic mode: OFF (same bugs each episode)")
        except (EOFError, KeyboardInterrupt):
            pass

        # Ask about episodes
        try:
            eps_input = input(f"  Number of episodes? (default={N_EPISODES}): ").strip()
            if eps_input.isdigit() and int(eps_input) > 0:
                N_EPISODES = int(eps_input)
        except (EOFError, KeyboardInterrupt):
            pass

    nets = {
        "classify":  PolicyNet(BUG_DIM, 32, len(SEVERITIES), LR, "Classify"),
        "assign":    PolicyNet(BUG_DIM + len(SEVERITIES), 32, len(TEAMS), LR, "Assign"),
        "info":      PolicyNet(BUG_DIM, 16, 2, LR, "Info"),
        "escalate":  PolicyNet(BUG_DIM, 16, 2, LR, "Escalate"),
        "duplicate": PolicyNet(BUG_DIM + 2, 16, 2, LR, "Duplicate"),
        "spam":      PolicyNet(BUG_DIM, 16, 2, LR, "Spam"),
    }
    total_params = sum(n.param_count for n in nets.values())

    print(f"\n{'='*70}")
    print(f"  ENTERPRISE RL TRAINING — BugTriage OpenEnv")
    print(f"  Task: {TASK}  |  Episodes: {N_EPISODES}  |  Params: {total_params}")
    print(f"{'='*70}")
    print(f"  Algorithm: REINFORCE + EMA baseline + entropy (β={ENTROPY_BETA})")
    print(f"  Policy heads: classify, assign, info, escalate, duplicate, spam")
    print(f"  Episode flow: {'Flag Spam → ' if TASK == 'adversarial-triage' else ''}Classify All → Assign All → Info/Esc/Dup → Submit All")
    print(f"  Duplicate targeting: Jaccard text similarity (no hardcoded targets)")
    print(f"  Exploration: ε = {EPS_START:.0%} → {EPS_END:.0%} over {int(N_EPISODES * EPS_DECAY_FRAC)} eps")
    print(f"  LR: {LR}")

    session = requests.Session()

    # Before training
    print(f"\n{'─'*70}")
    print(f"BEFORE training (random weights, ε=0):")
    before = run_episode(session, nets, TASK, 0.0, train=False, ep_idx=0)
    print(f"  Score: {before.score:.3f}  |  Components: {before.components}")

    # Training loop
    print(f"\n{'─'*70}")
    print(f"TRAINING ({N_EPISODES} episodes)\n")
    is_adv = TASK == "adversarial-triage"
    if is_adv:
        print(f"  {'Ep':>5}  {'Score':>6}  {'Avg50':>6}  {'ε':>5}  "
              f"{'Sev':>5}  {'Team':>5}  {'Dup':>5}  {'Esc':>5}  {'Spam':>5}  {'Progress'}")
        print(f"  {'─'*5}  {'─'*6}  {'─'*6}  {'─'*5}  "
              f"{'─'*5}  {'─'*5}  {'─'*5}  {'─'*5}  {'─'*5}  {'─'*22}")
    else:
        print(f"  {'Ep':>5}  {'Score':>6}  {'Avg50':>6}  {'ε':>5}  "
              f"{'Sev':>5}  {'Team':>5}  {'Dup':>5}  {'Esc':>5}  {'Info':>5}  {'Progress'}")
        print(f"  {'─'*5}  {'─'*6}  {'─'*6}  {'─'*5}  "
              f"{'─'*5}  {'─'*5}  {'─'*5}  {'─'*5}  {'─'*5}  {'─'*22}")

    scores = []
    best_avg = 0.0
    start = time.time()

    for ep in range(1, N_EPISODES + 1):
        eps = epsilon_schedule(ep, N_EPISODES)
        result = run_episode(session, nets, TASK, eps, train=True, ep_idx=ep)
        scores.append(result.score)

        if ep % PRINT_EVERY == 0 or ep <= 3:
            avg50 = np.mean(scores[-50:]) if len(scores) >= 50 else np.mean(scores)
            c = result.components
            filled = int(avg50 * 20)
            bar = "█" * filled + "░" * (20 - filled)
            marker = ""
            if avg50 > best_avg:
                best_avg = avg50
                if avg50 > 0.5: marker = " ★★"
                elif avg50 > 0.4: marker = " ★"

            if is_adv:
                print(f"  {ep:>5}  {result.score:>6.3f}  {avg50:>6.3f}  {eps:>5.2f}  "
                      f"{c.get('severity',0):>5.2f}  {c.get('team',0):>5.2f}  "
                      f"{c.get('duplicate_detection',0):>5.2f}  "
                      f"{c.get('sla_escalations',0):>5.2f}  "
                      f"{c.get('spam_detection',0):>5.2f}  "
                      f"|{bar}|{marker}")
            else:
                print(f"  {ep:>5}  {result.score:>6.3f}  {avg50:>6.3f}  {eps:>5.2f}  "
                      f"{c.get('severity',0):>5.2f}  {c.get('team',0):>5.2f}  "
                      f"{c.get('duplicate_detection',0):>5.2f}  "
                      f"{c.get('security_escalation', c.get('sla_escalations',0)):>5.2f}  "
                      f"{c.get('info_request', c.get('info_requests',0)):>5.2f}  "
                      f"|{bar}|{marker}")

    elapsed = time.time() - start

    # After training
    print(f"\n{'─'*70}")
    print(f"AFTER training (ε=0, pure exploitation):")
    after = run_episode(session, nets, TASK, 0.0, train=False, ep_idx=0)
    print(f"  Score: {after.score:.3f}")
    print(f"  Components: {after.components}")

    # Summary
    first50 = np.mean(scores[:50])
    last50 = np.mean(scores[-50:])
    improvement = last50 - first50
    window = max(1, N_EPISODES // 25)
    smooth = [np.mean(scores[max(0, i - window):i + 1]) for i in range(len(scores))]

    print(f"\nScore curve over {N_EPISODES} episodes:")
    rows, width = 8, 60
    for row in range(rows - 1, -1, -1):
        thresh = row / (rows - 1)
        line = ""
        for col in range(width):
            idx = int(col * len(smooth) / width)
            line += "█" if smooth[idx] >= thresh - 0.02 else " "
        print(f"  {thresh:.1f}│{line}")
    print(f"  0.0│{'─' * width}")
    print(f"     ep1{'':>{width - 8}}ep{N_EPISODES}")

    print(f"\n{'='*70}")
    print(f"  RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"  Task:          {TASK}")
    print(f"  Episodes:      {N_EPISODES}")
    print(f"  First 50 avg:  {first50:.3f}")
    print(f"  Last 50 avg:   {last50:.3f}")
    print(f"  Improvement:   {improvement:+.3f}  "
          f"{'✅ STRONG' if improvement > 0.15 else '✅ LEARNING' if improvement > 0.05 else '⚠️  marginal'}")
    print(f"  Peak (smooth): {max(smooth):.3f}")
    print(f"  Final greedy:  {after.score:.3f}")
    print(f"  Time:          {elapsed:.1f}s ({elapsed / N_EPISODES * 1000:.0f}ms/ep)")
    print(f"\n  Component breakdown (final greedy):")
    for k, v in sorted(after.components.items()):
        bar = "█" * int(v * 40)
        print(f"    {k:25} {bar:40} {v:.3f}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
