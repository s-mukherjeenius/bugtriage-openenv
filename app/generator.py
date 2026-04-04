"""
BugTriage OpenEnv — Dynamic Scenario Generator
================================================
Generates randomised but structurally valid bug-triage scenarios from
rich pools of realistic bug report templates.

Each generated scenario has:
  - Deterministic output for a given seed (reproducible)
  - Ground truth that is inferable from the bug text (no hidden tricks)
  - Correct difficulty structure (bug count, dup pairs, SLA breaches, etc.)

Usage:
    from app.generator import generate_scenario
    scenario = generate_scenario("batch-triage", seed=42)
"""
from __future__ import annotations

import random
import string
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from app.models import BugReport
from app.scenarios import BugGroundTruth, TaskScenario


# ═══════════════════════════════════════════════════════════════════════════
# Bug template pools — organised by team
# Each entry: (severity, title, description_template, product, escalation?)
# Description templates use {version}, {reporter}, {product} placeholders.
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class _BugTemplate:
    team: str
    severity: str
    title: str
    description: str
    product: str
    should_escalate: bool = False
    sla_critical: bool = False

# ── Backend bugs ──────────────────────────────────────────────────────────

_BACKEND_BUGS: List[_BugTemplate] = [
    _BugTemplate(
        team="backend", severity="critical",
        title="Payment gateway returning HTTP 500 on all transactions",
        description=(
            "Our payment processing pipeline is completely down since {time}. "
            "Every checkout attempt fails with HTTP 500 and 'upstream connect error'. "
            "The PaymentService.processCharge() method is throwing NullPointerException. "
            "This is affecting 100% of users across all plans. Revenue loss is approximately "
            "$8,000 per minute. No deployment was made in the last 6 hours. P0 incident."
        ),
        product="Checkout API",
        should_escalate=True, sla_critical=True,
    ),
    _BugTemplate(
        team="backend", severity="critical",
        title="All API endpoints returning 503 Service Unavailable",
        description=(
            "Production API is completely unreachable. Every request returns 503. "
            "Load balancer health checks are failing on all backend instances. "
            "Application logs show OutOfMemoryError in the main service pool. "
            "All customers affected, including enterprise accounts. This is a P0 — "
            "100% of traffic is being dropped."
        ),
        product="Core API",
        should_escalate=True, sla_critical=True,
    ),
    _BugTemplate(
        team="backend", severity="high",
        title="Webhook delivery failing for Stripe payment events",
        description=(
            "Stripe webhook processor has been silently dropping events since last night. "
            "OrderCompletionHandler throws ClassCastException when processing refund events. "
            "Approximately 340 orders are stuck in 'pending' state. Customers are getting "
            "charged but not receiving confirmation emails. Affects ~15% of transactions."
        ),
        product="Billing Service",
        should_escalate=False, sla_critical=False,
    ),
    _BugTemplate(
        team="backend", severity="high",
        title="CSV export producing corrupted files for large datasets",
        description=(
            "When exporting reports with more than 10,000 rows, the CSV file is truncated "
            "at approximately row 8,192. The file appears to cut off mid-line, making it "
            "unreadable in Excel. The export endpoint returns HTTP 200 but the Content-Length "
            "header doesn't match the actual body size. Affects all business-tier customers "
            "who use bulk exports."
        ),
        product="Analytics API",
        should_escalate=False, sla_critical=False,
    ),
    _BugTemplate(
        team="backend", severity="medium",
        title="Search API returning stale results after index update",
        description=(
            "After updating the product catalog, search results don't reflect changes for "
            "approximately 45 minutes. The Elasticsearch reindex job completes successfully "
            "but the cache layer (Redis) isn't being invalidated. Users see outdated prices "
            "and descriptions. Workaround: manually flush the cache via admin panel."
        ),
        product="Search Service",
        should_escalate=False, sla_critical=False,
    ),
    _BugTemplate(
        team="backend", severity="medium",
        title="Rate limiter incorrectly blocking legitimate API requests",
        description=(
            "The rate limiting middleware is counting OPTIONS preflight requests toward "
            "the rate limit quota. This causes legitimate POST requests from browser-based "
            "clients to get 429 Too Many Requests after only a few operations. Workaround: "
            "increase rate limit from 100 to 200 requests per minute."
        ),
        product="API Gateway",
        should_escalate=False, sla_critical=False,
    ),
]

# ── Frontend bugs ─────────────────────────────────────────────────────────

_FRONTEND_BUGS: List[_BugTemplate] = [
    _BugTemplate(
        team="frontend", severity="medium",
        title="Dark mode toggle breaks navigation sidebar layout",
        description=(
            "When switching from light mode to dark mode, the left navigation sidebar "
            "collapses to 0px width and all menu items stack vertically off-screen. "
            "The CSS transition is applying width:0 before the color variables update. "
            "Reproducible on Chrome 121+ and Firefox 122+. Light mode works fine. "
            "Workaround: refresh the page after toggling."
        ),
        product="Web Dashboard",
        should_escalate=False, sla_critical=False,
    ),
    _BugTemplate(
        team="frontend", severity="low",
        title="Tooltip text wrapping incorrectly on action buttons",
        description=(
            "The tooltip on the 'Export Report' button in the analytics dashboard wraps "
            "to three lines instead of displaying as a single line. The tooltip container "
            "has max-width:120px which is too narrow for the text. Purely cosmetic issue. "
            "Affects all browsers."
        ),
        product="Web Dashboard",
        should_escalate=False, sla_critical=False,
    ),
    _BugTemplate(
        team="frontend", severity="low",
        title="Company logo appears pixelated on retina displays",
        description=(
            "The header logo in the main navigation is using a 1x PNG image. On retina "
            "and HiDPI displays (MacBook Pro, 4K monitors), the logo appears blurry and "
            "pixelated. Need to serve a 2x or SVG version. Minor visual issue only — no "
            "functional impact."
        ),
        product="Marketing Site",
        should_escalate=False, sla_critical=False,
    ),
    _BugTemplate(
        team="frontend", severity="medium",
        title="Form validation messages not visible against dark background",
        description=(
            "Error messages below form fields use red text (#CC0000) on the dark blue "
            "background (#1A1A2E), resulting in a contrast ratio of only 2.1:1. This fails "
            "WCAG AA requirements. Users with visual impairments cannot read validation "
            "errors. Affects the signup and login pages."
        ),
        product="Auth Portal",
        should_escalate=False, sla_critical=False,
    ),
    _BugTemplate(
        team="frontend", severity="low",
        title="Button alignment off by 3px in modal dialogs",
        description=(
            "The 'Cancel' and 'Confirm' buttons in all modal dialogs are misaligned by 3px "
            "vertically. The Cancel button sits slightly lower than Confirm. This is caused "
            "by inconsistent padding values in the button component CSS. Purely cosmetic — "
            "buttons work correctly."
        ),
        product="Design System",
        should_escalate=False, sla_critical=False,
    ),
]

# ── Security bugs ─────────────────────────────────────────────────────────

_SECURITY_BUGS: List[_BugTemplate] = [
    _BugTemplate(
        team="security", severity="critical",
        title="Authentication bypass via crafted JWT token",
        description=(
            "A malicious actor can forge valid session tokens by exploiting a weakness "
            "in the JWT signature verification. The auth middleware accepts tokens signed "
            "with the 'none' algorithm when the header specifies alg:none. This allows "
            "complete authentication bypass. Proof of concept attached. All user accounts "
            "are vulnerable. This is a critical security breach requiring immediate action."
        ),
        product="Auth Service",
        should_escalate=True, sla_critical=True,
    ),
    _BugTemplate(
        team="security", severity="critical",
        title="SQL injection vulnerability in user search endpoint",
        description=(
            "The /api/v2/users/search endpoint is vulnerable to SQL injection via the "
            "'query' parameter. Unparameterized query: SELECT * FROM users WHERE name LIKE "
            "'%{input}%'. An attacker can extract the entire users table including hashed "
            "passwords. Active exploitation suspected — unusual query patterns detected in "
            "access logs. All user data is at risk."
        ),
        product="User API",
        should_escalate=True, sla_critical=True,
    ),
    _BugTemplate(
        team="security", severity="high",
        title="Session tokens not invalidated after password reset",
        description=(
            "When a user resets their password, existing session tokens remain valid "
            "indefinitely. An attacker who obtained a session token before the password "
            "reset can continue accessing the account. The token revocation job is not "
            "being triggered by the password reset flow. Affects all authentication methods."
        ),
        product="Auth Service",
        should_escalate=True, sla_critical=False,
    ),
    _BugTemplate(
        team="security", severity="high",
        title="Rate limit bypass allows brute-force login attempts",
        description=(
            "The login rate limiter can be bypassed by rotating the X-Forwarded-For header. "
            "An attacker can make unlimited login attempts by sending each request with a "
            "different spoofed IP. The rate limiter trusts the X-Forwarded-For header without "
            "validating it against the actual client IP. Brute-force attacks are possible."
        ),
        product="API Gateway",
        should_escalate=True, sla_critical=False,
    ),
]

# ── Database bugs ─────────────────────────────────────────────────────────

_DATABASE_BUGS: List[_BugTemplate] = [
    _BugTemplate(
        team="database", severity="high",
        title="PostgreSQL query planner choosing sequential scan on large table",
        description=(
            "The reports query is performing a full sequential scan on the events table "
            "(450M rows) instead of using the composite index on (tenant_id, created_at). "
            "Query time has increased from 200ms to 45 seconds. EXPLAIN ANALYZE shows the "
            "planner estimates 1 row but actual rows returned is 50,000. Statistics are stale. "
            "VACUUM ANALYZE hasn't run in 3 weeks."
        ),
        product="Analytics DB",
        should_escalate=False, sla_critical=False,
    ),
    _BugTemplate(
        team="database", severity="critical",
        title="Database connection pool exhausted — all queries timing out",
        description=(
            "The connection pool is completely saturated at 100/100 connections. All new "
            "queries are timing out after 30 seconds. Application logs show 'Cannot acquire "
            "connection from pool' errors. A long-running migration lock appears to be holding "
            "connections. All services depending on the primary database are affected. "
            "Production is down."
        ),
        product="Core Database",
        should_escalate=True, sla_critical=True,
    ),
    _BugTemplate(
        team="database", severity="medium",
        title="Duplicate records appearing in user_preferences table",
        description=(
            "The user_preferences table has duplicate rows for approximately 2% of users. "
            "The unique constraint on (user_id, preference_key) was accidentally dropped "
            "during last month's migration. The application layer doesn't enforce uniqueness. "
            "Queries return unpredictable results depending on which duplicate is selected."
        ),
        product="User Service DB",
        should_escalate=False, sla_critical=False,
    ),
]

# ── Infrastructure bugs ───────────────────────────────────────────────────

_INFRA_BUGS: List[_BugTemplate] = [
    _BugTemplate(
        team="infrastructure", severity="critical",
        title="Primary database node unreachable — cluster failover incomplete",
        description=(
            "The primary DB node (db-primary-01) became unreachable at 03:15 UTC. "
            "Automatic failover to the replica was initiated but did not complete — "
            "the replica is stuck in read-only mode. All write operations are failing. "
            "The monitoring dashboard shows the node as 'unreachable' for 47 minutes. "
            "This is a P0 infrastructure incident affecting all services."
        ),
        product="Database Cluster",
        should_escalate=True, sla_critical=True,
    ),
    _BugTemplate(
        team="infrastructure", severity="high",
        title="Kubernetes pod evictions causing service restarts every 20 minutes",
        description=(
            "Worker pods in the processing namespace are being evicted every 20 minutes "
            "due to memory pressure on node pool 'workers-a'. The pods are hitting the "
            "memory limit of 2Gi but the actual workload requires 3Gi. Each eviction "
            "causes a 90-second service interruption while the pod reschedules."
        ),
        product="K8s Cluster",
        should_escalate=False, sla_critical=False,
    ),
    _BugTemplate(
        team="infrastructure", severity="medium",
        title="Deployment pipeline failing intermittently on staging",
        description=(
            "The CI/CD pipeline for the staging environment fails approximately 30% of "
            "the time with a 'container image pull timeout' error. The container registry "
            "appears to be throttling requests. Retry usually succeeds. Does not affect "
            "production deployments which use a different registry mirror."
        ),
        product="CI/CD Pipeline",
        should_escalate=False, sla_critical=False,
    ),
]

# ── Mobile bugs ───────────────────────────────────────────────────────────

_MOBILE_BUGS: List[_BugTemplate] = [
    _BugTemplate(
        team="mobile", severity="high",
        title="iOS app crashes on launch after updating to iOS 18",
        description=(
            "Users who updated to iOS 18 are experiencing immediate crashes on app launch. "
            "The crash report shows EXC_BAD_ACCESS in UIKit's layout engine when the app "
            "tries to render the home screen. Approximately 12% of our iOS user base is "
            "affected. The app is unusable for these users — they cannot get past the "
            "splash screen."
        ),
        product="iOS App",
        should_escalate=False, sla_critical=False,
    ),
    _BugTemplate(
        team="mobile", severity="medium",
        title="Android push notifications arriving with 15-minute delay",
        description=(
            "Push notifications on Android devices are arriving with a consistent 15-minute "
            "delay. The FCM delivery receipts show messages are being queued by the OS "
            "doze mode. Our notification priority is set to 'normal' instead of 'high'. "
            "Affects Android 12+ devices. iOS notifications are delivered instantly."
        ),
        product="Android App",
        should_escalate=False, sla_critical=False,
    ),
    _BugTemplate(
        team="mobile", severity="low",
        title="App icon badge count not clearing after reading notifications",
        description=(
            "The app icon badge on both iOS and Android continues to show unread count "
            "even after the user has opened and read all notifications. The count resets "
            "only when force-closing and reopening the app. Minor annoyance — notifications "
            "themselves work correctly."
        ),
        product="Mobile App",
        should_escalate=False, sla_critical=False,
    ),
]

# ── All pools merged ──────────────────────────────────────────────────────

_ALL_BUGS: List[_BugTemplate] = (
    _BACKEND_BUGS + _FRONTEND_BUGS + _SECURITY_BUGS
    + _DATABASE_BUGS + _INFRA_BUGS + _MOBILE_BUGS
)

_BUGS_BY_SEVERITY: Dict[str, List[_BugTemplate]] = {}
for _b in _ALL_BUGS:
    _BUGS_BY_SEVERITY.setdefault(_b.severity, []).append(_b)

_BUGS_BY_TEAM: Dict[str, List[_BugTemplate]] = {}
for _b in _ALL_BUGS:
    _BUGS_BY_TEAM.setdefault(_b.team, []).append(_b)


# ═══════════════════════════════════════════════════════════════════════════
# Utility pools
# ═══════════════════════════════════════════════════════════════════════════

_REPORTERS = [
    "alice.chen@company.com", "bob.smith@company.com", "carla.dev@company.com",
    "david.ops@company.com", "elena.qa@company.com", "frank.sre@company.com",
    "grace.pm@company.com", "henry.sec@company.com", "iris.eng@company.com",
    "jake.support@company.com", "karen.leads@company.com", "leo.infra@company.com",
    "mia.mobile@company.com", "noah.data@company.com", "olivia.fe@company.com",
]

_TIERS: List[str] = ["enterprise", "enterprise", "business", "business", "starter", "free"]

_VERSIONS = [
    "v1.0.3", "v1.2.0", "v2.0.1", "v2.3.4", "v3.0.0", "v3.1.2",
    "v3.5.1", "v4.0.0-beta", "v4.1.0", "v4.2.3",
]

_TIMESTAMPS_BASE = [
    "2024-{month:02d}-{day:02d}T{hour:02d}:{minute:02d}:00Z"
]

_STEPS_TO_REPRODUCE_POOL = [
    "1. Log in to the application\n2. Navigate to {product}\n3. Perform the action\nResult: Error occurs",
    "1. Open {product} in browser\n2. Click the relevant button\n3. Observe the failure\nExpected: Success\nActual: Error",
    "1. Access {product} via API\n2. Send the request\n3. Check the response\nResult: Unexpected error code",
]

_ENV_INFO_POOL = [
    {"os": "Ubuntu 22.04", "runtime": "Python 3.11", "region": "us-east-1"},
    {"os": "macOS 14.2", "browser": "Chrome 121", "region": "eu-west-1"},
    {"os": "Windows 11", "browser": "Firefox 122", "region": "us-west-2"},
    {"os": "Alpine 3.19", "runtime": "Java 21", "region": "ap-southeast-1"},
    {"os": "Amazon Linux 2", "runtime": "Node.js 20", "region": "us-east-2"},
]


# ═══════════════════════════════════════════════════════════════════════════
# Duplicate pair generation
# ═══════════════════════════════════════════════════════════════════════════

# Each entry: (original_template_index, variant_title, variant_description)
# The variant describes the SAME root cause with different wording.

_DUPLICATE_VARIANTS: Dict[str, List[Tuple[str, str]]] = {
    "backend": [
        (
            "Checkout flow broken — orders not completing",
            "Tried to place an order but the checkout just spins and returns an error. "
            "Multiple customers have reported the same issue in the last hour. The payment "
            "page shows a generic 'Something went wrong' message. Seems like the same issue "
            "as the payment gateway failure reported earlier.",
        ),
        (
            "Billing webhook events not being processed",
            "Noticed that recent Stripe events are piling up in the dead letter queue. "
            "The webhook handler appears to be rejecting events with an unexpected format. "
            "This looks like it might be related to the webhook delivery failure that was "
            "reported — same symptoms, different reporter.",
        ),
    ],
    "frontend": [
        (
            "Sidebar disappears when switching themes",
            "After changing the theme from light to dark, the entire left navigation panel "
            "vanishes. The page content shifts to fill the full width. Toggling back to "
            "light mode doesn't restore it. Have to reload the page. Seems identical to "
            "the dark mode layout issue filed yesterday.",
        ),
    ],
    "security": [
        (
            "Forged tokens accepted by API — authentication broken",
            "Our penetration test discovered that the API accepts JWT tokens with alg:none "
            "in the header. This allows creating arbitrary sessions without credentials. "
            "Likely the same root cause as the JWT bypass vulnerability already reported.",
        ),
    ],
    "database": [
        (
            "Slow queries on the reporting page — everything times out",
            "The weekly report generation is timing out after 60 seconds. The dashboard "
            "shows the query hitting a full table scan on the events table. This seems "
            "related to the PostgreSQL performance issue flagged earlier — same table, "
            "same symptoms.",
        ),
    ],
    "infrastructure": [
        (
            "Services crashing due to database connectivity issues",
            "Multiple microservices are logging 'connection refused' errors when trying "
            "to reach the database. The connection pool is maxed out and new connections "
            "are being rejected. This appears to be the same incident as the primary DB "
            "node failure that was already reported.",
        ),
    ],
}


# ═══════════════════════════════════════════════════════════════════════════
# Generator core
# ═══════════════════════════════════════════════════════════════════════════

def _make_timestamp(rng: random.Random, month: int = 1) -> str:
    day = rng.randint(1, 28)
    hour = rng.randint(0, 23)
    minute = rng.randint(0, 59)
    return f"2024-{month:02d}-{day:02d}T{hour:02d}:{minute:02d}:00Z"


def _make_bug_id(prefix: str, index: int) -> str:
    return f"{prefix}-{index:03d}"


def _template_to_bug(
    rng: random.Random,
    template: _BugTemplate,
    bug_id: str,
    timestamp: str,
    customer_tier: str,
    sla_hours: Optional[float] = None,
    strip_info: bool = False,
) -> Tuple[BugReport, BugGroundTruth]:
    """Convert a template into a BugReport + ground truth pair."""

    reporter = rng.choice(_REPORTERS)
    version = rng.choice(_VERSIONS)

    # Generate steps_to_reproduce (unless stripped for info-incomplete bugs)
    steps = None
    if not strip_info:
        base = rng.choice(_STEPS_TO_REPRODUCE_POOL)
        steps = base.format(product=template.product)

    # Generate environment_info (unless stripped)
    env_info = None
    if not strip_info:
        env_info = rng.choice(_ENV_INFO_POOL).copy()

    # SLA: critical/high bugs from enterprise get tight SLAs
    if sla_hours is None:
        if template.severity == "critical":
            sla_hours = round(rng.uniform(0.5, 2.0), 1)
        elif template.severity == "high" and customer_tier == "enterprise":
            sla_hours = round(rng.uniform(2.0, 8.0), 1)
        elif template.severity == "high":
            sla_hours = round(rng.uniform(8.0, 24.0), 1)
        else:
            sla_hours = None  # no SLA for medium/low

    # Determine escalation based on actual conditions
    should_escalate = template.should_escalate
    sla_critical = False
    if sla_hours is not None and sla_hours < 2.0:
        sla_critical = True
        if template.severity in ("critical", "high"):
            should_escalate = True
    if customer_tier == "enterprise" and template.severity in ("critical", "high"):
        should_escalate = True

    bug = BugReport(
        id=bug_id,
        title=template.title,
        description=template.description.replace("{time}", timestamp[:16]),
        reporter=reporter,
        timestamp=timestamp,
        product=template.product,
        version=version,
        steps_to_reproduce=steps,
        expected_behavior="Normal operation without errors",
        actual_behavior=f"Failure as described: {template.title}",
        environment_info=env_info,
        customer_tier=customer_tier,
        sla_hours_remaining=sla_hours,
        linked_bug_ids=None,
    )

    gt = BugGroundTruth(
        severity=template.severity,
        team=template.team,
        needs_info=strip_info,
        is_duplicate_of=None,
        should_escalate=should_escalate,
        sla_critical=sla_critical,
    )

    return bug, gt


def _make_duplicate(
    rng: random.Random,
    original_bug: BugReport,
    original_gt: BugGroundTruth,
    dup_id: str,
    team: str,
) -> Tuple[BugReport, BugGroundTruth]:
    """Create a duplicate bug that describes the same root cause differently."""

    variants = _DUPLICATE_VARIANTS.get(team, [])
    if not variants:
        # Fallback: simple rewording
        title = f"[Possible duplicate] {original_bug.title}"
        desc = (
            f"I'm seeing what appears to be the same issue described in {original_bug.id}. "
            f"The symptoms are identical: {original_bug.actual_behavior}. "
            f"Different reporter, same root cause."
        )
    else:
        title, desc = rng.choice(variants)

    reporter = rng.choice(_REPORTERS)
    # Duplicate filed later than original
    dup_timestamp = original_bug.timestamp.replace("T", "T")  # same day
    hour = rng.randint(12, 23)
    minute = rng.randint(0, 59)
    dup_timestamp = dup_timestamp[:11] + f"{hour:02d}:{minute:02d}:00Z"

    dup_bug = BugReport(
        id=dup_id,
        title=title,
        description=desc,
        reporter=reporter,
        timestamp=dup_timestamp,
        product=original_bug.product,
        version=original_bug.version,
        steps_to_reproduce=original_bug.steps_to_reproduce,
        expected_behavior=original_bug.expected_behavior,
        actual_behavior=original_bug.actual_behavior,
        environment_info=rng.choice(_ENV_INFO_POOL).copy(),
        customer_tier=rng.choice(["business", "starter"]),
        sla_hours_remaining=None,
        linked_bug_ids=[original_bug.id],
    )

    dup_gt = BugGroundTruth(
        severity=original_gt.severity,
        team=original_gt.team,
        needs_info=False,
        is_duplicate_of=original_bug.id,
        should_escalate=False,
        sla_critical=False,
    )

    return dup_bug, dup_gt


# ═══════════════════════════════════════════════════════════════════════════
# Task-level generators
# ═══════════════════════════════════════════════════════════════════════════

_INSTRUCTIONS_EASY = (
    "You are a senior triage engineer. One bug report requires your attention.\n\n"
    "Available action_types (output exactly ONE JSON action per step — no extra text):\n"
    '  classify      → {"action_type": "classify",      "bug_id": "...", "severity": "critical|high|medium|low"}\n'
    '  assign        → {"action_type": "assign",         "bug_id": "...", "assigned_team": "backend|frontend|mobile|infrastructure|security|database|qa"}\n'
    '  request_info  → {"action_type": "request_info",   "bug_id": "...", "info_requested": ["item1", "item2"]}\n'
    '  mark_duplicate→ {"action_type": "mark_duplicate", "bug_id": "...", "duplicate_of": "ORIGINAL-ID"}\n'
    '  escalate      → {"action_type": "escalate",       "bug_id": "...", "escalation_reason": "reason"}\n'
    '  submit        → {"action_type": "submit",          "bug_id": "..."}\n'
    "\nRules:\n"
    "  - classify and assign a bug BEFORE submitting it.\n"
    "  - Only request info when critical information is genuinely missing.\n"
    "  - Submit once triage is complete to receive full credit.\n"
)

_INSTRUCTIONS_MULTI = (
    "You are a senior triage engineer handling multiple bug reports simultaneously.\n\n"
    "Available action_types (output exactly ONE JSON action per step — no extra text):\n"
    '  classify      → {"action_type": "classify",      "bug_id": "...", "severity": "critical|high|medium|low"}\n'
    '  assign        → {"action_type": "assign",         "bug_id": "...", "assigned_team": "backend|frontend|mobile|infrastructure|security|database|qa"}\n'
    '  request_info  → {"action_type": "request_info",   "bug_id": "...", "info_requested": ["item1", "item2"]}\n'
    '  mark_duplicate→ {"action_type": "mark_duplicate", "bug_id": "...", "duplicate_of": "ORIGINAL-ID"}\n'
    '  escalate      → {"action_type": "escalate",       "bug_id": "...", "escalation_reason": "reason"}\n'
    '  submit        → {"action_type": "submit",          "bug_id": "..."}\n'
    "\nRules:\n"
    "  - classify and assign each bug BEFORE submitting it.\n"
    "  - Detect duplicates: if two bugs describe the same root cause, mark the later one.\n"
    "  - Request info ONLY when steps_to_reproduce AND environment_info are both missing.\n"
    "  - Escalate if: SLA < 2h, enterprise + critical/high, or active security exploit.\n"
    "  - Submit all bugs to complete the episode.\n"
)


def generate_easy(rng: random.Random, prefix: str = "GEN") -> TaskScenario:
    """Generate an easy task: 1 critical bug, needs escalation."""
    # Pick a critical bug template
    critical_bugs = [b for b in _ALL_BUGS if b.severity == "critical"]
    template = rng.choice(critical_bugs)
    bug_id = _make_bug_id(prefix, 1)
    timestamp = _make_timestamp(rng)

    bug, gt = _template_to_bug(
        rng, template, bug_id, timestamp,
        customer_tier="enterprise",
        sla_hours=round(rng.uniform(0.5, 1.5), 1),
    )

    return TaskScenario(
        task_id="single-triage",
        name="Single Critical Bug Triage",
        description="Triage one critical production bug. Classify, assign, escalate, submit.",
        difficulty="easy",
        max_steps=5,
        reward_threshold=0.80,
        bug_reports=[bug],
        ground_truth={bug_id: gt},
        instructions=_INSTRUCTIONS_EASY,
    )


def generate_medium(rng: random.Random, prefix: str = "GEN") -> TaskScenario:
    """
    Generate a medium task: 8 bugs.
    Structure: 5 regular + 1 duplicate pair (2 bugs) + 1 info-incomplete.
    Must include: mix of severities, 1 security escalation bug.
    """
    bugs: List[BugReport] = []
    ground_truth: Dict[str, BugGroundTruth] = {}
    used_templates: set = set()
    idx = 1

    # 1. One security bug that needs escalation
    sec_templates = [b for b in _SECURITY_BUGS if b.should_escalate]
    sec_t = rng.choice(sec_templates)
    used_templates.add(id(sec_t))
    bid = _make_bug_id(prefix, idx); idx += 1
    bug, gt = _template_to_bug(rng, sec_t, bid, _make_timestamp(rng), "enterprise")
    bugs.append(bug); ground_truth[bid] = gt

    # 2. One info-incomplete bug (medium severity, missing steps + env)
    info_pool = [b for b in _ALL_BUGS if b.severity in ("medium", "high") and id(b) not in used_templates]
    info_t = rng.choice(info_pool)
    used_templates.add(id(info_t))
    bid = _make_bug_id(prefix, idx); idx += 1
    bug, gt = _template_to_bug(rng, info_t, bid, _make_timestamp(rng), rng.choice(_TIERS), strip_info=True)
    bugs.append(bug); ground_truth[bid] = gt

    # 3. One duplicate pair: pick a template, generate original + duplicate
    dup_pool = [b for b in _ALL_BUGS if b.team in _DUPLICATE_VARIANTS and id(b) not in used_templates]
    dup_t = rng.choice(dup_pool)
    used_templates.add(id(dup_t))
    orig_bid = _make_bug_id(prefix, idx); idx += 1
    orig_bug, orig_gt = _template_to_bug(rng, dup_t, orig_bid, _make_timestamp(rng, month=1), rng.choice(_TIERS))
    bugs.append(orig_bug); ground_truth[orig_bid] = orig_gt

    dup_bid = _make_bug_id(prefix, idx); idx += 1
    dup_bug, dup_gt = _make_duplicate(rng, orig_bug, orig_gt, dup_bid, dup_t.team)
    bugs.append(dup_bug); ground_truth[dup_bid] = dup_gt

    # 4. Fill remaining slots (4 more bugs) with varied severities
    remaining_needed = 8 - len(bugs)
    severity_targets = ["critical", "high", "medium", "low"]
    available = [b for b in _ALL_BUGS if id(b) not in used_templates]
    rng.shuffle(available)

    for i in range(remaining_needed):
        target_sev = severity_targets[i % len(severity_targets)]
        candidates = [b for b in available if b.severity == target_sev and id(b) not in used_templates]
        if not candidates:
            candidates = [b for b in available if id(b) not in used_templates]
        if not candidates:
            break
        t = candidates[0]
        used_templates.add(id(t))
        bid = _make_bug_id(prefix, idx); idx += 1
        bug, gt = _template_to_bug(rng, t, bid, _make_timestamp(rng), rng.choice(_TIERS))
        bugs.append(bug); ground_truth[bid] = gt

    # Shuffle bug order (so duplicates aren't adjacent)
    combined = list(zip(bugs, [ground_truth[b.id] for b in bugs]))
    rng.shuffle(combined)
    bugs = [b for b, _ in combined]
    ground_truth = {b.id: gt for b, gt in combined}

    return TaskScenario(
        task_id="batch-triage",
        name="Batch Bug Triage with Duplicate Detection",
        description=f"Process {len(bugs)} bug reports. Detect duplicates, request info, escalate.",
        difficulty="medium",
        max_steps=32,
        reward_threshold=0.65,
        bug_reports=bugs,
        ground_truth=ground_truth,
        instructions=_INSTRUCTIONS_MULTI,
    )


def generate_hard(rng: random.Random, prefix: str = "GEN") -> TaskScenario:
    """
    Generate a hard task: 15 bugs.
    Structure: 3 duplicate pairs (6 bugs) + 2 info-incomplete + 5 SLA-critical + rest normal.
    """
    bugs: List[BugReport] = []
    ground_truth: Dict[str, BugGroundTruth] = {}
    used_templates: set = set()
    idx = 1

    # 1. Three duplicate pairs from different teams
    dup_teams = rng.sample(list(_DUPLICATE_VARIANTS.keys()), min(3, len(_DUPLICATE_VARIANTS)))
    for team in dup_teams:
        pool = [b for b in _BUGS_BY_TEAM.get(team, []) if id(b) not in used_templates]
        if not pool:
            continue
        t = rng.choice(pool)
        used_templates.add(id(t))

        orig_bid = _make_bug_id(prefix, idx); idx += 1
        orig_bug, orig_gt = _template_to_bug(rng, t, orig_bid, _make_timestamp(rng, month=1), rng.choice(_TIERS))
        bugs.append(orig_bug); ground_truth[orig_bid] = orig_gt

        dup_bid = _make_bug_id(prefix, idx); idx += 1
        dup_bug, dup_gt = _make_duplicate(rng, orig_bug, orig_gt, dup_bid, team)
        bugs.append(dup_bug); ground_truth[dup_bid] = dup_gt

    # 2. Two info-incomplete bugs
    info_pool = [b for b in _ALL_BUGS if b.severity in ("medium", "high") and id(b) not in used_templates]
    for _ in range(2):
        if not info_pool:
            break
        t = rng.choice(info_pool)
        info_pool.remove(t)
        used_templates.add(id(t))
        bid = _make_bug_id(prefix, idx); idx += 1
        bug, gt = _template_to_bug(rng, t, bid, _make_timestamp(rng), rng.choice(_TIERS), strip_info=True)
        bugs.append(bug); ground_truth[bid] = gt

    # 3. SLA-critical bugs (at least 5 total, some may already exist from dup pairs)
    sla_count = sum(1 for gt in ground_truth.values() if gt.sla_critical)
    sla_needed = max(0, 5 - sla_count)
    critical_pool = [b for b in _ALL_BUGS if b.severity == "critical" and id(b) not in used_templates]
    for _ in range(sla_needed):
        if not critical_pool:
            break
        t = rng.choice(critical_pool)
        critical_pool.remove(t)
        used_templates.add(id(t))
        bid = _make_bug_id(prefix, idx); idx += 1
        bug, gt = _template_to_bug(
            rng, t, bid, _make_timestamp(rng), "enterprise",
            sla_hours=round(rng.uniform(0.3, 1.8), 1),
        )
        bugs.append(bug); ground_truth[bid] = gt

    # 4. Fill remaining to reach 15
    remaining = 15 - len(bugs)
    available = [b for b in _ALL_BUGS if id(b) not in used_templates]
    rng.shuffle(available)
    severity_cycle = ["high", "medium", "low", "medium", "high", "low", "medium"]
    for i in range(remaining):
        target = severity_cycle[i % len(severity_cycle)]
        candidates = [b for b in available if b.severity == target and id(b) not in used_templates]
        if not candidates:
            candidates = [b for b in available if id(b) not in used_templates]
        if not candidates:
            break
        t = candidates[0]
        used_templates.add(id(t))
        bid = _make_bug_id(prefix, idx); idx += 1
        bug, gt = _template_to_bug(rng, t, bid, _make_timestamp(rng), rng.choice(_TIERS))
        bugs.append(bug); ground_truth[bid] = gt

    # Shuffle order
    combined = list(zip(bugs, [ground_truth[b.id] for b in bugs]))
    rng.shuffle(combined)
    bugs = [b for b, _ in combined]
    ground_truth = {b.id: gt for b, gt in combined}

    return TaskScenario(
        task_id="sla-crisis",
        name="SLA Crisis: Mass Bug Surge with Escalations",
        description=f"{len(bugs)} simultaneous bugs with duplicates, SLA breaches, and escalations.",
        difficulty="hard",
        max_steps=50,
        reward_threshold=0.50,
        bug_reports=bugs,
        ground_truth=ground_truth,
        instructions=_INSTRUCTIONS_MULTI,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════

_GENERATORS = {
    "single-triage": generate_easy,
    "batch-triage":  generate_medium,
    "sla-crisis":    generate_hard,
}


def generate_scenario(task_id: str, seed: int) -> TaskScenario:
    """
    Generate a fresh scenario for the given task using the provided seed.
    Same seed + same task → identical scenario (deterministic).

    Args:
        task_id: One of "single-triage", "batch-triage", "sla-crisis"
        seed: Integer seed for reproducible generation

    Returns:
        A complete TaskScenario with bug reports and ground truth.
    """
    gen_fn = _GENERATORS.get(task_id)
    if gen_fn is None:
        raise ValueError(f"Unknown task '{task_id}'. Available: {list(_GENERATORS.keys())}")

    rng = random.Random(seed)
    prefix = f"S{seed % 10000:04d}"  # e.g. S0042, S1337
    return gen_fn(rng, prefix=prefix)
