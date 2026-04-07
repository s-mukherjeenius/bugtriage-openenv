"""
BugTriage OpenEnv — Scenario Definitions
All bug reports and ground-truth annotations for each task.

Ground truth is NEVER exposed via any API endpoint — it lives only here
and in graders.py. The agent receives only the raw BugReport objects.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from app.models import BugReport


# ---------------------------------------------------------------------------
# Ground-truth dataclass (internal use only — never serialised to the agent)
# ---------------------------------------------------------------------------

@dataclass
class BugGroundTruth:
    severity: str                           # critical / high / medium / low
    team: str                               # backend / frontend / security / ...
    needs_info: bool = False                # True → info request is expected
    is_duplicate_of: Optional[str] = None  # ID of the original bug if this is a dup
    should_escalate: bool = False           # True → escalation is expected / rewarded
    sla_critical: bool = False              # True → sla_hours_remaining < 2.0


@dataclass
class TaskScenario:
    task_id: str
    name: str
    description: str
    difficulty: str                         # easy / medium / hard
    max_steps: int
    reward_threshold: float
    bug_reports: List[BugReport]
    ground_truth: Dict[str, BugGroundTruth]
    instructions: str


# ===========================================================================
# TASK 1 — single-triage  (EASY)
# ===========================================================================
# One production-down payment processing failure.
# All info is present. Agent: classify → assign → (escalate) → submit.
# Max steps: 5   Reward threshold: 0.80
# ===========================================================================

_T1_BUGS: List[BugReport] = [
    BugReport(
        id="PAY-001",
        title="Payment processing completely broken — all transactions failing",
        description=(
            "Since 14:32 UTC today our entire payment flow is down. Every checkout "
            "attempt returns HTTP 500 with 'upstream connect error'. We are losing "
            "approximately $12,000 per minute in GMV. The Stripe webhook processor "
            "is throwing NullPointerException in PaymentService.processCharge() at "
            "line 247. Stack trace attached. This is affecting 100% of users on all "
            "plans including enterprise. No deployment or config change was made in "
            "the last 4 hours. This is a P0 production incident."
        ),
        reporter="jane.ops@company.com",
        timestamp="2024-01-15T14:35:00Z",
        product="Checkout API",
        version="v3.12.1",
        steps_to_reproduce=(
            "1. Add any item to cart\n"
            "2. Enter valid payment details\n"
            "3. Click 'Pay Now'\n"
            "Result: HTTP 500, transaction never completes"
        ),
        expected_behavior="Transaction completes and order is created",
        actual_behavior="HTTP 500 — NullPointerException in PaymentService.processCharge()",
        environment_info={
            "region": "us-east-1",
            "service": "checkout-api",
            "runtime": "Java 17",
        },
        customer_tier="enterprise",
        sla_hours_remaining=1.0,
        linked_bug_ids=None,
    )
]

_T1_GROUND_TRUTH: Dict[str, BugGroundTruth] = {
    "PAY-001": BugGroundTruth(
        severity="critical",
        team="backend",
        needs_info=False,
        is_duplicate_of=None,
        should_escalate=True,   # optional for easy task — still rewarded
        sla_critical=True,
    )
}

_T1_INSTRUCTIONS = """\
You are a senior triage engineer. One bug report requires your attention.

Available action_types (output exactly ONE JSON action per step — no extra text):
  classify      → {"action_type": "classify",      "bug_id": "...", "severity": "critical|high|medium|low"}
  assign        → {"action_type": "assign",         "bug_id": "...", "assigned_team": "backend|frontend|mobile|infrastructure|security|database|qa"}
  request_info  → {"action_type": "request_info",   "bug_id": "...", "info_requested": ["item1", "item2"]}
  mark_duplicate→ {"action_type": "mark_duplicate", "bug_id": "...", "duplicate_of": "ORIGINAL-ID"}
  escalate      → {"action_type": "escalate",       "bug_id": "...", "escalation_reason": "reason"}
  submit        → {"action_type": "submit",          "bug_id": "..."}

Rules:
  - classify and assign a bug BEFORE submitting it.
  - Only request info when critical information is genuinely missing.
  - Submit once triage is complete to receive full credit.
"""

SCENARIO_TASK1 = TaskScenario(
    task_id="single-triage",
    name="Single Critical Bug Triage",
    description=(
        "A production-down payment processing failure has been reported. "
        "Classify its severity, assign it to the correct team, and submit "
        "your triage decision as quickly as possible."
    ),
    difficulty="easy",
    max_steps=5,
    reward_threshold=0.80,
    bug_reports=_T1_BUGS,
    ground_truth=_T1_GROUND_TRUTH,
    instructions=_T1_INSTRUCTIONS,
)


# ===========================================================================
# TASK 2 — batch-triage  (MEDIUM)
# ===========================================================================
# 8 bug reports: 1 duplicate pair (BUG-003 is duplicate of BUG-006),
# 1 info-incomplete report (BUG-005), 1 security issue needing escalation.
# Max steps: 32   Reward threshold: 0.65
# ===========================================================================

_T2_BUGS: List[BugReport] = [
    BugReport(
        id="BUG-001",
        title="Login button misaligned on small mobile screens",
        description=(
            "On iPhone SE (375px width) the 'Sign In' button overflows its container "
            "by about 8px on the right side. It still works but looks unprofessional. "
            "Only visible below 380px viewport width."
        ),
        reporter="ui_qa@company.com",
        timestamp="2024-02-01T09:00:00Z",
        product="Web App",
        version="v2.8.0",
        steps_to_reproduce="1. Open app on iPhone SE (375px viewport)\n2. Navigate to login page",
        expected_behavior="Button fits within container",
        actual_behavior="Button overflows 8px on the right",
        environment_info={"device": "iPhone SE", "browser": "Safari 17"},
        customer_tier="free",
        sla_hours_remaining=None,
    ),
    BugReport(
        id="BUG-002",
        title="Application crashes when uploading files larger than 100 MB",
        description=(
            "Uploading any file over ~100 MB causes the backend to crash with an "
            "OutOfMemoryError. The upload progress bar freezes at 73% and the server "
            "returns 502 Bad Gateway. The worker process then restarts. This affects "
            "business customers who upload video assets regularly."
        ),
        reporter="bob.enterprise@bigcorp.com",
        timestamp="2024-02-01T10:15:00Z",
        product="File Storage API",
        version="v1.4.2",
        steps_to_reproduce=(
            "1. Log in as any user\n"
            "2. Navigate to Files > Upload\n"
            "3. Select a file > 100 MB\n"
            "4. Click Upload"
        ),
        expected_behavior="File uploads successfully",
        actual_behavior="502 Bad Gateway; server worker restarts with OOM error",
        environment_info={"os": "Windows 11", "browser": "Chrome 120"},
        customer_tier="business",
        sla_hours_remaining=12.0,
    ),
    BugReport(
        id="BUG-003",
        title="Dark mode: text impossible to read on Safari macOS",
        description=(
            "When dark mode is enabled, the main content area shows nearly-black text "
            "on a very dark background in Safari on macOS Sonoma. The contrast ratio is "
            "roughly 1.3:1, making the product unusable in dark mode on Mac. "
            "The issue does NOT appear on Chrome or Firefox."
        ),
        reporter="design_review@company.com",
        timestamp="2024-02-01T11:00:00Z",   # LATER than BUG-006 → this is the duplicate
        product="Web App",
        version="v2.8.0",
        steps_to_reproduce=(
            "1. Enable dark mode on macOS\n"
            "2. Open app in Safari\n"
            "3. Navigate to any content page"
        ),
        expected_behavior="Text is clearly readable with ≥4.5:1 contrast",
        actual_behavior="Text nearly invisible — contrast ~1.3:1",
        environment_info={"os": "macOS Sonoma", "browser": "Safari 17"},
        customer_tier="starter",
        sla_hours_remaining=None,
    ),
    BugReport(
        id="BUG-004",
        title="Analytics reports page takes 30+ seconds to load",
        description=(
            "The /dashboard/reports page times out for accounts with more than "
            "50,000 records. The query for 'Top Events' does a full table scan "
            "(confirmed via EXPLAIN ANALYZE). The index on event_timestamp is not "
            "being used. Enterprise customers relying on daily reports are frustrated."
        ),
        reporter="db_perf@company.com",
        timestamp="2024-02-01T11:30:00Z",
        product="Analytics Dashboard",
        version="v4.1.0",
        steps_to_reproduce=(
            "1. Log in as an account with >50k records\n"
            "2. Navigate to Dashboard > Reports\n"
            "3. Observe load time"
        ),
        expected_behavior="Page loads in <3 seconds",
        actual_behavior="30–45 second load time; sometimes HTTP 504",
        environment_info={"db": "PostgreSQL 15", "records": "~80,000"},
        customer_tier="enterprise",
        sla_hours_remaining=8.0,
    ),
    BugReport(
        id="BUG-005",
        title="App randomly crashes — can't reproduce consistently",
        description=(
            "The app just crashes sometimes. It happens maybe once a day? I'm not "
            "sure what causes it. It happened while I was using it. No error message, "
            "just closes. My device is fairly new."
        ),
        reporter="random.user123@gmail.com",
        timestamp="2024-02-01T12:00:00Z",
        product="Mobile App",
        version="unknown",
        steps_to_reproduce=None,        # MISSING — agent should request_info
        expected_behavior=None,
        actual_behavior="App closes unexpectedly",
        environment_info=None,          # MISSING — agent should request_info
        customer_tier="free",
        sla_hours_remaining=None,
    ),
    BugReport(
        id="BUG-006",
        title="Dark mode text contrast broken — affects both Safari and Chrome on Mac",
        description=(
            "When macOS dark mode is active, text throughout the application has almost "
            "no contrast against the background. Specifically the body text in article "
            "and dashboard views. Tested on Safari 17 AND Chrome 120 on macOS Sonoma. "
            "Clearly a CSS variable not resolving correctly in the dark theme."
        ),
        reporter="accessibility_team@company.com",
        timestamp="2024-02-01T08:45:00Z",  # EARLIER timestamp → this is the ORIGINAL
        product="Web App",
        version="v2.8.0",
        steps_to_reproduce=(
            "1. Set macOS to dark mode\n"
            "2. Open app in any browser\n"
            "3. Go to any content page"
        ),
        expected_behavior="Accessible contrast ratio ≥4.5:1",
        actual_behavior="Contrast ~1.3:1 — text nearly unreadable",
        environment_info={"os": "macOS Sonoma", "browser": "Safari 17 and Chrome 120"},
        customer_tier="starter",
        sla_hours_remaining=None,
    ),
    BugReport(
        id="BUG-007",
        title="API rate limiting completely bypassed with forged X-Forwarded-For header",
        description=(
            "Our public API rate limit (100 req/min per IP) can be trivially bypassed "
            "by setting X-Forwarded-For: 127.0.0.1 in any request. This spoofs the "
            "client IP and the rate limiter reads the forged header instead of the real "
            "IP. An attacker can make unlimited API calls — enabling data scraping or "
            "credential stuffing at scale. "
            "POC: curl -H 'X-Forwarded-For: 127.0.0.1' https://api.company.com/users"
        ),
        reporter="pentest@security-firm.com",
        timestamp="2024-02-01T13:00:00Z",
        product="Public API Gateway",
        version="v2.3.0",
        steps_to_reproduce=(
            "1. Send 101+ requests with X-Forwarded-For: 127.0.0.1\n"
            "2. Observe: rate limit is never triggered"
        ),
        expected_behavior="Rate limit enforced regardless of spoofed headers",
        actual_behavior="Rate limit completely bypassed",
        environment_info={"endpoint": "api.company.com", "method": "ANY"},
        customer_tier=None,
        sla_hours_remaining=2.0,
    ),
    BugReport(
        id="BUG-008",
        title="CSV export includes soft-deleted records",
        description=(
            "When exporting data via Reports > Export > CSV, the downloaded file "
            "includes records with deleted_at IS NOT NULL. This is a data integrity "
            "issue — deleted records must never appear in exports. Customers have "
            "noticed ghost entries in their data."
        ),
        reporter="data_integrity@company.com",
        timestamp="2024-02-01T14:00:00Z",
        product="Reports Module",
        version="v4.1.0",
        steps_to_reproduce=(
            "1. Soft-delete one or more records\n"
            "2. Navigate to Reports > Export > CSV\n"
            "3. Open exported file\n"
            "4. Observe deleted records present"
        ),
        expected_behavior="Only active (non-deleted) records in export",
        actual_behavior="Soft-deleted records included in CSV export",
        environment_info={"db": "PostgreSQL 15"},
        customer_tier="business",
        sla_hours_remaining=24.0,
    ),
]

_T2_GROUND_TRUTH: Dict[str, BugGroundTruth] = {
    "BUG-001": BugGroundTruth(severity="low",      team="frontend"),
    "BUG-002": BugGroundTruth(severity="high",     team="backend"),
    # BUG-003 is a duplicate of BUG-006 (same root cause, BUG-006 was filed earlier)
    "BUG-003": BugGroundTruth(severity="medium",   team="frontend",  is_duplicate_of="BUG-006"),
    "BUG-004": BugGroundTruth(severity="high",     team="database"),
    # BUG-005 has no steps or environment info — request_info expected
    "BUG-005": BugGroundTruth(severity="medium",   team="mobile",    needs_info=True),
    # BUG-006 is the ORIGINAL of the dark-mode contrast pair
    "BUG-006": BugGroundTruth(severity="medium",   team="frontend"),
    # BUG-007 is a critical security issue — escalation expected
    "BUG-007": BugGroundTruth(severity="critical", team="security",  should_escalate=True, sla_critical=True),
    "BUG-008": BugGroundTruth(severity="high",     team="backend"),
}

_T2_INSTRUCTIONS = """\
You are a senior triage engineer. Eight bug reports require processing.

Available action_types (one JSON action per step — no extra text):
  classify      → {"action_type": "classify",      "bug_id": "...", "severity": "critical|high|medium|low"}
  assign        → {"action_type": "assign",         "bug_id": "...", "assigned_team": "backend|frontend|mobile|infrastructure|security|database|qa"}
  request_info  → {"action_type": "request_info",   "bug_id": "...", "info_requested": ["item1", "item2"]}
  mark_duplicate→ {"action_type": "mark_duplicate", "bug_id": "...", "duplicate_of": "ORIGINAL-ID"}
  escalate      → {"action_type": "escalate",       "bug_id": "...", "escalation_reason": "..."}
  submit        → {"action_type": "submit",          "bug_id": "..."}

Strategy:
  - Look for reports describing the same root cause with different wording (duplicates).
  - Mark the LATER-submitted report as duplicate of the EARLIER one.
  - Only request_info when steps_to_reproduce or environment_info are completely absent.
  - All critical security vulnerabilities must be escalated.
  - Submit each bug once classify + assign (+ any optional actions) are done.
"""

SCENARIO_TASK2 = TaskScenario(
    task_id="batch-triage",
    name="Batch Bug Triage with Duplicate Detection",
    description=(
        "Process 8 incoming bug reports. Classify severity, assign to the correct team, "
        "detect duplicate reports, request missing information where necessary, "
        "escalate security issues, and submit all decisions."
    ),
    difficulty="medium",
    max_steps=32,
    reward_threshold=0.65,
    bug_reports=_T2_BUGS,
    ground_truth=_T2_GROUND_TRUTH,
    instructions=_T2_INSTRUCTIONS,
)


# ===========================================================================
# TASK 3 — sla-crisis  (HARD)
# ===========================================================================
# 15 bug reports with:
#   • 3 duplicate pairs:
#       CRI-009 is duplicate of CRI-003  (Firefox logo — same root cause)
#       CRI-011 is duplicate of CRI-004  (DB primary down — same incident)
#       CRI-012 is duplicate of CRI-007  (German tooltip wrap — same bug)
#   • 5 SLA-critical bugs (< 2h remaining): CRI-001, CRI-004, CRI-008, CRI-011, CRI-015
#   • Enterprise customers needing escalation: CRI-001, CRI-004, CRI-006, CRI-008, CRI-015
#   • 2 info-incomplete bugs: CRI-005, CRI-014
#   • Linked bug clusters: CRI-001 ↔ CRI-008 ↔ CRI-015 (auth service)
#                          CRI-004 ↔ CRI-011 (DB incident)
# Max steps: 50   Reward threshold: 0.50
# ===========================================================================

_T3_BUGS: List[BugReport] = [
    BugReport(
        id="CRI-001",
        title="Authentication tokens not expiring — persistent session vulnerability",
        description=(
            "JWT tokens issued by our auth service are not expiring even after the "
            "configured TTL (24h). Tokens issued over 72 hours ago are still valid. "
            "Any compromised token gives permanent access. Our SOC team detected this "
            "during a routine audit. Customer data is at risk."
        ),
        reporter="soc_team@company.com",
        timestamp="2024-03-01T06:00:00Z",
        product="Auth Service",
        version="v5.2.1",
        steps_to_reproduce="1. Issue a JWT\n2. Wait >24h\n3. Use token — still valid",
        expected_behavior="Token rejected after TTL",
        actual_behavior="Token valid indefinitely",
        environment_info={"service": "auth-service", "region": "us-east-1"},
        customer_tier="enterprise",
        sla_hours_remaining=1.5,
        linked_bug_ids=["CRI-008", "CRI-015"],
    ),
    BugReport(
        id="CRI-002",
        title="Memory leak in background job processor — server OOM every 6 hours",
        description=(
            "The async job processor leaks ~200 MB per hour. After ~6 hours "
            "the worker OOMs and all pending jobs are lost. The leak is in the "
            "Redis connection pool not being released on job completion."
        ),
        reporter="platform@company.com",
        timestamp="2024-03-01T07:00:00Z",
        product="Job Processor",
        version="v2.1.0",
        steps_to_reproduce="1. Monitor memory over 6+ hours\n2. Observe steady increase",
        expected_behavior="Stable memory footprint",
        actual_behavior="~200 MB/hr leak; OOM crash every 6h, jobs lost",
        environment_info={"runtime": "Python 3.11", "service": "job-worker"},
        customer_tier="business",
        sla_hours_remaining=6.0,
    ),
    BugReport(
        id="CRI-003",
        title="Navbar logo broken on Firefox — shows alt text",
        description=(
            "The company logo in the top navigation bar doesn't load in Firefox 121. "
            "It shows alt text 'Company Logo' instead of the image. "
            "Chrome and Safari are unaffected. Possibly a missing CORS header."
        ),
        reporter="qa_firefox@company.com",
        timestamp="2024-03-01T08:00:00Z",   # EARLIER → this is the ORIGINAL
        product="Web App",
        version="v3.0.0",
        steps_to_reproduce="1. Open app in Firefox 121\n2. Observe navbar",
        expected_behavior="Logo image renders",
        actual_behavior="Alt text 'Company Logo' shown",
        environment_info={"browser": "Firefox 121", "os": "Windows 11"},
        customer_tier="free",
        sla_hours_remaining=None,
    ),
    BugReport(
        id="CRI-004",
        title="Database primary node down — writes failing for all customers",
        description=(
            "Our primary PostgreSQL node went offline 20 minutes ago. All write "
            "operations are failing with 'could not connect to server'. Read replicas "
            "serve reads but we have zero write capability. This affects every "
            "enterprise customer. On-call DBA is unavailable."
        ),
        reporter="on_call_engineer@company.com",
        timestamp="2024-03-01T09:00:00Z",   # EARLIER → this is the ORIGINAL of CRI-011
        product="Database Cluster",
        version="PostgreSQL 15.4",
        steps_to_reproduce="1. Attempt any write operation\nResult: connection refused",
        expected_behavior="Writes succeed",
        actual_behavior="All writes failing — primary node down",
        environment_info={"cluster": "prod-primary", "region": "us-east-1"},
        customer_tier="enterprise",
        sla_hours_remaining=0.3,
        linked_bug_ids=["CRI-011"],
    ),
    BugReport(
        id="CRI-005",
        title="User reports weird behavior when switching accounts",
        description=(
            "Sometimes when I switch between my personal and work accounts things get "
            "mixed up. Not sure exactly when it happens. It fixed itself after I "
            "cleared cookies. Happens maybe 1 in 10 times."
        ),
        reporter="confused_user@example.com",
        timestamp="2024-03-01T09:30:00Z",
        product="Web App",
        version="v3.0.0",
        steps_to_reproduce=None,        # MISSING — request_info expected
        expected_behavior=None,
        actual_behavior="Account data appears mixed up",
        environment_info=None,          # MISSING
        customer_tier="starter",
        sla_hours_remaining=None,
    ),
    BugReport(
        id="CRI-006",
        title="Invoice PDF generation fails for amounts > $99,999.99",
        description=(
            "Any invoice with a total exceeding $99,999.99 fails to generate a PDF. "
            "The invoice shows in the UI but 'Download PDF' returns HTTP 500. "
            "Stack trace shows a number formatting overflow in the PDF renderer. "
            "Blocking enterprise customers from downloading invoices for large orders."
        ),
        reporter="billing@company.com",
        timestamp="2024-03-01T10:00:00Z",
        product="Billing Service",
        version="v1.8.3",
        steps_to_reproduce=(
            "1. Create invoice with total > $99,999.99\n"
            "2. Click 'Download PDF'\n"
            "Result: HTTP 500"
        ),
        expected_behavior="PDF downloads successfully",
        actual_behavior="HTTP 500 — number formatting overflow",
        environment_info={"service": "billing-service"},
        customer_tier="enterprise",
        sla_hours_remaining=4.0,
    ),
    BugReport(
        id="CRI-007",
        title="Tooltip text wrapping incorrectly in German locale",
        description=(
            "When the UI language is set to German, some tooltip texts wrap mid-word "
            "creating hyphenation artifacts. Affects ~12 tooltips in the settings panel. "
            "Cosmetic only — functionality not affected."
        ),
        reporter="i18n_qa@company.com",
        timestamp="2024-03-01T10:30:00Z",   # EARLIER → ORIGINAL of CRI-012
        product="Web App",
        version="v3.0.0",
        steps_to_reproduce="1. Set locale to German\n2. Hover over settings panel tooltips",
        expected_behavior="Clean word wrap",
        actual_behavior="Mid-word breaks",
        environment_info={"locale": "de-DE", "browser": "Chrome 120"},
        customer_tier="free",
        sla_hours_remaining=None,
    ),
    BugReport(
        id="CRI-008",
        title="Session tokens remain valid after explicit user logout",
        description=(
            "After a user clicks 'Log Out', their session token is not invalidated "
            "server-side. The token continues to authenticate API requests. "
            "Confirmed by replaying the token post-logout. If a token is intercepted, "
            "logging out does not protect the user."
        ),
        reporter="security_audit@company.com",
        timestamp="2024-03-01T06:30:00Z",
        product="Auth Service",
        version="v5.2.1",
        steps_to_reproduce=(
            "1. Log in and capture auth token\n"
            "2. Log out\n"
            "3. Replay captured token\n"
            "Result: API still accepts it"
        ),
        expected_behavior="Token rejected after logout",
        actual_behavior="Token still valid post-logout",
        environment_info={"service": "auth-service"},
        customer_tier="enterprise",
        sla_hours_remaining=1.5,
        linked_bug_ids=["CRI-001", "CRI-015"],
    ),
    BugReport(
        id="CRI-009",
        title="Logo not showing in Firefox — navigation bar",
        description=(
            "In Firefox, the logo image in the navigation bar is not displaying. "
            "It shows fallback text instead of the image. Works fine in Chrome. "
            "Might be a CORS issue with our image CDN."
        ),
        reporter="ux_review@company.com",
        timestamp="2024-03-01T08:45:00Z",   # LATER than CRI-003 → DUPLICATE of CRI-003
        product="Web App",
        version="v3.0.0",
        steps_to_reproduce="1. Open any page in Firefox\n2. Look at top nav",
        expected_behavior="Logo image renders",
        actual_behavior="Fallback alt text shown",
        environment_info={"browser": "Firefox 121"},
        customer_tier="free",
        sla_hours_remaining=None,
    ),
    BugReport(
        id="CRI-010",
        title="Webhook delivery failing silently — no retry, no error notification",
        description=(
            "Webhooks that fail (4xx/5xx from customer endpoint) are silently dropped "
            "with no retry and no notification. Customers believe their webhooks are "
            "working but events are being lost. Regression introduced in v3.0.0."
        ),
        reporter="integrations@company.com",
        timestamp="2024-03-01T11:00:00Z",
        product="Webhook Service",
        version="v3.0.0",
        steps_to_reproduce=(
            "1. Register a webhook pointing to an endpoint that returns 500\n"
            "2. Trigger a webhook event\n"
            "3. Observe: no retry, no notification"
        ),
        expected_behavior="Webhook retried with exponential backoff; customer notified on failure",
        actual_behavior="Silent drop — no retry, no notification",
        environment_info={"service": "webhook-service"},
        customer_tier="business",
        sla_hours_remaining=5.0,
    ),
    BugReport(
        id="CRI-011",
        title="Write operations returning 503 — possible DB connectivity issue",
        description=(
            "For the past 25 minutes, all API endpoints that perform writes are "
            "returning 503 Service Unavailable. Reads still work. Error logs show "
            "'could not connect to primary database host'. May be related to the "
            "earlier DB alert from the infrastructure team."
        ),
        reporter="api_monitoring@company.com",
        timestamp="2024-03-01T09:25:00Z",   # LATER than CRI-004 → DUPLICATE of CRI-004
        product="API Gateway",
        version="v3.0.0",
        steps_to_reproduce="1. Make any POST/PUT/DELETE request\nResult: 503",
        expected_behavior="Write operations succeed",
        actual_behavior="503 — cannot reach primary DB",
        environment_info={"service": "api-gateway"},
        customer_tier="enterprise",
        sla_hours_remaining=0.3,
        linked_bug_ids=["CRI-004"],
    ),
    BugReport(
        id="CRI-012",
        title="Tooltip word-wrapping broken in German translation",
        description=(
            "The German language setting causes tooltip text to break words incorrectly. "
            "Visible in the account settings area. Looks unprofessional but nothing "
            "is functionally broken. Reported by our German localization reviewer."
        ),
        reporter="localization@company.com",
        timestamp="2024-03-01T11:00:00Z",   # LATER than CRI-007 → DUPLICATE of CRI-007
        product="Web App",
        version="v3.0.0",
        steps_to_reproduce="Set language to Deutsch, hover settings tooltips",
        expected_behavior="Normal word wrap",
        actual_behavior="Words broken incorrectly",
        environment_info={"locale": "de-DE"},
        customer_tier="starter",
        sla_hours_remaining=None,
    ),
    BugReport(
        id="CRI-013",
        title="Export function crashes for datasets over 10k rows",
        description=(
            "The data export feature returns HTTP 500 for any dataset with more than "
            "~10,000 rows. Smaller exports work fine. No error shown to the user — "
            "the download just fails. Stack trace shows a memory buffer overflow in "
            "the CSV serialiser."
        ),
        reporter="data_team@company.com",
        timestamp="2024-03-01T12:00:00Z",
        product="Reports Module",
        version="v4.2.0",
        steps_to_reproduce="1. Export a dataset with >10k rows\nResult: HTTP 500",
        expected_behavior="Export completes",
        actual_behavior="HTTP 500 — buffer overflow in CSV serialiser",
        environment_info={"service": "reports-service"},
        customer_tier="business",
        sla_hours_remaining=10.0,
    ),
    BugReport(
        id="CRI-014",
        title="App crashes intermittently — no clear pattern",
        description=(
            "My app just crashes randomly. I think it might be when I switch screens "
            "quickly but I'm not sure. Happens on my phone. I'm on a recent version I think."
        ),
        reporter="mobile_user99@hotmail.com",
        timestamp="2024-03-01T12:30:00Z",
        product="Mobile App",
        version="unknown",
        steps_to_reproduce=None,        # MISSING — request_info expected
        expected_behavior=None,
        actual_behavior="Random crash",
        environment_info=None,          # MISSING
        customer_tier="free",
        sla_hours_remaining=None,
    ),
    BugReport(
        id="CRI-015",
        title="Two-factor authentication codes rejected even when entered correctly",
        description=(
            "Enterprise customers using TOTP-based 2FA are reporting that valid codes "
            "are being rejected. The issue is intermittent (~30% failure rate) and "
            "correlates with auth service clock drift. Enterprise customers are "
            "intermittently locked out of their accounts."
        ),
        reporter="enterprise_support@company.com",
        timestamp="2024-03-01T13:00:00Z",
        product="Auth Service",
        version="v5.2.1",
        steps_to_reproduce=(
            "1. Enable TOTP 2FA as an enterprise user\n"
            "2. Log in and enter a valid TOTP code\n"
            "3. ~30% of the time: 'Invalid code' error"
        ),
        expected_behavior="Valid TOTP codes always accepted",
        actual_behavior="~30% rejection rate due to clock drift",
        environment_info={"service": "auth-service"},
        customer_tier="enterprise",
        sla_hours_remaining=2.0,
        linked_bug_ids=["CRI-001", "CRI-008"],
    ),
]

# Ground truth — one clean definition, no duplicates
_T3_GROUND_TRUTH: Dict[str, BugGroundTruth] = {
    # ── Auth cluster (CRI-001 / CRI-008 / CRI-015 share root cause in auth-service)
    "CRI-001": BugGroundTruth(severity="critical", team="security",        should_escalate=True,  sla_critical=True),
    "CRI-008": BugGroundTruth(severity="critical", team="security",        should_escalate=True,  sla_critical=True),
    "CRI-015": BugGroundTruth(severity="critical", team="security",        should_escalate=True,  sla_critical=True),

    # ── DB cluster (CRI-004 is original; CRI-011 is a duplicate)
    "CRI-004": BugGroundTruth(severity="critical", team="infrastructure",  should_escalate=True,  sla_critical=True),
    "CRI-011": BugGroundTruth(severity="critical", team="infrastructure",  should_escalate=True,  sla_critical=True,
                               is_duplicate_of="CRI-004"),

    # ── Firefox logo pair (CRI-003 is original; CRI-009 is later → duplicate)
    "CRI-003": BugGroundTruth(severity="low",      team="frontend"),
    "CRI-009": BugGroundTruth(severity="low",      team="frontend",        is_duplicate_of="CRI-003"),

    # ── German tooltip pair (CRI-007 is original; CRI-012 is later → duplicate)
    "CRI-007": BugGroundTruth(severity="low",      team="frontend"),
    "CRI-012": BugGroundTruth(severity="low",      team="frontend",        is_duplicate_of="CRI-007"),

    # ── Enterprise high-value bugs requiring escalation
    "CRI-006": BugGroundTruth(severity="high",     team="backend",         should_escalate=True),

    # ── High severity, no escalation required
    "CRI-002": BugGroundTruth(severity="high",     team="backend"),
    "CRI-010": BugGroundTruth(severity="high",     team="backend"),
    "CRI-013": BugGroundTruth(severity="high",     team="backend"),

    # ── Incomplete info — request_info before classifying
    "CRI-005": BugGroundTruth(severity="medium",   team="frontend",        needs_info=True),
    "CRI-014": BugGroundTruth(severity="low",      team="mobile",          needs_info=True),
}

_T3_INSTRUCTIONS = """\
You are the lead triage engineer managing a surge of 15 bug reports during a critical incident window.

Available action_types (ONE JSON action per step — no explanation text):
  classify      → {"action_type": "classify",      "bug_id": "...", "severity": "critical|high|medium|low"}
  assign        → {"action_type": "assign",         "bug_id": "...", "assigned_team": "backend|frontend|mobile|infrastructure|security|database|qa"}
  request_info  → {"action_type": "request_info",   "bug_id": "...", "info_requested": ["item1", "item2"]}
  mark_duplicate→ {"action_type": "mark_duplicate", "bug_id": "...", "duplicate_of": "ORIGINAL-ID"}
  escalate      → {"action_type": "escalate",       "bug_id": "...", "escalation_reason": "..."}
  submit        → {"action_type": "submit",          "bug_id": "..."}

Critical triage rules:
  1. Bugs with sla_hours_remaining < 2.0 → escalate immediately.
  2. Enterprise customer + critical or high severity → always escalate.
  3. Mark the LATER-submitted report as duplicate of the EARLIER one when root cause matches.
  4. If steps_to_reproduce AND environment_info are both absent → request_info before classifying.
  5. Linked bugs (linked_bug_ids) often share a root cause — align severity decisions.
  6. Submit each bug after classify + assign (+ optional actions) to free up your queue.

You must handle all 15 bugs correctly within 50 steps. Prioritise by SLA urgency first.
"""

SCENARIO_TASK3 = TaskScenario(
    task_id="sla-crisis",
    name="SLA Crisis: Mass Bug Surge with Escalations",
    description=(
        "A critical incident window has triggered 15 simultaneous bug reports. "
        "Multiple reports involve SLA breaches, enterprise customer impact, duplicate "
        "submissions, incomplete information, and linked root causes. "
        "Triage all bugs correctly within the step budget."
    ),
    difficulty="hard",
    max_steps=50,
    reward_threshold=0.50,
    bug_reports=_T3_BUGS,
    ground_truth=_T3_GROUND_TRUTH,
    instructions=_T3_INSTRUCTIONS,
)


# ---------------------------------------------------------------------------
# Public registry
# ---------------------------------------------------------------------------

SCENARIOS: Dict[str, TaskScenario] = {
    "single-triage": SCENARIO_TASK1,
    "batch-triage":  SCENARIO_TASK2,
    "sla-crisis":    SCENARIO_TASK3,
}

ALL_TASK_IDS: List[str] = list(SCENARIOS.keys())
