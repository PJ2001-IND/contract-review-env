"""
Synthetic contract data with planted issues for 3 difficulty tiers.
Each contract contains clauses with known ground-truth issues for deterministic grading.
"""

from typing import List, Dict, Any


def _clause(id: str, title: str, text: str, issues: List[Dict[str, Any]] = None):
    """Helper to build a clause dict."""
    return {
        "id": id,
        "title": title,
        "text": text,
        "issues": issues or [],
    }


# ══════════════════════════════════════════════════════════════════════════════
# TASK 1 — EASY: Clause Identification
# Simple SaaS agreement with 5 clauses, 3 obvious red flags
# ══════════════════════════════════════════════════════════════════════════════

EASY_CONTRACT = {
    "id": "saas_basic_001",
    "title": "SaaS Subscription Agreement – CloudSync Pro",
    "clauses": [
        _clause(
            "c1", "Service Description",
            "CloudSync Pro provides cloud-based file synchronization and backup services. "
            "The service includes 100GB of storage, automatic syncing across devices, "
            "and 24/7 customer support via email. Uptime SLA of 99.5% applies.",
            # No issues — clean clause
        ),
        _clause(
            "c2", "Liability",
            "The Provider shall bear UNLIMITED LIABILITY for any and all damages, losses, "
            "or claims arising from the use of the Service, including but not limited to "
            "indirect, consequential, special, and punitive damages, regardless of whether "
            "such damages were foreseeable. There is no cap on the Provider's total liability.",
            issues=[{
                "type": "unlimited_liability",
                "severity": "critical",
                "description": "Unlimited liability clause exposes the provider to unbounded financial risk. Standard practice is to cap liability at the total fees paid.",
                "keywords": ["unlimited liability", "no cap", "punitive damages"],
                "expected_action": "flag_risk",
                "amendment_hint": "Liability should be capped at the total fees paid in the preceding 12 months.",
            }],
        ),
        _clause(
            "c3", "Term and Renewal",
            "This Agreement shall commence on the Effective Date and continue for an initial "
            "term of 12 months. Upon expiration, the Agreement SHALL AUTOMATICALLY RENEW for "
            "successive 3-year periods. Cancellation requires 180 days written notice prior to "
            "the renewal date. Failure to provide timely notice results in the Subscriber being "
            "locked into the full renewal term with no early termination option.",
            issues=[{
                "type": "auto_renewal_lock",
                "severity": "critical",
                "description": "Auto-renewal for 3-year periods with 180-day cancellation notice is highly unfavorable. Locks subscriber into long commitments.",
                "keywords": ["automatically renew", "3-year", "180 days", "locked", "no early termination"],
                "expected_action": "flag_risk",
                "amendment_hint": "Renewal should be annual (not 3-year), with 30-day cancellation notice and option for early termination with prorated refund.",
            }],
        ),
        _clause(
            "c4", "Data Privacy",
            "The Provider shall process Subscriber data in accordance with applicable data "
            "protection regulations including GDPR and CCPA. Data is encrypted at rest (AES-256) "
            "and in transit (TLS 1.3). The Provider shall notify the Subscriber of any data "
            "breach within 72 hours of discovery.",
            # No issues — clean clause
        ),
        _clause(
            "c5", "Intellectual Property",
            "All data, content, code, algorithms, and derivative works uploaded to or processed "
            "by the Service shall become the EXCLUSIVE INTELLECTUAL PROPERTY of the Provider. "
            "The Subscriber irrevocably assigns all rights, title, and interest in such materials "
            "to the Provider, including the right to use, modify, sell, or sublicense without "
            "restriction or compensation to the Subscriber.",
            issues=[{
                "type": "ip_transfer",
                "severity": "critical",
                "description": "Full IP transfer clause strips the subscriber of ownership of their own data and content. Highly unfavorable and unusual.",
                "keywords": ["exclusive intellectual property", "irrevocably assigns", "all rights", "without restriction"],
                "expected_action": "flag_risk",
                "amendment_hint": "Subscriber should retain all IP rights. Provider may have a limited license only for service operation purposes.",
            }],
        ),
    ],
    "metadata": {
        "difficulty": "easy",
        "total_issues": 3,
        "task_id": "clause_identification",
        "task_description": (
            "Review this SaaS Subscription Agreement. For each clause, decide whether to "
            "'approve' it (if it looks standard and fair) or 'flag_risk' (if it contains "
            "problematic terms). When flagging a risk, specify the severity (critical/moderate/minor) "
            "and explain your reasoning."
        ),
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# TASK 2 — MEDIUM: Risk Assessment
# Enterprise software license with 10 clauses, 5 issues of varying severity
# ══════════════════════════════════════════════════════════════════════════════

MEDIUM_CONTRACT = {
    "id": "enterprise_sw_002",
    "title": "Enterprise Software License Agreement – DataVault Analytics",
    "clauses": [
        _clause(
            "c1", "License Grant",
            "Provider grants Licensee a non-exclusive, non-transferable license to use "
            "DataVault Analytics software for internal business purposes. The license covers "
            "up to 50 named users. Additional users may be added at the then-current list price.",
            # Clean
        ),
        _clause(
            "c2", "Fees and Payment",
            "Licensee shall pay annual license fees of $50,000, due within 30 days of invoice. "
            "The Provider reserves the right to INCREASE FEES BY UP TO 40% upon each annual "
            "renewal without prior approval from the Licensee. Late payments accrue interest "
            "at 2% per month.",
            issues=[{
                "type": "excessive_price_increase",
                "severity": "moderate",
                "description": "40% annual price increase without approval is excessive. Industry standard is 3-5% or CPI-linked.",
                "keywords": ["increase fees", "40%", "without prior approval"],
                "expected_action": "flag_risk",
                "amendment_hint": "Cap annual fee increases at 5% or CPI, whichever is lower, with 60-day notice.",
            }],
        ),
        _clause(
            "c3", "Support and Maintenance",
            "Provider shall provide standard support (9am-5pm EST, weekdays) and quarterly "
            "software updates. Critical security patches will be released within 48 hours of "
            "vulnerability disclosure. Support tickets receive initial response within 4 business hours.",
            # Clean
        ),
        _clause(
            "c4", "Data Handling",
            "All data processed by the Software remains the property of the Licensee. "
            "However, the Provider may use ANONYMIZED AND AGGREGATED data derived from "
            "Licensee's usage for product improvement, benchmarking, and may SHARE SUCH "
            "DATA WITH THIRD PARTIES for research purposes without prior notice.",
            issues=[{
                "type": "data_sharing",
                "severity": "moderate",
                "description": "Sharing anonymized data with third parties without notice is concerning. Re-identification risk exists, and no opt-out is provided.",
                "keywords": ["share such data", "third parties", "without prior notice", "anonymized"],
                "expected_action": "flag_risk",
                "amendment_hint": "Require opt-in consent for any third-party data sharing, even anonymized. Provide quarterly transparency reports.",
            }],
        ),
        _clause(
            "c5", "Audit Rights",
            "Provider may audit Licensee's use of the Software at any time, with 7 days' "
            "written notice, to verify compliance with license terms. Audits shall be conducted "
            "during normal business hours on Licensee's premises.",
            # Clean
        ),
        _clause(
            "c6", "Indemnification",
            "Licensee shall INDEMNIFY, DEFEND, AND HOLD HARMLESS the Provider against any "
            "and all third-party claims, damages, losses, and expenses (including attorneys' "
            "fees) arising from Licensee's use of the Software, EVEN IF SUCH CLAIMS ARISE "
            "FROM DEFECTS OR ERRORS IN THE SOFTWARE itself.",
            issues=[{
                "type": "one_sided_indemnification",
                "severity": "critical",
                "description": "Licensee indemnifies Provider even for Provider's own software defects. This is extremely one-sided and unusual.",
                "keywords": ["indemnify", "hold harmless", "even if", "defects or errors"],
                "expected_action": "flag_risk",
                "amendment_hint": "Each party should indemnify the other for claims arising from their own actions. Provider should indemnify for IP infringement and software defects.",
            }],
        ),
        _clause(
            "c7", "Confidentiality",
            "Both parties agree to maintain confidentiality of proprietary information "
            "disclosed during the term of this Agreement. Confidential information shall "
            "not be disclosed to third parties without prior written consent. This obligation "
            "survives for 3 years after termination.",
            # Clean
        ),
        _clause(
            "c8", "Termination",
            "Either party may terminate this Agreement for material breach with 30 days' "
            "written notice and opportunity to cure. Upon termination, Provider shall return "
            "or destroy all Licensee data within 30 days.",
            # Clean — standard termination clause
        ),
        _clause(
            "c9", "Limitation of Liability",
            "Provider's total liability under this Agreement shall not exceed the fees paid "
            "in the preceding 6 months. PROVIDER SHALL NOT BE LIABLE FOR ANY LOSS OF DATA, "
            "EVEN IF DUE TO PROVIDER'S NEGLIGENCE OR WILLFUL MISCONDUCT.",
            issues=[{
                "type": "negligence_exclusion",
                "severity": "critical",
                "description": "Excluding liability for data loss due to Provider's own negligence or willful misconduct is unreasonable.",
                "keywords": ["not be liable", "loss of data", "negligence", "willful misconduct"],
                "expected_action": "flag_risk",
                "amendment_hint": "Liability exclusion should not apply in cases of gross negligence, willful misconduct, or breach of data protection obligations.",
            }],
        ),
        _clause(
            "c10", "Governing Law",
            "This Agreement shall be governed by and construed in accordance with the laws "
            "of the State of Delaware, USA. Any disputes shall be resolved through binding "
            "arbitration administered by the PROVIDER'S CHOSEN ARBITRATION BODY, with hearings "
            "held exclusively at Provider's headquarters.",
            issues=[{
                "type": "biased_dispute_resolution",
                "severity": "minor",
                "description": "Arbitration body chosen by Provider and location at Provider's headquarters creates bias. Should be neutral.",
                "keywords": ["provider's chosen", "exclusively at provider's headquarters"],
                "expected_action": "flag_risk",
                "amendment_hint": "Use a neutral arbitration body (e.g., AAA or JAMS). Allow hearings at a mutually agreed location or remote.",
            }],
        ),
    ],
    "metadata": {
        "difficulty": "medium",
        "total_issues": 5,
        "task_id": "risk_assessment",
        "task_description": (
            "Review this Enterprise Software License Agreement. For each clause, determine if "
            "it is acceptable ('approve') or contains risks ('flag_risk'). For flagged clauses, "
            "you MUST accurately classify the severity as 'critical', 'moderate', or 'minor'. "
            "Provide detailed reasoning explaining the specific risk."
        ),
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# TASK 3 — HARD: Negotiation
# Complex vendor agreement with 15 clauses, 7 issues, some interdependent
# ══════════════════════════════════════════════════════════════════════════════

HARD_CONTRACT = {
    "id": "vendor_complex_003",
    "title": "Master Services Agreement – OmniTech Solutions",
    "clauses": [
        _clause(
            "c1", "Scope of Services",
            "OmniTech shall provide managed IT infrastructure services including server "
            "management, network monitoring, cloud migration, and cybersecurity operations. "
            "Services are delivered as described in each Statement of Work (SOW) executed "
            "under this Agreement.",
            # Clean
        ),
        _clause(
            "c2", "Service Level Agreement",
            "OmniTech commits to 99.9% uptime for all managed services. However, PLANNED "
            "MAINTENANCE WINDOWS OF UP TO 72 HOURS PER MONTH are excluded from uptime "
            "calculations. OmniTech may schedule maintenance at its sole discretion without "
            "advance notice to the Client.",
            issues=[{
                "type": "excessive_maintenance_window",
                "severity": "moderate",
                "description": "72 hours/month of excluded maintenance effectively reduces the real SLA to ~90%. No advance notice requirement is problematic.",
                "keywords": ["72 hours", "excluded from uptime", "sole discretion", "without advance notice"],
                "expected_action": "suggest_amendment",
                "amendment_hint": "Limit planned maintenance to 8 hours/month, require 48-hour advance notice, and include only pre-scheduled windows in the exclusion.",
            }],
        ),
        _clause(
            "c3", "Fees and Payment Terms",
            "Monthly service fees are specified in each SOW. All fees are non-refundable. "
            "Client shall pay within 15 days of invoice receipt. A SERVICE MANAGEMENT FEE "
            "OF 25% OF TOTAL CONTRACT VALUE is payable upon signing and is non-refundable "
            "even if the Agreement is terminated within the first 30 days.",
            issues=[{
                "type": "upfront_nonrefundable_fee",
                "severity": "critical",
                "description": "25% non-refundable upfront fee with no performance guarantee creates significant financial risk for the Client.",
                "keywords": ["25%", "non-refundable", "upon signing", "total contract value"],
                "expected_action": "suggest_amendment",
                "amendment_hint": "Reduce upfront fee to 10% with 50% refundable within 60-day trial period. Or restructure as monthly payments.",
            }],
        ),
        _clause(
            "c4", "Staffing",
            "OmniTech shall assign qualified personnel to deliver the Services. OmniTech "
            "may reassign or replace any personnel at any time without Client approval. "
            "The Client shall have no right to request specific team members or object to "
            "personnel changes.",
            issues=[{
                "type": "no_staffing_control",
                "severity": "minor",
                "description": "Client has no say in personnel changes. For managed services, key personnel continuity matters.",
                "keywords": ["without client approval", "no right to request", "object to personnel changes"],
                "expected_action": "suggest_amendment",
                "amendment_hint": "Key personnel changes should require 14-day notice. Client may request (not require) specific team members for continuity.",
            }],
        ),
        _clause(
            "c5", "Intellectual Property",
            "Any tools, scripts, automation frameworks, or methodologies developed by OmniTech "
            "during the engagement remain OmniTech's exclusive property. Client receives a "
            "non-exclusive license to use deliverables specified in the SOW.",
            # Clean — standard vendor IP clause
        ),
        _clause(
            "c6", "Data Security",
            "OmniTech shall implement industry-standard security measures including encryption, "
            "access controls, and regular vulnerability assessments. OmniTech's security "
            "certifications include SOC 2 Type II and ISO 27001.",
            # Clean
        ),
        _clause(
            "c7", "Data Breach Notification",
            "In the event of a data breach affecting Client data, OmniTech shall notify the "
            "Client WITHIN 30 BUSINESS DAYS of discovering the breach. OmniTech shall have "
            "no obligation to notify affected end-users or regulatory bodies; such obligations "
            "rest solely with the Client.",
            issues=[{
                "type": "slow_breach_notification",
                "severity": "critical",
                "description": "30 business days (6+ weeks) for breach notification is far too slow. GDPR requires 72 hours. Shifting all notification obligations to Client is also problematic.",
                "keywords": ["30 business days", "no obligation to notify", "solely with the client"],
                "expected_action": "suggest_amendment",
                "amendment_hint": "Notify within 48 hours of discovery. Vendor should assist with regulatory notifications and provide forensic report within 5 business days.",
            }],
        ),
        _clause(
            "c8", "Subcontracting",
            "OmniTech may subcontract any portion of the Services to third parties without "
            "Client consent. Subcontractors are bound by confidentiality obligations equivalent "
            "to those in this Agreement.",
            # Mild concern but we'll leave it clean for balance
        ),
        _clause(
            "c9", "Limitation of Liability",
            "OmniTech's aggregate liability shall not exceed the fees paid in the 3 months "
            "preceding the claim. THIS CAP APPLIES EVEN TO DATA BREACHES, REGULATORY "
            "FINES, AND GROSS NEGLIGENCE. Client expressly waives any right to claim "
            "consequential, incidental, or exemplary damages.",
            issues=[{
                "type": "liability_cap_on_breach",
                "severity": "critical",
                "description": "3-month fee cap on liability even for data breaches and gross negligence is unreasonably low. Should be higher for data security incidents.",
                "keywords": ["3 months", "even to data breaches", "regulatory fines", "gross negligence", "waives"],
                "expected_action": "suggest_amendment",
                "amendment_hint": "General cap at 12 months' fees. Separate higher cap (2x annual fees) for data breaches and security incidents. Exclude gross negligence and willful misconduct from cap.",
                "depends_on": ["c7"],  # Related to breach notification clause
            }],
        ),
        _clause(
            "c10", "Term and Renewal",
            "Initial term of 36 months. Automatically renews for successive 24-month periods. "
            "Termination for convenience requires 12 months' written notice and PAYMENT OF "
            "AN EARLY TERMINATION FEE EQUAL TO 50% OF REMAINING CONTRACT VALUE.",
            issues=[{
                "type": "punitive_termination",
                "severity": "critical",
                "description": "50% early termination fee combined with 12-month notice on a 36-month contract is very punitive. Effectively locks client in.",
                "keywords": ["12 months' notice", "50%", "early termination fee", "remaining contract value"],
                "expected_action": "suggest_amendment",
                "amendment_hint": "Reduce notice to 90 days. Early termination fee should be 3 months' fees (not % of remaining). Allow termination for cause with 30 days' notice and no fee.",
                "depends_on": ["c3"],  # Related to upfront fee
            }],
        ),
        _clause(
            "c11", "Force Majeure",
            "Neither party shall be liable for delays or failures caused by events beyond "
            "reasonable control, including natural disasters, pandemics, government actions, "
            "or cyber attacks. During force majeure, OmniTech may suspend services without "
            "fee adjustments.",
            # Clean — standard force majeure (the no-fee-adjustment is borderline but not flagged for this task)
        ),
        _clause(
            "c12", "Insurance",
            "OmniTech maintains commercial general liability insurance of $1M per occurrence "
            "and professional liability (E&O) insurance of $2M aggregate. Client may request "
            "certificates of insurance upon written request.",
            # Clean
        ),
        _clause(
            "c13", "Non-Solicitation",
            "During the term and for 24 months after termination, neither party shall "
            "solicit or hire the other party's employees who were involved in the Services. "
            "Violation incurs a penalty of 100% of the hired employee's annual compensation.",
            issues=[{
                "type": "excessive_non_solicit",
                "severity": "minor",
                "description": "24-month non-solicitation with 100% salary penalty is aggressive. Industry standard is 12 months. Penalty should be reasonable.",
                "keywords": ["24 months", "100%", "annual compensation"],
                "expected_action": "suggest_amendment",
                "amendment_hint": "Reduce non-solicitation period to 12 months. Penalty should be 50% of 6 months' compensation. Apply only to direct solicitation, not unsolicited applications.",
            }],
        ),
        _clause(
            "c14", "Governing Law and Disputes",
            "This Agreement is governed by the laws of the State of California, USA. "
            "Disputes shall be resolved through mediation first, then binding arbitration "
            "under AAA Commercial Rules. Each party bears its own costs.",
            # Clean — standard dispute resolution
        ),
        _clause(
            "c15", "Entire Agreement",
            "This Agreement, together with all SOWs and amendments, constitutes the entire "
            "agreement between the parties. No modification shall be effective unless in "
            "writing and signed by both parties.",
            # Clean
        ),
    ],
    "metadata": {
        "difficulty": "hard",
        "total_issues": 7,
        "task_id": "negotiation",
        "task_description": (
            "Review this Master Services Agreement as a contract negotiator. For each clause, "
            "either 'approve' it or 'suggest_amendment' with specific replacement text. Your "
            "amendments must be legally sound and commercially reasonable. Consider how clauses "
            "relate to each other (e.g., liability caps affect breach notification remedies). "
            "Classify severity and provide detailed reasoning."
        ),
    },
}


# ── Task registry ─────────────────────────────────────────────────────────────

TASKS = {
    "clause_identification": {
        "contract": EASY_CONTRACT,
        "difficulty": "easy",
        "description": "Identify problematic clauses in a simple SaaS agreement",
    },
    "risk_assessment": {
        "contract": MEDIUM_CONTRACT,
        "difficulty": "medium",
        "description": "Assess risk severity in an enterprise software license",
    },
    "negotiation": {
        "contract": HARD_CONTRACT,
        "difficulty": "hard",
        "description": "Negotiate amendments in a complex vendor agreement",
    },
}


def get_task_ids() -> List[str]:
    return list(TASKS.keys())


def get_contract_for_task(task_id: str) -> dict:
    if task_id not in TASKS:
        raise ValueError(f"Unknown task: {task_id}. Available: {get_task_ids()}")
    return TASKS[task_id]["contract"]


def get_ground_truth_issues(contract: dict) -> Dict[str, List[dict]]:
    """Returns a mapping of clause_id -> list of issues for all clauses with issues."""
    result = {}
    for clause in contract["clauses"]:
        if clause["issues"]:
            result[clause["id"]] = clause["issues"]
    return result


from typing import Dict  # noqa: E402 (already imported above, just for clarity)
