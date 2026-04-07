"""
Deterministic graders for the three contract review tasks.
Each grader scores agent performance on a 0.0–1.0 scale.
Uses multi-factor scoring: keyword coverage, sequence similarity, and length signals.
"""

from difflib import SequenceMatcher
from typing import List, Dict, Any



def _keyword_match_score(text: str, keywords: List[str]) -> float:
    """Check how many expected keywords appear in the text (case-insensitive)."""
    if not keywords or not text:
        return 0.001
    text_lower = text.lower()
    matches = sum(1 for kw in keywords if kw.lower() in text_lower)
    raw = matches / len(keywords)
    return round(min(0.999, max(0.001, raw)), 4)


def _amendment_quality_score(suggested: str, hint: str, keywords: List[str]) -> float:
    """
    Multi-factor amendment quality scoring:
    - Sequence similarity to hint text (40%)
    - Keyword coverage (40%)
    - Length adequacy — longer and more specific is better (20%)
    Returns a score strictly in (0, 1).
    """
    if not suggested or not suggested.strip():
        return 0.001

    s = suggested.lower().strip()
    h = hint.lower().strip()

    # 1. Sequence similarity to amendment hint
    seq_score = SequenceMatcher(None, s, h).ratio()

    # 2. Keyword coverage
    kw_score = _keyword_match_score(s, keywords) if keywords else 0.5

    # 3. Length adequacy — reward substantive amendments (>20 words)
    word_count = len(s.split())
    if word_count >= 20:
        length_bonus = 0.999
    elif word_count >= 10:
        length_bonus = 0.7
    elif word_count >= 5:
        length_bonus = 0.4
    else:
        length_bonus = 0.1

    combined = 0.40 * seq_score + 0.40 * kw_score + 0.20 * length_bonus
    return round(min(0.999, max(0.001, combined)), 4)


def _reasoning_quality_score(reasoning: str, keywords: List[str]) -> float:
    """Score reasoning quality: keyword coverage + length signal."""
    if not reasoning or not reasoning.strip():
        return 0.001
    kw = _keyword_match_score(reasoning, keywords) if keywords else 0.3
    words = len(reasoning.split())
    length = min(0.999, words / 30.0)  # Saturates at 30 words
    return round(min(0.999, max(0.001, 0.6 * kw + 0.4 * length)), 4)



def clause_identification_grader(
    reviews: List[Dict[str, Any]],
    ground_truth: Dict[str, List[Dict[str, Any]]],
) -> float:
    """
    EASY task grader: Precision/recall of flagged clauses.

    Scoring:
    - Recall (60%): What fraction of actual problematic clauses were flagged?
    - Precision (40%): What fraction of flagged clauses were actually problematic?
    """
    true_positive_ids = set(ground_truth.keys())
    flagged_ids = set()
    correct_flags = set()

    for review in reviews:
        cid = review.get("clause_id", "")
        action = review.get("action_type", "")
        if action in ("flag_risk", "suggest_amendment", "reject"):
            flagged_ids.add(cid)
            if cid in true_positive_ids:
                correct_flags.add(cid)

    # Recall: of all issues, how many did we find?
    recall = len(correct_flags) / len(true_positive_ids) if true_positive_ids else 0.999

    # Precision: of all flags, how many were correct?
    precision = len(correct_flags) / len(flagged_ids) if flagged_ids else 0.001

    # If agent flagged nothing at all, return minimum score
    if not flagged_ids and true_positive_ids:
        return 0.001

    score = 0.6 * recall + 0.4 * precision
    return round(min(0.999, max(0.001, score)), 4)


def risk_assessment_grader(
    reviews: List[Dict[str, Any]],
    ground_truth: Dict[str, List[Dict[str, Any]]],
) -> float:
    """
    MEDIUM task grader: Clause detection + severity accuracy + reasoning quality.

    Scoring:
    - Clause detection (50%): Precision/recall of flagged clauses
    - Severity accuracy (30%): Did the agent correctly classify severity?
    - Reasoning quality (20%): Do the agent's reasons mention key concepts?
    """
    true_positive_ids = set(ground_truth.keys())

    flagged_ids = set()
    correct_flags = set()
    severity_correct = 0
    severity_total = 0
    reasoning_scores = []

    for review in reviews:
        cid = review.get("clause_id", "")
        action = review.get("action_type", "")

        if action in ("flag_risk", "suggest_amendment", "reject"):
            flagged_ids.add(cid)
            if cid in true_positive_ids:
                correct_flags.add(cid)

                # Check severity
                issues = ground_truth[cid]
                expected_severity = issues[0].get("severity", "")
                actual_severity = review.get("severity", "")
                severity_total += 1
                if actual_severity == expected_severity:
                    severity_correct += 1

                # Check reasoning quality: keyword coverage + length
                reasoning = review.get("reasoning", "")
                keywords = issues[0].get("keywords", [])
                reasoning_scores.append(_reasoning_quality_score(reasoning, keywords))

    # Detection: precision/recall
    recall = len(correct_flags) / len(true_positive_ids) if true_positive_ids else 0.999
    precision = len(correct_flags) / len(flagged_ids) if flagged_ids else 0.001
    detection = 0.6 * recall + 0.4 * precision

    # Severity accuracy
    severity_score = severity_correct / severity_total if severity_total > 0 else 0.001
    severity_score = round(min(0.999, max(0.001, severity_score)), 4)

    # Reasoning quality
    reasoning_avg = sum(reasoning_scores) / len(reasoning_scores) if reasoning_scores else 0.001
    reasoning_avg = round(min(0.999, max(0.001, reasoning_avg)), 4)

    score = 0.50 * detection + 0.30 * severity_score + 0.20 * reasoning_avg
    return round(min(0.999, max(0.001, score)), 4)


def negotiation_grader(
    reviews: List[Dict[str, Any]],
    ground_truth: Dict[str, List[Dict[str, Any]]],
) -> float:
    """
    HARD task grader: Detection + severity + amendment quality + reasoning + precision.

    Scoring:
    - Clause detection / recall (25%): Did agent find all problematic clauses?
    - Severity accuracy (15%): Was severity correctly classified?
    - Amendment quality (35%): Multi-factor: seq similarity + keyword overlap + length
    - Reasoning quality (10%): Were the explanations specific and on-point?
    - Precision / no false positives (15%): Avoiding approving bad clauses
    """
    true_positive_ids = set(ground_truth.keys())
    total_clauses_reviewed = len(reviews)

    flagged_ids = set()
    correct_flags = set()
    severity_correct = 0
    severity_total = 0
    amendment_scores = []
    reasoning_scores = []
    false_positives = 0

    for review in reviews:
        cid = review.get("clause_id", "")
        action = review.get("action_type", "")

        if action in ("flag_risk", "suggest_amendment", "reject"):
            flagged_ids.add(cid)
            if cid in true_positive_ids:
                correct_flags.add(cid)

                issues = ground_truth[cid]
                expected_severity = issues[0].get("severity", "")
                actual_severity = review.get("severity", "")
                severity_total += 1
                if actual_severity == expected_severity:
                    severity_correct += 1

                # Amendment quality — multi-factor scoring
                suggested = review.get("suggested_text", "") or ""
                hint = issues[0].get("amendment_hint", "")
                keywords = issues[0].get("keywords", [])
                if action == "suggest_amendment" and suggested.strip():
                    amend_score = _amendment_quality_score(suggested, hint, keywords)
                    amendment_scores.append(amend_score)
                elif action == "suggest_amendment":
                    amendment_scores.append(0.05)   # Attempted but empty text
                else:
                    amendment_scores.append(0.001)  # Only flagged, no amendment text

                # Also score reasoning quality for negotiation
                reasoning = review.get("reasoning", "") or ""
                reasoning_scores.append(_reasoning_quality_score(reasoning, keywords))
            else:
                false_positives += 1

    # Detection (recall)
    recall = len(correct_flags) / len(true_positive_ids) if true_positive_ids else 0.999

    # Severity
    severity_score = severity_correct / severity_total if severity_total > 0 else 0.001

    # Amendment quality
    amendment_avg = sum(amendment_scores) / len(amendment_scores) if amendment_scores else 0.001

    # Precision (penalize false positives)
    non_issue_clauses = total_clauses_reviewed - len(true_positive_ids)
    if non_issue_clauses > 0:
        precision_score = 1.0 - (false_positives / non_issue_clauses)
    else:
        precision_score = 0.999 if false_positives == 0 else 0.001

    # Reasoning quality
    reasoning_avg = sum(reasoning_scores) / len(reasoning_scores) if reasoning_scores else 0.001
    reasoning_avg = round(min(0.999, max(0.001, reasoning_avg)), 4)

    score = (
        0.25 * recall
        + 0.15 * severity_score
        + 0.35 * amendment_avg
        + 0.10 * reasoning_avg
        + 0.15 * max(0.001, precision_score)
    )
    return round(min(0.999, max(0.001, score)), 4)


# ── Grader registry ───────────────────────────────────────────────────────────

GRADERS = {
    "clause_identification": clause_identification_grader,
    "risk_assessment": risk_assessment_grader,
    "negotiation": negotiation_grader,
}


def grade_episode(
    task_id: str,
    reviews: List[Dict[str, Any]],
    ground_truth: Dict[str, List[Dict[str, Any]]],
) -> float:
    """Run the appropriate grader for the given task."""
    if task_id not in GRADERS:
        raise ValueError(f"Unknown task: {task_id}. Available: {list(GRADERS.keys())}")
    return GRADERS[task_id](reviews, ground_truth)
