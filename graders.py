"""
Deterministic graders for the three contract review tasks.
Each grader scores agent performance on a 0.0–1.0 scale.
"""

from typing import List, Dict, Any


def _keyword_match_score(text: str, keywords: List[str]) -> float:
    """Check how many expected keywords appear in the text (case-insensitive)."""
    if not keywords or not text:
        return 0.0
    text_lower = text.lower()
    matches = sum(1 for kw in keywords if kw.lower() in text_lower)
    return matches / len(keywords)


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
    recall = len(correct_flags) / len(true_positive_ids) if true_positive_ids else 1.0

    # Precision: of all flags, how many were correct?
    precision = len(correct_flags) / len(flagged_ids) if flagged_ids else 0.0

    # If agent flagged nothing, precision is 0 and recall is 0
    if not flagged_ids and true_positive_ids:
        return 0.0

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

                # Check reasoning quality via keyword match
                reasoning = review.get("reasoning", "")
                keywords = issues[0].get("keywords", [])
                reasoning_scores.append(_keyword_match_score(reasoning, keywords))

    # Detection: precision/recall
    recall = len(correct_flags) / len(true_positive_ids) if true_positive_ids else 1.0
    precision = len(correct_flags) / len(flagged_ids) if flagged_ids else 0.0
    detection = 0.6 * recall + 0.4 * precision

    # Severity accuracy
    severity_score = severity_correct / severity_total if severity_total > 0 else 0.0

    # Reasoning quality
    reasoning_avg = sum(reasoning_scores) / len(reasoning_scores) if reasoning_scores else 0.0

    score = 0.50 * detection + 0.30 * severity_score + 0.20 * reasoning_avg
    return round(min(0.999, max(0.001, score)), 4)


def negotiation_grader(
    reviews: List[Dict[str, Any]],
    ground_truth: Dict[str, List[Dict[str, Any]]],
) -> float:
    """
    HARD task grader: Detection + severity + amendment quality + precision.

    Scoring:
    - Clause detection (30%): Recall of problematic clauses
    - Severity accuracy (20%): Correct severity classification
    - Amendment quality (30%): Quality of suggested amendments
    - Precision / no false positives (20%): Avoiding false flags
    """
    true_positive_ids = set(ground_truth.keys())
    total_clauses_reviewed = len(reviews)

    flagged_ids = set()
    correct_flags = set()
    severity_correct = 0
    severity_total = 0
    amendment_scores = []
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

                # Amendment quality
                suggested = review.get("suggested_text", "") or ""
                hint = issues[0].get("amendment_hint", "")
                if action == "suggest_amendment" and suggested:
                    # Score based on keyword overlap with amendment hint
                    hint_words = set(hint.lower().split())
                    suggested_words = set(suggested.lower().split())
                    if hint_words:
                        overlap = len(hint_words & suggested_words) / len(hint_words)
                    else:
                        overlap = 0.0
                    amendment_scores.append(min(1.0, overlap * 1.5))  # Slight boost
                elif action == "suggest_amendment":
                    amendment_scores.append(0.2)  # Tried but empty
                else:
                    amendment_scores.append(0.0)  # Flagged but no amendment
            else:
                false_positives += 1

    # Detection (recall)
    recall = len(correct_flags) / len(true_positive_ids) if true_positive_ids else 1.0

    # Severity
    severity_score = severity_correct / severity_total if severity_total > 0 else 0.0

    # Amendment quality
    amendment_avg = sum(amendment_scores) / len(amendment_scores) if amendment_scores else 0.0

    # Precision (penalize false positives)
    non_issue_clauses = total_clauses_reviewed - len(true_positive_ids)
    if non_issue_clauses > 0:
        precision_score = 1.0 - (false_positives / non_issue_clauses)
    else:
        precision_score = 1.0 if false_positives == 0 else 0.0

    score = (
        0.30 * recall
        + 0.20 * severity_score
        + 0.30 * amendment_avg
        + 0.20 * max(0.0, precision_score)
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
