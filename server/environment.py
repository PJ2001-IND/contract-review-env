"""
Core Contract Review environment implementing the OpenEnv Environment interface.
Manages state, processes actions, computes rewards, and drives episode flow.
"""

import uuid
from typing import Optional, Any, List

# Support both in-repo and standalone imports
try:
    from openenv.core.env_server.interfaces import Action, Environment, Observation
except ImportError:
    Environment = object  # Fallback: no ABC constraint

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import (
    ContractAction,
    ContractObservation,
    ContractState,
)
from contracts import get_contract_for_task, get_ground_truth_issues, get_task_ids
from graders import grade_episode


class ContractReviewEnvironment(Environment):
    """
    OpenEnv-compatible environment for contract review.

    The agent reviews a contract clause-by-clause, taking actions
    (approve, flag_risk, suggest_amendment, reject) on each clause.
    """

    SUPPORTS_CONCURRENT_SESSIONS = True
    
    # Store the last completed score globally for the stateless /grader HTTP endpoint
    _global_last_grader_score: Optional[float] = None
    _global_last_task_id: Optional[str] = None

    # ── Reward shaping constants ──────────────────────────────────────────
    REWARD_CORRECT_FLAG = 0.15
    REWARD_CORRECT_SEVERITY = 0.10
    REWARD_GOOD_AMENDMENT = 0.10
    PENALTY_FALSE_POSITIVE = -0.05
    PENALTY_PER_STEP = -0.02
    REWARD_COMPLETION_BONUS = 0.20

    def __init__(self):
        self._state = ContractState()
        self._contract: Optional[dict] = None
        self._clauses: List[dict] = []
        self._current_clause_idx: int = 0
        self._ground_truth: dict = {}
        self._reviews: list = []
        self._raw_reward: float = 0.0
        self._done: bool = False
        self._last_grader_score: Optional[float] = None

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_id: str = "clause_identification",
        **kwargs: Any,
    ) -> ContractObservation:
        """Reset the environment and return initial observation."""
        if task_id not in get_task_ids():
            task_id = "clause_identification"

        self._contract = get_contract_for_task(task_id)
        self._clauses = self._contract["clauses"]
        self._ground_truth = get_ground_truth_issues(self._contract)
        self._current_clause_idx = 0
        self._reviews = []
        self._raw_reward = 0.0
        self._done = False
        self._last_grader_score = None

        self._state = ContractState(
            episode_id=episode_id or str(uuid.uuid4()),
            step_count=0,
            task_id=task_id,
            contract_id=self._contract["id"],
            total_issues=self._contract["metadata"]["total_issues"],
            issues_found=0,
            correct_severities=0,
            false_positives=0,
            amendments_suggested=0,
            cumulative_reward=0.0,
        )

        clause = self._clauses[0]
        return ContractObservation(
            done=False,
            reward=None,
            contract_title=self._contract["title"],
            contract_text=self._format_full_contract(),
            current_clause_id=clause["id"],
            current_clause_title=clause["title"],
            current_clause_text=clause["text"],
            clause_index=0,
            total_clauses=len(self._clauses),
            reviewed_clauses=[],
            task_id=task_id,
            task_description=self._contract["metadata"]["task_description"],
            message=f"Review started. You have {len(self._clauses)} clauses to review. Begin with clause '{clause['title']}'.",
        )

    def step(
        self,
        action: ContractAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> ContractObservation:
        """Take a step in the environment."""
        if self._done:
            return self._make_done_observation("Episode already completed.")

        # Validate action type
        if not isinstance(action, ContractAction):
            # Try to coerce from dict/Action
            if isinstance(action, dict):
                action = ContractAction(**action)
            elif hasattr(action, 'model_dump'):
                action = ContractAction(**action.model_dump())

        self._state.step_count += 1
        step_reward = self.PENALTY_PER_STEP  # Time pressure

        current_clause = self._clauses[self._current_clause_idx]
        clause_id = current_clause["id"]

        # Score the action against ground truth
        has_issue = clause_id in self._ground_truth
        is_flag = action.action_type in ("flag_risk", "suggest_amendment", "reject")

        message_parts = []

        if has_issue and is_flag:
            # TRUE POSITIVE
            step_reward += self.REWARD_CORRECT_FLAG
            self._state.issues_found += 1
            message_parts.append("Correctly identified a problematic clause.")

            expected_severity = self._ground_truth[clause_id][0].get("severity", "")
            if action.severity == expected_severity:
                step_reward += self.REWARD_CORRECT_SEVERITY
                self._state.correct_severities += 1
                message_parts.append(f"Severity '{action.severity}' is correct.")
            else:
                message_parts.append(f"Severity '{action.severity}' — expected '{expected_severity}'.")

            if action.action_type == "suggest_amendment" and action.suggested_text:
                self._state.amendments_suggested += 1
                hint = self._ground_truth[clause_id][0].get("amendment_hint", "")
                hint_words = set(hint.lower().split())
                suggested_words = set(action.suggested_text.lower().split())
                if hint_words:
                    overlap = len(hint_words & suggested_words) / len(hint_words)
                    if overlap > 0.2:
                        step_reward += self.REWARD_GOOD_AMENDMENT
                        message_parts.append("Amendment suggestion addresses key concerns.")

        elif not has_issue and is_flag:
            # FALSE POSITIVE
            step_reward += self.PENALTY_FALSE_POSITIVE
            self._state.false_positives += 1
            message_parts.append("This clause appears standard — flagging may be a false positive.")

        elif has_issue and not is_flag:
            # FALSE NEGATIVE
            message_parts.append("Clause approved.")

        else:
            # TRUE NEGATIVE
            message_parts.append("Clause approved — looks standard.")

        # Record review
        review = {
            "clause_id": clause_id,
            "clause_title": current_clause["title"],
            "action_type": action.action_type,
            "severity": action.severity,
            "reasoning": action.reasoning,
            "suggested_text": action.suggested_text,
            "reward_earned": step_reward,
        }
        self._reviews.append(review)

        # Accumulate reward
        self._raw_reward += step_reward
        self._state.cumulative_reward = self._raw_reward

        # Advance to next clause
        self._current_clause_idx += 1

        if self._current_clause_idx >= len(self._clauses):
            # Episode complete
            self._done = True
            self._raw_reward += self.REWARD_COMPLETION_BONUS

            # Run grader
            self._last_grader_score = grade_episode(
                self._state.task_id, self._reviews, self._ground_truth
            )
            ContractReviewEnvironment._global_last_grader_score = self._last_grader_score
            ContractReviewEnvironment._global_last_task_id = self._state.task_id

            final_reward = max(0.001, min(0.999, self._last_grader_score))

            message_parts.append(
                f"All clauses reviewed! Final grader score: {self._last_grader_score:.4f}. "
                f"Issues found: {self._state.issues_found}/{self._state.total_issues}. "
                f"False positives: {self._state.false_positives}."
            )

            return ContractObservation(
                done=True,
                reward=final_reward,
                contract_title=self._contract["title"],
                contract_text=self._format_full_contract(),
                current_clause_id=clause_id,
                current_clause_title=current_clause["title"],
                current_clause_text=current_clause["text"],
                clause_index=self._current_clause_idx - 1,
                total_clauses=len(self._clauses),
                reviewed_clauses=self._reviews,
                task_id=self._state.task_id,
                task_description=self._contract["metadata"]["task_description"],
                message=" ".join(message_parts),
            )

        # Not done — show next clause
        next_clause = self._clauses[self._current_clause_idx]
        remaining = len(self._clauses) - self._current_clause_idx
        message_parts.append(f"{remaining} clause(s) remaining. Next: '{next_clause['title']}'.")

        return ContractObservation(
            done=False,
            reward=step_reward,
            contract_title=self._contract["title"],
            contract_text=self._format_full_contract(),
            current_clause_id=next_clause["id"],
            current_clause_title=next_clause["title"],
            current_clause_text=next_clause["text"],
            clause_index=self._current_clause_idx,
            total_clauses=len(self._clauses),
            reviewed_clauses=self._reviews,
            task_id=self._state.task_id,
            task_description=self._contract["metadata"]["task_description"],
            message=" ".join(message_parts),
        )

    @property
    def state(self) -> ContractState:
        """Get the current environment state."""
        return self._state

    def get_last_grader_score(self) -> Optional[float]:
        """Returns the grader score from the most recently completed episode across any session."""
        return ContractReviewEnvironment._global_last_grader_score

    def _format_full_contract(self) -> str:
        """Format the entire contract as readable text."""
        lines = [f"CONTRACT: {self._contract['title']}", "=" * 60, ""]
        for clause in self._clauses:
            lines.append(f"CLAUSE {clause['id'].upper()} — {clause['title']}")
            lines.append("-" * 40)
            lines.append(clause["text"])
            lines.append("")
        return "\n".join(lines)

    def _make_done_observation(self, message: str) -> ContractObservation:
        """Return a done observation when the episode is already complete."""
        return ContractObservation(
            done=True,
            reward=0.0,
            contract_title=self._contract["title"] if self._contract else "",
            contract_text="",
            current_clause_id="",
            current_clause_title="",
            current_clause_text="",
            clause_index=len(self._clauses) - 1 if self._clauses else 0,
            total_clauses=len(self._clauses),
            reviewed_clauses=self._reviews,
            task_id=self._state.task_id,
            task_description="",
            message=message,
        )
