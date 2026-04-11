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
    Now acts as a non-linear search and retrieval environment where the agent
    must actively decide what to read, search, and when to conclude the review.
    """

    SUPPORTS_CONCURRENT_SESSIONS = True
    
    _global_last_grader_score: Optional[float] = None
    _global_last_task_id: Optional[str] = None

    # ── Reward shaping constants ──────────────────────────────────────────
    REWARD_CORRECT_FLAG = 0.15
    REWARD_CORRECT_SEVERITY = 0.10
    REWARD_GOOD_AMENDMENT = 0.10
    PENALTY_FALSE_POSITIVE = -0.05
    PENALTY_PER_STEP = -0.01
    REWARD_COMPLETION_BONUS = 0.20
    MAX_STEPS = 20

    def __init__(self):
        self._state = ContractState()
        self._contract: Optional[dict] = None
        self._clauses: List[dict] = []
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

        toc = [{"id": c["id"], "title": c["title"]} for c in self._clauses]
        toc_text = "\n".join([f"- [{c['id']}] {c['title']}" for c in self._clauses])

        return ContractObservation(
            done=False,
            reward=0.01,
            contract_title=self._contract["title"],
            table_of_contents=toc,
            active_view="toc",
            view_content=f"Table of Contents:\n{toc_text}",
            flagged_issues=[],
            steps_remaining=self.MAX_STEPS,
            total_clauses=len(self._clauses),
            task_id=task_id,
            task_description=self._contract["metadata"]["task_description"],
            message=f"Review started. {self.MAX_STEPS} steps remaining. You can use 'read_clause', 'search_contract', 'flag_issue', 'suggest_amendment', or 'finish_review'.",
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

        if not isinstance(action, ContractAction):
            if isinstance(action, dict):
                action = ContractAction(**action)
            elif hasattr(action, 'model_dump'):
                action = ContractAction(**action.model_dump())

        self._state.step_count += 1
        steps_left = max(0, self.MAX_STEPS - self._state.step_count)
        step_reward = self.PENALTY_PER_STEP  # Time pressure
        message_parts = []
        active_view = "toc"
        view_content = ""

        # Base TOC logic
        toc = [{"id": c["id"], "title": c["title"]} for c in self._clauses]
        toc_text = "\n".join([f"- [{c['id']}] {c['title']}" for c in self._clauses])

        act = action.action_type
        if act == "read_clause":
            active_view = "clause_detail"
            clause = next((c for c in self._clauses if c["id"] == action.clause_id), None)
            if clause:
                view_content = f"CLAUSE {clause['id'].upper()} - {clause['title']}\n{('-'*40)}\n{clause['text']}"
                message_parts.append(f"Reading clause {clause['id']}.")
            else:
                view_content = f"Error: Clause ID '{action.clause_id}' not found."
                message_parts.append("Clause not found.")
                step_reward += self.PENALTY_FALSE_POSITIVE # Slight penalty for hallucinating ID

        elif act == "search_contract":
            active_view = "search_results"
            query = (action.search_query or "").lower()
            if not query:
                view_content = "Please provide a valid search_query."
                message_parts.append("Empty search query.")
            else:
                matches = []
                for c in self._clauses:
                    if query in c["text"].lower() or query in c["title"].lower():
                        text = c["text"]
                        idx = text.lower().find(query)
                        start = max(0, idx - 40)
                        end = min(len(text), idx + 40)
                        excerpt = text[start:end].replace("\n", " ")
                        matches.append(f"- [{c['id']}] {c['title']}: ...{excerpt}...")
                if matches:
                    view_content = f"Search results for '{query}':\n" + "\n".join(matches)
                    message_parts.append(f"Found {len(matches)} matches.")
                else:
                    view_content = f"No matches found for '{query}'."
                    message_parts.append("No search matches.")

        elif act in ("flag_issue", "suggest_amendment"):
            active_view = "toc"
            view_content = toc_text
            cid = action.clause_id
            
            # Record it
            issue_record = {
                "clause_id": cid,
                "action_type": act,
                "severity": action.severity,
                "reasoning": action.reasoning,
                "suggested_text": action.suggested_text,
            }
            
            # Identify duplicates
            is_dup = any(r["clause_id"] == cid for r in self._reviews)
            if not is_dup and cid:
                self._reviews.append(issue_record)
                has_issue = cid in self._ground_truth
                
                if has_issue:
                    step_reward += self.REWARD_CORRECT_FLAG
                    self._state.issues_found += 1
                    message_parts.append(f"Flagged clause {cid}. Valid issue detected.")
                    expected_severity = self._ground_truth[cid][0].get("severity", "")
                    if action.severity == expected_severity:
                        step_reward += self.REWARD_CORRECT_SEVERITY
                        self._state.correct_severities += 1
                    if act == "suggest_amendment" and action.suggested_text:
                        self._state.amendments_suggested += 1
                        hint = self._ground_truth[cid][0].get("amendment_hint", "")
                        overlap = len(set(hint.lower().split()) & set(action.suggested_text.lower().split()))
                        if overlap > 3:
                            step_reward += self.REWARD_GOOD_AMENDMENT
                else:
                    step_reward += self.PENALTY_FALSE_POSITIVE
                    self._state.false_positives += 1
                    message_parts.append(f"Flagged clause {cid}. This might be a false positive.")
            else:
                if not cid:
                    message_parts.append("No clause ID provided.")
                else:
                    message_parts.append(f"Clause {cid} was already flagged. Ignored duplicate.")

        elif act == "finish_review":
            self._done = True
            message_parts.append("Review finished by agent.")

        else:
            active_view = "toc"
            view_content = toc_text
            message_parts.append("Unknown action type.")

        self._raw_reward += step_reward
        self._state.cumulative_reward = self._raw_reward

        if steps_left <= 0:
            self._done = True
            message_parts.append("Step limit reached.")

        if self._done:
            self._raw_reward += self.REWARD_COMPLETION_BONUS
            raw_grader = grade_episode(self._state.task_id, self._reviews, self._ground_truth)
            self._last_grader_score = round(min(0.999, max(0.01, float(raw_grader))), 4)
            ContractReviewEnvironment._global_last_grader_score = self._last_grader_score
            ContractReviewEnvironment._global_last_task_id = self._state.task_id
            
            final_reward = round(min(0.999, max(0.01, self._last_grader_score)), 4)
            msg = f"Episode completed! Final Grader Score: {self._last_grader_score:.4f}. Issues found: {self._state.issues_found}/{self._state.total_issues}."
            
            return ContractObservation(
                done=True,
                reward=final_reward,
                contract_title=self._contract["title"],
                table_of_contents=toc,
                active_view="toc",
                view_content=toc_text,
                flagged_issues=self._reviews,
                steps_remaining=0,
                total_clauses=len(self._clauses),
                task_id=self._state.task_id,
                task_description=self._contract["metadata"]["task_description"],
                message=msg,
            )

        return ContractObservation(
            done=False,
            reward=round(min(0.999, max(0.01, float(step_reward))), 4),
            contract_title=self._contract["title"],
            table_of_contents=toc,
            active_view=active_view,
            view_content=view_content,
            flagged_issues=self._reviews,
            steps_remaining=steps_left,
            total_clauses=len(self._clauses),
            task_id=self._state.task_id,
            task_description=self._contract["metadata"]["task_description"],
            message=" ".join(message_parts),
        )

    @property
    def state(self) -> ContractState:
        return self._state

    def get_last_grader_score(self) -> Optional[float]:
        score = ContractReviewEnvironment._global_last_grader_score
        if score is None:
            return None
        return round(min(0.999, max(0.01, float(score))), 4)

    def _make_done_observation(self, message: str) -> ContractObservation:
        return ContractObservation(
            done=True,
            reward=0.01,
            contract_title=self._contract["title"] if self._contract else "",
            table_of_contents=[],
            active_view="toc",
            view_content="",
            flagged_issues=self._reviews,
            steps_remaining=0,
            total_clauses=len(self._clauses),
            task_id=self._state.task_id,
            task_description="",
            message=message,
        )
