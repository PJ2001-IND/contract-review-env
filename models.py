"""
Typed models for the Contract Review OpenEnv environment.
Defines Action, Observation, and State schemas following the OpenEnv spec.
"""

from __future__ import annotations

from typing import List, Optional, Literal
from pydantic import Field

# Import OpenEnv base types — support both in-repo and standalone
try:
    from openenv.core.env_server.interfaces import Action, Observation, State
except ImportError:
    try:
        from openenv.core.env_server.types import Action, Observation, State
    except ImportError:
        # Fallback for standalone usage without openenv-core installed
        from pydantic import BaseModel

        class Action(BaseModel):
            """Base action model."""
            pass

        class Observation(BaseModel):
            """Base observation model."""
            done: bool = False
            reward: Optional[float] = 0.01

        class State(BaseModel):
            """Base state model."""
            episode_id: Optional[str] = None
            step_count: int = 0


# ── Contract Review Models ────────────────────────────────────────────────────


class ContractAction(Action):
    """Action the agent takes on the contract."""
    action_type: Literal["read_clause", "search_contract", "flag_issue", "suggest_amendment", "finish_review"] = Field(
        ..., description="Type of action to take"
    )
    clause_id: Optional[str] = Field(None, description="ID of the clause (used for read_clause, flag_issue, suggest_amendment)")
    search_query: Optional[str] = Field(None, description="Query string (used for search_contract)")
    severity: Optional[Literal["critical", "moderate", "minor"]] = Field(
        None, description="Severity (required when action_type is 'flag_issue' or 'suggest_amendment')"
    )
    reasoning: Optional[str] = Field(
        None, description="Explanation for the issue or amendment"
    )
    suggested_text: Optional[str] = Field(
        None, description="Proposed amendment text (used when action_type is 'suggest_amendment')"
    )


class ClauseReview(Action):
    """Record of the agent's review of a single clause."""
    clause_id: str = ""
    clause_title: str = ""
    action_type: str = ""
    severity: Optional[str] = None
    reasoning: str = ""
    suggested_text: Optional[str] = None
    reward_earned: float = 0.01


class ContractObservation(Observation):
    """What the agent observes after each step."""
    contract_title: str = Field("", description="Title of the contract being reviewed")
    table_of_contents: List[dict] = Field(default_factory=list, description="List of all clauses {id, title}")
    active_view: str = Field("toc", description="What the agent is currently looking at: 'toc', 'search_results', 'clause_detail'")
    view_content: str = Field("", description="Text content corresponding to the active_view")
    flagged_issues: List[dict] = Field(default_factory=list, description="History of issues flagged by agent")
    steps_remaining: int = Field(0, description="Steps left before forced termination")
    total_clauses: int = Field(0, description="Total number of clauses in the contract")
    task_id: str = Field("", description="ID of the current task")
    task_description: str = Field("", description="Description of what the agent should do")
    message: str = Field("", description="Feedback message from the environment")


class ContractState(State):
    """Internal environment state."""
    task_id: str = Field(default="", description="Current task ID")
    contract_id: str = Field(default="", description="Current contract ID")
    total_issues: int = Field(default=0, description="Total known issues in contract")
    issues_found: int = Field(default=0, description="Number of issues correctly identified")
    correct_severities: int = Field(default=0, description="Severities classified correctly")
    false_positives: int = Field(default=0, description="Number of incorrect flags")
    amendments_suggested: int = Field(default=0, description="Number of amendments suggested")
    cumulative_reward: float = Field(default=0.01, description="Total accumulated reward this episode")

    model_config = {"populate_by_name": True}

    def model_dump(self, **kwargs) -> dict:
        """Always serialize all fields, even if unset or equal to defaults."""
        kwargs.pop("exclude_unset", None)
        kwargs.pop("exclude_defaults", None)
        return super().model_dump(exclude_unset=False, exclude_defaults=False, **kwargs)
