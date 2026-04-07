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
            reward: Optional[float] = None

        class State(BaseModel):
            """Base state model."""
            episode_id: Optional[str] = None
            step_count: int = 0


# ── Contract Review Models ────────────────────────────────────────────────────


class ContractAction(Action):
    """Action the agent takes on a contract clause."""
    clause_id: str = Field(..., description="ID of the clause to act on (e.g. 'c1')")
    action_type: Literal["flag_risk", "suggest_amendment", "approve", "reject"] = Field(
        ..., description="Type of action to take on the clause"
    )
    severity: Optional[Literal["critical", "moderate", "minor"]] = Field(
        None, description="Severity of the identified risk (required when action_type is 'flag_risk' or 'suggest_amendment')"
    )
    reasoning: str = Field(
        ..., description="Explanation of why this action is being taken"
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
    reward_earned: float = 0.0


class ContractObservation(Observation):
    """What the agent observes after each step."""
    contract_title: str = Field("", description="Title of the contract being reviewed")
    contract_text: str = Field("", description="Full contract text")
    current_clause_id: str = Field("", description="ID of the current clause under review")
    current_clause_title: str = Field("", description="Title of the current clause")
    current_clause_text: str = Field("", description="Text of the current clause")
    clause_index: int = Field(0, description="0-based index of current clause")
    total_clauses: int = Field(0, description="Total number of clauses in the contract")
    reviewed_clauses: List[dict] = Field(default_factory=list, description="History of reviewed clauses")
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
    cumulative_reward: float = Field(default=0.0, description="Total accumulated reward this episode")

    model_config = {"populate_by_name": True}

    def model_dump(self, **kwargs) -> dict:
        """Always serialize all fields, even if unset or equal to defaults."""
        kwargs.pop("exclude_unset", None)
        kwargs.pop("exclude_defaults", None)
        return super().model_dump(exclude_unset=False, exclude_defaults=False, **kwargs)
