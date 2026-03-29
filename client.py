"""
Contract Review Environment Client.
Extends OpenEnv's EnvClient for WebSocket communication,
with HTTP fallback for standalone use.
"""

from __future__ import annotations

from typing import Optional

from models import ContractAction, ContractObservation, ContractState

# Try to use OpenEnv's official EnvClient
try:
    from openenv.core.client_types import StepResult
    from openenv.core.env_client import EnvClient

    class ContractReviewEnv(EnvClient[ContractAction, ContractObservation, ContractState]):
        """OpenEnv-compatible WebSocket client for the Contract Review environment."""

        def _step_payload(self, action: ContractAction) -> dict:
            """Shape expected by the server's /step endpoint."""
            return {
                "clause_id": action.clause_id,
                "action_type": action.action_type,
                "severity": action.severity,
                "reasoning": action.reasoning,
                "suggested_text": action.suggested_text,
            }

        def _parse_result(self, payload: dict) -> StepResult[ContractObservation]:
            """Parse step response into StepResult."""
            obs = ContractObservation(**payload["observation"])
            return StepResult(
                observation=obs,
                reward=payload.get("reward"),
                done=bool(payload.get("done", False)),
            )

        def _parse_state(self, payload: dict) -> ContractState:
            """Parse state response."""
            return ContractState(
                episode_id=payload.get("episode_id"),
                step_count=payload.get("step_count", 0),
                task_id=payload.get("task_id", ""),
                contract_id=payload.get("contract_id", ""),
                total_issues=payload.get("total_issues", 0),
                issues_found=payload.get("issues_found", 0),
                correct_severities=payload.get("correct_severities", 0),
                false_positives=payload.get("false_positives", 0),
                amendments_suggested=payload.get("amendments_suggested", 0),
                cumulative_reward=payload.get("cumulative_reward", 0.0),
            )

except ImportError:
    # Fallback: HTTP-based client for standalone use
    from pydantic import BaseModel

    class StepResult(BaseModel):
        observation: ContractObservation
        reward: Optional[float] = None
        done: bool = False

    class ContractReviewEnv:
        """HTTP-based client for the Contract Review environment (standalone)."""

        def __init__(self, base_url: str = "http://localhost:7860"):
            self.base_url = base_url.rstrip("/")

        def reset(self, task_id: str = "clause_identification") -> StepResult:
            import requests
            resp = requests.post(f"{self.base_url}/reset", json={"task_id": task_id})
            resp.raise_for_status()
            data = resp.json()
            obs = ContractObservation(**data)
            return StepResult(observation=obs, reward=data.get("reward"), done=data.get("done", False))

        def step(self, action: ContractAction) -> StepResult:
            import requests
            resp = requests.post(f"{self.base_url}/step", json={
                "clause_id": action.clause_id,
                "action_type": action.action_type,
                "severity": action.severity,
                "reasoning": action.reasoning,
                "suggested_text": action.suggested_text,
            })
            resp.raise_for_status()
            data = resp.json()
            obs = ContractObservation(**data)
            return StepResult(observation=obs, reward=data.get("reward"), done=data.get("done", False))

        def get_state(self) -> ContractState:
            import requests
            resp = requests.get(f"{self.base_url}/state")
            resp.raise_for_status()
            return ContractState(**resp.json())

        def get_tasks(self) -> dict:
            import requests
            resp = requests.get(f"{self.base_url}/tasks")
            resp.raise_for_status()
            return resp.json()

        def get_grader_score(self) -> dict:
            import requests
            resp = requests.get(f"{self.base_url}/grader")
            resp.raise_for_status()
            return resp.json()

        def close(self):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()
