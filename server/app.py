"""
FastAPI application for the Contract Review OpenEnv environment.
Follows the official OpenEnv pattern: create_app(EnvClass, ActionType, ObsType).
Adds required additional endpoints: /tasks, /grader, /baseline.
"""

import os
import sys
import subprocess

# Support both in-repo and standalone imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from models import ContractAction, ContractObservation, ContractState
from contracts import TASKS, get_task_ids

# Try to use OpenEnv's create_app, fall back to manual FastAPI setup
try:
    from openenv.core.env_server import create_app
    from environment import ContractReviewEnvironment

    # create_app calls env_factory() on every request.
    # We must pass a singleton factory so /reset and /step share the same instance.
    _singleton_env = ContractReviewEnvironment()

    def _env_factory() -> ContractReviewEnvironment:
        return _singleton_env

    from fastapi import Request
    from fastapi.responses import JSONResponse
    from fastapi.routing import APIRoute

    app = create_app(
        _env_factory,
        ContractAction,
        ContractObservation,
        env_name="contract_review_env",
    )

    # Override /state: remove OpenEnv's partial-serialization route and insert our full one
    async def _full_state_endpoint(request: Request):
        """Return all ContractState fields (OpenEnv omits fields equal to defaults)."""
        s = _singleton_env.state
        return JSONResponse({
            "episode_id": getattr(s, "episode_id", None),
            "step_count": s.step_count,
            "task_id": s.task_id,
            "contract_id": s.contract_id,
            "total_issues": s.total_issues,
            "issues_found": s.issues_found,
            "correct_severities": s.correct_severities,
            "false_positives": s.false_positives,
            "amendments_suggested": s.amendments_suggested,
            "cumulative_reward": s.cumulative_reward,
        })

    # Remove existing /state GET route registered by create_app
    app.router.routes = [
        r for r in app.router.routes
        if not (isinstance(r, APIRoute) and r.path == "/state" and "GET" in r.methods)
    ]
    # Re-add our full version first so it matches before any fallback
    new_state_route = APIRoute(
        "/state",
        _full_state_endpoint,
        methods=["GET"],
        summary="Get current environment state",
        tags=["State Management"],
    )
    app.router.routes.insert(0, new_state_route)

except ImportError:
    # Fallback: create FastAPI app manually (for standalone use)
    from environment import ContractReviewEnvironment

    app = FastAPI(
        title="Contract Review Environment",
        description="OpenEnv environment for AI agent contract review and negotiation",
        version="1.0.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Global environment instance
    _env = ContractReviewEnvironment()

    class ResetRequest(BaseModel):
        task_id: str = "clause_identification"
        episode_id: Optional[str] = None
        seed: Optional[int] = None

    class StepRequest(BaseModel):
        clause_id: str
        action_type: str
        severity: Optional[str] = None
        reasoning: str = ""
        suggested_text: Optional[str] = None

    @app.get("/health")
    async def health():
        return {"status": "healthy", "environment": "contract_review_env", "version": "1.0.0"}

    @app.post("/reset")
    async def reset(request: ResetRequest = None):
        if request is None:
            request = ResetRequest()
        obs = _env.reset(seed=request.seed, episode_id=request.episode_id, task_id=request.task_id)
        return obs.model_dump()

    @app.post("/step")
    async def step(request: StepRequest):
        action = ContractAction(
            clause_id=request.clause_id,
            action_type=request.action_type,
            severity=request.severity,
            reasoning=request.reasoning,
            suggested_text=request.suggested_text,
        )
        obs = _env.step(action)
        return obs.model_dump()

    @app.get("/state")
    async def state():
        return _env.state.model_dump()


# ── Additional required endpoints (added on top of OpenEnv's standard ones) ──

class TaskInfo(BaseModel):
    id: str
    difficulty: str
    description: str
    action_schema: dict


class GraderResponse(BaseModel):
    task_id: str
    score: float
    episode_completed: bool


class BaselineResponse(BaseModel):
    results: dict
    status: str


@app.get("/")
async def root():
    return {
        "status": "Active", 
        "message": "Contract Review OpenEnv API is running. Please navigate to /docs to view the Swagger API interface."
    }




@app.get("/tasks")
async def tasks():
    """Returns list of all tasks with their action schema."""
    action_schema = ContractAction.model_json_schema()
    task_list = []
    for task_id, task_info in TASKS.items():
        task_list.append(TaskInfo(
            id=task_id,
            difficulty=task_info["difficulty"],
            description=task_info["description"],
            action_schema=action_schema,
        ).model_dump())
    return {"tasks": task_list}


@app.get("/grader")
async def grader():
    """Returns grader score from the last globally completed episode."""
    score = ContractReviewEnvironment._global_last_grader_score
    task_id = ContractReviewEnvironment._global_last_task_id
    if score is None:
        return GraderResponse(task_id=task_id or "none", score=0.0, episode_completed=False).model_dump()
    return GraderResponse(task_id=task_id, score=score, episode_completed=True).model_dump()


@app.post("/baseline")
async def baseline():
    """Trigger inference script and return baseline scores for all 3 tasks."""
    try:
        script_paths = [
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "inference.py"),
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "inference.py"),
            "/app/inference.py",
        ]
        script_path = next((p for p in script_paths if os.path.exists(p)), None)
        if script_path is None:
            raise HTTPException(status_code=404, detail="inference.py not found")

        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True, text=True, timeout=1200,
            env={**os.environ},
        )
        if result.returncode != 0:
            return BaselineResponse(results={"error": result.stderr[-500:]}, status="failed").model_dump()

        scores = {}
        for line in result.stdout.split("\n"):
            if "SCORE" in line.upper():
                try:
                    parts = line.split(":")
                    if len(parts) >= 2:
                        key = parts[0].strip().lower().replace("score", "").strip(" -_")
                        val = float(parts[-1].strip())
                        scores[key] = val
                except (ValueError, IndexError):
                    pass

        return BaselineResponse(
            results=scores if scores else {"raw_output": result.stdout[-1000:]},
            status="completed",
        ).model_dump()
    except subprocess.TimeoutExpired:
        return BaselineResponse(results={"error": "Timeout (20 min)"}, status="timeout").model_dump()
    except Exception as e:
        return BaselineResponse(results={"error": str(e)}, status="error").model_dump()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
