"""
FastAPI application for the Contract Review OpenEnv environment.
Implements the full OpenEnv HTTP interface with stateful session management.
Endpoints: /health, /reset, /step, /state, /tasks, /grader, /baseline, /docs, /ws
"""

import os
import sys
import subprocess

# Support both in-repo and standalone imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any
import json
import uuid

from models import ContractAction, ContractObservation, ContractState
from contracts import TASKS, get_task_ids
from environment import ContractReviewEnvironment


# ── Create FastAPI app ──

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


# ── Shared stateful environment for HTTP endpoints ──
_env = ContractReviewEnvironment()


# ── Request/Response models ──

class ResetRequest(BaseModel):
    task_id: str = "clause_identification"
    episode_id: Optional[str] = None
    seed: Optional[int] = None

class ActionRequest(BaseModel):
    clause_id: str
    action_type: str
    severity: Optional[str] = None
    reasoning: str = ""
    suggested_text: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class StepRequestBody(BaseModel):
    action: Optional[ActionRequest] = None
    # Also accept flat action fields
    clause_id: Optional[str] = None
    action_type: Optional[str] = None
    severity: Optional[str] = None
    reasoning: Optional[str] = None
    suggested_text: Optional[str] = None

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


# ── Core OpenEnv endpoints ──

@app.get("/")
async def root():
    """Root endpoint with environment info."""
    return {
        "name": "contract_review_env",
        "version": "1.0.0",
        "description": "OpenEnv environment for contract review, risk assessment, and negotiation",
        "endpoints": ["/health", "/reset", "/step", "/state", "/tasks", "/grader", "/baseline", "/docs"],
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post("/reset")
async def reset(request: ResetRequest = None):
    """Reset the environment and start a new episode."""
    if request is None:
        request = ResetRequest()
    obs = _env.reset(
        seed=request.seed,
        episode_id=request.episode_id,
        task_id=request.task_id,
    )
    return {
        "observation": obs.model_dump(),
        "reward": None,
        "done": False,
    }


@app.post("/step")
async def step(body: StepRequestBody):
    """Execute an action on the current clause."""
    if body.action is not None:
        action = ContractAction(
            clause_id=body.action.clause_id,
            action_type=body.action.action_type,
            severity=body.action.severity,
            reasoning=body.action.reasoning,
            suggested_text=body.action.suggested_text,
        )
    elif body.clause_id is not None:
        action = ContractAction(
            clause_id=body.clause_id,
            action_type=body.action_type or "approve",
            severity=body.severity,
            reasoning=body.reasoning or "",
            suggested_text=body.suggested_text,
        )
    else:
        raise HTTPException(status_code=422, detail="Must provide action in request body")

    obs = _env.step(action)
    return {
        "observation": obs.model_dump(),
        "reward": obs.reward,
        "done": obs.done,
    }


@app.get("/state")
async def state():
    """Get current environment state."""
    return _env.state.model_dump()


@app.get("/schema")
async def schema():
    """Return action and observation JSON schemas."""
    return {
        "action_schema": ContractAction.model_json_schema(),
        "observation_schema": ContractObservation.model_json_schema(),
        "state_schema": ContractState.model_json_schema(),
    }


# ── WebSocket endpoint (matches OpenEnv /ws pattern) ──

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for persistent session interaction."""
    await websocket.accept()
    ws_env = ContractReviewEnvironment()
    try:
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type", "")

            if msg_type == "reset":
                task_id = data.get("task_id", "clause_identification")
                obs = ws_env.reset(task_id=task_id)
                await websocket.send_json({
                    "type": "reset_result",
                    "observation": obs.model_dump(),
                    "reward": None,
                    "done": False,
                })
            elif msg_type == "step":
                action_data = data.get("action", data)
                action = ContractAction(
                    clause_id=action_data.get("clause_id", ""),
                    action_type=action_data.get("action_type", "approve"),
                    severity=action_data.get("severity"),
                    reasoning=action_data.get("reasoning", ""),
                    suggested_text=action_data.get("suggested_text"),
                )
                obs = ws_env.step(action)
                await websocket.send_json({
                    "type": "step_result",
                    "observation": obs.model_dump(),
                    "reward": obs.reward,
                    "done": obs.done,
                })
            elif msg_type == "state":
                await websocket.send_json({
                    "type": "state_result",
                    **ws_env.state.model_dump(),
                })
            else:
                await websocket.send_json({"error": f"Unknown type: {msg_type}"})
    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"error": str(e)})
        except Exception:
            pass


# ── Additional required endpoints ──

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
    """Returns grader score after an episode is completed."""
    score = _env.get_last_grader_score()
    task_id = _env.state.task_id
    if score is None:
        return GraderResponse(task_id=task_id or "none", score=0.0, episode_completed=False).model_dump()
    return GraderResponse(task_id=task_id, score=score, episode_completed=True).model_dump()


@app.post("/baseline")
async def baseline():
    """Trigger inference script and return baseline scores for all 3 tasks."""
    try:
        script_paths = [
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "inference.py"),
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


# ── Entry point ──

def main():
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
