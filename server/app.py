"""
FastAPI application for the Contract Review OpenEnv environment.
Follows the official OpenEnv pattern: create_app(EnvClass, ActionType, ObsType).
Adds required additional endpoints: /tasks, /grader, /baseline.
"""

import os
import re
import sys
import subprocess

# Support both in-repo and standalone imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
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


def _clamp_score(value: float) -> float:
    """Ensure baseline-reported task scores always stay strictly within (0, 1)."""
    return round(min(0.999, max(0.001, float(value))), 4)


def _parse_baseline_scores(stdout: str) -> dict:
    """
    Parse canonical hackathon logs:
    [START] task=<task> env=<env> model=<model>
    [END] success=<bool> steps=<n> score=<score> rewards=<...>

    Also keep support for older summary lines like "SCORE - task_id: value".
    """
    scores = {}
    current_task = None

    for raw_line in stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        if line.startswith("[START]"):
            match = re.search(r"\btask=([^\s]+)", line)
            if match:
                current_task = match.group(1)
            continue

        if line.startswith("[END]"):
            match = re.search(r"\bscore=([0-9]*\.?[0-9]+)", line)
            if match and current_task:
                scores[current_task] = _clamp_score(float(match.group(1)))
            continue

        if "SCORE" in line.upper():
            try:
                parts = line.split(":")
                if len(parts) >= 2:
                    key = parts[0].strip().lower().replace("score", "").strip(" -_")
                    scores[key] = _clamp_score(float(parts[-1].strip()))
            except (ValueError, IndexError):
                continue

    return scores


@app.get("/")
async def root():
    return {
        "status": "Active",
        "message": "Contract Review OpenEnv API. Navigate to /docs for Swagger UI, /web for interactive demo."
    }


@app.get("/web", response_class=HTMLResponse, include_in_schema=True,
         summary="Interactive Web UI", tags=["default"])
async def web_ui():
    """Interactive web interface for manually testing the Contract Review environment."""
    html = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Contract Review Env — Interactive Demo</title>
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         background: #0f1117; color: #e2e8f0; margin: 0; padding: 20px; }
  h1 { color: #7c3aed; } h2 { color: #a78bfa; border-bottom: 1px solid #374151; padding-bottom: 8px; }
  .card { background: #1e2130; border-radius: 12px; padding: 20px; margin: 16px 0;
          border: 1px solid #374151; }
  button { background: #7c3aed; color: white; border: none; padding: 10px 20px;
           border-radius: 8px; cursor: pointer; font-size: 14px; margin: 4px; transition: all 0.2s; }
  button:hover { background: #6d28d9; transform: translateY(-1px); }
  button.danger { background: #dc2626; } button.danger:hover { background: #b91c1c; }
  button.success { background: #059669; } button.success:hover { background: #047857; }
  select, input, textarea { background: #111827; color: #e2e8f0; border: 1px solid #374151;
    border-radius: 8px; padding: 8px 12px; width: 100%; box-sizing: border-box; margin: 6px 0; }
  pre { background: #111827; border-radius: 8px; padding: 16px; overflow-x: auto;
        font-size: 12px; max-height: 400px; overflow-y: auto; border: 1px solid #374151; }
  .badge { display: inline-block; padding: 2px 8px; border-radius: 12px; font-size: 11px; font-weight: bold; }
  .easy { background: #064e3b; color: #34d399; } .medium { background: #78350f; color: #fbbf24; }
  .hard { background: #7f1d1d; color: #f87171; }
  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
  @media (max-width: 768px) { .grid { grid-template-columns: 1fr; } }
  .status { padding: 8px 16px; border-radius: 8px; background: #064e3b; color: #34d399;
            display: inline-block; margin: 8px 0; }
</style>
</head>
<body>
<h1>📝 Contract Review Environment</h1>
<p>Interactive demo for the OpenEnv hackathon submission. Use the controls below to test all tasks and endpoints.</p>

<div class="card">
  <h2>🎯 Start Episode</h2>
  <label>Task:</label>
  <select id="taskSelect">
    <option value="clause_identification">🟢 clause_identification (Easy — 5 clauses)</option>
    <option value="risk_assessment">🟡 risk_assessment (Medium — 10 clauses)</option>
    <option value="negotiation">🔴 negotiation (Hard — 15 clauses)</option>
  </select>
  <br><button onclick="reset()">🔄 Reset Environment</button>
  <button onclick="getState()">📊 Get State</button>
  <button onclick="getGrader()">🏆 Get Score</button>
</div>

<div class="card">
  <h2>⚡ Submit Action</h2>
  <div class="grid">
    <div>
      <label>Clause ID:</label>
      <input id="clauseId" placeholder="e.g. c1, c2, c3" value="c1">
      <label>Action Type:</label>
      <select id="actionType">
        <option value="approve">✅ approve</option>
        <option value="flag_risk">⚠️ flag_risk</option>
        <option value="suggest_amendment">✏️ suggest_amendment</option>
      </select>
    </div>
    <div>
      <label>Severity (for flag/amend):</label>
      <select id="severity">
        <option value="null">null (for approve)</option>
        <option value="critical">🔴 critical</option>
        <option value="moderate">🟡 moderate</option>
        <option value="minor">🟢 minor</option>
      </select>
      <label>Reasoning:</label>
      <input id="reasoning" placeholder="Explain your decision..." value="Clause appears standard and fair">
    </div>
  </div>
  <label>Suggested Amendment (for suggest_amendment only):</label>
  <textarea id="suggestedText" rows="2" placeholder="Proposed replacement text..."></textarea>
  <br><button class="success" onclick="step()">▶️ Submit Step</button>
</div>

<div class="card">
  <h2>📡 Response</h2>
  <pre id="output">← Click a button to see the response here</pre>
</div>

<div class="card">
  <h2>🔌 Quick API Reference</h2>
  <div class="grid">
    <div>
      <button onclick="fetch('/health').then(r=>r.json()).then(d=>show(d))">GET /health</button>
      <button onclick="fetch('/tasks').then(r=>r.json()).then(d=>show(d))">GET /tasks</button>
      <button onclick="fetch('/metadata').then(r=>r.json()).then(d=>show(d))">GET /metadata</button>
      <button onclick="fetch('/schema').then(r=>r.json()).then(d=>show(d))">GET /schema</button>
    </div>
    <div>
      <a href="/docs" target="_blank"><button>📖 Swagger UI /docs</button></a>
      <a href="/openapi.json" target="_blank"><button>📄 OpenAPI JSON</button></a>
    </div>
  </div>
</div>

<script>
const show = (data) => { document.getElementById('output').textContent = JSON.stringify(data, null, 2); };

async function reset() {
  const task = document.getElementById('taskSelect').value;
  const r = await fetch('/reset', {method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({task_id: task})});
  const d = await r.json();
  if (d.current_clause_id) document.getElementById('clauseId').value = d.current_clause_id;
  show(d);
}

async function step() {
  const sevRaw = document.getElementById('severity').value;
  const sev = sevRaw === 'null' ? null : sevRaw;
  const txt = document.getElementById('suggestedText').value.trim() || null;
  const body = {
    action: {
      clause_id: document.getElementById('clauseId').value,
      action_type: document.getElementById('actionType').value,
      severity: sev, reasoning: document.getElementById('reasoning').value,
      suggested_text: txt,
    }
  };
  const r = await fetch('/step', {method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify(body)});
  const d = await r.json();
  if (d.observation && d.observation.current_clause_id)
    document.getElementById('clauseId').value = d.observation.current_clause_id;
  show(d);
}

async function getState() {
  const r = await fetch('/state');
  show(await r.json());
}
async function getGrader() {
  const r = await fetch('/grader');
  show(await r.json());
}
</script>
</body></html>
"""
    return HTMLResponse(content=html)


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
        # Return 0.001 (not 0.0) — validator requires scores strictly > 0
        return GraderResponse(task_id=task_id or "none", score=0.001, episode_completed=False).model_dump()
    
    # Absolute final safety clamp before JSON serialization
    safe_score = round(min(0.999, max(0.001, float(score))), 4)
    return GraderResponse(task_id=task_id, score=safe_score, episode_completed=True).model_dump()


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
        default_scores = {"clause_identification": 0.001, "risk_assessment": 0.001, "negotiation": 0.001}

        if result.returncode != 0:
            return BaselineResponse(results=default_scores, status=f"failed: {result.stderr[-200:]}").model_dump()

        scores = _parse_baseline_scores(result.stdout)
        
        # Merge parsed scores with defaults to ensure all tasks report a valid score > 0
        final_scores = {**default_scores, **scores}
        status_msg = "completed" if scores else f"completed_no_scores: {result.stdout[-200:]}"

        return BaselineResponse(
            results=final_scores,
            status=status_msg,
        ).model_dump()
    except subprocess.TimeoutExpired:
        return BaselineResponse(
            results={"clause_identification": 0.001, "risk_assessment": 0.001, "negotiation": 0.001}, 
            status="timeout_20min"
        ).model_dump()
    except Exception as e:
        return BaselineResponse(
            results={"clause_identification": 0.001, "risk_assessment": 0.001, "negotiation": 0.001}, 
            status=f"error: {str(e)}"
        ).model_dump()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
