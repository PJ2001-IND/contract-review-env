---
title: Contract Review Environment
emoji: 📝
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 7860
base_path: /web
tags:
  - openenv
license: mit
---

# Contract Review Environment — OpenEnv

> **An AI agent environment for contract review, risk assessment, and negotiation.**

Built for the [Meta PyTorch OpenEnv Hackathon](https://www.scaler.com/school-of-technology/meta-pytorch-hackathon/).

## Overview

This environment simulates a real-world contract review task where an AI agent must:

1. **Read** software/business contracts clause by clause
2. **Identify** problematic or risky clauses
3. **Classify** risk severity (critical, moderate, minor)
4. **Suggest** specific amendments with better wording
5. **Negotiate** terms while considering clause interdependencies

## Quick Start

The simplest way to use the Contract Review environment is through the `ContractReviewEnv` client class:

```python
from contract_review_env import ContractAction, ContractReviewEnv

try:
    # Create environment from Docker image
    env = ContractReviewEnv.from_docker_image("contract-review-env:latest")

    # Reset with a specific task
    result = env.reset(task_id="clause_identification")
    print(f"Contract: {result.observation.contract_title}")
    print(f"First Clause: {result.observation.current_clause_title}")

    # Agent reviews clauses step by step
    while not result.done:
        action = ContractAction(
            clause_id=result.observation.current_clause_id,
            action_type="flag_risk",
            severity="critical",
            reasoning="This clause exposes the subscriber to unlimited liability.",
            suggested_text=None
        )
        result = env.step(action)
        print(f"Reward: {result.reward}")

    print(f"Final Score: {result.observation.message}")

finally:
    env.close()
```

That's it! The `ContractReviewEnv.from_docker_image()` method handles:
- Starting the Docker container
- Waiting for the server to be ready
- Connecting to the environment
- Container cleanup when you call `close()`

## Tasks

| Task ID | Difficulty | Contract | Clauses | Issues | Description |
|---------|-----------|----------|---------|--------|-------------|
| `clause_identification` | Easy | SaaS Agreement | 5 | 3 | Spot obvious red flags (unlimited liability, auto-renewal, IP transfer) |
| `risk_assessment` | Medium | Enterprise License | 10 | 5 | Classify severity correctly, provide reasoning |
| `negotiation` | Hard | Vendor Agreement | 15 | 7 | Suggest legally sound amendments, handle interdependencies |

## Action Space

```python
class ContractAction:
    clause_id: str          # "c1", "c2", etc.
    action_type: str        # "approve" | "flag_risk" | "suggest_amendment" | "reject"
    severity: str | None    # "critical" | "moderate" | "minor" (required for flag/amend)
    reasoning: str          # Why you're taking this action
    suggested_text: str | None  # Proposed amendment (for suggest_amendment)
```

## Observation Space

```python
class ContractObservation:
    contract_title: str         # Title of the contract
    contract_text: str          # Full contract text
    current_clause_id: str      # ID of clause under review (e.g. "c3")
    current_clause_title: str   # Human-readable title
    current_clause_text: str    # Full clause text
    clause_index: int           # 0-based position
    total_clauses: int          # Total clauses in contract
    reviewed_clauses: list      # History of previous reviews
    task_id: str                # Current task identifier
    task_description: str       # What the agent should do
    message: str                # Feedback from environment
    done: bool                  # Whether episode is complete
    reward: float | None        # Reward for the step
```

## Reward Function

The reward function provides **partial progress signals** throughout the episode:

| Signal | Value | Condition |
|--------|-------|-----------|
| Correct flag | +0.15 | Identified a genuinely problematic clause |
| Correct severity | +0.10 | Severity matches ground truth |
| Good amendment | +0.10 | Suggested text addresses key concerns |
| False positive | −0.05 | Flagged a clean clause as problematic |
| Time pressure | −0.02 | Per step (prevents infinite loops) |
| Completion bonus | +0.20 | Reviewed all clauses |

Final score is determined by the task-specific **grader** (0.0–1.0).

## Setup

### Prerequisites

- Python 3.10+
- Docker (for containerized deployment)
- Hugging Face CLI (`pip install huggingface-hub`)
- OpenAI API key or HF Token

### Install

```bash
cd contract_review_env
pip install -e .
```

### Run Locally

```bash
# Start the server (from repo root)
uv run uvicorn server.app:app --host 0.0.0.0 --port 7860

# Or with plain pip
uvicorn server.app:app --host 0.0.0.0 --port 7860

# Or with Docker
docker build -t contract-review-env .
docker run -p 8990:7860 contract-review-env
```

### Run Tests

```bash
# Verify all 33 API tests pass
uv run python run_tests.py
```

### Run Inference

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
export HF_TOKEN="your_token_here"

python inference.py
```

## Deploying to Hugging Face Spaces

You can deploy your OpenEnv environment to Hugging Face Spaces using the `openenv push` command:

```bash
# From the environment directory (where openenv.yaml is located)
openenv push

# Or specify options
openenv push --namespace my-org --private
```

The `openenv push` command will:
1. Validate that the directory is an OpenEnv environment (checks for `openenv.yaml`)
2. Prepare a custom build for Hugging Face Docker space (enables web interface)
3. Upload to Hugging Face (ensuring you're logged in)

After deployment, your space will be available at:
`https://huggingface.co/spaces/<repo-id>`

The deployed space includes:
- **Web Interface** at `/web` — Interactive UI for exploring the environment
- **API Documentation** at `/docs` — Full OpenAPI/Swagger interface
- **Health Check** at `/health` — Container health monitoring
- **WebSocket** at `/ws` — Persistent session endpoint for low-latency interactions

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check — returns `{"status": "healthy"}` |
| `/reset` | POST | Start new episode (body: `{"task_id": "clause_identification"}`) |
| `/step` | POST | Submit action for current clause |
| `/state` | GET | Get current environment state |
| `/tasks` | GET | List all tasks with full action schema |
| `/grader` | GET | Get grader score after episode (0.0–1.0) |
| `/baseline` | POST | Run LLM inference and return scores |
| `/docs` | GET | Interactive Swagger UI |

## Advanced Usage

### Connecting to an Existing Server

If you already have a Contract Review environment server running, you can connect directly:

```python
from contract_review_env import ContractReviewEnv

# Connect to existing server
env = ContractReviewEnv(base_url="<ENV_HTTP_URL_HERE>")

# Use as normal
result = env.reset(task_id="clause_identification")
result = env.step(ContractAction(
    clause_id="c1",
    action_type="approve",
    reasoning="Standard service description clause."
))
```

### Using the Context Manager

```python
from contract_review_env import ContractAction, ContractReviewEnv

with ContractReviewEnv(base_url="http://localhost:7860") as env:
    result = env.reset(task_id="risk_assessment")
    while not result.done:
        # Agent logic here
        result = env.step(ContractAction(...))
```

### Concurrent WebSocket Sessions

The server supports multiple concurrent WebSocket connections:

```python
# In server/app.py - factory mode for concurrent sessions
app = create_app(
    ContractReviewEnvironment,   # Pass class, not instance
    ContractAction,
    ContractObservation,
    max_concurrent_envs=4,       # Allow 4 concurrent sessions
)
```

## Project Structure

```
contract_review_env/
├── .dockerignore          # Docker build exclusions
├── .gitignore             # Git exclusions
├── __init__.py            # Module exports
├── README.md              # This file
├── Dockerfile             # Main container image (port 7860)
├── openenv.yaml           # OpenEnv manifest
├── pyproject.toml         # Project metadata and dependencies
├── uv.lock                # Locked dependencies (generated)
├── client.py              # ContractReviewEnv client
├── models.py              # Action and Observation models
├── contracts.py           # 3 synthetic contracts with embedded issues
├── graders.py             # Deterministic graders (0.0–1.0)
├── inference.py           # Baseline LLM agent script
├── play_demo.py           # Interactive terminal demo (human player)
├── run_tests.py           # Comprehensive API test suite (33 tests)
└── server/
    ├── __init__.py        # Server module exports
    ├── environment.py     # Core environment logic
    ├── app.py             # FastAPI application (HTTP + WebSocket)
    ├── requirements.txt   # Server dependencies
    └── Dockerfile         # Server-only container
```

## Evaluation Criteria

- **Real-world utility (30%)**: Contract review is a $40B+ industry
- **Task & grader quality (25%)**: 3 deterministic graders with clear difficulty progression
- **Environment design (20%)**: Clean state management, partial-progress rewards
- **Code quality (15%)**: Typed Pydantic models, OpenEnv spec compliant
- **Creativity (10%)**: Novel legal/contract domain for OpenEnv

## License

MIT
