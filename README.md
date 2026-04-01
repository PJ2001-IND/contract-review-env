---
title: Contract Review Environment
emoji: 📝
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
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
# Start the server
cd server
uvicorn app:app --host 0.0.0.0 --port 7860

# Or with Docker
docker build -t contract-review-env .
docker run -p 8990:7860 contract-review-env
```

### Run Inference

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
export HF_TOKEN="your_token_here"

python inference.py
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/reset` | POST | Start new episode (body: `{"task_id": "clause_identification"}`) |
| `/step` | POST | Submit action for current clause |
| `/state` | GET | Get current environment state |
| `/tasks` | GET | List all tasks with action schema |
| `/grader` | GET | Get grader score after episode |
| `/baseline` | POST | Run inference and return scores |

## Evaluation Criteria

- **Real-world utility (30%)**: Contract review is a $40B+ industry
- **Task & grader quality (25%)**: 3 deterministic graders with clear difficulty progression
- **Environment design (20%)**: Clean state management, partial-progress rewards
- **Code quality (15%)**: Typed Pydantic models, OpenEnv spec compliant
- **Creativity (10%)**: Novel legal/contract domain for OpenEnv

## License

MIT
