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
..
# 📝 Contract Review Environment — OpenEnv

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green?style=flat-square&logo=fastapi)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=flat-square&logo=docker)
![OpenEnv](https://img.shields.io/badge/OpenEnv-Compliant-purple?style=flat-square)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Space-yellow?style=flat-square&logo=huggingface)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)

> **An AI agent environment for contract review, risk assessment, and legal clause negotiation** — built for the [Meta PyTorch OpenEnv Hackathon](https://www.scaler.com/school-of-technology/meta-pytorch-hackathon/).

---

## 📌 Overview

This environment simulates a real-world contract review task where an AI agent must:

1. **Read** software/business contracts clause by clause
2. **Identify** problematic or risky clauses
3. **Classify** risk severity (critical, moderate, minor)
4. **Suggest** specific amendments with better wording
5. **Negotiate** terms while considering clause interdependencies

The environment is fully compliant with the **OpenEnv specification** and exposes a standard HTTP + WebSocket API for agent interaction, deterministic graders, and partial reward shaping throughout each episode.

---

## 🎯 Tasks

| Task ID | Difficulty | Contract | Clauses | Issues | Description |
|---------|-----------|----------|---------|--------|-------------|
| `clause_identification` | 🟢 Easy | SaaS Agreement | 5 | 3 | Spot obvious red flags (unlimited liability, auto-renewal, IP transfer) |
| `risk_assessment` | 🟡 Medium | Enterprise License | 10 | 5 | Classify severity correctly, provide reasoning |
| `negotiation` | 🔴 Hard | Vendor Agreement | 15 | 7 | Suggest legally sound amendments, handle interdependencies |

---

## 🤖 Action & Observation Space

### Action Space

```python
class ContractAction:
    clause_id: str          # "c1", "c2", etc.
    action_type: str        # "approve" | "flag_risk" | "suggest_amendment" | "reject"
    severity: str | None    # "critical" | "moderate" | "minor" (required for flag/amend)
    reasoning: str          # Why you're taking this action
    suggested_text: str | None  # Proposed amendment (for suggest_amendment)
```

### Observation Space

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

---

## 🏆 Reward Function

The reward function provides **partial progress signals** throughout the episode — not just at the end:

| Signal | Value | Condition |
|--------|-------|-----------|
| ✅ Correct flag | `+0.15` | Identified a genuinely problematic clause |
| ✅ Correct severity | `+0.10` | Severity matches ground truth |
| ✅ Good amendment | `+0.10` | Suggested text addresses key concerns |
| ❌ False positive | `−0.05` | Flagged a clean clause as problematic |
| ⏱️ Time pressure | `−0.02` | Per step (prevents infinite loops) |
| 🎉 Completion bonus | `+0.20` | Reviewed all clauses |

> 📊 Final score is determined by the task-specific **grader** returning a float between `0.0` and `1.0`.

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- Docker (for containerized deployment)
- OpenAI API key **or** Hugging Face Token (for running inference)

### Install

```bash
git clone https://github.com/PJ2001-IND/contract-review-env.git
cd contract-review-env
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

### Run Inference (AI Agent)

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
export HF_TOKEN="your_token_here"

python inference.py
```

### Play Interactively (Human Demo)

```bash
# Play the role of the AI agent yourself in the terminal
uv run python play_demo.py
```

---

## 🔌 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | `GET` | Health check — returns `{"status": "healthy"}` |
| `/reset` | `POST` | Start new episode (body: `{"task_id": "clause_identification"}`) |
| `/step` | `POST` | Submit action for current clause |
| `/state` | `GET` | Get current environment state |
| `/tasks` | `GET` | List all tasks with full action schema |
| `/grader` | `GET` | Get grader score after episode (0.0–1.0) |
| `/baseline` | `POST` | Run LLM inference and return scores |
| `/docs` | `GET` | Interactive Swagger UI |

---

## 📁 Project Structure

```
contract_review_env/
├── .dockerignore          # Docker build exclusions
├── .gitignore             # Git exclusions
├── __init__.py            # Module exports
├── README.md              # This file
├── Dockerfile             # Main container image (port 7860)
├── openenv.yaml           # OpenEnv manifest
├── pyproject.toml         # Project metadata and dependencies
├── uv.lock                # Locked dependencies
├── client.py              # ContractReviewEnv client
├── models.py              # Action and Observation Pydantic models
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

---

## 📊 Evaluation Criteria

| Criteria | Weight | Our Implementation |
|----------|--------|-------------------|
| **Real-world utility** | 30% | Contract review is a $40B+ legal industry use case |
| **Task & grader quality** | 25% | 3 deterministic graders with clear difficulty progression |
| **Environment design** | 20% | Clean state management, partial-progress rewards |
| **Code quality** | 15% | Typed Pydantic models, fully OpenEnv spec compliant |
| **Creativity** | 10% | Novel legal/contract domain, first of its kind in OpenEnv |

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.10+ | Core programming language |
| FastAPI | HTTP API server and WebSocket endpoint |
| Pydantic v2 | Strict action/observation schema validation |
| Uvicorn | High-performance ASGI web server |
| OpenEnv Core | Base environment interface and `create_app` factory |
| OpenAI SDK | Universal LLM client (works with OpenAI and HF models) |
| Docker | Containerized deployment for Hugging Face Spaces |
| uv | Ultra-fast Python package manager |

---

## ⚠️ Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `API_BASE_URL` | Optional | LLM endpoint (e.g. `https://router.huggingface.co/v1`) |
| `MODEL_NAME` | Optional | Model identifier (default: `gpt-4o-mini`) |
| `OPENAI_API_KEY` | Optional | OpenAI API key for inference |
| `HF_TOKEN` | Optional | Hugging Face token (alternative to OpenAI key) |
| `PORT` | Optional | Server port (default: `7860`) |

> 🔐 **Security:** Never hardcode API keys. Always use environment variables or Hugging Face Secrets.

---

## 👤 Author

**Praasuk Jain**
- GitHub: [@PJ2001-IND](https://github.com/PJ2001-IND)
- Hugging Face: [@praasukjain2001](https://huggingface.co/praasukjain2001)
- LinkedIn: [praasuk-jain](https://www.linkedin.com/in/praasuk-jain-425b6b1a3/)

---

## 📄 License

MIT

---

> ⭐ If you found this project useful, consider giving it a star!
