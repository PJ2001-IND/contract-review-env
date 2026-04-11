"""
Inference Script — Contract Review OpenEnv Environment
===================================
MANDATORY
- Before submitting, ensure the following variables are defined:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.

- This script must be named `inference.py` and placed in the root directory
- Uses OpenAI Client for all LLM calls
"""

import os
import sys
import json
import re
import textwrap
from typing import List, Dict, Any, Optional

from openai import OpenAI

# Add the environment package to path
_script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _script_dir)
sys.path.insert(0, os.path.join(_script_dir, "server"))
sys.path.insert(0, os.path.join(_script_dir, "contract_review_env"))
sys.path.insert(0, os.path.join(_script_dir, "contract_review_env", "server"))

from models import ContractAction
from contracts import get_task_ids
from server.environment import ContractReviewEnvironment


# ── Configuration ─────────────────────────────────────────────────────────────

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

TEMPERATURE = 0.1
MAX_TOKENS = 600

DEBUG = os.getenv("DEBUG", "false").lower() == "true"


# ── System prompt for the contract review agent ──────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""
    You are a legal contract review AI agent. Your goal is to review a contract, locate problematic clauses, and flag them or suggest amendments before you run out of steps. You do this by actively searching, reading specific clauses, and flagging risks.

    You MUST respond with a single valid JSON object representing your next action. No additional text or markdown formatting outside the JSON is allowed.

    Available Actions (choose ONE per turn):

    1. Read a Clause:
    { "action_type": "read_clause", "clause_id": "<id>" }

    2. Search the Contract:
    { "action_type": "search_contract", "search_query": "<query>" }

    3. Flag an Issue:
    { "action_type": "flag_issue", "clause_id": "<id>", "severity": "<critical | moderate | minor>", "reasoning": "<explain why>" }

    4. Suggest an Amendment:
    { "action_type": "suggest_amendment", "clause_id": "<id>", "severity": "<critical | moderate | minor>", "reasoning": "<explain why>", "suggested_text": "<new text>" }

    5. Finish Review (call this when you have flagged all risks you could find):
    { "action_type": "finish_review" }

    Important Strategy:
    - You cannot read the entire contract easily. Use the Table of Contents or `search_contract` (e.g. for "liability", "indemnify", "renew", "data") to jump directly to risky clauses.
    - Always output raw JSON.
""").strip()


def build_user_prompt(obs: Any) -> str:
    """Build the user prompt from the observation."""
    flags_text = "None"
    if obs.flagged_issues:
        flags_text = ", ".join(f"{f['clause_id']} ({f['action_type']})" for f in obs.flagged_issues)

    return textwrap.dedent(f"""
        Task: {obs.task_description}

        Contract Title: {obs.contract_title}
        Total Clauses: {obs.total_clauses}
        Steps Remaining: {obs.steps_remaining}

        --- CURRENT VIEW ({obs.active_view}) ---
        {obs.view_content}
        ----------------------------------------

        Environment Message: {obs.message}
        Issues Flagged So Far: {flags_text}

        Based on the current view, choose your next action. Output a valid JSON object.
    """).strip()


def parse_llm_response(response_text: str) -> ContractAction:
    """Parse the LLM's response into a ContractAction."""
    text = response_text.strip()
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()

    try:
        data = json.loads(text)
        return ContractAction(**data)
    except Exception as e:
        if DEBUG:
            print(f"  [DEBUG] JSON parse failed: {e}")
        # Fallback heuristic
        if "search" in text.lower():
            return ContractAction(action_type="search_contract", search_query="liability")
        elif "finish" in text.lower() or "done" in text.lower():
            return ContractAction(action_type="finish_review")
        else:
            return ContractAction(action_type="read_clause", clause_id="c1")


def _clamp_score(value: float) -> float:
    return round(min(0.999, max(0.01, float(value))), 4)


def _sanitize_single_line(value: Optional[str]) -> str:
    if value is None:
        return "null"
    cleaned = re.sub(r"\s+", " ", str(value)).strip()
    return cleaned if cleaned else "null"


def _format_action(action: ContractAction) -> str:
    a = action.action_type
    if a == "search_contract":
        return f"search('{action.search_query}')"
    elif a == "read_clause":
        return f"read('{action.clause_id}')"
    elif a in ("flag_issue", "suggest_amendment"):
        return f"flag('{action.clause_id}','{action.severity}')"
    return "finish()"


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: Any, reward: float, done: bool, error: Optional[str] = None) -> None:
    error_val = _sanitize_single_line(error)
    done_val = str(done).lower()
    action_str = _sanitize_single_line(action)
    print(
        f"[STEP] step={step} action={action_str} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    safe_score = _clamp_score(score)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={safe_score:.3f} rewards={rewards_str}",
        flush=True,
    )


def run_task(
    client: OpenAI,
    env: ContractReviewEnvironment,
    task_id: str,
) -> float:
    log_start(task=task_id, env="contract_review_env", model=MODEL_NAME)

    step = 0
    rewards = []
    grader_score = 0.01
    success = False

    try:
        obs = env.reset(task_id=task_id)

        while not obs.done:
            step += 1
            if DEBUG:
                print(f"  [DEBUG] Active view: {obs.active_view}")

            user_prompt = build_user_prompt(obs)

            try:
                completion = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS,
                    stream=False,
                )
                response_text = completion.choices[0].message.content or ""
                error_msg = None
            except Exception as exc:
                error_msg = str(exc)
                response_text = ""

            action = parse_llm_response(response_text)

            obs = env.step(action)
            reward = obs.reward if obs.reward is not None else 0.01
            rewards.append(reward)

            log_step(
                step=step,
                action=_format_action(action),
                reward=reward,
                done=obs.done,
                error=error_msg
            )

        grader_score = env.get_last_grader_score()
        if grader_score is None:
            grader_score = 0.01
        grader_score = _clamp_score(grader_score)
        success = grader_score >= 0.5
        log_end(success=success, steps=step, score=grader_score, rewards=rewards)
        return grader_score

    except Exception as e:
        if DEBUG:
            print(f"  [CRITICAL ERROR] {e}")
        log_end(success=False, steps=step, score=0.01, rewards=rewards)
        return 0.01


def main() -> None:
    if DEBUG:
        print("=" * 60)
        print("CONTRACT REVIEW ENVIRONMENT — BASELINE INFERENCE")
        print("=" * 60)

    client_kwargs = {"api_key": HF_TOKEN or "dummy"}
    if API_BASE_URL:
        client_kwargs["base_url"] = API_BASE_URL
    client = OpenAI(**client_kwargs)

    env = ContractReviewEnvironment()

    task_ids = get_task_ids()
    scores = {}

    for task_id in task_ids:
        try:
            score = run_task(client, env, task_id)
            scores[task_id] = score
        except Exception as exc:
            if DEBUG:
                print(f"\n  [ERROR] Task {task_id} failed: {exc}")
            scores[task_id] = 0.01

    if DEBUG:
        print("\n" + "=" * 60)
        for task_id, score in scores.items():
            print(f"  SCORE - {task_id}: {score:.4f}")
        print("=" * 60)

if __name__ == "__main__":
    main()
