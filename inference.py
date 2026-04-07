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

# Add the environment package to path — works from repo root or parent
_script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _script_dir)
sys.path.insert(0, os.path.join(_script_dir, "server"))
# Also check parent (when inference.py is run from outside)
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
MAX_TOKENS = 500

DEBUG = os.getenv("DEBUG", "false").lower() == "true"


# ── System prompt for the contract review agent ──────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""
    You are a contract review AI agent. You analyze contract clauses and determine
    whether they are acceptable (approve) or problematic (flag_risk/suggest_amendment).

    For EACH clause you must respond with a single JSON object (no other text):
    {
        "clause_id": "<the clause ID, e.g. c1>",
        "action_type": "<approve | flag_risk | suggest_amendment>",
        "severity": "<critical | moderate | minor | null>",
        "reasoning": "<your explanation>",
        "suggested_text": "<proposed amendment text or null>"
    }

    Rules:
    - If a clause looks standard and fair, use "approve" with severity null.
    - If you spot problematic terms, use "flag_risk" with appropriate severity.
    - If you can propose better wording, use "suggest_amendment" with suggested_text.
    - severity is required for flag_risk and suggest_amendment.
    - Be specific in reasoning — mention the exact problematic language.
    - Respond ONLY with the JSON object, no markdown, no explanation outside the JSON.
""").strip()


def build_user_prompt(
    task_description: str,
    contract_title: str,
    clause_id: str,
    clause_title: str,
    clause_text: str,
    clause_index: int,
    total_clauses: int,
    reviewed_clauses: List[Dict[str, Any]],
    message: str,
) -> str:
    """Build the user prompt for the LLM."""
    # Summarize previous reviews
    history = ""
    if reviewed_clauses:
        history_lines = []
        for r in reviewed_clauses[-5:]:  # Last 5 reviews for context
            history_lines.append(
                f"  - {r.get('clause_id', '?')} ({r.get('clause_title', '?')}): "
                f"{r.get('action_type', '?')}"
                + (f" [severity: {r.get('severity')}]" if r.get('severity') else "")
            )
        history = "\nPrevious reviews:\n" + "\n".join(history_lines)

    return textwrap.dedent(f"""
        Contract: {contract_title}
        Task: {task_description}

        Clause {clause_index + 1} of {total_clauses}
        Clause ID: {clause_id}
        Clause Title: {clause_title}

        --- CLAUSE TEXT ---
        {clause_text}
        --- END CLAUSE ---
        {history}

        Environment message: {message}

        Respond with a JSON object for your review of clause {clause_id}.
    """).strip()


def parse_llm_response(response_text: str, clause_id: str) -> ContractAction:
    """Parse the LLM's response into a ContractAction."""
    # Try to extract JSON from the response
    text = response_text.strip()

    # Handle markdown code blocks
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()

    try:
        data = json.loads(text)
        return ContractAction(
            clause_id=data.get("clause_id", clause_id),
            action_type=data.get("action_type", "approve"),
            severity=data.get("severity"),
            reasoning=data.get("reasoning", "No reasoning provided"),
            suggested_text=data.get("suggested_text"),
        )
    except (json.JSONDecodeError, Exception) as e:
        if DEBUG:
            print(f"  [DEBUG] JSON parse failed: {e}")
            print(f"  [DEBUG] Raw response: {text[:200]}")

        # Fallback: try to infer from text
        text_lower = text.lower()
        if any(kw in text_lower for kw in ["problematic", "risk", "concern", "unfair", "unlimited", "critical"]):
            return ContractAction(
                clause_id=clause_id,
                action_type="flag_risk",
                severity="moderate",
                reasoning=text[:200] if text else "Detected potential risk",
                suggested_text=None,
            )
        else:
            return ContractAction(
                clause_id=clause_id,
                action_type="approve",
                severity=None,
                reasoning=text[:200] if text else "Clause appears standard",
                suggested_text=None,
            )


def _clamp_score(value: float) -> float:
    """Ensure all reported task scores stay strictly within (0, 1)."""
    return round(min(0.999, max(0.001, float(value))), 4)


def _sanitize_single_line(value: Optional[str]) -> str:
    """Collapse whitespace so log values always stay on one line."""
    if value is None:
        return "null"
    cleaned = re.sub(r"\s+", " ", str(value)).strip()
    return cleaned if cleaned else "null"


def _format_action(action: ContractAction) -> str:
    """Return a compact validator-friendly action string."""
    severity = action.severity or "null"
    return f"review('{action.clause_id}','{action.action_type}','{severity}')"


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
    """Run a single task and return the grader score."""
    log_start(task=task_id, env="contract_review_env", model=MODEL_NAME)

    step = 0
    rewards = []
    grader_score = 0.001
    success = False

    try:
        # Reset environment
        obs = env.reset(task_id=task_id)

        while not obs.done:
            step += 1
            if DEBUG:
                print(f"  [DEBUG] Reviewing clause {obs.current_clause_id} — '{obs.current_clause_title}'")

            # Build prompt
            user_prompt = build_user_prompt(
                task_description=obs.task_description,
                contract_title=obs.contract_title,
                clause_id=obs.current_clause_id,
                clause_title=obs.current_clause_title,
                clause_text=obs.current_clause_text,
                clause_index=obs.clause_index,
                total_clauses=obs.total_clauses,
                reviewed_clauses=obs.reviewed_clauses,
                message=obs.message,
            )

            # Call LLM
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

            # Parse action
            action = parse_llm_response(response_text, obs.current_clause_id)

            # Step environment
            obs = env.step(action)
            reward = obs.reward if obs.reward is not None else 0.0
            rewards.append(reward)

            log_step(
                step=step,
                action=_format_action(action),
                reward=reward,
                done=obs.done,
                error=error_msg
            )

        # Get grader score
        grader_score = env.get_last_grader_score()
        if grader_score is None:
            grader_score = 0.001
        grader_score = _clamp_score(grader_score)
        success = grader_score >= 0.5
        log_end(success=success, steps=step, score=grader_score, rewards=rewards)
        return grader_score

    except Exception as e:
        if DEBUG:
            print(f"  [CRITICAL ERROR] {e}")
        # ALWAYS print an [END] log even if crash, to prevent Validator falling back to exactly 0.0
        log_end(success=False, steps=step, score=0.001, rewards=rewards)
        return 0.001


def main() -> None:
    """Run inference on all 3 tasks and report scores."""
    if DEBUG:
        print("=" * 60)
        print("CONTRACT REVIEW ENVIRONMENT — BASELINE INFERENCE")
        print("=" * 60)
        print(f"API_BASE_URL: {API_BASE_URL}")
        print(f"MODEL_NAME:   {MODEL_NAME}")
        print(f"HF_TOKEN:     {'***' + HF_TOKEN[-4:] if HF_TOKEN else 'NOT SET'}")
        if LOCAL_IMAGE_NAME:
            print(f"LOCAL_IMAGE_NAME: {LOCAL_IMAGE_NAME}")
        print()

    if not HF_TOKEN and DEBUG:
        print("WARNING: No HF_TOKEN set. Running with empty key — LLM calls may fail.")

    # Initialize OpenAI client
    client_kwargs = {"api_key": HF_TOKEN or "dummy"}
    if API_BASE_URL:
        client_kwargs["base_url"] = API_BASE_URL
    client = OpenAI(**client_kwargs)

    # Initialize environment (direct, no server needed)
    env = ContractReviewEnvironment()

    # Run all tasks
    task_ids = get_task_ids()
    scores = {}

    for task_id in task_ids:
        try:
            score = run_task(client, env, task_id)
            scores[task_id] = score
        except Exception as exc:
            if DEBUG:
                print(f"\n  [ERROR] Task {task_id} failed: {exc}")
            scores[task_id] = 0.001  # Strictly > 0 per validator requirement


    # Summary
    if DEBUG:
        print("\n" + "=" * 60)
        print("BASELINE RESULTS SUMMARY")
        print("=" * 60)
        for task_id, score in scores.items():
            print(f"  SCORE - {task_id}: {score:.4f}")
        avg = sum(scores.values()) / len(scores) if scores else 0.0
        print(f"\n  AVERAGE SCORE: {avg:.4f}")
        print("=" * 60)


if __name__ == "__main__":
    main()
