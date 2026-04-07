"""
Interactive CLI for the Contract Review OpenEnv Environment.
Allows a human to play the role of the AI agent!
"""

import sys
import os

# Ensure we can import the environment
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "server"))

from server.environment import ContractReviewEnvironment
from models import ContractAction


def main():
    print("=" * 60)
    print("📝 CONTRACT REVIEW ENVIRONMENT — INTERACTIVE TERMINAL")
    print("=" * 60)
    
    env = ContractReviewEnvironment()
    
    # Let user pick difficulty
    print("Select a task:")
    print("  1. clause_identification (Easy)")
    print("  2. risk_assessment (Medium)")
    print("  3. negotiation (Hard)")
    
    choice = input("\nEnter 1, 2, or 3 (default: 1): ").strip()
    task_id = "clause_identification"
    if choice == "2":
        task_id = "risk_assessment"
    elif choice == "3":
        task_id = "negotiation"
        
    print(f"\nStarting episode for task: {task_id}...\n")
    obs = env.reset(task_id=task_id)
    
    print(f"Contract: {obs.contract_title}")
    print(obs.message)
    print("-" * 60)
    
    step = 0
    while not obs.done:
        step += 1
        print(f"\n[Step {step}] Reviewing Clause '{obs.current_clause_id}'")
        print(f"Title: {obs.current_clause_title}")
        print("\n--- Clause Text ---")
        print(obs.current_clause_text)
        print("-------------------\n")
        
        valid_actions = ["approve", "flag_risk", "suggest_amendment"]
        a_type = ""
        while a_type not in valid_actions:
            a_type = input("Action [approve / flag_risk / suggest_amendment]: ").strip().lower()
            
        severity = None
        if a_type in ["flag_risk", "suggest_amendment"]:
            valid_sevs = ["critical", "moderate", "minor"]
            sev_input = ""
            while sev_input not in valid_sevs:
                sev_input = input("Severity [critical / moderate / minor]: ").strip().lower()
            severity = sev_input
            
        reasoning = input("Reasoning (Why?): ").strip() or "No reasoning provided"
        
        suggested_text = None
        if a_type == "suggest_amendment":
            suggested_text = input("Suggested Amendment Text: ").strip() or None
            
        action = ContractAction(
            clause_id=obs.current_clause_id,
            action_type=a_type,
            severity=severity,
            reasoning=reasoning,
            suggested_text=suggested_text
        )
        
        obs = env.step(action)
        reward_display = obs.reward if obs.reward is not None else 0.0
        print(f"\n✅ Action Processed. Reward Earned: {reward_display:+.4f}")
        print(f"Response: {obs.message}")
        print("-" * 60)
        
    print("\n🎉 EPISODE COMPLETE!")
    print(obs.message)
    print(f"Final Grader Score: {env.get_last_grader_score():.4f}")


if __name__ == "__main__":
    main()
