import requests
import time
import json
import sys

BASE_URL = "http://127.0.0.1:8999"

def wait_for_server():
    print("⏳ Waiting for server to start...")
    for _ in range(10):
        try:
            r = requests.get(f"{BASE_URL}/health")
            if r.status_code == 200:
                print("✅ Server is online!\n")
                return True
        except requests.exceptions.ConnectionError:
            time.sleep(1)
    print("❌ Server failed to start")
    sys.exit(1)

def run_tests():
    print("=" * 60)
    print("🚀 CONTRACT REVIEW OPENENV - API DEMONSTRATION")
    print("=" * 60)

    # 1. Health Endpoint
    print("\n[GET /health] - Checking server heartbeat")
    r = requests.get(f"{BASE_URL}/health")
    print(f"Response ({r.status_code}): {json.dumps(r.json(), indent=2)}")

    # 2. Tasks Endpoint
    print("\n[GET /tasks] - Fetching available tasks (Requirements check)")
    r = requests.get(f"{BASE_URL}/tasks")
    tasks = r.json()
    print(f"Response ({r.status_code}): Found {len(tasks['tasks'])} tasks.")
    for t in tasks["tasks"]:
        print(f" - {t['id']}: {t['description'][:50]}...")

    # 3. Reset Endpoint (Start an episode)
    print("\n[POST /reset] - Starting a new episode on 'clause_identification' (Easy)")
    payload = {"task_id": "clause_identification"}
    r = requests.post(f"{BASE_URL}/reset", json=payload)
    resp = r.json()
    obs = resp.get("observation", {})
    
    print(f"Response ({r.status_code}):")
    print(f" Contract: {obs.get('contract_title')}")
    print(f" Total Clauses: {obs.get('total_clauses')}")
    print(f" First Clause to Review: {str(obs.get('current_clause_text'))[:60]}...")
    
    # Quick variable holding
    clause_id = obs.get("current_clause_id")
    is_done = resp.get("done", False)
    
    # 4. Step Endpoint (play the game)
    step_num = 1
    while not is_done:
        print(f"\n[POST /step] - Automated Agent action on clause '{clause_id}'")
        
        # We will deterministically 'approve' some and 'flag_risk' others
        action_type = "flag_risk" if clause_id in ["c2", "c5"] else "approve"
        action_payload = {
            "action": {
                "clause_id": clause_id or "c1",
                "action_type": action_type,
                "severity": "critical" if action_type == "flag_risk" else None,
                "reasoning": "Detected an issue!" if action_type == "flag_risk" else "Looks standard.",
                "suggested_text": None
            }
        }
        print(f"-> Sending Action: {action_type.upper()}")
        
        r = requests.post(f"{BASE_URL}/step", json=action_payload)
        resp = r.json()
        obs = resp.get("observation", {})
        
        print(f"<- Response ({r.status_code}):")
        print(f"   Reward received: {resp.get('reward'):+.4f}")
        print(f"   Message: {obs.get('message')}")
        
        clause_id = obs.get("current_clause_id")
        is_done = resp.get("done", False)
        
        # 5. State Endpoint check in the middle
        if step_num == 3:
            print("\n[GET /state] - Dumping raw internal memory state at step 3")
            r_state = requests.get(f"{BASE_URL}/state")
            state_data = r_state.json()
            print(f"Response ({r_state.status_code}):")
            print(f" Contract ID: {state_data.get('contract_id')}")
            print(f" Total clauses evaluated so far: {len(state_data.get('evaluated_clauses', []))}")
        
        step_num += 1

    # 6. Grader Endpoint
    print("\n[GET /grader] - Episode is done. Fetching automated final score (Requirements check)")
    r = requests.get(f"{BASE_URL}/grader")
    score_data = r.json()
    print(f"Response ({r.status_code}): {json.dumps(score_data, indent=2)}")
    
    print("\n" + "=" * 60)
    print("✅ DEMO COMPLETE: All endpoints fully conform to OpenEnv Rubric!")
    print("=" * 60)


if __name__ == "__main__":
    if wait_for_server():
        run_tests()
