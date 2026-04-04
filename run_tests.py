"""
Comprehensive API Test Suite for Contract Review OpenEnv Environment
"""
import requests
import json
import sys
import os

BASE = "http://localhost:8999"
PASS = "✅ PASS"
FAIL = "❌ FAIL"
results = []

def test(name, condition, detail=""):
    status = PASS if condition else FAIL
    results.append((name, condition))
    print(f"{status} {name}")
    if detail:
        print(f"     → {detail}")
    return condition

print("=" * 60)
print("CONTRACT REVIEW ENV — COMPREHENSIVE API TEST SUITE")
print("=" * 60)

# ── TEST 1: /health ─────────────────────────────────────────
print("\n[1] GET /health")
try:
    r = requests.get(f"{BASE}/health", timeout=5)
    data = r.json()
    test("Returns HTTP 200", r.status_code == 200, f"Status: {r.status_code}")
    test("Returns healthy status", data.get("status") == "healthy", f"Response: {data}")
except Exception as e:
    test("GET /health reachable", False, str(e))

# ── TEST 2: /tasks ──────────────────────────────────────────
print("\n[2] GET /tasks")
try:
    r = requests.get(f"{BASE}/tasks", timeout=5)
    data = r.json()
    tasks = data.get("tasks", [])
    test("Returns HTTP 200", r.status_code == 200)
    test("Returns 3 tasks", len(tasks) == 3, f"Found: {len(tasks)} tasks")
    task_ids = [t["id"] for t in tasks]
    test("Contains clause_identification", "clause_identification" in task_ids)
    test("Contains risk_assessment", "risk_assessment" in task_ids)
    test("Contains negotiation", "negotiation" in task_ids)
    test("Each task has action_schema", all("action_schema" in t for t in tasks))
    for t in tasks:
        print(f"     → [{t.get('difficulty','?').upper()}] {t['id']}: {t.get('description','')[:60]}")
except Exception as e:
    test("GET /tasks reachable", False, str(e))

# ── TEST 3: /reset via HTTP ──────────────────────────────────
print("\n[3] POST /reset — clause_identification (Easy)")
try:
    r = requests.post(f"{BASE}/reset", json={"task_id": "clause_identification"}, timeout=10)
    raw = r.json()
    test("Returns HTTP 200", r.status_code == 200, f"Status: {r.status_code}")
    # OpenEnv wraps observation in {"observation": {...}, "reward": null, "done": false}
    obs = raw.get("observation", raw)
    test("Has contract_title", "contract_title" in obs, obs.get("contract_title", "MISSING"))
    test("Has current_clause_id", "current_clause_id" in obs, obs.get("current_clause_id", "MISSING"))
    test("Has total_clauses", "total_clauses" in obs, f"{obs.get('total_clauses','?')} clauses")
    test("Episode not done initially", raw.get("done") == False)
    print(f"     → Contract: {obs.get('contract_title','?')}")
    print(f"     → First Clause: {obs.get('current_clause_id','?')} — {obs.get('current_clause_title','?')}")
    print(f"     → Total Clauses: {obs.get('total_clauses','?')}")
except Exception as e:
    test("POST /reset reachable", False, str(e))

# ── TEST 4: /state ───────────────────────────────────────────
print("\n[4] GET /state")
try:
    r = requests.get(f"{BASE}/state", timeout=5)
    test("Returns HTTP 200", r.status_code == 200, f"Status: {r.status_code}")
    data = r.json()
    test("Has episode_id or task_id", "episode_id" in data or "task_id" in data)
    print(f"     → State keys: {list(data.keys())}")
except Exception as e:
    test("GET /state reachable", False, str(e))

# ── TEST 5: /grader ──────────────────────────────────────────
print("\n[5] GET /grader")
try:
    r = requests.get(f"{BASE}/grader", timeout=5)
    data = r.json()
    test("Returns HTTP 200", r.status_code == 200, f"Status: {r.status_code}")
    test("Has score field", "score" in data, f"Keys: {list(data.keys())}")
    print(f"     → Response: {data}")
except Exception as e:
    test("GET /grader reachable", False, str(e))

# ── TEST 6: /docs (Swagger UI) ───────────────────────────────
print("\n[6] GET /docs — Swagger UI")
try:
    r = requests.get(f"{BASE}/docs", timeout=5)
    test("Swagger UI returns HTTP 200", r.status_code == 200, f"Status: {r.status_code}")
    test("Contains swagger/openapi HTML", "openapi" in r.text.lower() or "swagger" in r.text.lower())
except Exception as e:
    test("GET /docs reachable", False, str(e))

# ── TEST 7: /openapi.json ────────────────────────────────────
print("\n[7] GET /openapi.json — Schema Validation")
try:
    r = requests.get(f"{BASE}/openapi.json", timeout=5)
    data = r.json()
    test("Returns HTTP 200", r.status_code == 200)
    test("Has paths defined", "paths" in data)
    paths = list(data.get("paths", {}).keys())
    test("Has /health path", any("/health" in p for p in paths))
    test("Has /tasks path", any("/tasks" in p for p in paths))
    test("Has /reset path", any("/reset" in p for p in paths))
    test("Has /step path", any("/step" in p for p in paths))
    test("Has /grader path", any("/grader" in p for p in paths))
    print(f"     → Registered paths: {paths}")
except Exception as e:
    test("GET /openapi.json reachable", False, str(e))

# ── TEST 8: Full Episode via Direct Python ───────────────────
print("\n[8] FULL EPISODE — Direct Python (Session-aware)")
try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "server"))
    from server.environment import ContractReviewEnvironment
    from models import ContractAction

    env = ContractReviewEnvironment()
    obs = env.reset(task_id="clause_identification")

    test("Reset returns observation", obs is not None)
    test("Observation has contract_title", hasattr(obs, "contract_title"), obs.contract_title)
    test("Episode starts with 5 clauses", obs.total_clauses == 5, f"{obs.total_clauses} clauses")

    steps = 0
    total_reward = 0.0
    while not obs.done and steps < 20:
        steps += 1
        action = ContractAction(
            clause_id=obs.current_clause_id,
            action_type="flag_risk",
            severity="critical",
            reasoning="This clause contains highly problematic terms exposing the subscriber to unlimited risk.",
            suggested_text=None
        )
        obs = env.step(action)
        reward = obs.reward or 0
        total_reward += reward
        print(f"     → Step {steps}: clause={obs.current_clause_id if not obs.done else 'DONE'}, reward={reward:.4f}")

    test("Episode completes in 5 steps", steps == 5, f"Steps taken: {steps}")
    test("Episode is marked done", obs.done)
    score = env.get_last_grader_score()
    test("Grader score is float 0.0–1.0", score is not None and 0.0 <= score <= 1.0, f"Score: {score}")
    print(f"     → Final Grader Score: {score:.4f}")
    print(f"     → Total Episode Reward: {total_reward:.4f}")

    # Verify /grader endpoint reflects updated score
    r = requests.get(f"{BASE}/grader", timeout=5)
    grader_data = r.json()
    test("/grader endpoint has score key", "score" in grader_data)
    print(f"     → /grader API response: {grader_data}")

except Exception as e:
    test("Full episode direct Python", False, str(e))
    import traceback; traceback.print_exc()

# ── SUMMARY ──────────────────────────────────────────────────
print("\n" + "=" * 60)
print("FINAL TEST SUMMARY")
print("=" * 60)
passed = sum(1 for _, ok in results if ok)
total = len(results)
for name, ok in results:
    print(f"  {'✅' if ok else '❌'} {name}")
print(f"\n  RESULT: {passed}/{total} tests passed")
if passed == total:
    print("  🏆 ALL TESTS PASSED — Project is fully functional!")
elif passed >= int(total * 0.85):
    print(f"  ✅ {passed}/{total} passed — Project is working well!")
else:
    print(f"  ⚠️  {total - passed} test(s) failed — review above output")
print("=" * 60)
