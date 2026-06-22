import sys
sys.path.insert(0, r"C:\Users\hp\PycharmProjects\PythonProject")

from agent.graph import agent_brain
from agent.state import initialize_agent_state

# Test 1: "latest" should return only 2024-25 docs
print("=" * 60)
print("TEST 1: LATEST")
state = initialize_agent_state("What is the latest repo rate?")
out = agent_brain.invoke(state)
print("Docs:", [p.get("doc_id") for p in out.get("retrieved_passages", [])[:5]])
print("Response:", out.get("final_response")[:400])

# Test 2: Specific year should return only that year
print("\n" + "=" * 60)
print("TEST 2: 2022-23")
state = initialize_agent_state("What was repo rate in 2022-23?")
out = agent_brain.invoke(state)
print("Docs:", [p.get("doc_id") for p in out.get("retrieved_passages", [])[:5]])
print("Response:", out.get("final_response")[:400])

# Test 3: No year filter
print("\n" + "=" * 60)
print("TEST 3: NO FILTER")
state = initialize_agent_state("What is repo rate?")
out = agent_brain.invoke(state)
print("Docs:", [p.get("doc_id") for p in out.get("retrieved_passages", [])[:5]])
print("Response:", out.get("final_response")[:400])