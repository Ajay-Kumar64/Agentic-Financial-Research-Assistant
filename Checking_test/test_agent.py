import sys
sys.path.insert(0, r"C:\Users\hp\PycharmProjects\PythonProject")

from agent.graph import agent_brain
from agent.state import initialize_agent_state

state = initialize_agent_state("What is the current repo rate?")
out = agent_brain.invoke(state)

print("=== RETRIEVED DOCUMENTS ===")
for i, p in enumerate(out.get("retrieved_passages", [])[:5], 1):
    doc = p.get("doc_id", "unknown")
    year = doc.replace(".pdf", "") if doc else "unknown"
    text_preview = p.get("text", "")[:100].replace("\n", " ")
    print(f"{i}. [{year}] {text_preview}...")

print("\n=== RESPONSE ===")
print(out.get("final_response")[:500])