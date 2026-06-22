import time
from agent.tools.rag_search import retrieve_passages

queries = [
    "What is RBI repo rate?",
    "Compare Q1 and Q2 money market rates",
    "What is WACR and how does it relate to repo?",
]

print("=" * 60)
print("RAG BENCHMARK (Reranker: DISABLED)")
print("=" * 60)

# Warm-up query (absorbs the embedder load time)
print("\n[Warming up embedder...]")
_ = retrieve_passages("warmup", top_k=1)
print("[Warm-up complete]\n")

for q in queries:
    print(f"Query: {q}")
    start = time.time()
    result = retrieve_passages(q, top_k=5)
    total = time.time() - start
    print(f"Total: {total:.2f}s | Passages: {len(result)}")
    if result:
        print(f"Top doc: {result[0]['doc_id']} (score: {result[0]['score']:.3f})")
    print()