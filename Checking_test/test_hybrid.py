from rag.retriever import dual, load_faiss

# MUST load the index first (this was done by @app.on_event("startup") in your API)
load_faiss(
    "../artifacts/faiss_index/index.faiss",
    "artifacts/faiss_index/meta.pkl"
)

query = "What is RBI repo rate?"

bm25_res, dense_res = dual(query, k=5)

print(f"BM25 hits: {len(bm25_res)}")
print(f"Dense hits: {len(dense_res)}")

if bm25_res:
    print("\n--- First BM25 result ---")
    print(bm25_res[0]["text"][:300])
else:
    print("\n--- BM25 returned NOTHING (ES may have no data) ---")

if dense_res:
    print("\n--- First Dense result ---")
    print(dense_res[0]["text"][:300])