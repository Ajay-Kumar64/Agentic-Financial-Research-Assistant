# rag/retriever.py
import time
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from rag.es_index import bm25_search
from rag.fusion import rrf

# -------- GLOBAL STATE (LOADED ON STARTUP) --------
dense_index = None
faiss_meta = None

# IMPORTANT: same model used during FAISS build
#embedder = SentenceTransformer("BAAI/bge-base-en-v1.5")
embedder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

def load_faiss(index_path: str, meta_path: str):
    """
    Called ONCE at app startup. Loads FAISS + warms up embedder.
    """
    global dense_index, faiss_meta

    dense_index = faiss.read_index(index_path)

    with open(meta_path, "rb") as f:
        faiss_meta = pickle.load(f)

    print("✅ FAISS index + meta loaded")

    # WARM-UP: First encode() loads weights into memory/cache — pay it once here
    print("[Retriever] Warming up BGE embedder...")
    t0 = time.time()
    _ = embedder.encode("warmup query", normalize_embeddings=True)
    print(f"[Retriever] ✅ Embedder ready in {time.time() - t0:.2f}s")


def dual(query: str, k: int = 10):
    """
    Hybrid retrieval: BM25 + Dense
    """
    if dense_index is None or faiss_meta is None:
        raise RuntimeError("Dense index not loaded")

    # ---------- BM25 ----------
    bm25_results = bm25_search(query, k)

    # ---------- Dense ----------
    query_vec = embedder.encode(
        query,
        normalize_embeddings=True
    ).astype("float32")

    scores, indices = dense_index.search(
        np.expand_dims(query_vec, axis=0),
        k
    )

    dense_results = []
    for rank, idx in enumerate(indices[0]):
        if idx == -1:
            continue
        meta = faiss_meta[idx]
        dense_results.append({
            "chunk_id": meta.get("chunk_id", f"idx_{idx}"),
            "text": meta.get("text", ""),
            "score": float(scores[0][rank]),
            "doc_id": meta.get("doc_id", meta.get("source", "unknown")),
            "page": meta.get("page", 0),
        })

    return bm25_results, dense_results


def fuse(bm25_results, dense_results, k=60):
    """
    Reciprocal Rank Fusion
    """
    bm25_rank = {c["chunk_id"]: i for i, c in enumerate(bm25_results)}
    dense_rank = {c["chunk_id"]: i for i, c in enumerate(dense_results)}

    return rrf([bm25_rank, dense_rank], k)