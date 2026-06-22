import os
import re
import time
import asyncio
from typing import Dict, Any, List

# Pre-load RAG modules
from rag.retriever import dual, load_faiss
from rag.fusion import rrf

# === RAG CACHE ===
import hashlib
_rag_local_cache: Dict[str, Any] = {}
_MAX_RAG_CACHE = 50
_RAG_CACHE_TTL = 600  # 10 minutes


def _check_relevance(passages: list, query: str) -> bool:
    """Check if retrieved passages are actually relevant to the query."""
    query_words = set(re.findall(r'\w+', query.lower()))
    # Keep only meaningful words (length > 3)
    query_words = {w for w in query_words if len(w) > 3 and w not in {
        "what", "when", "where", "which", "between", "from", "with", "have", "were",
        "they", "them", "their", "there", "about", "this", "that", "than", "then"
    }}
    if not query_words:
        return True  # too short to judge

    relevant_count = 0
    for p in passages[:3]:
        text = p.get("text", "").lower()
        matches = sum(1 for w in query_words if w in text)
        if matches >= max(1, len(query_words) // 3):
            relevant_count += 1

    return relevant_count >= 1  # at least 1 of top 3 must be relevant

def _rag_cache_key(query: str, year_filter: str | None) -> str:
    return hashlib.sha256(f"rag:{year_filter or 'all'}:{query.lower().strip()}".encode()).hexdigest()

def _get_cached_rag(query: str, year_filter: str | None) -> list | None:
    key = _rag_cache_key(query, year_filter)
    # Local cache
    if key in _rag_local_cache:
        entry = _rag_local_cache[key]
        if time.time() - entry.get("ts", 0) < _RAG_CACHE_TTL:
            print(f"[RAG Cache] Hit for: {query[:50]}...")
            return entry["passages"]
        del _rag_local_cache[key]
    return None

def _set_cached_rag(query: str, year_filter: str | None, passages: list):
    key = _rag_cache_key(query, year_filter)
    _rag_local_cache[key] = {"passages": passages, "ts": time.time()}
    if len(_rag_local_cache) > _MAX_RAG_CACHE:
        oldest = next(iter(_rag_local_cache))
        del _rag_local_cache[oldest]

# =============================================================================
# CONFIG
# =============================================================================
USE_RERANKER = False

# =============================================================================
# PRE-LOAD
# =============================================================================
print("[RAG Tool] Pre-loading FAISS index...")
_faiss_loaded = False

def _preload():
    global _faiss_loaded
    if not _faiss_loaded:
        index_path = os.path.join("artifacts", "faiss_index", "index.faiss")
        meta_path = os.path.join("artifacts", "faiss_index", "meta.pkl")
        if os.path.exists(index_path) and os.path.exists(meta_path):
            load_faiss(index_path, meta_path)
            _faiss_loaded = True
            print("[RAG Tool] ✅ FAISS pre-loaded")
        else:
            print(f"[RAG Tool] ⚠️ FAISS files not found")

    print("[RAG Tool] ⚡ Reranker DISABLED for fast CPU inference")

_preload()


# =============================================================================
# YEAR FILTERING LOGIC
# =============================================================================
def _extract_year_from_doc(doc_id: str) -> str:
    """Extract year like '2024-25' from '2024-25.pdf' or '2024-25.pdf_77'"""
    if not doc_id:
        return ""
    # FIXED: Match 20XX-YY where YY is exactly 2 digits
    match = re.search(r'20\d{2}[-]?\d{2}', str(doc_id))
    return match.group() if match else ""


def _sort_by_recency(passages: List[Dict], year_filter: str = None) -> List[Dict]:
    """
    If year_filter is 'latest', keep only the newest year's passages.
    If year_filter is specific (e.g., '2022-23'), keep only that year.
    """
    if not year_filter:
        return passages

    # Extract year from each passage
    for p in passages:
        p["_year"] = _extract_year_from_doc(p.get("doc_id", ""))

    if year_filter == "latest":
        # Sort by year descending (newest first)
        passages.sort(key=lambda x: x.get("_year", ""), reverse=True)
        newest_year = passages[0].get("_year", "") if passages else ""
        if newest_year:
            filtered = [p for p in passages if p.get("_year") == newest_year]
            if len(filtered) >= 2:  # lowered threshold
                print(f"[RAG] 'latest' → filtered to {len(filtered)} passages from {newest_year}")
                return filtered
        print(f"[RAG] 'latest' → not enough passages from {newest_year}, returning all")
        return passages
    else:
        # Specific year requested
        filtered = [p for p in passages if year_filter in p.get("_year", "")]
        if len(filtered) >= 1:
            print(f"[RAG] '{year_filter}' → filtered to {len(filtered)} passages")
            return filtered
        print(f"[RAG] '{year_filter}' → no matches found, returning all")
        return passages


# =============================================================================
# RETRIEVAL
# =============================================================================
async def retrieve_passages_async(query: str, top_k: int = 5, year_filter: str = None) -> List[Dict[str, Any]]:
    cached = _get_cached_rag(query, year_filter)
    if cached:
        return cached[:top_k]

    timings = {}
    t0 = time.time()

    # 1. Hybrid retrieval
    t1 = time.time()
    try:
        bm25_res, dense_res = await asyncio.to_thread(dual, query, k=top_k * 4)
    except Exception as e:
        print(f"[RAG] Hybrid failed ({e}), dense-only fallback...")
        bm25_res = []
        try:
            from rag.retriever import dense_index, faiss_meta, embedder
            import numpy as np
            query_vec = embedder.encode(query, normalize_embeddings=True).astype("float32")
            scores, indices = dense_index.search(np.expand_dims(query_vec, axis=0), top_k * 4)
            dense_res = []
            for rank, idx in enumerate(indices[0]):
                if idx == -1:
                    continue
                meta = faiss_meta[idx]
                dense_res.append({
                    "chunk_id": meta.get("chunk_id", f"idx_{idx}"),
                    "text": meta.get("text", ""),
                    "score": float(scores[0][rank]),
                    "doc_id": meta.get("doc_id", meta.get("source", "unknown")),
                    "page": meta.get("page", 0),
                })
        except Exception as e2:
            print(f"[RAG] Dense fallback failed: {e2}")
            return []

    timings["retrieval"] = round(time.time() - t1, 3)

    if not bm25_res and not dense_res:
        return []

    # 2. RRF Fusion
    t2 = time.time()
    if bm25_res and dense_res:
        bm25_rank = {d["chunk_id"]: i for i, d in enumerate(bm25_res)}
        dense_rank = {d["chunk_id"]: i for i, d in enumerate(dense_res)}
        fused = rrf([bm25_rank, dense_rank], k=60)
        fused_ids = [doc_id for doc_id, _ in fused[:top_k * 4]]
        doc_map = {d["chunk_id"]: d for d in bm25_res + dense_res}
        docs = [doc_map[cid] for cid in fused_ids if cid in doc_map]
    elif dense_res:
        docs = dense_res[:top_k * 4]
    else:
        docs = bm25_res[:top_k * 4]
    timings["fusion"] = round(time.time() - t2, 3)

    if not docs:
        return []

    # 3. Format
    passages = []
    for d in docs[:top_k * 2]:  # keep more for filtering
        passages.append({
            "chunk_id": d.get("chunk_id", "unknown"),
            "text": d.get("text", ""),
            "score": float(d.get("score", 0.0)),
            "doc_id": d.get("doc_id", d.get("source", "unknown")),
            "page": d.get("page", 0),
        })

    # 4. RECENCY FILTERING
    t4 = time.time()
    if year_filter:
        passages = _sort_by_recency(passages, year_filter)
        timings["recency_filter"] = round(time.time() - t4, 3)
    else:
        timings["recency_filter"] = 0.0

    # Take final top_k
    final_passages = passages[:top_k]

    total_time = round(time.time() - t0, 3)
    print(f"[RAG Timing] Total: {total_time}s | Retrieval: {timings['retrieval']}s | Fusion: {timings['fusion']}s | Recency: {timings['recency_filter']}s")

    _set_cached_rag(query, year_filter, final_passages)
    return final_passages


def retrieve_passages(query: str, top_k: int = 5, year_filter: str = None) -> List[Dict[str, Any]]:
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Running inside Uvicorn — schedule coroutine and block-wait
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(
                    asyncio.run,
                    retrieve_passages_async(query, top_k, year_filter)
                )
                return future.result(timeout=60)
        else:
            return loop.run_until_complete(retrieve_passages_async(query, top_k, year_filter))
    except RuntimeError:
        return asyncio.run(retrieve_passages_async(query, top_k, year_filter))


class RagSearchTool:
    name = "rag_search"
    description = "Retrieve factual information from RBI financial reports using hybrid BM25+FAISS retrieval."

    def run(self, query: str, top_k: int = 5, year_filter: str = None) -> Dict[str, Any]:
        start = time.time()
        passages = retrieve_passages(query, top_k=top_k, year_filter=year_filter)

        if not passages:
            return {
                "success": False,
                "retrieved_passages": [],
                "confidence_score": 0.0,
                "error": "No relevant documents found",
                "latency_sec": round(time.time() - start, 3),
                "text_summary": "[No relevant documents found]",
                "needs_fallback": True,
            }

        # Check relevance
        is_relevant = _check_relevance(passages, query)

        text = "\n\n".join([
                               f"[{i + 1}] Source: {p['doc_id']} (Year: {_extract_year_from_doc(p['doc_id'])}, Page {p['page']})\n{p['text'][:400]}"
                               for i, p in enumerate(passages)])
        avg_score = sum(p["score"] for p in passages) / len(passages) if passages else 0

        return {
            "success": True,
            "retrieved_passages": passages,
            "confidence_score": min(avg_score, 1.0),
            "text_summary": text,
            "latency_sec": round(time.time() - start, 3),
            "error": None,
            "needs_fallback": not is_relevant,  # True if docs seem irrelevant
        }