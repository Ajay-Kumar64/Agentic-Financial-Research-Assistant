"""
agent/tools/rag_search.py
==========================
OpenSearch-based RAG search tool for the agent.

Replaces FAISS+BM25 with OpenSearch hybrid search.
Keeps backward-compatible async/sync helper functions for the agent graph.
"""

import os
import re
import time
import asyncio
import hashlib
from typing import Dict, Any, List

from rag.retriever import SmartRetriever
from rag.cache import CacheManager


# =============================================================================
# LOCAL CACHE (kept for fast in-memory hits before Redis)
# =============================================================================
_rag_local_cache: Dict[str, Any] = {}
_MAX_RAG_CACHE = 50
_RAG_CACHE_TTL = 600  # 10 minutes


def _rag_cache_key(query: str, year_filter: str | None) -> str:
    return hashlib.sha256(f"rag:{year_filter or 'all'}:{query.lower().strip()}".encode()).hexdigest()


def _get_cached_rag(query: str, year_filter: str | None) -> list | None:
    key = _rag_cache_key(query, year_filter)
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
# RELEVANCE CHECK (kept for agent guardrails)
# =============================================================================
def _check_relevance(passages: list, query: str) -> bool:
    """Check if retrieved passages are actually relevant to the query."""
    query_words = set(re.findall(r'\w+', query.lower()))
    query_words = {w for w in query_words if len(w) > 3 and w not in {
        "what", "when", "where", "which", "between", "from", "with", "have", "were",
        "they", "them", "their", "there", "about", "this", "that", "than", "then"
    }}
    if not query_words:
        return True

    relevant_count = 0
    for p in passages[:3]:
        text = p.get("text", "").lower()
        matches = sum(1 for w in query_words if w in text)
        if matches >= max(1, len(query_words) // 3):
            relevant_count += 1

    return relevant_count >= 1


# =============================================================================
# YEAR FILTERING (kept for agent recency queries)
# =============================================================================
def _extract_year_from_doc(doc_id: str) -> str:
    """Extract year like '2024-25' from '2024-25.pdf' or '2024-25.pdf_77'"""
    if not doc_id:
        return ""
    match = re.search(r'20\d{2}[-]?\d{2}', str(doc_id))
    return match.group() if match else ""


def _sort_by_recency(passages: List[Dict], year_filter: str = None) -> List[Dict]:
    """
    If year_filter is 'latest', keep only the newest year's passages.
    If year_filter is specific (e.g., '2022-23'), keep only that year.
    """
    if not year_filter:
        return passages

    for p in passages:
        p["_year"] = _extract_year_from_doc(p.get("doc_id", ""))

    if year_filter == "latest":
        passages.sort(key=lambda x: x.get("_year", ""), reverse=True)
        newest_year = passages[0].get("_year", "") if passages else ""
        if newest_year:
            filtered = [p for p in passages if p.get("_year") == newest_year]
            # Return only the single most recent passage
            return filtered[:1] if filtered else []
        return []
    else:
        filtered = [p for p in passages if year_filter in p.get("_year", "")]
        if len(filtered) >= 1:
            print(f"[RAG] '{year_filter}' → filtered to {len(filtered)} passages")
            return filtered
        print(f"[RAG] '{year_filter}' → no matches found, returning all")
        return passages


# =============================================================================
# SMART RETRIEVER (OpenSearch — loaded once)
# =============================================================================
_retriever: SmartRetriever = None


def _get_retriever() -> SmartRetriever:
    """Lazy-load SmartRetriever with config from environment."""
    global _retriever
    if _retriever is None:
        from rag.config import RAGConfig
        config = RAGConfig()
        _retriever = SmartRetriever(
            embedder_model=config.embedder_model,
            reranker_model=config.reranker_model,
        )
        _retriever.warmup()
        print("[RAG Tool] ✅ SmartRetriever ready (OpenSearch)")
    return _retriever


# =============================================================================
# ASYNC RETRIEVAL (rewired to OpenSearch — used by agent graph nodes)
# =============================================================================
async def retrieve_passages_async(query: str, top_k: int = 5, year_filter: str = None) -> List[Dict[str, Any]]:
    """
    Async retrieval using OpenSearch hybrid search.
    Used by agent graph nodes.
    """
    cached = _get_cached_rag(query, year_filter)
    if cached:
        return cached[:top_k]

    t0 = time.time()

    retriever = _get_retriever()
    result = retriever.retrieve(query, top_k=top_k, year_filter=year_filter)

    docs = result.docs
    if not docs:
        return []

    # Format to match old interface
    passages = []
    for d in docs:
        passages.append({
            "chunk_id": d.get("chunk_id", "unknown"),
            "text": d.get("text", ""),
            "score": float(d.get("score", 0.0)),
            "doc_id": d.get("doc_id", d.get("source", "unknown")),
            "page": d.get("page", 0),
            "year": d.get("year", ""),
        })

    # Year filtering
    if year_filter:
        passages = _sort_by_recency(passages, year_filter)

    final_passages = passages[:top_k]

    total_time = round(time.time() - t0, 3)
    print(f"[RAG Timing] Total: {total_time}s | Strategy: {result.strategy} | Cached: {result.cached}")

    _set_cached_rag(query, year_filter, final_passages)
    return final_passages


async def parallel_retrieve(queries: list, top_k: int = 5) -> list:
    """
    Retrieve passages for multiple queries in parallel.
    Cuts latency when the planner needs multiple independent retrievals.
    """
    if not queries:
        return []

    unique_queries = list(dict.fromkeys(queries))
    tasks = [retrieve_passages_async(q, top_k=top_k) for q in unique_queries]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    all_passages = []
    for r in results:
        if isinstance(r, list):
            all_passages.extend(r)
        # Exceptions skipped gracefully

    return all_passages


def retrieve_passages(query: str, top_k: int = 5, year_filter: str = None) -> List[Dict[str, Any]]:
    """
    Synchronous wrapper for retrieve_passages_async.
    Handles both running and non-running event loops.
    """
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
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


# =============================================================================
# MAIN RAG SEARCH TOOL (OpenSearch — used by agent planner)
# =============================================================================
class RagSearchTool:
    """
    RAG search tool for LangGraph agent.

    Usage:
        tool = RagSearchTool()
        result = tool.run("What is the repo rate?", top_k=5)

        if result["needs_fallback"]:
            # Route to web_search
            pass
        else:
            # Use result["text_summary"] as LLM context
            pass
    """

    name = "rag_search"
    description = (
        "Retrieve factual information from RBI financial reports using "
        "OpenSearch hybrid retrieval (BM25 + kNN) with reranking, "
        "query expansion, and self-evaluation."
    )

    def __init__(self):
        self.retriever = _get_retriever()

    def run(self, query: str, top_k: int = 5, year_filter: str = None) -> Dict[str, Any]:
        """
        Execute RAG search.

        Returns:
            {
                "success": bool,
                "retrieved_passages": List[Dict],
                "confidence_score": float,
                "text_summary": str,
                "latency_sec": float,
                "error": str | None,
                "needs_fallback": bool,
                "strategy": str,
                "crag_confidence": float,
                "crag_reason": str,
                "cached": bool,
                "timings": Dict,
            }
        """
        start = time.time()

        # Check local cache first
        cached = _get_cached_rag(query, year_filter)
        if cached:
            passages = cached[:top_k]
            is_relevant = _check_relevance(passages, query)

            text_lines = []
            for i, doc in enumerate(passages):
                year = doc.get("year", "") or _extract_year_from_doc(doc.get("doc_id", ""))
                page = doc.get("page", 0)
                doc_id = doc.get("doc_id", "unknown")
                doc_text = doc.get("text", "")[:400]
                text_lines.append(
                    f"[{i+1}] Source: {doc_id} (Year: {year}, Page {page})\n{doc_text}"
                )

            text = "\n\n".join(text_lines)
            avg_score = sum(d.get("score", 0) for d in passages) / len(passages) if passages else 0

            return {
                "success": True,
                "retrieved_passages": passages,
                "confidence_score": min(avg_score, 1.0),
                "text_summary": text,
                "latency_sec": round(time.time() - start, 3),
                "error": None,
                "needs_fallback": not is_relevant,
                "strategy": "cache_hit",
                "crag_confidence": 1.0 if is_relevant else 0.5,
                "crag_reason": "retrieved_from_local_cache",
                "cached": True,
                "timings": {"cache_lookup": round(time.time() - start, 3)},
            }

        try:
            result = self.retriever.retrieve(
                query=query,
                top_k=top_k,
                year_filter=year_filter,
            )
        except Exception as e:
            print(f"[RAG Tool] SmartRetriever failed: {e}")
            return {
                "success": False,
                "retrieved_passages": [],
                "confidence_score": 0.0,
                "error": f"Retrieval error: {str(e)}",
                "latency_sec": round(time.time() - start, 3),
                "text_summary": "[Retrieval system error]",
                "needs_fallback": True,
                "strategy": "error",
                "crag_confidence": 0.0,
                "crag_reason": "system_error",
                "cached": False,
                "timings": {},
            }

        docs = result.docs

        if not docs:
            return {
                "success": False,
                "retrieved_passages": [],
                "confidence_score": 0.0,
                "error": "No relevant documents found",
                "latency_sec": round(time.time() - start, 3),
                "text_summary": "[No relevant documents found]",
                "needs_fallback": True,
                "strategy": result.strategy,
                "crag_confidence": result.confidence,
                "crag_reason": result.reason,
                "cached": result.cached,
                "timings": result.timings,
            }

        # Build text summary for LLM context
        text_lines = []
        for i, doc in enumerate(docs):
            year = doc.get("year", "")
            page = doc.get("page", 0)
            doc_id = doc.get("doc_id", "unknown")
            doc_text = doc.get("text", "")[:400]
            text_lines.append(
                f"[{i+1}] Source: {doc_id} (Year: {year}, Page {page})\n{doc_text}"
            )

        text = "\n\n".join(text_lines)
        avg_score = sum(d.get("score", 0) for d in docs) / len(docs) if docs else 0

        # Cache successful results
        _set_cached_rag(query, year_filter, docs)

        return {
            "success": True,
            "retrieved_passages": docs,
            "confidence_score": min(avg_score, 1.0),
            "text_summary": text,
            "latency_sec": round(time.time() - start, 3),
            "error": None,
            "needs_fallback": result.fallback_needed,
            "strategy": result.strategy,
            "crag_confidence": result.confidence,
            "crag_reason": result.reason,
            "cached": result.cached,
            "timings": result.timings,
        }

    def run_batch(self, queries: List[str], top_k: int = 5) -> List[Dict[str, Any]]:
        """Run multiple queries in sequence."""
        return [self.run(q, top_k=top_k) for q in queries]