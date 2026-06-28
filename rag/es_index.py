# rag/es_index.py — SAFE VERSION with graceful fallback
"""
Elasticsearch BM25 index with fallback to in-memory or disabled mode.
"""

import os
from typing import List, Dict, Any

# Try to import elasticsearch, but don't fail if not available
try:
    from elasticsearch import Elasticsearch
    _ES_AVAILABLE = True
except ImportError:
    _ES_AVAILABLE = False
    print("[ES] elasticsearch package not installed. BM25 disabled.")
    print("[ES] Install: pip install elasticsearch>=8.0.0")

ES_HOST = os.getenv("ES_HOST", "http://localhost:9200")
_es_client = None

def _get_es_client():
    global _es_client
    if not _ES_AVAILABLE:
        return None
    if _es_client is None:
        try:
            _es_client = Elasticsearch([ES_HOST])
            if not _es_client.ping():
                print(f"[ES] Cannot ping Elasticsearch at {ES_HOST}")
                _es_client = None
        except Exception as e:
            print(f"[ES] Connection failed: {e}")
            _es_client = None
    return _es_client

def bm25_search(query: str, k: int = 10) -> List[Dict[str, Any]]:
    """BM25 search via Elasticsearch. Returns empty list if ES unavailable."""
    client = _get_es_client()
    if client is None:
        return []

    try:
        response = client.search(
            index="financial_docs",
            body={
                "query": {
                    "match": {
                        "content": query
                    }
                },
                "size": k
            }
        )

        results = []
        for hit in response["hits"]["hits"]:
            results.append({
                "chunk_id": hit["_id"],
                "text": hit["_source"].get("content", ""),
                "score": hit["_score"],
                "doc_id": hit["_source"].get("doc_id", "unknown"),
                "page": hit["_source"].get("page", 0),
                "source": "bm25",
            })
        return results
    except Exception as e:
        print(f"[ES] Search failed: {e}")
        return []

def index_document(doc_id: str, text: str, metadata: Dict = None) -> bool:
    """Index a document into Elasticsearch."""
    client = _get_es_client()
    if client is None:
        return False

    try:
        client.index(
            index="financial_docs",
            id=doc_id,
            body={
                "content": text,
                "doc_id": doc_id,
                **(metadata or {})
            }
        )
        return True
    except Exception as e:
        print(f"[ES] Indexing failed: {e}")
        return False