"""
rag/__init__.py
===============
Production RAG package for Agentic Financial Research Assistant.

This package provides end-to-end RAG capabilities:
    - Document processing (PDF → chunks → structured data)
    - Indexing (embed → bulk index → OpenSearch)
    - Retrieval (route → search → rerank → evaluate → cache)

Main exports:
    - SmartRetriever: Primary retrieval interface
    - IndexingPipeline: Document indexing
    - OpenSearchRAGClient: Low-level OpenSearch operations
    - DocumentProcessor: PDF processing and chunking
    - RAGConfig: Centralized configuration

Usage:
    from rag import SmartRetriever, RAGConfig

    config = RAGConfig()
    retriever = SmartRetriever(config)
    result = retriever.retrieve("What is the repo rate?")
"""

# Config (no external deps, safe to import)
from rag.config import RAGConfig

# Document processing (pdfplumber is optional with try/except)
from rag.document_processor import DocumentProcessor, ProcessingConfig

# Reranker (sentence-transformers needed, but lazy-loaded)
from rag.reranker import FastReranker

# Cache (redis optional, safe)
from rag.cache import CacheManager

# These require opensearch-py - wrapped in try/except for graceful failure
try:
    from rag.opensearch_client import OpenSearchRAGClient
    from rag.indexing_pipeline import IndexingPipeline
    from rag.retriever import SmartRetriever, RetrievalResult
    _OPENSEARCH_OK = True
except ImportError as e:
    print(f"[rag] ⚠️ OpenSearch components not available: {e}")
    print("[rag] Install: pip install opensearch-py sentence-transformers")
    OpenSearchRAGClient = None
    IndexingPipeline = None
    SmartRetriever = None
    RetrievalResult = None
    _OPENSEARCH_OK = False

__all__ = [
    "RAGConfig",
    "OpenSearchRAGClient",
    "DocumentProcessor",
    "ProcessingConfig",
    "IndexingPipeline",
    "SmartRetriever",
    "RetrievalResult",
    "FastReranker",
    "CacheManager",
]

__version__ = "2.0.0"