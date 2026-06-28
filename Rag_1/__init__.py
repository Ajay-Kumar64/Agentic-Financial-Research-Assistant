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

from rag.config import RAGConfig
from rag.opensearch_client import OpenSearchRAGClient
from rag.document_processor import DocumentProcessor, ProcessingConfig
from rag.indexing_pipeline import IndexingPipeline
from rag.retriever import SmartRetriever, RetrievalResult
from rag.reranker import FastReranker
from rag.cache import CacheManager

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