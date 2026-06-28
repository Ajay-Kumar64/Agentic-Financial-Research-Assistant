"""
rag/config.py
=============
Centralized configuration for the RAG system.

All settings in one place — environment variables override defaults.
This makes the system easy to configure per environment (dev/staging/prod).
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class RAGConfig:
    """
    Production RAG configuration.

    Usage:
        from rag.config import RAGConfig

        # Default config (loads from environment)
        config = RAGConfig()

        # Override specific values
        config = RAGConfig(
            opensearch_index="my_index",
            use_hyde=False,
        )
    """

    # =====================================================================
    # OpenSearch Connection
    # =====================================================================
    opensearch_host: str = field(default_factory=lambda: os.getenv("OPENSEARCH_HOST", "opensearch"))
    opensearch_port: int = field(default_factory=lambda: int(os.getenv("OPENSEARCH_PORT", "9200")))
    opensearch_user: str = field(default_factory=lambda: os.getenv("OPENSEARCH_USER", "admin"))
    opensearch_password: str = field(default_factory=lambda: os.getenv("OPENSEARCH_PASSWORD", "admin"))
    opensearch_index: str = field(default_factory=lambda: os.getenv("OPENSEARCH_INDEX", "rbi_reports"))

    # OpenSearch performance
    opensearch_num_shards: int = 1
    opensearch_num_replicas: int = 0
    opensearch_refresh_interval: str = "1s"

    # =====================================================================
    # Models
    # =====================================================================
    embedder_model: str = field(default_factory=lambda: os.getenv("EMBEDDER_MODEL", "BAAI/bge-base-en-v1.5"))
    reranker_model: str = field(default_factory=lambda: os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3"))
    embedding_dim: int = 768

    # =====================================================================
    # Chunking
    # =====================================================================
    parent_size: int = 1024      # words per parent chunk (returned to LLM)
    child_size: int = 256        # words per child chunk (indexed for retrieval)
    overlap: int = 50            # word overlap between chunks

    # =====================================================================
    # Retrieval
    # =====================================================================
    bm25_k1: float = 1.2         # BM25 term frequency saturation
    bm25_b: float = 0.75         # BM25 length normalization
    dense_k: int = 50            # kNN candidates before fusion
    hybrid_k: int = 50           # Final results from hybrid search
    rerank_topn: int = 50        # Cross-encoder rerank candidates
    final_topk: int = 5          # Final results to return to LLM

    # Hybrid fusion weights [BM25, Dense]
    # BM25 catches exact terms, Dense catches semantic similarity
    # For financial docs: 30% BM25, 70% Dense (tune based on evaluation)
    hybrid_weights: List[float] = field(default_factory=lambda: [0.3, 0.7])

    # =====================================================================
    # CRAG (Self-Evaluation)
    # =====================================================================
    crag_threshold: float = 0.35       # Confidence threshold for fallback
    crag_use_llm: bool = True          # Use LLM judge for borderline cases

    # =====================================================================
    # Cache
    # =====================================================================
    cache_ttl: int = 300               # 5 minutes
    cache_max_memory: int = 100        # Max in-memory entries

    # =====================================================================
    # Feature Flags
    # =====================================================================
    use_hyde: bool = field(default_factory=lambda: os.getenv("USE_HYDE", "true").lower() == "true")
    use_reranker: bool = field(default_factory=lambda: os.getenv("USE_RERANKER", "true").lower() == "true")
    use_crag: bool = field(default_factory=lambda: os.getenv("USE_CRAG", "true").lower() == "true")
    use_router: bool = field(default_factory=lambda: os.getenv("USE_ROUTER", "true").lower() == "true")

    # =====================================================================
    # LLM (for HyDE and CRAG)
    # =====================================================================
    google_api_key: str = field(default_factory=lambda: os.getenv("GOOGLE_API_KEY", ""))
    llm_model: str = "gemini-2.0-flash-lite"

    # =====================================================================
    # Redis
    # =====================================================================
    redis_host: str = field(default_factory=lambda: os.getenv("REDIS_HOST", "redis"))
    redis_port: int = field(default_factory=lambda: int(os.getenv("REDIS_PORT", "6379")))
    redis_db: int = field(default_factory=lambda: int(os.getenv("REDIS_DB", "0")))

    def to_dict(self) -> dict:
        """Export config as dict (for logging/debugging)."""
        return {
            "opensearch": {
                "host": self.opensearch_host,
                "port": self.opensearch_port,
                "index": self.opensearch_index,
            },
            "models": {
                "embedder": self.embedder_model,
                "reranker": self.reranker_model,
                "embedding_dim": self.embedding_dim,
            },
            "chunking": {
                "parent_size": self.parent_size,
                "child_size": self.child_size,
                "overlap": self.overlap,
            },
            "retrieval": {
                "hybrid_weights": self.hybrid_weights,
                "final_topk": self.final_topk,
            },
            "features": {
                "hyde": self.use_hyde,
                "reranker": self.use_reranker,
                "crag": self.use_crag,
                "router": self.use_router,
            },
        }