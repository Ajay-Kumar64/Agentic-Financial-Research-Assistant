"""
rag/opensearch_client.py
==========================
Production OpenSearch client with connection pooling, retry logic,
health checks, and all CRUD operations needed for RAG.

WHY OPENSEARCH (not Elasticsearch):
- Apache 2.0 license — fully free, no feature gating, SaaS-safe
- Native hybrid search (BM25 + kNN) — no Enterprise plan needed
- All security features free (RBAC, TLS, audit logging)
- k-NN plugin supports FAISS, NMSLIB, Lucene engines
- Industry standard — 400+ organizations contributing

WHY NOT ELASTICSEARCH:
- AGPL/ELv2/SSPL license — legal risk for SaaS products
- Native RRF requires Enterprise subscription ($$$)
- Advanced security, alerting gated behind Platinum+
- Same Lucene engine underneath — performance difference <20%

SPEED COMPARISON:
- FAISS FlatIP (current): ~1ms for 3K vectors (exact search)
- OpenSearch HNSW: ~2-5ms for 3K vectors (ANN, effectively exact at this scale)
- OpenSearch HNSW: ~10-20ms for 50K vectors
- OpenSearch HNSW: ~50ms for 500K vectors
- At your scale (5 docs ~500 chunks), difference is negligible (<5ms)
- At 500 docs (~50K chunks), OpenSearch is faster due to native C++ implementation
- At 5000+ docs, OpenSearch HNSW is the only viable option (FAISS requires manual sharding)

ARTIFACT CREATION:
Unlike FAISS which creates .faiss and .pkl files, OpenSearch stores everything
in the cluster. "Artifacts" are:
1. Index mappings/settings (exportable as JSON)
2. Bulk index data (exportable as NDJSON)
3. Snapshots (stored in S3/minio for backup/restore)

For Colab workflow, we export the index data as NDJSON + mapping JSON,
then import into your production OpenSearch cluster.
"""

import os
import time
import json
from typing import List, Dict, Optional, Any
from datetime import datetime

from opensearchpy import OpenSearch, helpers


class OpenSearchRAGClient:
    """
    Production-grade OpenSearch client for RAG applications.

    Features:
    - Connection pooling with automatic retry
    - Health checking and cluster monitoring
    - Index lifecycle management (create, delete, alias)
    - Bulk indexing with progress tracking
    - Hybrid search (BM25 + kNN) with search pipelines
    - Metadata filtering (year, doc_id, page, metric_type)
    - Snapshot/restore for backup
    """

    # Default connection config
    DEFAULT_HOST = os.getenv("OPENSEARCH_HOST", "opensearch")
    DEFAULT_PORT = int(os.getenv("OPENSEARCH_PORT", "9200"))
    DEFAULT_USER = os.getenv("OPENSEARCH_USER", "admin")
    DEFAULT_PASS = os.getenv("OPENSEARCH_PASSWORD", "admin")
    DEFAULT_INDEX = os.getenv("OPENSEARCH_INDEX", "rbi_reports")

    def __init__(
        self,
        host: str = None,
        port: int = None,
        username: str = None,
        password: str = None,
        index_name: str = None,
        embedding_dim: int = 768,
        bm25_k1: float = 1.2,
        bm25_b: float = 0.75,
    ):
        self.host = host or self.DEFAULT_HOST
        self.port = port or self.DEFAULT_PORT
        self.username = username or self.DEFAULT_USER
        self.password = password or self.DEFAULT_PASS
        self.index_name = index_name or self.DEFAULT_INDEX
        self.embedding_dim = embedding_dim
        self.bm25_k1 = bm25_k1
        self.bm25_b = bm25_b

        self._client: Optional[OpenSearch] = None
        self._connect()

    def _connect(self, max_retries: int = 5):
        """Establish connection with exponential backoff retry."""
        for attempt in range(max_retries):
            try:
                self._client = OpenSearch(
                    hosts=[{
                        "host": self.host,
                        "port": self.port,
                        "scheme": "https" if self.port == 9200 else "http"
                    }],
                    http_auth=(self.username, self.password),
                    verify_certs=False,  # TODO: Enable in production with proper certs
                    ssl_show_warn=False,
                    timeout=30,
                    retry_on_timeout=True,
                    max_retries=3,
                    # Connection pooling
                    connections_per_node=10,
                )

                # Health check
                health = self._client.cluster.health()
                print(
                    f"[OpenSearch] ✅ Connected to {self.host}:{self.port} | "
                    f"Status: {health['status']} | Nodes: {health['number_of_nodes']} | "
                    f"Shards: {health.get('active_primary_shards', 'N/A')}"
                )
                return

            except Exception as e:
                wait = min(2 ** attempt, 30)  # Cap at 30s
                print(f"[OpenSearch] Connection attempt {attempt + 1}/{max_retries} failed: {e}. Retrying in {wait}s...")
                time.sleep(wait)

        raise ConnectionError(
            f"Failed to connect to OpenSearch at {self.host}:{self.port} after {max_retries} retries. "
            f"Ensure OpenSearch is running and credentials are correct."
        )

    @property
    def client(self) -> OpenSearch:
        """Get underlying OpenSearch client, reconnect if needed."""
        if self._client is None:
            self._connect()
        return self._client

    # =====================================================================
    # HEALTH & MONITORING
    # =====================================================================

    def health(self) -> Dict:
        """Get cluster health status."""
        return self.client.cluster.health()

    def stats(self) -> Dict:
        """Get cluster statistics."""
        return self.client.cluster.stats()

    def index_stats(self, index_name: str = None) -> Dict:
        """Get index-level statistics (doc count, size, etc.)."""
        idx = index_name or self.index_name
        return self.client.indices.stats(index=idx)

    def is_healthy(self) -> bool:
        """Quick health check."""
        try:
            health = self.health()
            return health["status"] in ("green", "yellow")
        except Exception:
            return False

    # =====================================================================
    # INDEX MANAGEMENT
    # =====================================================================

    def index_exists(self, index_name: str = None) -> bool:
        """Check if index exists."""
        return self.client.indices.exists(index=index_name or self.index_name)

    def create_index(
        self,
        index_name: str = None,
        embedding_dim: int = None,
        num_shards: int = 1,
        num_replicas: int = 0,
    ) -> bool:
        """
        Create RAG index with:
        - BM25 text field with custom k1/b parameters
        - dense_vector field with HNSW k-NN
        - Metadata fields for filtering and structured data

        WHY HNSW (not Flat/IVF):
        - HNSW is the default ANN in OpenSearch, optimized for Lucene
        - At 5-500 docs, HNSW is effectively exact (all vectors cached)
        - At 5000+ docs, HNSW provides sub-linear search with <1% accuracy loss
        - No manual tuning needed — works out of the box

        WHY Lucene engine (not FAISS/NMSLIB):
        - Lucene is native to OpenSearch — no extra plugins or dependencies
        - FAISS engine requires additional C++ libraries, better for GPU
        - NMSLIB is deprecated in newer OpenSearch versions
        - For CPU-only deployments, Lucene HNSW is the production standard
        """
        idx = index_name or self.index_name
        dim = embedding_dim or self.embedding_dim

        if self.index_exists(idx):
            print(f"[OpenSearch] Index '{idx}' already exists")
            return False

        mapping = {
            "settings": {
                "index": {
                    "number_of_shards": num_shards,
                    "number_of_replicas": num_replicas,
                    "knn": True,  # Enable k-NN plugin
                    "knn.algo_param.ef_search": 100,  # Higher = more accurate, slower
                    "similarity": {
                        "default": {
                            "type": "BM25",
                            "k1": self.bm25_k1,
                            "b": self.bm25_b,
                        }
                    },
                    # Refresh interval: 1s for near-real-time search
                    # Increase to 30s for bulk indexing performance
                    "refresh_interval": "1s",
                }
            },
            "mappings": {
                "dynamic": "strict",  # Reject unknown fields
                "properties": {
                    # Main text content (BM25 searchable)
                    "text": {
                        "type": "text",
                        "analyzer": "standard",
                        "fields": {
                            "keyword": {"type": "keyword"},  # For exact match aggregations
                            "english": {  # English stemmer for better recall
                                "type": "text",
                                "analyzer": "english"
                            }
                        }
                    },
                    # Dense vector embedding (kNN searchable)
                    "text_embedding": {
                        "type": "knn_vector",
                        "dimension": dim,
                        "method": {
                            "name": "hnsw",
                            "space_type": "cosinesimil",  # Cosine similarity
                            "engine": "lucene",
                            "parameters": {
                                "ef_construction": 128,  # Higher = better index quality, slower build
                                "m": 16,  # Higher = better recall, more memory
                            }
                        }
                    },
                    # Chunk identifiers
                    "chunk_id": {"type": "keyword"},
                    "doc_id": {"type": "keyword"},
                    "parent_id": {"type": "keyword"},

                    # Chunk type for filtering
                    "type": {
                        "type": "keyword",
                        # child = small chunk (indexed for retrieval)
                        # parent = large chunk (returned to LLM for context)
                        # table = extracted table
                        # metric = extracted financial metric
                    },

                    # Document metadata
                    "page": {"type": "integer"},
                    "year": {"type": "keyword"},
                    "source": {"type": "keyword"},  # PDF filename

                    # Structured data fields
                    "metric_type": {"type": "keyword"},
                    "metric_value": {"type": "float"},
                    "structured": {"type": "boolean"},

                    # Contextual retrieval
                    "context_prefix": {"type": "text"},

                    # Timestamps
                    "indexed_at": {"type": "date"},
                }
            }
        }

        self.client.indices.create(index=idx, body=mapping)
        print(
            f"[OpenSearch] ✅ Created index '{idx}' | "
            f"Shards: {num_shards} | Replicas: {num_replicas} | "
            f"Embedding dim: {dim} | BM25 k1={self.bm25_k1} b={self.bm25_b}"
        )
        return True

    def delete_index(self, index_name: str = None):
        """Delete index (use with caution)."""
        idx = index_name or self.index_name
        if self.index_exists(idx):
            self.client.indices.delete(index=idx)
            print(f"[OpenSearch] 🗑️ Deleted index '{idx}'")

    def refresh_index(self, index_name: str = None):
        """Force refresh to make latest docs searchable."""
        self.client.indices.refresh(index=index_name or self.index_name)

    # =====================================================================
    # BULK INDEXING
    # =====================================================================

    def bulk_index(
        self,
        documents: List[Dict],
        index_name: str = None,
        chunk_size: int = 100,
    ) -> Dict[str, Any]:
        """
        Bulk index documents with embeddings.

        documents: List of dicts with keys:
            - text (str): chunk text
            - text_embedding (List[float]): embedding vector
            - chunk_id (str): unique chunk ID
            - doc_id (str): source document ID
            - type (str): child/parent/table/metric
            - [optional] page, year, metric_type, metric_value, etc.

        Returns: {"success": int, "errors": List, "took_ms": int}
        """
        idx = index_name or self.index_name

        actions = []
        for doc in documents:
            # Ensure required fields
            if "text_embedding" not in doc:
                raise ValueError(f"Document {doc.get('chunk_id')} missing 'text_embedding'")

            action = {
                "_index": idx,
                "_id": doc.get("chunk_id", doc.get("chunk_id")),
                "_source": {
                    **doc,
                    "indexed_at": datetime.utcnow().isoformat(),
                }
            }
            actions.append(action)

        t0 = time.time()
        success, errors = helpers.bulk(
            self.client,
            actions,
            chunk_size=chunk_size,
            raise_on_error=False,
            raise_on_exception=False,
        )
        elapsed_ms = int((time.time() - t0) * 1000)

        if errors:
            print(f"[OpenSearch] ⚠️ Bulk index: {success} succeeded, {len(errors)} errors")
            # Log first few errors
            for err in errors[:3]:
                print(f"  Error: {err}")
        else:
            print(f"[OpenSearch] ✅ Bulk indexed {success} docs in {elapsed_ms}ms")

        return {
            "success": success,
            "errors": errors,
            "took_ms": elapsed_ms,
        }

    def count_docs(self, index_name: str = None) -> int:
        """Get document count in index."""
        return self.client.count(index=index_name or self.index_name)["count"]

    # =====================================================================
    # SEARCH METHODS
    # =====================================================================

    def search_bm25(
        self,
        query: str,
        k: int = 50,
        filters: Dict = None,
        index_name: str = None,
    ) -> List[Dict]:
        """
        BM25 keyword search with optional metadata filters.

        filters: Dict of term filters, e.g. {"year": "2023-24", "type": "child"}
        """
        idx = index_name or self.index_name

        must_clauses = [
            {
                "multi_match": {
                    "query": query,
                    "fields": [
                        "text^3",           # Boost main text
                        "context_prefix^2",  # Boost context
                        "metric_type^4",     # Strong boost for metric types
                    ],
                    "type": "best_fields",
                    "operator": "or",
                }
            }
        ]

        # Add filters
        if filters:
            for field, value in filters.items():
                must_clauses.append({"term": {field: value}})

        body = {
            "size": k,
            "query": {"bool": {"must": must_clauses}},
            "_source": True,
        }

        response = self.client.search(index=idx, body=body)
        return self._parse_hits(response)

    def search_knn(
        self,
        query_vector: List[float],
        k: int = 50,
        filters: Dict = None,
        index_name: str = None,
    ) -> List[Dict]:
        """
        Dense vector kNN search with optional metadata filters.

        WHY kNN (not brute force):
        - kNN uses HNSW ANN index — sub-linear search time
        - At 500 chunks: ~2ms (effectively exact)
        - At 50K chunks: ~15ms
        - At 500K chunks: ~50ms
        - Brute force would be O(n) and unusable at scale
        """
        idx = index_name or self.index_name

        knn_query = {
            "field": "text_embedding",
            "query_vector": query_vector,
            "k": k,
            "num_candidates": max(k * 2, 100),  # Internal candidates before k filter
        }

        # Add filters as pre-filter (applied before kNN, more efficient)
        if filters:
            filter_clauses = [{"term": {f: v}} for f, v in filters.items()]
            knn_query["filter"] = {"bool": {"must": filter_clauses}}

        body = {
            "size": k,
            "query": {"knn": knn_query},
            "_source": True,
        }

        response = self.client.search(index=idx, body=body)
        return self._parse_hits(response)

    def search_hybrid(
        self,
        query: str,
        query_vector: List[float],
        k: int = 50,
        filters: Dict = None,
        index_name: str = None,
    ) -> List[Dict]:
        """
        Hybrid search: BM25 + kNN fused via OpenSearch search pipeline.

        This is the PRIMARY search method for production RAG.

        HOW IT WORKS:
        1. OpenSearch runs BM25 and kNN queries in parallel on the same index
        2. Results are normalized (min-max) to [0,1] scale
        3. Combined via arithmetic mean with weights [0.3, 0.7] (BM25: 30%, Dense: 70%)
        4. Final ranked list returned

        WHY THESE WEIGHTS:
        - Dense (70%): Better for semantic/paraphrase queries
        - BM25 (30%): Catches exact terms, identifiers, numbers
        - For financial docs where exact terms matter, you might use [0.4, 0.6]
        - Tune based on your evaluation metrics

        SPEED:
        - Parallel execution: max(BM25_time, kNN_time) + fusion_overhead
        - At 500 chunks: ~5-10ms total
        - At 50K chunks: ~20-30ms total
        - Fusion overhead: <1ms
        """
        idx = index_name or self.index_name

        # Build hybrid query
        hybrid_queries = [
            {"match": {"text": query}},
            {
                "knn": {
                    "text_embedding": {
                        "vector": query_vector,
                        "k": k,
                    }
                }
            }
        ]

        # Build filter query
        bool_query = {"hybrid": {"queries": hybrid_queries}}
        if filters:
            filter_clauses = [{"term": {f: v}} for f, v in filters.items()]
            bool_query = {
                "bool": {
                    "must": [bool_query],
                    "filter": filter_clauses,
                }
            }

        body = {
            "size": k,
            "query": bool_query,
            "_source": True,
        }

        # Use search pipeline for normalization + combination
        # OpenSearch 2.10+ supports this natively
        try:
            response = self.client.search(
                index=idx,
                body=body,
                params={"search_pipeline": f"{idx}_hybrid_pipeline"},
            )
        except Exception as e:
            # Fallback: if pipeline doesn't exist, search without it
            # (results will be un-fused, but still functional)
            print(f"[OpenSearch] Hybrid pipeline not available: {e}")
            response = self.client.search(index=idx, body=body)

        return self._parse_hits(response)

    def _parse_hits(self, response: Dict) -> List[Dict]:
        """Parse OpenSearch response into standardized format."""
        hits = []
        for hit in response.get("hits", {}).get("hits", []):
            source = hit.get("_source", {})
            hits.append({
                "chunk_id": source.get("chunk_id"),
                "text": source.get("text", ""),
                "score": hit.get("_score", 0.0),
                "doc_id": source.get("doc_id", ""),
                "parent_id": source.get("parent_id"),
                "type": source.get("type", "child"),
                "page": source.get("page", 0),
                "year": source.get("year", ""),
                "metric_type": source.get("metric_type"),
                "metric_value": source.get("metric_value"),
                "structured": source.get("structured", False),
                "context_prefix": source.get("context_prefix", ""),
            })
        return hits

    # =====================================================================
    # SEARCH PIPELINE (Hybrid Fusion)
    # =====================================================================

    def create_hybrid_pipeline(
        self,
        pipeline_name: str = None,
        weights: List[float] = None,
    ) -> bool:
        """
        Create search pipeline for hybrid result normalization and combination.

        weights: [bm25_weight, dense_weight] — default [0.3, 0.7]

        WHY min_max NORMALIZATION:
        - BM25 scores are unbounded positive numbers
        - Cosine similarity is bounded [-1, 1]
        - Min-max brings both to [0, 1] scale for fair combination

        WHY arithmetic_mean:
        - Simple, interpretable
        - Weights directly control contribution of each signal
        - Alternative: geometric_mean (punishes low scores more)
        """
        name = pipeline_name or f"{self.index_name}_hybrid_pipeline"
        weights = weights or [0.3, 0.7]

        body = {
            "description": f"Hybrid search pipeline for {self.index_name}",
            "phase_results_processors": [
                {
                    "normalization-processor": {
                        "normalization": {
                            "technique": "min_max"
                        },
                        "combination": {
                            "technique": "arithmetic_mean",
                            "parameters": {
                                "weights": weights
                            }
                        }
                    }
                }
            ]
        }

        try:
            self.client.search_pipeline.put(id=name, body=body)
            print(f"[OpenSearch] ✅ Created hybrid pipeline '{name}' with weights {weights}")
            return True
        except Exception as e:
            print(f"[OpenSearch] ⚠️ Could not create pipeline: {e}")
            return False

    # =====================================================================
    # ARTIFACT EXPORT/IMPORT (For Colab → Production workflow)
    # =====================================================================

    def export_index_artifacts(self, output_dir: str, index_name: str = None):
        """
        Export index as "artifacts" for transfer between environments.

        Unlike FAISS which creates .faiss/.pkl files, OpenSearch artifacts are:
        1. mapping.json — Index settings and mappings
        2. data.jsonl — All documents as NDJSON (one per line)
        3. pipeline.json — Search pipeline definition

        This lets you:
        - Build index in Colab, export, import into production cluster
        - Version control your index configuration
        - Backup/restore without cluster access
        """
        import os
        os.makedirs(output_dir, exist_ok=True)

        idx = index_name or self.index_name

        # 1. Export mapping
        mapping = self.client.indices.get_mapping(index=idx)
        settings = self.client.indices.get_settings(index=idx)

        artifact = {
            "index_name": idx,
            "mapping": mapping[idx]["mappings"],
            "settings": settings[idx]["settings"],
            "exported_at": datetime.utcnow().isoformat(),
        }

        mapping_path = os.path.join(output_dir, "mapping.json")
        with open(mapping_path, "w") as f:
            json.dump(artifact, f, indent=2)
        print(f"[Artifacts] Exported mapping → {mapping_path}")

        # 2. Export documents (scroll API for large datasets)
        data_path = os.path.join(output_dir, "data.jsonl")
        doc_count = 0

        with open(data_path, "w") as f:
            # Use scan() for memory-efficient scrolling
            for doc in helpers.scan(self.client, index=idx, _source=True):
                f.write(json.dumps(doc["_source"]) + "\n")
                doc_count += 1
                if doc_count % 1000 == 0:
                    print(f"[Artifacts] Exported {doc_count} docs...")

        print(f"[Artifacts] Exported {doc_count} docs → {data_path}")

        # 3. Export pipeline
        try:
            pipeline = self.client.search_pipeline.get(
                id=f"{idx}_hybrid_pipeline"
            )
            pipeline_path = os.path.join(output_dir, "pipeline.json")
            with open(pipeline_path, "w") as f:
                json.dump(pipeline, f, indent=2)
            print(f"[Artifacts] Exported pipeline → {pipeline_path}")
        except Exception as e:
            print(f"[Artifacts] No pipeline to export: {e}")

        return {
            "mapping": mapping_path,
            "data": data_path,
            "doc_count": doc_count,
        }

    def import_index_artifacts(
        self,
        mapping_path: str,
        data_path: str,
        index_name: str = None,
    ) -> Dict:
        """
        Import artifacts into OpenSearch.

        Usage:
            client.import_index_artifacts(
                "artifacts/mapping.json",
                "artifacts/data.jsonl",
                index_name="rbi_reports"
            )
        """
        idx = index_name or self.index_name

        # 1. Load mapping
        with open(mapping_path) as f:
            artifact = json.load(f)

        # 2. Create index with mapping
        if self.index_exists(idx):
            print(f"[Import] Index '{idx}' exists, deleting...")
            self.delete_index(idx)

        self.client.indices.create(
            index=idx,
            body={
                "settings": artifact["settings"]["index"],
                "mappings": artifact["mapping"],
            }
        )
        print(f"[Import] Created index '{idx}' from mapping")

        # 3. Bulk import documents
        documents = []
        with open(data_path) as f:
            for line in f:
                doc = json.loads(line.strip())
                documents.append(doc)

                if len(documents) >= 100:
                    self.bulk_index(documents, index_name=idx)
                    documents = []

            if documents:
                self.bulk_index(documents, index_name=idx)

        # 4. Refresh
        self.refresh_index(idx)

        count = self.count_docs(idx)
        print(f"[Import] ✅ Imported {count} docs into '{idx}'")

        return {"index": idx, "doc_count": count}

    # =====================================================================
    # SNAPSHOT / RESTORE (Production Backup)
    # =====================================================================

    def create_snapshot_repo(self, repo_name: str, repo_type: str = "fs",
                             settings: Dict = None):
        """
        Register snapshot repository.

        For S3: settings={"bucket": "my-bucket", "base_path": "opensearch-snapshots"}
        For filesystem: settings={"location": "/mnt/snapshots"}
        """
        body = {"type": repo_type, "settings": settings or {}}
        self.client.snapshot.create_repository(repository=repo_name, body=body)
        print(f"[Snapshot] Created repository '{repo_name}' ({repo_type})")

    def create_snapshot(self, repo_name: str, snapshot_name: str,
                        index_name: str = None):
        """Create snapshot of index."""
        idx = index_name or self.index_name
        body = {"indices": idx}
        self.client.snapshot.create(
            repository=repo_name,
            snapshot=snapshot_name,
            body=body,
            wait_for_completion=True,
        )
        print(f"[Snapshot] Created '{snapshot_name}' for '{idx}'")

    def restore_snapshot(self, repo_name: str, snapshot_name: str):
        """Restore index from snapshot."""
        self.client.snapshot.restore(
            repository=repo_name,
            snapshot=snapshot_name,
            wait_for_completion=True,
        )
        print(f"[Snapshot] Restored '{snapshot_name}'")