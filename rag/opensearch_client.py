"""
rag/opensearch_client.py
==========================
Production OpenSearch client with connection pooling, retry logic,
health checks, and all CRUD operations needed for RAG.
"""

import os
import time
import json
from typing import List, Dict, Optional, Any
from datetime import datetime

try:
    from opensearchpy import OpenSearch, helpers
    _OPENSEARCH_AVAILABLE = True
except ImportError:
    _OPENSEARCH_AVAILABLE = False
    print("[OpenSearch] opensearch-py not installed. Install: pip install opensearch-py")


class OpenSearchRAGClient:
    """Production-grade OpenSearch client for RAG applications."""

    DEFAULT_HOST = os.getenv("OPENSEARCH_HOST", "localhost")
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
        auto_connect: bool = True,
    ):
        self.host = host or self.DEFAULT_HOST
        self.port = port or self.DEFAULT_PORT
        self.username = username or self.DEFAULT_USER
        self.password = password or self.DEFAULT_PASS
        self.index_name = index_name or self.DEFAULT_INDEX
        self.embedding_dim = embedding_dim
        self.bm25_k1 = bm25_k1
        self.bm25_b = bm25_b
        self.auto_connect = auto_connect
        self._client: Optional[OpenSearch] = None

        if auto_connect:
            self._connect()

    def _connect(self, max_retries: int = 5):
        """Establish connection with exponential backoff retry."""
        if not _OPENSEARCH_AVAILABLE:
            raise ImportError("opensearch-py is required. Install: pip install opensearch-py")

        for attempt in range(max_retries):
            try:
                scheme = "http" if self.host in ("localhost", "127.0.0.1", "opensearch") else "https"

                self._client = OpenSearch(
                    hosts=[{"host": self.host, "port": self.port, "scheme": scheme}],
                    http_auth=(self.username, self.password),
                    verify_certs=False,
                    ssl_show_warn=False,
                    timeout=30,
                    retry_on_timeout=True,
                    max_retries=3,
                    connections_per_node=10,
                )

                health = self._client.cluster.health()
                print(
                    f"[OpenSearch] ✅ Connected to {self.host}:{self.port} ({scheme}) | "
                    f"Status: {health['status']} | Nodes: {health['number_of_nodes']}"
                )
                return

            except Exception as e:
                wait = min(2 ** attempt, 30)
                print(f"[OpenSearch] Connection attempt {attempt + 1}/{max_retries} failed: {e}. Retrying in {wait}s...")
                time.sleep(wait)

        raise ConnectionError(f"Failed to connect to OpenSearch at {self.host}:{self.port} after {max_retries} retries.")

    @property
    def client(self) -> OpenSearch:
        if self._client is None:
            self._connect()
        return self._client

    def health(self) -> Dict:
        return self.client.cluster.health()

    def stats(self) -> Dict:
        return self.client.cluster.stats()

    def index_stats(self, index_name: str = None) -> Dict:
        return self.client.indices.stats(index=index_name or self.index_name)

    def is_healthy(self) -> bool:
        try:
            return self.health()["status"] in ("green", "yellow")
        except Exception:
            return False

    def index_exists(self, index_name: str = None) -> bool:
        return self.client.indices.exists(index=index_name or self.index_name)

    def create_index(self, index_name: str = None, embedding_dim: int = None, num_shards: int = 1, num_replicas: int = 0) -> bool:
        idx = index_name or self.index_name
        dim = embedding_dim or self.embedding_dim

        if self.index_exists(idx):
            print(f"[OpenSearch] Index '{idx}' already exists")
            return False

        mapping = {
            "settings": {
                "index": {
                    "number_of_shards": num_shards, "number_of_replicas": num_replicas,
                    "knn": True, "knn.algo_param.ef_search": 100,
                    "similarity": {"default": {"type": "BM25", "k1": self.bm25_k1, "b": self.bm25_b}},
                    "refresh_interval": "1s",
                }
            },
            "mappings": {
                "dynamic": "strict",
                "properties": {
                    "text": {"type": "text", "analyzer": "standard", "fields": {"keyword": {"type": "keyword"}, "english": {"type": "text", "analyzer": "english"}}},
                    "text_embedding": {"type": "knn_vector", "dimension": dim, "method": {"name": "hnsw", "space_type": "cosinesimil", "engine": "lucene", "parameters": {"ef_construction": 128, "m": 16}}},
                    "chunk_id": {"type": "keyword"}, "doc_id": {"type": "keyword"}, "parent_id": {"type": "keyword"},
                    "type": {"type": "keyword"}, "page": {"type": "integer"}, "year": {"type": "keyword"},
                    "source": {"type": "keyword"}, "metric_type": {"type": "keyword"}, "metric_value": {"type": "float"},
                    "structured": {"type": "boolean"}, "context_prefix": {"type": "text"}, "indexed_at": {"type": "date"},
                }
            }
        }
        self.client.indices.create(index=idx, body=mapping)
        print(f"[OpenSearch] ✅ Created index '{idx}'")
        return True

    def delete_index(self, index_name: str = None):
        idx = index_name or self.index_name
        if self.index_exists(idx):
            self.client.indices.delete(index=idx)
            print(f"[OpenSearch] 🗑️ Deleted index '{idx}'")

    def refresh_index(self, index_name: str = None):
        self.client.indices.refresh(index=index_name or self.index_name)

    def bulk_index(self, documents: List[Dict], index_name: str = None, chunk_size: int = 100) -> Dict[str, Any]:
        idx = index_name or self.index_name
        actions = []
        for doc in documents:
            if "text_embedding" not in doc:
                raise ValueError(f"Document {doc.get('chunk_id')} missing 'text_embedding'")
            actions.append({"_index": idx, "_id": doc.get("chunk_id"), "_source": {**doc, "indexed_at": datetime.utcnow().isoformat()}})

        t0 = time.time()
        success, errors = helpers.bulk(self.client, actions, chunk_size=chunk_size, raise_on_error=False, raise_on_exception=False)
        elapsed_ms = int((time.time() - t0) * 1000)

        if errors:
            print(f"[OpenSearch] ⚠️ Bulk index: {success} succeeded, {len(errors)} errors")
        else:
            print(f"[OpenSearch] ✅ Bulk indexed {success} docs in {elapsed_ms}ms")
        return {"success": success, "errors": errors, "took_ms": elapsed_ms}

    def count_docs(self, index_name: str = None) -> int:
        return self.client.count(index=index_name or self.index_name)["count"]

    def search_bm25(self, query: str, k: int = 50, filters: Dict = None, index_name: str = None) -> List[Dict]:
        idx = index_name or self.index_name
        must_clauses = [{"multi_match": {"query": query, "fields": ["text^3", "context_prefix^2", "metric_type^4"], "type": "best_fields", "operator": "or"}}]
        if filters:
            for field, value in filters.items():
                must_clauses.append({"term": {field: value}})
        body = {"size": k, "query": {"bool": {"must": must_clauses}}, "_source": True}
        return self._parse_hits(self.client.search(index=idx, body=body))

    def search_knn(self, query_vector: List[float], k: int = 50, filters: Dict = None, index_name: str = None) -> List[Dict]:
        idx = index_name or self.index_name
        if isinstance(query_vector, str):
            query_vector = json.loads(query_vector)
        query_vector = [float(x) for x in query_vector]

        knn_clause = {"text_embedding": {"vector": query_vector, "k": k}}
        if filters:
            filter_clauses = [{"term": {f: v}} for f, v in filters.items()]
            body = {"size": k, "query": {"bool": {"must": [{"knn": knn_clause}], "filter": filter_clauses}}, "_source": True}
        else:
            body = {"size": k, "query": {"knn": knn_clause}, "_source": True}

        try:
            return self._parse_hits(self.client.search(index=idx, body=body))
        except Exception as e:
            print(f"[OpenSearch] kNN query failed: {e}")
            raise

    def search_hybrid(self, query: str, query_vector: List[float], k: int = 50, filters: Dict = None, index_name: str = None) -> List[Dict]:
        idx = index_name or self.index_name
        if isinstance(query_vector, str):
            query_vector = json.loads(query_vector)
        query_vector = [float(x) for x in query_vector]

        try:
            hybrid_body = {
                "size": k,
                "query": {"hybrid": {"queries": [{"match": {"text": query}}, {"knn": {"text_embedding": {"vector": query_vector, "k": k}}}]}},
                "_source": True,
            }
            if filters:
                filter_clauses = [{"term": {f: v}} for f, v in filters.items()]
                hybrid_body["query"] = {"bool": {"must": [hybrid_body["query"]], "filter": filter_clauses}}

            return self._parse_hits(self.client.search(index=idx, body=hybrid_body, params={"search_pipeline": f"{idx}_hybrid_pipeline"}))
        except Exception:
            print(f"[OpenSearch] Hybrid search failed. Falling back to parallel BM25 + kNN.")
            return self._fallback_hybrid_search(query, query_vector, k, filters, idx)

    def _fallback_hybrid_search(self, query: str, query_vector: List[float], k: int, filters: Dict, idx: str) -> List[Dict]:
        try:
            bm25_results = self.search_bm25(query, k=k, filters=filters, index_name=idx)
        except Exception:
            bm25_results = []
        try:
            knn_results = self.search_knn(query_vector, k=k, filters=filters, index_name=idx)
        except Exception:
            knn_results = []

        rrf_k = 60
        scores = {}
        for rank, doc in enumerate(bm25_results):
            doc_id = doc.get("chunk_id")
            if doc_id:
                scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (rank + 1 + rrf_k)
        for rank, doc in enumerate(knn_results):
            doc_id = doc.get("chunk_id")
            if doc_id:
                scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (rank + 1 + rrf_k)

        all_docs = {d.get("chunk_id"): d for d in bm25_results + knn_results if d.get("chunk_id")}
        merged = []
        for doc_id, rrf_score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
            if doc_id in all_docs:
                doc = all_docs[doc_id].copy()
                doc["score"] = rrf_score
                merged.append(doc)
        return merged[:k]

    def _parse_hits(self, response: Dict) -> List[Dict]:
        hits = []
        for hit in response.get("hits", {}).get("hits", []):
            source = hit.get("_source", {})
            hits.append({
                "chunk_id": source.get("chunk_id"), "text": source.get("text", ""),
                "score": hit.get("_score", 0.0), "doc_id": source.get("doc_id", ""),
                "parent_id": source.get("parent_id"), "type": source.get("type", "child"),
                "page": source.get("page", 0), "year": source.get("year", ""),
                "metric_type": source.get("metric_type"), "metric_value": source.get("metric_value"),
                "structured": source.get("structured", False), "context_prefix": source.get("context_prefix", ""),
            })
        return hits

    def create_hybrid_pipeline(self, pipeline_name: str = None, weights: List[float] = None) -> bool:
        name = pipeline_name or f"{self.index_name}_hybrid_pipeline"
        weights = weights or [0.3, 0.7]
        body = {"description": f"Hybrid search pipeline for {self.index_name}", "phase_results_processors": [{"normalization-processor": {"normalization": {"technique": "min_max"}, "combination": {"technique": "arithmetic_mean", "parameters": {"weights": weights}}}}]}
        try:
            self.client.search_pipeline.put(id=name, body=body)
            print(f"[OpenSearch] ✅ Created hybrid pipeline '{name}'")
            return True
        except Exception as e:
            print(f"[OpenSearch] ⚠️ Could not create pipeline: {e}")
            return False

    def export_index_artifacts(self, output_dir: str, index_name: str = None):
        os.makedirs(output_dir, exist_ok=True)
        idx = index_name or self.index_name
        mapping = self.client.indices.get_mapping(index=idx)
        settings = self.client.indices.get_settings(index=idx)
        artifact = {"index_name": idx, "mapping": mapping[idx]["mappings"], "settings": settings[idx]["settings"], "exported_at": datetime.utcnow().isoformat()}
        mapping_path = os.path.join(output_dir, "mapping.json")
        with open(mapping_path, "w") as f:
            json.dump(artifact, f, indent=2)
        data_path = os.path.join(output_dir, "data.jsonl")
        doc_count = 0
        with open(data_path, "w") as f:
            for doc in helpers.scan(self.client, index=idx, _source=True):
                f.write(json.dumps(doc["_source"]) + "\n")
                doc_count += 1
        return {"mapping": mapping_path, "data": data_path, "doc_count": doc_count}

    def import_index_artifacts(self, mapping_path: str, data_path: str, index_name: str = None) -> Dict:
        idx = index_name or self.index_name
        with open(mapping_path) as f:
            artifact = json.load(f)
        if self.index_exists(idx):
            self.delete_index(idx)
        self.client.indices.create(index=idx, body={"settings": artifact["settings"]["index"], "mappings": artifact["mapping"]})
        documents = []
        with open(data_path) as f:
            for line in f:
                documents.append(json.loads(line.strip()))
                if len(documents) >= 100:
                    self.bulk_index(documents, index_name=idx)
                    documents = []
            if documents:
                self.bulk_index(documents, index_name=idx)
        self.refresh_index(idx)
        return {"index": idx, "doc_count": self.count_docs(idx)}

    def create_snapshot_repo(self, repo_name: str, repo_type: str = "fs", settings: Dict = None):
        self.client.snapshot.create_repository(repository=repo_name, body={"type": repo_type, "settings": settings or {}})

    def create_snapshot(self, repo_name: str, snapshot_name: str, index_name: str = None):
        self.client.snapshot.create(repository=repo_name, snapshot=snapshot_name, body={"indices": index_name or self.index_name}, wait_for_completion=True)

    def restore_snapshot(self, repo_name: str, snapshot_name: str):
        self.client.snapshot.restore(repository=repo_name, snapshot=snapshot_name, wait_for_completion=True)