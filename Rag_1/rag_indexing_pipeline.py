"""
rag/indexing_pipeline.py
========================
Complete indexing pipeline for OpenSearch-based RAG.

Pipeline:
    PDFs → DocumentProcessor → Embed → Bulk Index → OpenSearch

ARTIFACT CREATION (Colab → Production):
Unlike FAISS which creates .faiss/.pkl files, OpenSearch artifacts are:
1. mapping.json — Index settings and mappings (version-controlled config)
2. data.jsonl — All documents as NDJSON (one doc per line)
3. pipeline.json — Search pipeline definition (hybrid fusion config)

These artifacts let you:
- Build index in Colab, export, import into production cluster
- Version control your index configuration in git
- Backup/restore without cluster access
- Reproduce index builds deterministically

SPEED:
- Processing 5 PDFs (~500 pages total): ~30-60s
- Embedding ~3000 chunks: ~45s (CPU, batch_size=32)
- Bulk indexing ~3000 docs: ~5-10s
- Total: ~2 minutes for 5 RBI annual reports
"""

import os
import json
import time
from typing import List, Dict, Tuple
from datetime import datetime

import numpy as np
from sentence_transformers import SentenceTransformer

from rag.opensearch_client import OpenSearchRAGClient
from rag.document_processor import DocumentProcessor, ProcessingConfig


class IndexingPipeline:
    """
    Production indexing pipeline for financial document RAG.

    Usage:
        pipeline = IndexingPipeline()
        stats = pipeline.index_pdfs(["rbi_2023-24.pdf", "rbi_2024-25.pdf"])
        print(f"Indexed {stats['total_docs']} documents")
    """

    def __init__(
        self,
        embedder_model: str = "BAAI/bge-base-en-v1.5",
        os_client: OpenSearchRAGClient = None,
        processor: DocumentProcessor = None,
    ):
        self.embedder_model = embedder_model
        self.os = os_client or OpenSearchRAGClient()
        self.processor = processor or DocumentProcessor()

        # Lazy-loaded embedder
        self._embedder = None

    @property
    def embedder(self) -> SentenceTransformer:
        """Lazy-load embedder."""
        if self._embedder is None:
            print(f"[Indexing] Loading embedder: {self.embedder_model}")
            self._embedder = SentenceTransformer(self.embedder_model)
            print(f"[Indexing] ✅ Embedder ready | Dim: {self._embedder.get_sentence_embedding_dimension()}")
        return self._embedder

    def index_pdfs(
        self,
        pdf_paths: List[str],
        index_name: str = None,
        recreate_index: bool = False,
    ) -> Dict:
        """
        Full pipeline: Process PDFs → Embed → Index in OpenSearch.

        Args:
            pdf_paths: List of PDF file paths
            index_name: Target OpenSearch index (default from env)
            recreate_index: If True, delete and recreate index

        Returns:
            Stats dict with counts, timings, errors
        """
        t0 = time.time()
        stats = {
            "started_at": datetime.utcnow().isoformat(),
            "pdf_count": len(pdf_paths),
            "errors": [],
        }

        # 1. Create/recreate index
        if recreate_index:
            self.os.delete_index(index_name)

        created = self.os.create_index(
            index_name=index_name,
            embedding_dim=self.embedder.get_sentence_embedding_dimension(),
        )

        if created:
            # Create hybrid search pipeline
            self.os.create_hybrid_pipeline(
                pipeline_name=f"{self.os.index_name}_hybrid_pipeline",
                weights=[0.3, 0.7],  # BM25: 30%, Dense: 70%
            )

        # 2. Process all PDFs
        t1 = time.time()
        children, parents, structured, metadata = self.processor.process_multiple(pdf_paths)
        stats["processing_time_sec"] = round(time.time() - t1, 2)

        # Combine for indexing: children + structured (parents are lookup-only)
        to_index = children + structured
        stats["children_count"] = len(children)
        stats["parents_count"] = len(parents)
        stats["structured_count"] = len(structured)
        stats["total_indexable"] = len(to_index)

        if not to_index:
            stats["errors"].append("No documents to index")
            return stats

        # 3. Embed
        t2 = time.time()
        print(f"\n[Indexing] Embedding {len(to_index)} documents...")
        texts = [doc["text"] for doc in to_index]
        embeddings = self.embedder.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True,
            batch_size=32,
            convert_to_numpy=True,
        )
        stats["embedding_time_sec"] = round(time.time() - t2, 2)

        # 4. Add embeddings to docs
        for doc, emb in zip(to_index, embeddings):
            doc["text_embedding"] = emb.tolist()

        # 5. Bulk index
        t3 = time.time()
        print(f"\n[Indexing] Bulk indexing into OpenSearch...")
        result = self.os.bulk_index(to_index, index_name=index_name, chunk_size=100)
        stats["indexing_time_sec"] = round(time.time() - t3, 2)
        stats["indexed_count"] = result["success"]
        stats["index_errors"] = len(result["errors"])

        # 6. Store parents for lookup (in-memory or separate index)
        # Parents are not indexed for search, but stored for retrieval-time lookup
        self._store_parents(parents)

        # 7. Final stats
        stats["total_time_sec"] = round(time.time() - t0, 2)
        stats["completed_at"] = datetime.utcnow().isoformat()

        print(f"\n{'='*60}")
        print("INDEXING COMPLETE")
        print(f"{'='*60}")
        print(f"PDFs processed: {stats['pdf_count']}")
        print(f"Children: {stats['children_count']}")
        print(f"Parents: {stats['parents_count']}")
        print(f"Structured: {stats['structured_count']}")
        print(f"Indexed: {stats['indexed_count']}")
        print(f"Processing: {stats['processing_time_sec']}s")
        print(f"Embedding: {stats['embedding_time_sec']}s")
        print(f"Indexing: {stats['indexing_time_sec']}s")
        print(f"Total: {stats['total_time_sec']}s")

        return stats

    def _store_parents(self, parents: List[Dict]):
        """
        Store parent chunks for retrieval-time lookup.

        Options:
        1. In-memory dict (current): Fast, but lost on restart
        2. Separate OpenSearch index: Persistent, searchable
        3. Redis: Fast, persistent, shared across instances

        For production, use option 2 or 3.
        """
        # TODO: Store in Redis or separate index for persistence
        self._parent_map = {p["chunk_id"]: p for p in parents}
        print(f"[Indexing] Stored {len(parents)} parents for lookup")

    def get_parent(self, parent_id: str) -> Dict:
        """Get parent chunk by ID."""
        return self._parent_map.get(parent_id)

    # =====================================================================
    # ARTIFACT EXPORT/IMPORT (Colab → Production)
    # =====================================================================

    def export_artifacts(self, output_dir: str, index_name: str = None):
        """
        Export index as artifacts for transfer between environments.

        This replaces the FAISS .faiss + .pkl workflow.

        Output:
            output_dir/
                mapping.json      — Index configuration
                data.jsonl        — All documents (one per line)
                pipeline.json     — Search pipeline
                stats.json        — Build statistics
        """
        os.makedirs(output_dir, exist_ok=True)

        # Export via OpenSearch client
        result = self.os.export_index_artifacts(output_dir, index_name)

        # Add stats
        stats_path = os.path.join(output_dir, "stats.json")
        with open(stats_path, "w") as f:
            json.dump({
                "embedder_model": self.embedder_model,
                "exported_at": datetime.utcnow().isoformat(),
                "total_docs": result.get("doc_count", 0),
            }, f, indent=2)

        print(f"\n[Artifacts] Exported to {output_dir}/")
        print(f"  mapping.json  — Index settings and mappings")
        print(f"  data.jsonl    — {result['doc_count']} documents")
        print(f"  pipeline.json — Hybrid search pipeline")
        print(f"  stats.json    — Build metadata")

        return result

    def import_artifacts(self, artifact_dir: str, index_name: str = None):
        """
        Import artifacts into OpenSearch.

        Usage:
            pipeline = IndexingPipeline()
            pipeline.import_artifacts("artifacts/", index_name="rbi_reports")
        """
        mapping_path = os.path.join(artifact_dir, "mapping.json")
        data_path = os.path.join(artifact_dir, "data.jsonl")

        if not os.path.exists(mapping_path):
            raise FileNotFoundError(f"Mapping not found: {mapping_path}")
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data not found: {data_path}")

        return self.os.import_index_artifacts(mapping_path, data_path, index_name)