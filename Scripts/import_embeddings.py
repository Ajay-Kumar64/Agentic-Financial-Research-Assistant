#!/usr/bin/env python3
"""
scripts/import_embeddings.py
============================
Import pre-computed embeddings (from Colab) into local OpenSearch.

Usage:
    python scripts/import_embeddings.py --input embeddings/ --index rbi_reports --recreate
"""

import os
import sys
import json
import argparse
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.opensearch_client import OpenSearchRAGClient

# Fields that OpenSearch mapping allows. Anything else gets stripped.
ALLOWED_FIELDS = {
    "text", "text_embedding", "chunk_id", "doc_id", "parent_id",
    "type", "page", "year", "source", "metric_type", "metric_value",
    "structured", "context_prefix", "indexed_at",
}


def clean_document(doc: dict) -> dict:
    """Remove fields not in the OpenSearch mapping to avoid strict_dynamic_mapping_exception."""
    return {k: v for k, v in doc.items() if k in ALLOWED_FIELDS}


def main():
    parser = argparse.ArgumentParser(description="Import Colab embeddings into local OpenSearch")
    parser.add_argument("--input", "-i", required=True, help="Input directory")
    parser.add_argument("--index", "-idx", default="rbi_reports", help="OpenSearch index name")
    parser.add_argument("--recreate", "-r", action="store_true", help="Delete and recreate index")
    parser.add_argument("--batch-size", type=int, default=100, help="Bulk index batch size")
    args = parser.parse_args()

    docs_path = os.path.join(args.input, "documents.jsonl")
    mapping_path = os.path.join(args.input, "mapping.json")

    if not os.path.exists(docs_path):
        print(f"❌ documents.jsonl not found in {args.input}")
        sys.exit(1)
    if not os.path.exists(mapping_path):
        print(f"❌ mapping.json not found in {args.input}")
        sys.exit(1)

    print(f"📁 Importing from: {args.input}")
    print(f"🎯 Target index: {args.index}")

    # Load metadata
    meta_path = os.path.join(args.input, "metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        print(f"\n📋 Build info:")
        print(f"   Embedder: {metadata.get('embedder_model', 'unknown')}")
        print(f"   Dimension: {metadata.get('embedding_dim', 'unknown')}")
        print(f"   PDFs: {metadata.get('pdf_count', 'unknown')}")

    # Connect
    print("\n🔌 Connecting to OpenSearch...")
    client = OpenSearchRAGClient(index_name=args.index)
    health = client.health()
    print(f"   Status: {health['status']} | Nodes: {health['number_of_nodes']}")

    # Load mapping
    print("\n📋 Loading mapping...")
    with open(mapping_path, "r", encoding="utf-8") as f:
        mapping = json.load(f)

    # Create/recreate index
    if args.recreate and client.index_exists(args.index):
        print(f"\n🗑️  Deleting existing index '{args.index}'...")
        client.delete_index(args.index)

    if not client.index_exists(args.index):
        print(f"\n🏗️  Creating index '{args.index}'...")
        client.client.indices.create(index=args.index, body=mapping)
        print("   ✅ Index created")

        print("\n🔧 Creating hybrid search pipeline...")
        client.create_hybrid_pipeline(
            pipeline_name=f"{args.index}_hybrid_pipeline",
            weights=[0.3, 0.7]
        )
    else:
        print(f"   Index '{args.index}' already exists")

    # Load and clean documents
    print(f"\n📄 Loading documents from {docs_path}...")
    raw_documents = []
    with open(docs_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                raw_documents.append(json.loads(line))

    print(f"   Loaded {len(raw_documents)} documents")

    # Strip unknown fields
    documents = [clean_document(d) for d in raw_documents]
    stripped_count = sum(1 for raw, clean in zip(raw_documents, documents) if len(raw) != len(clean))
    if stripped_count:
        print(f"   🧹 Stripped extra fields from {stripped_count} documents (start_word, end_word, etc.)")

    # Verify embeddings
    missing = [d for d in documents if "text_embedding" not in d]
    if missing:
        print(f"   ⚠️  {len(missing)} documents missing embeddings!")

    # Bulk index
    print(f"\n🚀 Bulk indexing (batch_size={args.batch_size})...")
    total_success = 0
    total_errors = 0

    for i in range(0, len(documents), args.batch_size):
        batch = documents[i:i + args.batch_size]
        result = client.bulk_index(batch, index_name=args.index, chunk_size=args.batch_size)
        total_success += result["success"]
        total_errors += len(result["errors"])

        if result["errors"]:
            for err in result["errors"][:2]:
                print(f"   ⚠️  Error: {err}")

        if (i // args.batch_size + 1) % 10 == 0 or i + args.batch_size >= len(documents):
            print(f"   Progress: {total_success}/{len(documents)}")

    # Refresh and verify
    print("\n🔄 Refreshing index...")
    client.refresh_index(args.index)

    count = client.count_docs(args.index)

    print("\n" + "="*60)
    print("✅ IMPORT COMPLETE")
    print("="*60)
    print(f"   Documents indexed: {total_success}")
    print(f"   Errors: {total_errors}")
    print(f"   Total in index: {count}")
    print(f"\nTest with:")
    print(f"   curl http://localhost:9200/{args.index}/_count")


if __name__ == "__main__":
    main()