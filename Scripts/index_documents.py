#!/usr/bin/env python3
"""
scripts/index_documents.py
===========================
Standalone script to index RBI PDFs into OpenSearch.

Usage:
    python scripts/index_documents.py --pdfs data/*.pdf --index rbi_reports --recreate

WHY STANDALONE SCRIPT:
- Run once during deployment or CI/CD
- Can be run in Colab for indexing, then export artifacts
- Separate from API runtime — indexing is a batch job
- Easy to schedule as cron job for periodic reindexing

WORKFLOW:
1. Place RBI PDFs in data/ directory
2. Run this script: python scripts/index_documents.py --pdfs data/*.pdf
3. Verify: curl http://localhost:9200/rbi_reports/_count
4. (Optional) Export artifacts for backup: python scripts/export_artifacts.py
"""

import os
import sys
import argparse
import glob
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.opensearch_client import OpenSearchRAGClient
from rag.document_processor import DocumentProcessor, ProcessingConfig
from rag.indexing_pipeline import IndexingPipeline


def main():
    parser = argparse.ArgumentParser(description="Index RBI PDFs into OpenSearch")
    parser.add_argument(
        "--pdfs", "-p",
        nargs="+",
        required=True,
        help="PDF file paths (supports wildcards, e.g., data/*.pdf)"
    )
    parser.add_argument(
        "--index", "-i",
        default="rbi_reports",
        help="OpenSearch index name (default: rbi_reports)"
    )
    parser.add_argument(
        "--recreate", "-r",
        action="store_true",
        help="Delete and recreate index before indexing"
    )
    parser.add_argument(
        "--embedder",
        default="BAAI/bge-base-en-v1.5",
        help="Embedding model name"
    )
    parser.add_argument(
        "--parent-size",
        type=int,
        default=1024,
        help="Parent chunk size in words"
    )
    parser.add_argument(
        "--child-size",
        type=int,
        default=256,
        help="Child chunk size in words"
    )
    parser.add_argument(
        "--no-contextualize",
        action="store_true",
        help="Skip contextual retrieval"
    )
    parser.add_argument(
        "--no-structured",
        action="store_true",
        help="Skip structured extraction (tables, metrics)"
    )

    args = parser.parse_args()

    # Expand wildcards
    pdf_paths = []
    for pattern in args.pdfs:
        matches = glob.glob(pattern)
        if matches:
            pdf_paths.extend(matches)
        elif os.path.exists(pattern):
            pdf_paths.append(pattern)

    if not pdf_paths:
        print("❌ No PDF files found!")
        sys.exit(1)

    print(f"📄 Found {len(pdf_paths)} PDF(s):")
    for p in pdf_paths:
        print(f"  - {p}")

    # Initialize components
    print("\n🔧 Initializing...")
    os_client = OpenSearchRAGClient(index_name=args.index)

    processor_config = ProcessingConfig(
        parent_size=args.parent_size,
        child_size=args.child_size,
        contextualize=not args.no_contextualize,
        extract_tables=not args.no_structured,
        extract_metrics=not args.no_structured,
    )
    processor = DocumentProcessor(config=processor_config)

    pipeline = IndexingPipeline(
        embedder_model=args.embedder,
        os_client=os_client,
        processor=processor,
    )

    # Index
    print(f"\n🚀 Starting indexing into '{args.index}'...")
    stats = pipeline.index_pdfs(
        pdf_paths=pdf_paths,
        index_name=args.index,
        recreate_index=args.recreate,
    )

    # Summary
    print("\n" + "="*60)
    print("INDEXING SUMMARY")
    print("="*60)
    print(f"PDFs processed: {stats['pdf_count']}")
    print(f"Children indexed: {stats['children_count']}")
    print(f"Parents stored: {stats['parents_count']}")
    print(f"Structured chunks: {stats['structured_count']}")
    print(f"Total indexed: {stats['indexed_count']}")
    print(f"Errors: {stats.get('index_errors', 0)}")
    print(f"Total time: {stats['total_time_sec']}s")
    print("="*60)

    # Verify
    count = os_client.count_docs(args.index)
    print(f"\n✅ Index '{args.index}' now has {count} documents")


if __name__ == "__main__":
    main()