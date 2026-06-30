#!/usr/bin/env python3
"""
scripts/export_artifacts.py
===========================
Export OpenSearch index as artifacts for transfer between environments.

Usage:
    python scripts/export_artifacts.py --index rbi_reports --output artifacts/

WHY EXPORT ARTIFACTS:
- Build index in Colab, export, import into production cluster
- Version control index configuration in git
- Backup without cluster access
- Reproduce index builds deterministically

OUTPUT:
    artifacts/
        mapping.json      — Index settings and mappings
        data.jsonl        — All documents (one per line, NDJSON)
        pipeline.json     — Search pipeline definition
        stats.json        — Build metadata

IMPORT:
    python scripts/import_artifacts.py --input artifacts/ --index rbi_reports
"""

import os
import sys
import argparse
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.opensearch_client import OpenSearchRAGClient


def main():
    parser = argparse.ArgumentParser(description="Export OpenSearch index as artifacts")
    parser.add_argument("--index", "-i", default="rbi_reports", help="Index name")
    parser.add_argument("--output", "-o", default="artifacts", help="Output directory")

    args = parser.parse_args()

    print(f"📦 Exporting index '{args.index}' to {args.output}/")

    client = OpenSearchRAGClient(index_name=args.index)
    result = client.export_index_artifacts(args.output, args.index)

    print(f"\n✅ Export complete:")
    print(f"  Documents: {result['doc_count']}")
    print(f"  Mapping: {result['mapping']}")
    print(f"  Data: {result['data']}")


if __name__ == "__main__":
    main()