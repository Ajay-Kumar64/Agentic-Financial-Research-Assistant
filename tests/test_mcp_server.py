# tests/test_mcp_server.py
import pytest
import asyncio
import sys
import os
import inspect

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp_server.server import search_financial_documents, calculate_financial_metric, get_document_metadata


def test_calculate_financial_metric():
    """Test the calculator tool via MCP interface."""
    result = asyncio.run(calculate_financial_metric("growth_rate(100, 150)"))
    assert result["success"] is True
    assert result["result"] == 50.0


def test_calculate_cagr():
    result = asyncio.run(calculate_financial_metric("cagr(1000, 1500, 3)"))
    assert result["success"] is True
    assert abs(result["result"] - 14.47) < 0.1


def test_calculate_invalid():
    result = asyncio.run(calculate_financial_metric("import os"))
    assert result["success"] is False
    assert "error" in result


def test_get_document_metadata_not_found():
    result = asyncio.run(get_document_metadata("nonexistent.pdf"))
    assert result["found"] is False


def test_search_financial_documents_structure():
    """Test that search returns correct schema even if no results."""
    result = asyncio.run(search_financial_documents("RBI repo rate", top_k=3))
    assert "passages" in result
    assert "doc_ids" in result
    assert "avg_confidence" in result
    assert "count" in result
    assert "source" in result
    assert result["source"] == "rag_hybrid"


def test_mcp_tools_registered():
    """Verify that tool functions are importable and callable."""
    assert inspect.iscoroutinefunction(search_financial_documents)
    assert inspect.iscoroutinefunction(calculate_financial_metric)
    assert inspect.iscoroutinefunction(get_document_metadata)