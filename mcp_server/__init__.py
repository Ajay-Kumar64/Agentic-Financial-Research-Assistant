# mcp_server/__init__.py
from mcp_server.server import mcp, search_financial_documents, calculate_financial_metric, get_document_metadata

__all__ = ["mcp", "search_financial_documents", "calculate_financial_metric", "get_document_metadata"]