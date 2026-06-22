"""
MCP Server — Exposes the RAG pipeline and calculator as universal tools
via JSON-RPC 2.0 (Model Context Protocol).

Any agent framework (LangGraph, CrewAI, Claude Agents) can connect
to this server and use the retrieval tools without code changes.
"""

from fastmcp import FastMCP
from agent.tools.rag_search import RagSearchTool
from agent.tools.calculator import calc_tool
from agent.tools.comparator import DocumentComparatorTool

mcp = FastMCP(
    name="Financial RAG Server",
    description="Search and analyze RBI financial reports using hybrid BM25+FAISS retrieval with BGE cross-encoder reranking."
)

rag = RagSearchTool()
comp = DocumentComparatorTool()


@mcp.tool()
async def search_financial_documents(query: str, top_k: int = 5, doc_filter: str = None) -> dict:
    """
    Search RBI financial reports using hybrid BM25+FAISS retrieval
    with BGE cross-encoder reranking.

    Args:
        query: Natural language search query about financial topics
        top_k: Number of results to return (default 5)
        doc_filter: Optional document ID to restrict search

    Returns:
        Retrieved passages with citations and confidence scores
    """
    result = rag.run(query=query, top_k=top_k, year_filter=doc_filter)
    passages = result.get("retrieved_passages", [])

    return {
        "passages": [p.get("text", "") for p in passages],
        "doc_ids": [p.get("doc_id", "") for p in passages],
        "chunk_ids": [p.get("chunk_id", "") for p in passages],
        "scores": [p.get("score", 0.0) for p in passages],
        "avg_confidence": result.get("confidence_score", 0.0),
    }


@mcp.tool()
async def calculate_financial_metric(expression: str) -> dict:
    """
    Perform safe financial calculations.

    Supports: basic arithmetic, growth_rate, yoy_change, ratio,
    percentage, cagr.

    Args:
        expression: Math expression or named function call
        Examples: "((6.5 - 4.0) / 4.0) * 100", "cagr(1000, 1500, 3)"

    Returns:
        Computed result with formula and formatted output
    """
    result = calc_tool.run(expression)
    return {
        "result": result.get("result"),
        "formula": result.get("expression", expression),
        "success": result.get("success", False),
        "error": result.get("error"),
    }


@mcp.tool()
async def compare_documents(doc_a: str, doc_b: str, metric: str = "financial metrics") -> dict:
    """
    Compare two financial documents on a specific metric.

    Args:
        doc_a: Content of first document
        doc_b: Content of second document
        metric: Dimension to compare on (e.g., "repo rate policy")

    Returns:
        Structured comparison with summary, differences, similarities, and table
    """
    result = comp.run(doc_a=doc_a, doc_b=doc_b, metric=metric)
    return {
        "summary": result.get("summary", ""),
        "differences": result.get("differences", []),
        "similarities": result.get("similarities", []),
        "structured_table": result.get("structured_table", []),
        "success": result.get("success", False),
    }