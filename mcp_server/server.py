"""
MCP Server — Exposes all 5 agent tools via JSON-RPC 2.0 (Model Context Protocol).

Tools exposed:
1. search_financial_documents — RAG search (BM25 + FAISS hybrid)
2. calculate_financial_metric — Safe AST-based calculator
3. compare_documents — Document comparison
4. yahoo_finance — Live stock data (quotes, history, returns, fundamentals)
5. portfolio_analyzer — Multi-asset Sharpe ratio, volatility, max drawdown
6. get_document_metadata — Document metadata lookup

Any agent framework (LangGraph, CrewAI, Claude Agents) can connect
to this server and use these tools without code changes.
"""

import os
from fastmcp import FastMCP
from agent.tools.rag_search import RagSearchTool
from agent.tools.calculator import calc_tool
from agent.tools.comparator import DocumentComparatorTool
from agent.tools.yahoo_finance import yahoo_finance_tool
from agent.tools.portfolio_analyzer import portfolio_analyzer_tool

# FIX: Removed 'description' kwarg — no longer supported in FastMCP
mcp = FastMCP("Financial Agent MCP Server")

rag = RagSearchTool()
comp = DocumentComparatorTool()

# In-memory doc registry for metadata lookups (extend as needed)
DOC_REGISTRY = {
    "2024-25.pdf": {"title": "RBI Annual Report 2024-25", "year": "2024-25", "pages": 300},
    "2023-24.pdf": {"title": "RBI Annual Report 2023-24", "year": "2023-24", "pages": 280},
    "2022-23.pdf": {"title": "RBI Annual Report 2022-23", "year": "2022-23", "pages": 260},
    "2021-22.pdf": {"title": "RBI Annual Report 2021-22", "year": "2021-22", "pages": 250},
    "2020-21.pdf": {"title": "RBI Annual Report 2020-21", "year": "2020-21", "pages": 240},
}

# Tool registry for introspection
TOOL_REGISTRY = {
    "search_financial_documents": "RAG hybrid search (BM25 + FAISS)",
    "calculate_financial_metric": "Safe financial calculator (AST-based)",
    "compare_documents": "Document comparison on metrics",
    "yahoo_finance": "Live stock quotes, history, returns, fundamentals",
    "portfolio_analyzer": "Multi-asset portfolio risk analysis (Sharpe, volatility, drawdown)",
    "get_document_metadata": "Retrieve metadata for a financial document",
}


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

    # FIX: Added 'count' and 'source' fields expected by tests
    return {
        "passages": [p.get("text", "") for p in passages],
        "doc_ids": [p.get("doc_id", "") for p in passages],
        "chunk_ids": [p.get("chunk_id", "") for p in passages],
        "scores": [p.get("score", 0.0) for p in passages],
        "avg_confidence": result.get("confidence_score", 0.0),
        "count": len(passages),
        "source": "rag_hybrid",
    }


@mcp.tool()
async def get_document_metadata(doc_id: str) -> dict:
    """
    Retrieve metadata for a financial document by its ID.

    Args:
        doc_id: Document identifier (e.g., "2024-25.pdf")

    Returns:
        Document metadata including title, year, pages, and found status
    """
    meta = DOC_REGISTRY.get(doc_id)
    if meta:
        return {"found": True, "doc_id": doc_id, **meta}
    return {"found": False, "doc_id": doc_id}


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


@mcp.tool()
async def yahoo_finance(ticker: str, operation: str = "quote", period: str = "1y") -> dict:
    """
    Fetch live stock data from Yahoo Finance for Indian (.NS, .BO) and US equities.

    Args:
        ticker: Stock ticker symbol (e.g., RELIANCE.NS, TCS.NS, AAPL, ^NSEI)
        operation: Type of data — quote, history, returns, fundamentals
        period: Time period for history/returns (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y)

    Returns:
        Stock data including price, returns, volatility, or fundamentals
    """
    result = yahoo_finance_tool.run(ticker=ticker, operation=operation, period=period)

    if result.success:
        return {
            "success": True,
            "ticker": ticker.upper(),
            "operation": operation,
            "data": result.result_data,
        }
    else:
        return {
            "success": False,
            "ticker": ticker.upper(),
            "operation": operation,
            "error": result.error_message,
        }


@mcp.tool()
async def portfolio_analyzer(tickers: str, weights: str = None, period: str = "1y") -> dict:
    """
    Analyze a multi-asset portfolio: Sharpe ratio, volatility, max drawdown.

    Args:
        tickers: Comma-separated ticker symbols (e.g., "RELIANCE.NS,INFY.NS,HDFCBANK.NS")
        weights: Optional comma-separated weights (e.g., "0.4,0.3,0.3"). Equal if omitted.
        period: Analysis period (1y default)

    Returns:
        Portfolio metrics: Sharpe ratio, annualized return, volatility, max drawdown,
        and per-asset contribution breakdown
    """
    result = portfolio_analyzer_tool.run(tickers=tickers, weights=weights, period=period)

    if result.success:
        return {
            "success": True,
            "tickers": tickers,
            "period": period,
            "portfolio": result.result_data.get("portfolio", {}),
            "per_asset": result.result_data.get("per_asset", []),
        }
    else:
        return {
            "success": False,
            "tickers": tickers,
            "error": result.error_message,
        }


@mcp.tool()
async def list_available_tools() -> dict:
    """
    List all available tools on this MCP server with descriptions.

    Returns:
        Dictionary of tool names and their descriptions
    """
    return {
        "tools": TOOL_REGISTRY,
        "count": len(TOOL_REGISTRY),
        "version": "2.0.0",
    }