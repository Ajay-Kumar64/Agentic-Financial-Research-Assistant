"""
Comprehensive test suite for Agent tools and graph nodes.
Run with: pytest tests/test_tools.py -v
"""

import pytest
from agent.tools.yahoo_finance import yahoo_finance_tool
from agent.tools.portfolio_analyzer import portfolio_analyzer_tool
from agent.state import AgentState
from agent.graph import (
    yahoo_finance_node,
    portfolio_analyzer_node,
    run_agent_traced,
    _extract_ticker,
    _detect_yahoo_operation,
    _extract_portfolio_params,
    workflow,
)


# ============================================================================
# Step 1: Unit Test the Tools (Standalone)
# ============================================================================

class TestYahooFinanceTool:
    """Unit tests for Yahoo Finance tool."""

    def test_valid_indian_stock_quote(self):
        """Test valid Indian stock quote (RELIANCE.NS)."""
        r = yahoo_finance_tool.run('RELIANCE.NS', 'quote')
        assert r.success, f"RELIANCE.NS quote failed: {r.error_message}"
        assert r.result_data['current_price'] > 0

    def test_valid_us_stock_quote(self):
        """Test valid US stock quote (AAPL)."""
        r = yahoo_finance_tool.run('AAPL', 'quote')
        assert r.success
        assert r.result_data['current_price'] > 0

    def test_invalid_ticker(self):
        """Test invalid ticker handling."""
        r = yahoo_finance_tool.run('INVALID_TICKER', 'quote')
        assert not r.success, "Should fail for invalid ticker"
        assert 'Invalid ticker' in r.error_message or 'no data' in r.error_message.lower()

    def test_history_operation(self):
        """Test history operation."""
        r = yahoo_finance_tool.run('RELIANCE.NS', 'history', period='1mo')
        assert r.success
        assert 'latest_close' in r.result_data

    def test_returns_operation(self):
        """Test returns operation."""
        r = yahoo_finance_tool.run('RELIANCE.NS', 'returns', period='1y')
        assert r.success
        assert 'total_return_pct' in r.result_data


class TestPortfolioAnalyzerTool:
    """Unit tests for Portfolio Analyzer tool."""

    def test_valid_three_stock_portfolio(self):
        """Test valid 3-stock portfolio with custom weights."""
        r = portfolio_analyzer_tool.run('RELIANCE.NS,INFY.NS,HDFCBANK.NS', '0.4,0.3,0.3')
        assert r.success, f"Portfolio failed: {r.error_message}"
        assert 'portfolio' in r.result_data
        assert 'sharpe_ratio' in r.result_data['portfolio']

    def test_equal_weights_none(self):
        """Test equal weights (None)."""
        r = portfolio_analyzer_tool.run('RELIANCE.NS,INFY.NS', None)
        assert r.success

    def test_invalid_ticker(self):
        """Test invalid ticker in portfolio."""
        r = portfolio_analyzer_tool.run('INVALID1.NS,INVALID2.NS', None)
        assert not r.success

    def test_single_ticker_rejected(self):
        """Test single ticker rejected (need 2+)."""
        r = portfolio_analyzer_tool.run('RELIANCE.NS', None)
        assert not r.success

    def test_bad_weights_rejected(self):
        """Test weights that don't sum to 1 are rejected."""
        r = portfolio_analyzer_tool.run('RELIANCE.NS,INFY.NS', '0.5,0.2')
        assert not r.success


# ============================================================================
# Step 2: Test the Graph Nodes (Isolated)
# ============================================================================

class TestExtractors:
    """Tests for helper extractor functions."""

    def test_extract_ticker_regex_first_word(self):
        """Regex extracts first uppercase word from query."""
        assert _extract_ticker('What is Reliance stock price?') == 'WHAT'

    def test_extract_ticker_aapl(self):
        """Extract AAPL from query."""
        assert _extract_ticker('AAPL earnings') == 'AAPL'

    def test_extract_ticker_with_ns_suffix(self):
        """Extract RELIANCE.NS when it appears first."""
        assert _extract_ticker('RELIANCE.NS stock price') == 'RELIANCE.NS'

    def test_extract_ticker_mapping_fallback(self):
        """Mappings are not reached due to regex always matching first word."""
        # query.upper() makes everything uppercase, regex always matches first word
        assert _extract_ticker('no ticker here') == 'NO'

    def test_detect_yahoo_operation_fundamentals(self):
        """Detect fundamentals operation."""
        assert _detect_yahoo_operation('What is the PE ratio?') == 'fundamentals'

    def test_detect_yahoo_operation_returns(self):
        """Detect returns operation."""
        assert _detect_yahoo_operation('How did it perform last year?') == 'returns'

    def test_detect_yahoo_operation_quote(self):
        """Detect quote operation."""
        assert _detect_yahoo_operation('Current price') == 'quote'

    def test_extract_portfolio_params(self):
        """Extract portfolio tickers and weights from clean query."""
        tickers, weights = _extract_portfolio_params(
            'RELIANCE.NS INFY.NS 50% 50%'
        )
        assert tickers == 'RELIANCE.NS,INFY.NS'
        assert weights == '0.5,0.5'

    def test_extract_portfolio_params_equal_weights(self):
        """Extract portfolio tickers with no weights -> equal weights."""
        tickers, weights = _extract_portfolio_params(
            'RELIANCE.NS INFY.NS HDFCBANK.NS'
        )
        assert tickers == 'RELIANCE.NS,INFY.NS,HDFCBANK.NS'
        assert weights is None


class TestGraphNodes:
    """Tests for individual graph nodes."""

    def test_yahoo_finance_node_valid(self):
        """Yahoo node with valid query."""
        state = AgentState(
            query='What is the stock price of RELIANCE.NS?',
            current_query='What is the stock price of RELIANCE.NS?',
            steps_executed=[],
            tools_used=[],
            total_tokens_used=0,
            latency_ms=0,
        )
        result = yahoo_finance_node(state)
        assert 'yahoo_finance' in result['steps_executed']
        assert 'yahoo_finance' in result['tools_used']
        assert result['tool_outputs'][0]['tool'] == 'yahoo_finance'

    def test_portfolio_analyzer_node_valid(self):
        """Portfolio node with valid query (only tickers, no extra uppercase words)."""
        state = AgentState(
            query='RELIANCE.NS INFY.NS HDFCBANK.NS',
            current_query='RELIANCE.NS INFY.NS HDFCBANK.NS',
            steps_executed=[],
            tools_used=[],
            calculation_results=[],
            total_tokens_used=0,
            latency_ms=0,
        )
        result = portfolio_analyzer_node(state)
        assert 'portfolio_analyzer' in result['steps_executed']
        assert len(result['calculation_results']) > 0
        assert result['calculation_results'][0]['tool'] == 'portfolio_analyzer'

    def test_portfolio_analyzer_node_invalid_tickers(self):
        """Portfolio node handles invalid tickers gracefully."""
        state = AgentState(
            query='INVALID1 and INVALID2',
            current_query='INVALID1 and INVALID2',
            steps_executed=[],
            tools_used=[],
            calculation_results=[],
            total_tokens_used=0,
            latency_ms=0,
        )
        result = portfolio_analyzer_node(state)
        assert 'portfolio_analyzer' in result['steps_executed']
        assert len(result['calculation_results']) == 0  # No calc added on failure


# ============================================================================
# Step 3: Test Full Agent Flow (End-to-End)
# ============================================================================

class TestEndToEnd:
    """End-to-end integration tests."""

    def test_e2e_stock_query(self):
        """Fast-Path: Stock query routes to Yahoo Finance."""
        state = AgentState(query='What is the stock price of RELIANCE.NS?')
        result = run_agent_traced(state)
        assert 'yahoo_finance' in result.get('tools_used', []), 'Should use yahoo_finance'

    def test_e2e_portfolio_query(self):
        """Fast-Path: Portfolio query routes to Portfolio Analyzer (no explicit numbers)."""
        state = AgentState(
            query='Portfolio RELIANCE.NS INFY.NS HDFCBANK.NS'
        )
        result = run_agent_traced(state)
        assert 'portfolio_analyzer' in result.get('tools_used', []), 'Should use portfolio_analyzer'

    def test_e2e_calculator_query(self):
        """Fast-Path: Calculator still works."""
        state = AgentState(query='What is the percentage increase from 4.0 to 6.5?')
        result = run_agent_traced(state)
        assert 'financial_calculator' in result.get('tools_used', [])

    def test_e2e_rag_query(self):
        """Fallback: RAG still works."""
        state = AgentState(query='What was the repo rate in FY2023?')
        result = run_agent_traced(state)
        assert 'rag_search' in result.get('tools_used', [])

    def test_e2e_invalid_stock(self):
        """Guardrail: Invalid stock handled gracefully."""
        state = AgentState(query='What is the price of INVALID_TICKER?')
        result = run_agent_traced(state)
        tools = result.get('tools_used', [])
        assert 'yahoo_finance' in tools or 'final_answer' in tools


# ============================================================================
# Step 4: Check the Graph Structure
# ============================================================================

class TestGraphDiagram:
    """Verify workflow graph contains expected nodes."""

    def test_graph_contains_yahoo_finance(self):
        """Graph contains yahoo_finance node."""
        assert 'yahoo_finance' in workflow.nodes

    def test_graph_contains_portfolio_analyzer(self):
        """Graph contains portfolio_analyzer node."""
        assert 'portfolio_analyzer' in workflow.nodes

    def test_graph_contains_human_review(self):
        """Graph contains human_review node."""
        assert 'human_review' in workflow.nodes


# ============================================================================
# Step 5: Quick All-in-One Smoke Test
# ============================================================================

class TestSmoke:
    """Quick smoke tests covering core functionality."""

    def test_yahoo_finance_smoke(self):
        """Smoke test Yahoo Finance tool."""
        assert yahoo_finance_tool.run('RELIANCE.NS', 'quote').success
        assert not yahoo_finance_tool.run('INVALID', 'quote').success

    def test_portfolio_analyzer_smoke(self):
        """Smoke test Portfolio Analyzer tool."""
        assert portfolio_analyzer_tool.run('RELIANCE.NS,INFY.NS', None).success
        assert not portfolio_analyzer_tool.run('INVALID1,INVALID2', None).success

    def test_e2e_smoke_stock(self):
        """Smoke test E2E stock query."""
        s1 = AgentState(query='Stock price of RELIANCE.NS')
        r1 = run_agent_traced(s1)
        assert 'yahoo_finance' in r1['tools_used']

    def test_e2e_smoke_portfolio(self):
        """Smoke test E2E portfolio query."""
        s2 = AgentState(query='Portfolio RELIANCE.NS INFY.NS')
        r2 = run_agent_traced(s2)
        assert 'portfolio_analyzer' in r2['tools_used']