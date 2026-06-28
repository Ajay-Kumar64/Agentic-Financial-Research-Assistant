# test_graph_patches.py — run with: python test_graph_patches.py
import sys
import re
from types import ModuleType

# ── Create mock modules BEFORE any import of agent.graph ──
class MockAgentState(dict):
    pass

# ── Mock langgraph with a proper StateGraph dummy ──
class MockStateGraph:
    def __init__(self, *a, **k): pass
    def add_node(self, *a, **k): return self
    def add_edge(self, *a, **k): return self
    def add_conditional_edges(self, *a, **k): return self
    def set_entry_point(self, *a, **k): return self
    def compile(self): return type('MockCompiled', (), {'invoke': lambda self, s: s})()

mock_langgraph = ModuleType('langgraph.graph')
mock_langgraph.StateGraph = MockStateGraph
mock_langgraph.END = None
sys.modules['langgraph'] = ModuleType('langgraph')
sys.modules['langgraph.graph'] = mock_langgraph

# Mock agent package
sys.modules['agent'] = ModuleType('agent')
sys.modules['agent.state'] = ModuleType('agent.state')
sys.modules['agent.state'].AgentState = MockAgentState

mock_llm = ModuleType('agent.llm_provider')
mock_llm.call_llm_sync = lambda *a, **k: ("", 0)
sys.modules['agent.llm_provider'] = mock_llm

# ── Mock ALL tool modules with dummy classes/objects ──
sys.modules['agent.tools'] = ModuleType('agent.tools')

# rag_search
mock_rag = ModuleType('agent.tools.rag_search')
class RagSearchTool:
    def run(self, **kwargs): return {}
mock_rag.RagSearchTool = RagSearchTool
sys.modules['agent.tools.rag_search'] = mock_rag

# calculator
mock_calc = ModuleType('agent.tools.calculator')
class MockCalcTool:
    def run(self, expr): return {"success": True, "expression": expr, "result": 0}
mock_calc.calc_tool = MockCalcTool()
sys.modules['agent.tools.calculator'] = mock_calc

# comparator
mock_comp = ModuleType('agent.tools.comparator')
class DocumentComparatorTool:
    def run(self, **kwargs): return type('R', (), {'result_data': {}})()
mock_comp.DocumentComparatorTool = DocumentComparatorTool
sys.modules['agent.tools.comparator'] = mock_comp

# web_search
mock_web = ModuleType('agent.tools.web_search')
class WebSearchTool:
    def run(self, **kwargs): return type('R', (), {'result_data': []})()
mock_web.WebSearchTool = WebSearchTool
sys.modules['agent.tools.web_search'] = mock_web

# memory
mock_mem = ModuleType('agent.tools.memory')
class MockMemoryTool:
    def resolve_query(self, q, h): return q
mock_mem.memory_tool = MockMemoryTool()
sys.modules['agent.tools.memory'] = mock_mem

# yahoo_finance
mock_yf = ModuleType('agent.tools.yahoo_finance')
class MockYFResult:
    success = True
    result_data = {}
    error_message = ""
class MockYFTool:
    def run(self, **kwargs): return MockYFResult()
mock_yf.yahoo_finance_tool = MockYFTool()
sys.modules['agent.tools.yahoo_finance'] = mock_yf

# portfolio_analyzer
mock_pa = ModuleType('agent.tools.portfolio_analyzer')
class MockPAResult:
    success = True
    result_data = {"portfolio": {"sharpe_ratio": 1.5}}
    error_message = ""
class MockPATool:
    def run(self, **kwargs): return MockPAResult()
mock_pa.portfolio_analyzer_tool = MockPATool()
sys.modules['agent.tools.portfolio_analyzer'] = mock_pa

# guardrails
mock_gr = ModuleType('agent.guardrails')
def check_guardrails(state): return ("continue", "")
mock_gr.check_guardrails = check_guardrails
sys.modules['agent.guardrails'] = mock_gr

# ── Now import graph.py ──
import importlib.util
spec = importlib.util.spec_from_file_location("agent.graph", r"agent\graph.py")
agent_graph = importlib.util.module_from_spec(spec)
sys.modules['agent.graph'] = agent_graph
spec.loader.exec_module(agent_graph)

# Extract the functions we need to test
_extract_ticker = agent_graph._extract_ticker
_extract_portfolio_params = agent_graph._extract_portfolio_params
_detect_yahoo_operation = agent_graph._detect_yahoo_operation
PLANNER_FALLBACK = agent_graph.PLANNER_FALLBACK


def test_planner_prompt_has_critical_instructions():
    assert "CRITICAL: tool_input MUST be ONLY the stock ticker" in PLANNER_FALLBACK
    assert "RELIANCE.NS" in PLANNER_FALLBACK
    assert "tickers:RELIANCE.NS,INFY.NS,HDFCBANK.NS|weights:0.4,0.3,0.3" in PLANNER_FALLBACK
    print("✅ PLANNER_FALLBACK contains correct yahoo_finance & portfolio_analyzer instructions")

def test_extract_ticker_from_planner():
    assert _extract_ticker({"tool_input": "TCS.NS"}) == "TCS.NS"
    assert _extract_ticker({"tool_input": "AAPL"}) == "AAPL"
    assert _extract_ticker({"tool_input": "  INFY.NS  "}) == "INFY.NS"
    assert _extract_ticker({"tool_input": ""}) == "RELIANCE.NS"
    assert _extract_ticker({"tool_input": "What is the price?"}) == "RELIANCE.NS"
    print("✅ _extract_ticker correctly reads planner tool_input")

def test_extract_portfolio_params_from_planner():
    t, w = _extract_portfolio_params({
        "tool_input": "tickers:RELIANCE.NS,INFY.NS,HDFCBANK.NS|weights:0.4,0.3,0.3"
    })
    assert t == "RELIANCE.NS,INFY.NS,HDFCBANK.NS"
    assert w == "0.4,0.3,0.3"

    t, w = _extract_portfolio_params({
        "tool_input": "tickers:TCS.NS,SBIN.NS"
    })
    assert t == "TCS.NS,SBIN.NS"
    assert w is None

    t, w = _extract_portfolio_params({"tool_input": "RELIANCE.NS, INFY.NS"})
    assert t == "RELIANCE.NS,INFY.NS"
    assert w is None

    t, w = _extract_portfolio_params({"tool_input": "AAPL"})
    assert t == "AAPL"
    assert w is None

    t, w = _extract_portfolio_params({"tool_input": ""})
    assert t == "RELIANCE.NS,INFY.NS,HDFCBANK.NS"
    assert w == "0.4,0.3,0.3"
    print("✅ _extract_portfolio_params parses all planner formats correctly")

def test_detect_yahoo_operation():
    assert _detect_yahoo_operation("What is the PE ratio of TCS?") == "fundamentals"
    assert _detect_yahoo_operation("How did AAPL perform last year?") == "returns"
    assert _detect_yahoo_operation("Show me the history of INFY") == "history"
    assert _detect_yahoo_operation("Price of RELIANCE") == "quote"
    print("✅ _detect_yahoo_operation maps keywords correctly")

if __name__ == "__main__":
    test_planner_prompt_has_critical_instructions()
    test_extract_ticker_from_planner()
    test_extract_portfolio_params_from_planner()
    test_detect_yahoo_operation()
    print("\n🎉 All local smoke tests passed! Ready for Docker.")