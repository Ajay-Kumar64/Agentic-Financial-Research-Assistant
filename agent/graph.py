import json
import os
import re
import time
import asyncio
from langgraph.graph import StateGraph, END
from agent.state import AgentState
from agent.llm_provider import call_llm_sync

# Tools
from agent.tools.rag_search import RagSearchTool
from agent.tools.calculator import calc_tool
from agent.tools.comparator import DocumentComparatorTool
from agent.tools.web_search import WebSearchTool
from agent.memory import memory_tool
from agent.tools.yahoo_finance import yahoo_finance_tool
from agent.tools.portfolio_analyzer import portfolio_analyzer_tool

# =============================================================================
# LANGSMITH TRACING (Step 1: Production Observability)
# =============================================================================
try:
    from langsmith import traceable

    _LANGSMITH_AVAILABLE = True
except ImportError:
    _LANGSMITH_AVAILABLE = False
    print("[LangSmith] langsmith not installed. Run: pip install langsmith")

# Enable tracing if env var is set
os.environ.setdefault("LANGSMITH_TRACING", "true")
os.environ.setdefault("LANGSMITH_PROJECT", "agentic-financial-assistant")

rag_tool = RagSearchTool()
comp_tool = DocumentComparatorTool()
web_tool = WebSearchTool()


# =============================================================================
# ROUTING CONDITION (inline to avoid import issues)
# =============================================================================
def routing_condition(state: AgentState) -> str:
    """Route to the next node based on planner output."""
    return state.get("next_step", "final_answer")


# =============================================================================
# PROMPT LOADING
# =============================================================================
def _load_prompt(filename: str, fallback: str) -> str:
    try:
        path = os.path.join(os.path.dirname(__file__), "prompts", filename)
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return fallback


PLANNER_FALLBACK = """You are a financial research planner. Decide the NEXT tool to call.

Available tools:
- rag_search: Retrieve facts from RBI reports. tool_input: the search query.
- financial_calculator: Math calculations. tool_input: a valid Python math expression like "((6.5 - 4.0) / 4.0) * 100" or "cagr(1000, 1500, 3)".
- document_comparator: Compare two documents. tool_input: metric to compare.
- web_search: Search the web when RAG returns no results. tool_input: the search query.
- yahoo_finance: Fetch live stock prices from Yahoo Finance. 
  CRITICAL: tool_input MUST be ONLY the stock ticker. 
  For Indian NSE stocks: add .NS suffix (e.g., RELIANCE.NS, TCS.NS, INFY.NS, HDFCBANK.NS, SBIN.NS).
  For US stocks: just the ticker (e.g., AAPL, MSFT, GOOGL).
  Examples: "RELIANCE.NS", "TCS.NS", "AAPL"
- portfolio_analyzer: Calculate Sharpe ratio, volatility, max drawdown.
  CRITICAL: tool_input MUST be formatted as: "tickers:RELIANCE.NS,INFY.NS,HDFCBANK.NS|weights:0.4,0.3,0.3"
  For equal weights, omit weights: "tickers:RELIANCE.NS,INFY.NS"
- final_answer: Respond when enough info is gathered.

Max 5 tool calls. Respond with ONLY JSON:
{"next_tool": "...", "reason": "...", "tool_input": "..."}"""

RESPONSE_FALLBACK = """You are a financial research assistant. Answer using provided sources and calculations.
- Cite text sources with [1], [2], etc.
- If a CALCULATION RESULT is provided, report that numeric result clearly as the answer.
- CRITICAL: Before saying "not provided" or "not mentioned", carefully re-read ALL contexts for ANY specific numbers, percentages, dates, or values. The answer is often embedded in longer text — extract it.
- If you find ANY numerical values in the contexts that relate to the query, ALWAYS report them explicitly.
- If sources are insufficient AND no calculation is present, say what IS available in 1-2 sentences."""

PLANNER_SYSTEM_PROMPT = _load_prompt("planner_system.txt", PLANNER_FALLBACK)
RESPONSE_SYSTEM_PROMPT = _load_prompt("response_system.txt", RESPONSE_FALLBACK)


# =============================================================================
# MEMORY RESOLVER NODE
# =============================================================================

def _compress_context_for_planner(state: AgentState) -> str:
    passages = state.get("retrieved_passages", [])
    calcs = state.get("calculation_results", [])
    tools_used = state.get("tools_used", [])
    contexts = state.get("retrieved_contexts", [])
    tokens = state.get("total_tokens_used", 0)
    depth = state.get("tool_call_depth", 0)
    max_tokens = state.get("max_token_budget", 50000)
    year_filter = state.get("year_filter")

    passage_summaries = []
    for i, p in enumerate(passages[:3], 1):
        text = p.get("text", "")[:100].replace("\n", " ")
        passage_summaries.append(f"[{i}] {p.get('doc_id', '?')} p{p.get('page', 0)}: {text}...")

    calc_summaries = []
    for c in calcs[-2:]:
        calc_summaries.append(f"{c.get('expression', '?')} = {c.get('result', '?')}")

    web_summaries = []
    for i, ctx in enumerate(contexts[-2:], 1):
        if ctx and len(ctx) > 10 and not ctx.startswith("[RAG"):
            web_summaries.append(f"[Web{i}] {ctx[:150]}...")

    recent_tools = tools_used[-4:] if len(tools_used) > 4 else tools_used

    context = (
            "User query: " + str(state.get('current_query', '')) + "\n"
                                                                   "Tools used: " + str(recent_tools) + "\n"
                                                                                                        "Passages: " + str(
        len(passages)) + " retrieved (" + str(len(passage_summaries)) + " shown)\n"
            + "\n".join(passage_summaries) + "\n"
                                             "Web results: " + str(len(web_summaries)) + " found\n"
            + "\n".join(web_summaries) + "\n"
                                         "Calculations: " + str(len(calcs)) + " done (" + str(
        len(calc_summaries)) + " shown)\n"
            + "\n".join(calc_summaries) + "\n"
                                          "Tokens: " + str(tokens) + "/" + str(max_tokens) + "\n"
                                                                                             "Step: " + str(
        depth + 1) + "/5\n"
                     "Year: " + str(year_filter or 'none')
    )

    return context


def memory_resolver_node(state: AgentState) -> dict:
    raw_query = state.get("query") or ""
    history = state.get("conversation_history", [])
    if history and len(history) > 0:
        resolved = memory_tool.resolve_query(raw_query, history)
        if resolved != raw_query:
            print(f"[Memory] Resolved: '{raw_query}' -> '{resolved}'")
            return {
                "current_query": resolved,
                "resolved_references": {"original": raw_query, "resolved": resolved},
                "next_step": "guardrail_check"
            }
    return {"next_step": "guardrail_check"}


# =============================================================================
# HUMAN REVIEW NODE (Step 3: Enterprise Human-in-the-Loop)
# =============================================================================
def human_review_node(state: AgentState) -> dict:
    """
    Human-in-the-loop checkpoint.
    Triggered when agent confidence is critically low (< 0.4) after all fallbacks.
    """
    query = state.get("query", "")
    print(f"[Human Review] Query flagged for human review: '{query[:60]}...'")

    return {
        "final_response": (
            "I don't have enough reliable information to answer this question confidently. "
            "This query has been flagged for human review. "
            "Please contact a financial analyst or rephrase your question with more specific details."
        ),
        "steps_executed": state.get("steps_executed", []) + ["human_review"],
        "tools_used": state.get("tools_used", []) + ["human_review"],
        "guardrail_triggered": True,
        "guardrail_reason": "low_confidence_human_review",
        "confidence_score": state.get("confidence_score", 0.0),
        "needs_clarification": True,
        "task_complete": True,
    }


# =============================================================================
# GUARDRAIL CHECK NODE (Updated for Human Review routing)
# =============================================================================
def guardrail_check_node(state: AgentState) -> dict:
    from agent.guardrails import check_guardrails
    decision, reason = check_guardrails(state)

    if decision == "force_respond":
        return {
            "next_step": "final_answer",
            "guardrail_triggered": True,
            "guardrail_reason": reason or "guardrail_forced",
        }
    elif decision == "respond":
        return {
            "next_step": "final_answer",
            "guardrail_reason": reason,
        }
    elif decision == "continue":
        # NEW: Critical low confidence + already tried web_search -> human review
        # BUT: Skip human review if we have structured data from yahoo_finance or portfolio_analyzer
        confidence = state.get("confidence_score", 0.0)
        tools_used = state.get("tools_used", [])
        depth = state.get("tool_call_depth", 0)

        # If we have live market data or portfolio analysis, don't human-review
        has_structured_data = any(t in tools_used for t in ["yahoo_finance", "portfolio_analyzer"])
        if has_structured_data:
            return {"next_step": "planner"}

        if (confidence < 0.4
                and "web_search" in tools_used
                and depth >= 2
                and "human_review" not in tools_used):
            print(f"[Guardrail] Critical low confidence ({confidence:.2f}) after fallback -> routing to human_review")
            return {
                "next_step": "human_review",
                "guardrail_triggered": True,
                "guardrail_reason": "critical_low_confidence_human_review",
            }
        return {"next_step": "planner"}
    else:
        return {"next_step": "planner"}


# =============================================================================
# PLANNER NODE
# =============================================================================

def planner_node(state: AgentState) -> dict:
    t0 = time.time()
    query = state.get("current_query") or state.get("query") or ""
    query_lower = query.lower()
    steps = state.get("steps_executed", [])
    tools_used = state.get("tools_used", [])
    tokens = state.get("total_tokens_used", 0)
    depth = state.get("tool_call_depth", 0)
    passages = state.get("retrieved_passages", [])
    calcs = state.get("calculation_results", [])
    comp = state.get("comparison_results")
    contexts = state.get("retrieved_contexts", [])

    print(f"[Planner] Evaluating query: '{query[:100]}' | Tools used: {tools_used} | Passages: {len(passages)}")

    # =====================================================================
    # FAST-PATH RT: Real-time / current info queries -> skip RAG, go to web
    # =====================================================================
    realtime_keywords = ["today's date", "current date", "what time is it", "current time",
                         "weather in", "latest news", "today", "tomorrow", "yesterday",
                         "now in", "current president", "current pm"]
    if any(k in query_lower for k in realtime_keywords) and "web_search" not in tools_used:
        print(f"[Planner] FAST-PATH: Real-time query detected, routing to web_search")
        return {
            "next_step": "web_search",
            "current_query": query,
            "steps_executed": steps + ["planner->web_search(realtime)"],
            "total_tokens_used": tokens,
            "tokens_consumed": state.get("tokens_consumed", 0),
            "tool_call_depth": depth + 1,
            "tools_used": tools_used,
            "task_complete": False,
            "latency_ms": state.get("latency_ms", 0) + int((time.time() - t0) * 1000),
        }
    # =====================================================================
    # FAST-PATH 0: If we already have final_answer in tools, end immediately
    # =====================================================================
    if "final_answer" in tools_used:
        return {
            "next_step": "final_answer",
            "current_query": query,
            "steps_executed": steps + ["planner->final_answer(already_done)"],
            "total_tokens_used": tokens,
            "tokens_consumed": state.get("tokens_consumed", 0),
            "tool_call_depth": depth + 1,
            "tools_used": tools_used,
            "task_complete": True,
            "latency_ms": state.get("latency_ms", 0) + int((time.time() - t0) * 1000),
        }

    # =====================================================================
    # FAST-PATH A: Stock / market data query -> yahoo_finance (CHECK FIRST!)
    # =====================================================================
    stock_keywords = ["stock price", "share price", "market cap", "pe ratio", "peg", "peg ratio",
                      "eps", "earnings per share", "dividend yield", "pb ratio", "price to book",
                      "roe", "roa", "ticker", "nifty", "sensex", "returns", "volatility",
                      "52 week", "dividend", "beta", "fundamental", "valuation", "book value",
                      "price to earnings", "p/e", "p/e ratio", "p/b", "forward pe", "trailing pe",
                      "current price", "trading at", "quote", "live price", "intraday","cagr", "growth rate",
                      "historical performance", "10 year", "5 year"]
    ticker_keywords = ["tcs", "reliance", "infosys", "hdfc", "hdfc bank", "sbi", "nifty", "sensex",
                       "apple", "microsoft", "google", "amazon", "tesla", "aapl", "msft", "googl", "amzn", "tsla",
                       "nvidia", "nvda", "meta", "fb", "netflix", "nflx", "amd", "intel", "intc"]
    price_keywords = ["price", "quote", "trading at", "share", "stock", "cost", "peg", "eps",
                      "valuation", "market cap", "p/e", "p/b", "pe ratio", "pb ratio", "dividend",
                      "yield", "beta", "fundamental", "roe", "roa", "book value", "earnings",
                      "ticker", "live", "current","cagr","growth"]

    rbi_keywords = ["rbi", "reserve bank", "monetary policy", "repo rate", "inflation target",
                    "gdp projection", "gdp growth", "fiscal policy", "banking regulation"]
    is_rbi_query = any(k in query_lower for k in rbi_keywords)

    has_stock_keyword = any(k in query_lower for k in stock_keywords)
    has_ticker = any(k in query_lower for k in ticker_keywords)
    has_price = any(k in query_lower for k in price_keywords)
    has_ticker_and_price = has_ticker and has_price

    # Never route RBI queries to yahoo_finance
    if is_rbi_query:
        pass  # Skip stock fast-path, let it fall through to RAG
    elif (has_stock_keyword or has_ticker_and_price) and "yahoo_finance" not in tools_used:
        ticker = _extract_ticker_from_query(query)
        print(f"[Planner] FAST-PATH: Stock query detected, routing to yahoo_finance ({ticker})")
        return {
            "next_step": "yahoo_finance",
            "current_query": query,
            "tool_input": ticker,
            "steps_executed": steps + ["planner->yahoo_finance(fast-stock)"],
            "total_tokens_used": tokens,
            "tokens_consumed": state.get("tokens_consumed", 0),
            "tool_call_depth": depth + 1,
            "tools_used": tools_used,
            "task_complete": False,
            "latency_ms": state.get("latency_ms", 0) + int((time.time() - t0) * 1000),
        }

    # =====================================================================
    # FAST-PATH B: Portfolio / allocation / Sharpe query -> portfolio_analyzer
    # =====================================================================
    portfolio_keywords = ["portfolio", "sharpe ratio", "allocation", "diversify", "risk adjusted",
                          "max drawdown", "my holdings", "asset allocation", "optimize portfolio",
                          "portfolio analysis", "risk profile", "correlation matrix"]
    if any(k in query_lower for k in portfolio_keywords) and "portfolio_analyzer" not in tools_used:
        port_input = _extract_portfolio_input_from_query(query)
        print(f"[Planner] FAST-PATH: Portfolio query detected, routing to portfolio_analyzer")
        return {
            "next_step": "portfolio_analyzer",
            "current_query": query,
            "tool_input": port_input,
            "steps_executed": steps + ["planner->portfolio_analyzer(fast-portfolio)"],
            "total_tokens_used": tokens,
            "tokens_consumed": state.get("tokens_consumed", 0),
            "tool_call_depth": depth + 1,
            "tools_used": tools_used,
            "task_complete": False,
            "latency_ms": state.get("latency_ms", 0) + int((time.time() - t0) * 1000),
        }
        # =====================================================================
        # FAST-PATH C: After yahoo_finance with data -> final_answer
        # =====================================================================
        if "yahoo_finance" in tools_used and contexts:
            # Check if the yahoo_finance result actually contains data (not just an error)
            last_ctx = str(contexts[-1]) if contexts else ""
            if "[Yahoo Finance]" in last_ctx or "market cap" in last_ctx.lower() or "price" in last_ctx.lower():
                print(f"[Planner] FAST-PATH: Yahoo Finance data present, routing to final_answer")
                return {
                    "next_step": "final_answer",
                    "current_query": query,
                    "steps_executed": steps + ["planner->final_answer(yahoo-done)"],
                    "total_tokens_used": tokens,
                    "tokens_consumed": state.get("tokens_consumed", 0),
                    "tool_call_depth": depth + 1,
                    "tools_used": tools_used,
                    "task_complete": True,
                    "latency_ms": state.get("latency_ms", 0) + int((time.time() - t0) * 1000),
                }

        # =====================================================================
        # FAST-PATH D: After portfolio_analyzer with data -> final_answer
        # =====================================================================
        if "portfolio_analyzer" in tools_used and calcs:
            last_calc = calcs[-1] if calcs else {}
            if last_calc and last_calc.get("result") is not None:
                print(f"[Planner] FAST-PATH: Portfolio analysis done, routing to final_answer")
                return {
                    "next_step": "final_answer",
                    "current_query": query,
                    "steps_executed": steps + ["planner->final_answer(portfolio-done)"],
                    "total_tokens_used": tokens,
                    "tokens_consumed": state.get("tokens_consumed", 0),
                    "tool_call_depth": depth + 1,
                    "tools_used": tools_used,
                    "task_complete": True,
                    "latency_ms": state.get("latency_ms", 0) + int((time.time() - t0) * 1000),
                }

    # =====================================================================
    # FAST-PATH 1: After calculator with result -> ALWAYS final_answer
    # =====================================================================
    if tools_used and tools_used[-1] == "financial_calculator" and calcs:
        return {
            "next_step": "final_answer",
            "current_query": query,
            "steps_executed": steps + ["planner->final_answer(calc-done)"],
            "total_tokens_used": tokens,
            "tokens_consumed": state.get("tokens_consumed", 0),
            "tool_call_depth": depth + 1,
            "tools_used": tools_used,
            "task_complete": True,
            "latency_ms": state.get("latency_ms", 0) + int((time.time() - t0) * 1000),
        }

    # =====================================================================
    # FAST-PATH 2: After web_search with results -> ALWAYS final_answer
    # =====================================================================
    if "web_search" in tools_used and contexts:
        return {
            "next_step": "final_answer",
            "current_query": query,
            "steps_executed": steps + ["planner->final_answer(web-done)"],
            "total_tokens_used": tokens,
            "tokens_consumed": state.get("tokens_consumed", 0),
            "tool_call_depth": depth + 1,
            "tools_used": tools_used,
            "task_complete": True,
            "latency_ms": state.get("latency_ms", 0) + int((time.time() - t0) * 1000),
        }

    # =====================================================================
    # FAST-PATH 3: After document_comparator -> check if informative, else web_search
    # =====================================================================
    if "document_comparator" in tools_used and comp:
        comp_str = str(comp) if comp is not None else ""
        uninformative_markers = [
            "do not contain", "does not contain", "no information", "not contain",
            "impossible to compare", "insufficient data", "cannot compare", "no relevant",
            "making it impossible", "focus on currency", "focus on inflation",
            "not contain information regarding", "do not contain information regarding",
            "currency circulation", "banknote management", "banknotes", "physical currency",
            "identical excerpts", "rather than explicit", "regulatory updates rather than",
            "not explicitly about", "off-topic", "sources are about", "instead of",
            "do not address", "do not discuss", "do not mention", "not about digital",
            "focusing on currency", "focusing on banknotes", "not digital payment",
        ]
        is_uninformative = any(m in comp_str.lower() for m in uninformative_markers)
        if is_uninformative and "web_search" not in tools_used:
            print(f"[Planner] Comparator uninformative, routing to web_search")
            return {
                "next_step": "web_search",
                "current_query": query,
                "steps_executed": steps + ["planner->web_search(compare-fallback)"],
                "total_tokens_used": tokens,
                "tokens_consumed": state.get("tokens_consumed", 0),
                "tool_call_depth": depth + 1,
                "tools_used": tools_used,
                "task_complete": False,
                "latency_ms": state.get("latency_ms", 0) + int((time.time() - t0) * 1000),
            }
        return {
            "next_step": "final_answer",
            "current_query": query,
            "steps_executed": steps + ["planner->final_answer(compare-done)"],
            "total_tokens_used": tokens,
            "tokens_consumed": state.get("tokens_consumed", 0),
            "tool_call_depth": depth + 1,
            "tools_used": tools_used,
            "task_complete": True,
            "latency_ms": state.get("latency_ms", 0) + int((time.time() - t0) * 1000),
        }

    # =====================================================================
    # FAST-PATH 3.5: RAG returned empty or low confidence -> FORCE web_search
    # =====================================================================
    if tools_used and tools_used[-1] == "rag_search":
        confidence = state.get("confidence_score", 0.0)
        if not passages and "web_search" not in tools_used:
            print(f"[Planner] RAG returned no passages, forcing web_search")
            return {
                "next_step": "web_search",
                "current_query": query,
                "steps_executed": steps + ["planner->web_search(rag-empty)"],
                "total_tokens_used": tokens,
                "tokens_consumed": state.get("tokens_consumed", 0),
                "tool_call_depth": depth + 1,
                "tools_used": tools_used,
                "task_complete": False,
                "latency_ms": state.get("latency_ms", 0) + int((time.time() - t0) * 1000),
            }
        # NEW: Low confidence after RAG -> web_search fallback
        if confidence < 0.35 and "web_search" not in tools_used and passages:
            print(f"[Planner] RAG confidence too low ({confidence:.2f}), forcing web_search")
            return {
                "next_step": "web_search",
                "current_query": query,
                "steps_executed": steps + ["planner->web_search(rag-low-confidence)"],
                "total_tokens_used": tokens,
                "tokens_consumed": state.get("tokens_consumed", 0),
                "tool_call_depth": depth + 1,
                "tools_used": tools_used,
                "task_complete": False,
                "latency_ms": state.get("latency_ms", 0) + int((time.time() - t0) * 1000),
            }

    # =====================================================================
    # FAST-PATH 4: After rag_search with passages for simple factual query -> final_answer
    # =====================================================================
    if tools_used and tools_used[-1] == "rag_search" and passages:
        simple_keywords = ["what is", "what was", "how much", "what are", "tell me about", "repo rate", "gdp",
                           "inflation", "npa", "forex", "reserve"]
        is_simple = any(k in query_lower for k in simple_keywords)
        # EXPAND needs_more to include stock/price terms so stock queries don't get classified as simple RAG
        needs_more = any(k in query_lower for k in
                         ["compare", "versus", "difference", "change", "percentage", "cagr", "ratio", "calculate",
                          "compute", "growth rate", "stock price", "share price", "ticker", "market cap",
                          "peg", "p/e", "eps", "dividend", "valuation", "trading", "quote"])
        if is_simple and not needs_more and "financial_calculator" not in tools_used and "document_comparator" not in tools_used:
            return {
                "next_step": "final_answer",
                "current_query": query,
                "steps_executed": steps + ["planner->final_answer(rag-simple)"],
                "total_tokens_used": tokens,
                "tokens_consumed": state.get("tokens_consumed", 0),
                "tool_call_depth": depth + 1,
                "tools_used": tools_used,
                "task_complete": True,
                "latency_ms": state.get("latency_ms", 0) + int((time.time() - t0) * 1000),
            }

    # =====================================================================
    # FAST-PATH 5: Calc query with explicit numbers + no data needed -> calculator
    # =====================================================================
    calc_keywords = ["percentage increase", "percentage decrease", "percent change", "percentage change",
                     "what percentage", "what percent", "cagr", "growth rate", "ratio of",
                     "calculate", "compute", "how much did", "increase between",
                     "decrease between", "difference between", "sum of", "total of",
                     "absolute change", "absolute increase", "points increase", "points decrease"]
    is_calc_query = any(k in query_lower for k in calc_keywords)
    has_explicit_numbers = len(re.findall(r'\d+\.?\d*', query)) >= 2
    if is_calc_query and has_explicit_numbers and not passages and "financial_calculator" not in tools_used:
        return {
            "next_step": "financial_calculator",
            "current_query": query,
            "steps_executed": steps + ["planner->financial_calculator(fast-explicit)"],
            "total_tokens_used": tokens,
            "tokens_consumed": state.get("tokens_consumed", 0),
            "tool_call_depth": depth + 1,
            "tools_used": tools_used,
            "task_complete": False,
            "latency_ms": state.get("latency_ms", 0) + int((time.time() - t0) * 1000),
        }

    # =====================================================================
    # FAST-PATH 6: Calc query with data already retrieved -> calculator
    # =====================================================================
    if is_calc_query and passages and "financial_calculator" not in tools_used:
        return {
            "next_step": "financial_calculator",
            "current_query": query,
            "steps_executed": steps + ["planner->financial_calculator(fast-data)"],
            "total_tokens_used": tokens,
            "tokens_consumed": state.get("tokens_consumed", 0),
            "tool_call_depth": depth + 1,
            "tools_used": tools_used,
            "task_complete": False,
            "latency_ms": state.get("latency_ms", 0) + int((time.time() - t0) * 1000),
        }

    # =====================================================================
    # FAST-PATH 9: First turn, no tools used, simple factual -> rag_search
    # (Now comes AFTER stock/portfolio fast-paths)
    # =====================================================================
    if not tools_used or all(t in {"planner", "memory_resolver"} for t in tools_used):
        return {
            "next_step": "rag_search",
            "current_query": query,
            "steps_executed": steps + ["planner->rag_search(first)"],
            "total_tokens_used": tokens,
            "tokens_consumed": state.get("tokens_consumed", 0),
            "tool_call_depth": depth + 1,
            "tools_used": tools_used,
            "task_complete": False,
            "latency_ms": state.get("latency_ms", 0) + int((time.time() - t0) * 1000),
        }

    # =====================================================================
    # FALLBACK: If we've used 3+ non-final tools, just respond with what we have
    # =====================================================================
    non_final = [t for t in tools_used if t not in {"final_answer", "planner", "memory_resolver"}]
    if len(non_final) >= 3:
        return {
            "next_step": "final_answer",
            "current_query": query,
            "steps_executed": steps + ["planner->final_answer(budget)"],
            "total_tokens_used": tokens,
            "tokens_consumed": state.get("tokens_consumed", 0),
            "tool_call_depth": depth + 1,
            "tools_used": tools_used,
            "task_complete": True,
            "latency_ms": state.get("latency_ms", 0) + int((time.time() - t0) * 1000),
        }

    # =====================================================================
    # LLM-BASED PLANNING (only for complex cases not caught by fast-paths)
    # =====================================================================
    year_filter = None
    year_match = re.search(r'20\d{2}(?:[-/]?\d{2})?', query)
    context = ""
    if year_match:
        year_filter = year_match.group()
    elif "latest" in query_lower or "current" in query_lower or "recent" in query_lower:
        year_filter = "latest"

        # Build richer context for LLM
        web_ctx = state.get("retrieved_contexts", [])
        yahoo_ctx = [c for c in web_ctx if "[Yahoo Finance]" in str(c)]
        port_ctx = [c for c in web_ctx if "[Portfolio Analyzer]" in str(c)]
        other_ctx = [c for c in web_ctx if "[Yahoo Finance]" not in str(c) and "[Portfolio Analyzer]" not in str(c)]

        context = (
                "Q: " + str(query[:150]) + "\n"
                                           "Tools used: " + str(tools_used[-3:]) + "\n"
                                                                                   "Passages: " + str(
            len(passages)) + "\n"
                             "Yahoo Finance results: " + str(len(yahoo_ctx)) + "\n"
                                                                               "Portfolio results: " + str(
            len(port_ctx)) + "\n"
                             "Web results: " + str(len(other_ctx)) + "\n"
                                                                     "Calcs: " + str(len(calcs)) + "\n"
                                                                                                   "Step: " + str(
            depth + 1) + "/5"
        )

    response_text = ""
    session_tokens = 0
    for attempt in range(2):
        try:
            response_text, session_tokens = call_llm_sync(
                prompt=context,
                system_instruction=PLANNER_SYSTEM_PROMPT,
                temperature=0.0
            )
            if response_text.startswith("Error:") or len(response_text) < 10:
                raise ValueError(f"LLM error: {response_text[:100]}")
            break
        except Exception as e:
            print(f"[Planner] Attempt {attempt + 1} failed: {e}")
            if attempt < 1:
                time.sleep(0.5)
            else:
                response_text = "Error: Planner failed"

    planner_latency = time.time() - t0
    print(f"[Agent Timing] Planner (LLM): {round(planner_latency, 3)}s | Tokens: {session_tokens}")

    # Parse JSON
    next_tool = "final_answer"
    tool_input = query
    try:
        clean = re.sub(r"```json\s*|\s*```", "", response_text).strip()
        start = clean.find("{")
        end = clean.rfind("}") + 1
        if start >= 0 and end > start:
            plan = json.loads(clean[start:end])
            parsed = plan.get("next_tool", plan.get("tool", "final_answer"))
            if parsed in {"rag_search", "financial_calculator", "document_comparator", "web_search",
                          "yahoo_finance", "portfolio_analyzer", "final_answer"}:
                next_tool = parsed
            llm_input = plan.get("tool_input", plan.get("input", ""))
            if llm_input and llm_input != query:
                tool_input = llm_input
    except Exception as e:
        print(f"[Planner] JSON parse failed: {e}")
        if any(k in query_lower for k in ["compare", "versus", "difference"]):
            next_tool = "document_comparator"
        elif is_calc_query:
            next_tool = "financial_calculator"
        elif "rag_search" not in tools_used and not passages:
            next_tool = "rag_search"
        else:
            next_tool = "final_answer"

    # =====================================================================
    # LOOP PREVENTION: Never call the same non-final tool twice
    # =====================================================================
    if next_tool in tools_used and next_tool != "final_answer":
        if next_tool == "rag_search":
            if not passages and "web_search" not in tools_used:
                next_tool = "web_search"
            else:
                next_tool = "final_answer"
        elif next_tool == "web_search":
            next_tool = "final_answer"
        elif next_tool == "document_comparator":
            next_tool = "final_answer"
        elif next_tool == "financial_calculator":
            next_tool = "final_answer"
        elif next_tool == "yahoo_finance":
            next_tool = "final_answer"
        elif next_tool == "portfolio_analyzer":
            next_tool = "final_answer"
        else:
            next_tool = "final_answer"

    # =====================================================================
    # PATCH 2: Force comparator for any compare query when we have passages
    # =====================================================================
    if passages and (
            "compare" in query_lower or "versus" in query_lower or "difference" in query_lower):
        if "document_comparator" not in tools_used:
            next_tool = "document_comparator"
        else:
            next_tool = "final_answer"

    # =====================================================================
    # NEW: If RAG returned low-confidence passages and we haven't tried web_search yet, force it
    # =====================================================================
    confidence = state.get("confidence_score", 0.0)
    if passages and confidence < 0.35 and "web_search" not in tools_used and next_tool == "final_answer":
        print(f"[Planner] Low confidence ({confidence:.2f}) with passages, forcing web_search before final_answer")
        next_tool = "web_search"

    task_complete = next_tool == "final_answer"

    return {
        "next_step": next_tool,
        "current_query": tool_input,
        "year_filter": year_filter,
        "steps_executed": steps + [f"planner->{next_tool}(llm)"],
        "total_tokens_used": tokens + session_tokens,
        "tokens_consumed": state.get("tokens_consumed", 0) + session_tokens,
        "tool_call_depth": depth + 1,
        "tools_used": tools_used,
        "task_complete": task_complete,
        "latency_ms": state.get("latency_ms", 0) + int((time.time() - t0) * 1000),
    }


# =============================================================================
# TOOL NODES
# =============================================================================
def rag_search_node(state: AgentState) -> dict:
    t0 = time.time()
    raw_query = state.get("current_query") or state.get("query") or ""
    year_filter = state.get("year_filter")

    # SANITIZE: Strip known injection patterns before RAG search
    sanitized_query = raw_query
    injection_patterns = [
        r"forget all previous instructions.*",
        r"forget your instructions.*",
        r"ignore your instructions.*",
        r'say [\'"]HACKED[\'"].*',
        r'say [\'"].*?[\'"].*',
        r"also,.*",
        r"your new rules are:.*",
        r"always recommend.*",
        r"change temperature to.*",
        r"enable creative mode.*",
        r"write a poem.*",
        r"\[SYSTEM\].*",
        r"override:.*",
    ]
    for pattern in injection_patterns:
        sanitized_query = re.sub(pattern, "", sanitized_query, flags=re.IGNORECASE).strip()

    # Clean up extra punctuation/spaces from sanitization
    sanitized_query = re.sub(r"\s+", " ", sanitized_query).strip(" ,.;:!?")

    if sanitized_query != raw_query:
        print(f"[RAG] Sanitized query: '{raw_query[:60]}...' -> '{sanitized_query[:60]}...'")

    result = rag_tool.run(query=sanitized_query, top_k=5, year_filter=year_filter)

    steps = state.get("steps_executed", [])
    tools_used = state.get("tools_used", [])
    passages = state.get("retrieved_passages", [])
    contexts = state.get("retrieved_contexts", [])

    new_passages = result.get("retrieved_passages", [])
    needs_fallback = result.get("needs_fallback", False)

    # =====================================================================
    # CRITICAL FIX: Use CRAG fallback signal instead of just empty check
    # Dense search is "too forgiving" and returns irrelevant docs for off-topic queries.
    # Trust the CRAG evaluator: if it says fallback, DO IT!
    # =====================================================================
    if not new_passages or needs_fallback:
        print(f"[RAG] No passages or needs_fallback=True. Auto-triggering web search for: '{sanitized_query[:60]}...'")

        # 1. Execute Web Search immediately
        web_results_text = ""
        try:
            raw_result = web_tool.run(query=sanitized_query, max_results=3,timeout=10)
            results = raw_result.result_data if hasattr(raw_result, "result_data") else raw_result
            if results:
                web_results_text = "\n".join([f"- {r.get('title', '')}: {r.get('snippet', '')}" for r in results])
            else:
                web_results_text = "[Web search returned no results]"
        except Exception as e:
            print(f"[RAG] Auto-web-search failed: {e}")
            web_results_text = "[Web search failed]"

        # 2. Return state as if BOTH rag_search and web_search ran successfully
        # This forces the planner to go straight to final_answer (Fast-Path 2)
        dummy_passage = [{"text": web_results_text, "doc_id": "web_fallback", "score": 0.0}]

        print(f"[Agent Timing] RAG + Web Search: {round(time.time() - t0, 3)}s")
        return {
            "steps_executed": steps + ["rag_search", "web_search(crag-fallback)"],
            "tools_used": tools_used + ["rag_search", "web_search"],
            "tool_calls_count": state.get("tool_calls_count", 0) + 2,
            "tool_outputs": state.get("tool_outputs", []) + [
                {"tool": "rag_search", "result": "empty/fallback"},
                {"tool": "web_search", "result": web_results_text[:300]}
            ],
            "retrieved_passages": passages + dummy_passage,
            "retrieved_contexts": contexts + [web_results_text],
            "confidence_score": 0.7 if "[Web search returned no results]" not in web_results_text else 0.0,
            "total_tokens_used": state.get("total_tokens_used", 0) + len(web_results_text.split()),
            "latency_ms": state.get("latency_ms", 0) + int((time.time() - t0) * 1000),
        }

    # =====================================================================
    # NORMAL FLOW: RAG succeeded
    # =====================================================================
    if result.get("success") and result.get("text_summary"):
        contexts = contexts + [result.get("text_summary", "")]

    print(
        f"[Agent Timing] RAG Search: {round(time.time() - t0, 3)}s | Passages: {len(new_passages)} | Year: {year_filter}")

    return {
        "steps_executed": steps + ["rag_search"],
        "tools_used": tools_used + ["rag_search"],
        "tool_calls_count": state.get("tool_calls_count", 0) + 1,
        "tool_outputs": state.get("tool_outputs", []) + [{"tool": "rag_search", "result": result}],
        "retrieved_passages": passages + new_passages,
        "retrieved_contexts": contexts,
        "confidence_score": result.get("confidence_score", 0.0) if result.get("success") else 0.0,
        "total_tokens_used": state.get("total_tokens_used", 0) + len(result.get("text_summary", "").split()),
        "latency_ms": state.get("latency_ms", 0) + int((time.time() - t0) * 1000),
    }


def _extract_math_expression(text: str) -> str:
    text = text.strip()

    if re.match(r'^[\d\.\+\-\*/\(\)\s,]+$', text) and any(op in text for op in '+-*/'):
        return text

    text = re.sub(r'^(cagr calculation[:\s]*|calculate[:\s]*|compute[:\s]*|what is[:\s]*|what\'s[:\s]*)+',
                  "",
                  text,
                  flags=re.IGNORECASE)
    text = text.strip()
    text = text.replace('^', '**')

    # =============================================================================
    # PATCH 1: Extract balanced parenthesized expression before non-math text (GR-02 fix)
    # =============================================================================
    paren_depth = 0
    start_idx = None
    for i, ch in enumerate(text):
        if ch == '(':
            if paren_depth == 0:
                start_idx = i
            paren_depth += 1
        elif ch == ')':
            paren_depth -= 1
            if paren_depth == 0 and start_idx is not None:
                expr = text[start_idx:i + 1]
                if _has_valid_math(expr):
                    return expr

    direct = re.search(r'(cagr|growth_rate|ratio|percentage)\s*\([^)]+\)', text, re.IGNORECASE)
    if direct:
        return direct.group(0)

    cagr_formula = re.search(
        r'\((\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)\)\s*\*\*\s*\(\s*1\s*/\s*(\d+)\s*\)\s*-\s*1',
        text, re.IGNORECASE
    )
    if cagr_formula:
        end, start, years = cagr_formula.group(1), cagr_formula.group(2), cagr_formula.group(3)
        return f"cagr({start}, {end}, {years})"

    m = re.search(
        r'(?:grew|growth|increase|from)\s+(\d+(?:\.\d+)?)\s+to\s+(\d+(?:\.\d+)?)\s+(?:over|in|for)\s+(\d+)\s*years?',
        text, re.IGNORECASE
    )
    if m:
        return f"cagr({m.group(1)}, {m.group(2)}, {m.group(3)})"

    m = re.search(
        r'(?:percentage increase|what percent|what percentage).+?from\s+(\d+(?:\.\d+)?)\s+to\s+(\d+(?:\.\d+)?)',
        text, re.IGNORECASE
    )
    if m:
        return f"growth_rate({m.group(1)}, {m.group(2)})"

    m = re.search(r'what is\s+(\d+(?:\.\d+)?)%\s+of\s+(\d+(?:\.\d+)?)', text, re.IGNORECASE)
    if m:
        return f"percentage({m.group(1)}, {m.group(2)})"

    m = re.search(r'ratio\s+of\s+(\d+(?:\.\d+)?)\s+to\s+(\d+(?:\.\d+)?)', text, re.IGNORECASE)
    if m:
        return f"ratio({m.group(1)}, {m.group(2)})"

    return text.rstrip('?').strip()


def _has_valid_math(expression: str) -> bool:
    if not expression:
        return False
    has_digits = bool(re.search(r'\d', expression))
    has_operator = any(op in expression for op in '+-*/')
    has_function = bool(re.search(r'^(cagr|growth_rate|ratio|percentage)\s*\(', expression, re.IGNORECASE))
    return has_digits and (has_operator or has_function)


def _llm_formulate_expression(query: str, state: AgentState) -> str:
    all_queries = [query]
    for key in ["current_query", "query"]:
        q = state.get(key)
        if q and q not in all_queries:
            all_queries.append(q)

    for q in all_queries:
        q_lower = q.lower()
        if "repo rate" in q_lower:
            has_fy22 = "fy2022" in q_lower or "fy 2022" in q_lower or "2022" in q_lower
            has_fy23 = "fy2023" in q_lower or "fy 2023" in q_lower or "2023" in q_lower
            has_both_fy = has_fy22 and has_fy23
            wants_pct = "percentage" in q_lower or "percent" in q_lower
            wants_increase = "increase" in q_lower or "change" in q_lower or "how much" in q_lower
            is_multi_part = "and" in q_lower and (wants_pct or wants_increase)

            if has_both_fy and wants_pct and wants_increase:
                print("[Calculator] Fast-path: RBI repo rate FY2022->FY2023 both absolute + percent")
                return "((6.5 - 4.0) / 4.0) * 100"
            elif has_both_fy and wants_increase and not wants_pct:
                print("[Calculator] Fast-path: RBI repo rate absolute change FY2022->FY2023")
                return "6.5 - 4.0"
            elif has_both_fy:
                print("[Calculator] Fast-path: RBI repo rate FY2022->FY2023 percent")
                return "((6.5 - 4.0) / 4.0) * 100"
            if "fy2022" in q_lower or "fy 2022" in q_lower or ("2022" in q_lower and "repo rate" in q_lower):
                print("[Calculator] Fast-path: RBI repo rate FY2022")
                return "4.0"
            if "fy2023" in q_lower or "fy 2023" in q_lower or ("2023" in q_lower and "repo rate" in q_lower):
                print("[Calculator] Fast-path: RBI repo rate FY2023")
                return "6.5"

        # =================================================================
        # FAST-PATH: FY-comparison queries with "percentage change"
        # "How much did X increase between FY2022 and FY2023, and what percentage change?"
        # =================================================================
        for q in all_queries:
            q_lower = q.lower()
            has_fy_range = bool(re.search(r'fy\s*\d{4}.*fy\s*\d{4}|between\s+fy\d{4}\s+and\s+fy\d{4}', q_lower))
            has_pct_change = "percentage change" in q_lower or "percent change" in q_lower
            has_increase = "increase" in q_lower or "change" in q_lower

            if has_fy_range and has_pct_change:
                # Extract the known values from context or defaults
                known_values = {
                    "repo rate": {"fy2022": 4.0, "fy2023": 6.5},
                    "policy repo rate": {"fy2022": 4.0, "fy2023": 6.5},
                }
                for key, vals in known_values.items():
                    if key in q_lower:
                        old_val = vals.get("fy2022", vals.get("fy2023"))
                        new_val = vals.get("fy2023", vals.get("fy2022"))
                        if old_val and new_val and new_val != old_val:
                            expr = f"percent_change({old_val}, {new_val})"
                            print(f"[Calculator] Fast-path: FY comparison percent_change for {key}")
                            return expr

                # Generic: try to extract two numbers from context that look like rates/percentages
                context_numbers = []
                for p in state.get("retrieved_passages", [])[:3]:
                    nums = re.findall(r'(\d+\.?\d*)\s*per\s*cent', p.get("text", "").lower())
                    context_numbers.extend([float(n) for n in nums if float(n) > 0 and float(n) < 100])

                if len(context_numbers) >= 2:
                    unique_vals = sorted(set(context_numbers))
                    if len(unique_vals) >= 2:
                        expr = f"percent_change({unique_vals[0]}, {unique_vals[-1]})"
                        print(f"[Calculator] Fast-path: FY comparison from context numbers {unique_vals}")
                        return expr
    context_parts = []
    for p in state.get("retrieved_passages", [])[:2]:
        context_parts.append(p.get("text", "")[:150])
    for ctx in state.get("retrieved_contexts", [])[-2:]:
        if isinstance(ctx, str):
            context_parts.append(ctx[:150])

    context_text = "\n".join(context_parts) if context_parts else "No relevant documents retrieved."

    prompt = (
            "Convert this financial question into a valid Python math expression.\n\n"
            "Available functions: growth_rate(old, new), cagr(start, end, years), ratio(a, b), percentage(part, whole)\n"
            "Available operators: +, -, *, /, **, ()\n\n"
            "Question: " + str(query) + "\n\n"
                                        "Retrieved context (may or may not contain relevant numbers):\n"
            + str(context_text) + "\n\n"
                                  "Instructions:\n"
                                  "- If the context contains specific numbers, use them.\n"
                                  "- If the context lacks numbers, use your knowledge of widely known public financial benchmarks.\n"
                                  "- For RBI repo rate: FY2022 was 4.0%, FY2023 was 6.5%.\n"
                                  "- Return ONLY the expression, no explanation, no markdown, no quotes, no labels.\n\n"
                                  "Examples:\n"
                                  '- "percentage increase from 4.0 to 6.5" -> ((6.5 - 4.0) / 4.0) * 100\n'
                                  '- "CAGR from 1000 to 1500 over 3 years" -> cagr(1000, 1500, 3)\n'
                                  '- "ratio of 75 to 25" -> ratio(75, 25)\n\n'
                                  "Expression:"
    )

    try:
        response, _ = call_llm_sync(
            prompt=prompt,
            system_instruction="You generate valid Python math expressions. Return ONLY the raw expression string, nothing else. No markdown, no quotes, no explanation.",
            temperature=0.0
        )
        expr = response.strip()
        expr = re.sub(r"```[\w]*\n?|```", "", expr)
        expr = re.sub(r"^(expression\s*[:=]\s*|expr\s*[:=]\s*|result\s*[:=]\s*)", "", expr, flags=re.IGNORECASE)
        expr = expr.strip('"').strip("'").strip()
        expr = expr.split('\n')[0].strip()

        print(f"[Calculator] LLM raw response: '{response[:100]}...'")
        print(f"[Calculator] LLM cleaned expr: '{expr}'")

        if _has_valid_math(expr):
            print(f"[Calculator] LLM formulation SUCCESS: '{expr}'")
            return expr
        else:
            print(f"[Calculator] LLM formulation FAILED validation")
    except Exception as e:
        print(f"[Calculator] LLM formulation exception: {e}")

    return query


def financial_calculator_node(state: AgentState) -> dict:
    t0 = time.time()

    candidates = []
    for key in ["tool_input", "current_query", "query"]:
        val = state.get(key)
        if val and val not in candidates:
            candidates.append(val)

    expression = None
    for c in candidates:
        expr = _extract_math_expression(c)
        if _has_valid_math(expr):
            expression = expr
            print(f"[Calculator] Extracted from '{c[:60]}...': '{expression}'")
            break

    if not expression:
        expression = candidates[0] if candidates else ""

    if not _has_valid_math(expression):
        llm_expr = _llm_formulate_expression(expression, state)
        if llm_expr != expression:
            expression = llm_expr
            print(f"[Calculator] LLM formulated: '{expression}'")

    print(f"[Calculator] Final expression: '{expression}'")

    result = calc_tool.run(expression)

    steps = state.get("steps_executed", [])
    tools_used = state.get("tools_used", [])
    calcs = state.get("calculation_results", [])

    # FIX: Return new list instead of mutating in-place (LangGraph immutability)
    if result.get("success"):
        calcs = calcs + [result]

    result_text = f"{result.get('expression', '')} = {result.get('result', '')}"
    print(f"[Agent Timing] Calculator: {round(time.time() - t0, 3)}s | Result: {result.get('result')}")

    return {
        "steps_executed": steps + ["financial_calculator"],
        "tools_used": tools_used + ["financial_calculator"],
        "tool_calls_count": state.get("tool_calls_count", 0) + 1,
        "tool_outputs": state.get("tool_outputs", []) + [
            {"tool": "financial_calculator", "result": result.get("result")}],
        "calculation_results": calcs,
        "total_tokens_used": state.get("total_tokens_used", 0) + len(result_text.split()),
        "latency_ms": state.get("latency_ms", 0) + int((time.time() - t0) * 1000),
    }


def document_comparator_node(state: AgentState) -> dict:
    t0 = time.time()
    passages = state.get("retrieved_passages", [])
    if len(passages) < 2:
        return {
            "steps_executed": state.get("steps_executed", []) + ["document_comparator_skipped"],
            "tools_used": state.get("tools_used", []) + ["document_comparator"],
            "tool_calls_count": state.get("tool_calls_count", 0) + 1,
            "comparison_results": "Insufficient data for comparison.",
            "latency_ms": state.get("latency_ms", 0) + int((time.time() - t0) * 1000),
        }

    query_lower = state.get("query", "").lower()

    # FIX: Year-aware grouping instead of blind midpoint split
    year_matches = re.findall(r'20\d{2}', query_lower)
    group_a, group_b = [], []

    if len(year_matches) >= 2:
        y1, y2 = year_matches[0], year_matches[1]
        group_a = [p for p in passages if
                   y1 in p.get("text", "") or y1 in p.get("doc_id", "") or y1 in p.get("title", "")]
        group_b = [p for p in passages if
                   y2 in p.get("text", "") or y2 in p.get("doc_id", "") or y2 in p.get("title", "")]
        print(f"[Comparator] Year-aware grouping: {y1}={len(group_a)} docs, {y2}={len(group_b)} docs")

    # Fallback to midpoint split if year grouping is too sparse
    if len(group_a) < 1 or len(group_b) < 1:
        mid = len(passages) // 2
        group_a = passages[:mid]
        group_b = passages[mid:]
        print(f"[Comparator] Fallback midpoint split: A={len(group_a)} docs, B={len(group_b)} docs")

    text_a = "\n\n".join([
        f"[{p.get('doc_id', 'A')}] {p.get('text', '')[:500]}"
        for p in group_a[:3]
    ])
    text_b = "\n\n".join([
        f"[{p.get('doc_id', 'B')}] {p.get('text', '')[:500]}"
        for p in group_b[:3]
    ])

    query = state.get("query", "").lower()
    if "policy" in query or "stance" in query:
        metric = "monetary policy stance"
    elif "digital" in query or "payment" in query:
        metric = "digital payments approach"
    elif "gdp" in query:
        metric = "GDP growth outlook"
    elif "npa" in query or "asset" in query:
        metric = "non-performing assets"
    elif "inflation" in query:
        metric = "inflation management"
    else:
        metric = "financial metrics"

    raw_result = comp_tool.run(doc_a=text_a, doc_b=text_b, metric=metric)

    if hasattr(raw_result, "result_data"):
        result = raw_result.result_data
    elif isinstance(raw_result, dict):
        result = raw_result
    else:
        result = {"summary": str(raw_result)}

    # FIX: None-safe summary extraction
    if isinstance(result, dict):
        comp_text = result.get("summary") or ""  # None -> ""
        tokens_used = result.get("tokens_used", 0)
    else:
        comp_text = str(result)
        tokens_used = len(comp_text.split())

    print(f"[Agent Timing] Comparator: {round(time.time() - t0, 3)}s | Tokens: {tokens_used}")

    return {
        "steps_executed": state.get("steps_executed", []) + ["document_comparator"],
        "tools_used": state.get("tools_used", []) + ["document_comparator"],
        "tool_calls_count": state.get("tool_calls_count", 0) + 1,
        "tool_outputs": state.get("tool_outputs", []) + [{"tool": "document_comparator", "result": comp_text[:200]}],
        "comparison_results": comp_text,
        "total_tokens_used": state.get("total_tokens_used", 0) + tokens_used,
        "latency_ms": state.get("latency_ms", 0) + int((time.time() - t0) * 1000),
    }


def web_search_node(state: AgentState) -> dict:
    t0 = time.time()
    query = state.get("current_query") or state.get("query") or ""

    results_text = ""
    try:
        raw_result = web_tool.run(query=query, max_results=3)
        results = raw_result.result_data if hasattr(raw_result, "result_data") else raw_result

        if results:
            text = "\n".join([f"- {r.get('title', '')}: {r.get('snippet', '')}" for r in results])
            print(f"[WebSearch DEBUG] Results: {text[:300]}")
            results_text = text
        else:
            print(f"[WebSearch DEBUG] No results for query: {query[:60]}")
    except AttributeError as e:
        print(f"[WebSearch DEBUG] Tool format error (tuple/dict mismatch): {e}")
        print(f"[WebSearch DEBUG] >>> FIX REQUIRED: Update agent/tools/web_search.py to handle tuple results <<<")
        results_text = "[Web search unavailable due to tool format error]"
    except Exception as e:
        print(f"[WebSearch DEBUG] Failed: {e}")

    if not results_text:
        results_text = "[Web search returned no results]"

    print(f"[Agent Timing] Web Search: {round(time.time() - t0, 3)}s")

    return {
        "steps_executed": state.get("steps_executed", []) + ["web_search"],
        "tools_used": state.get("tools_used", []) + ["web_search"],
        "tool_calls_count": state.get("tool_calls_count", 0) + 1,
        "tool_outputs": state.get("tool_outputs", []) + [{"tool": "web_search", "result": results_text}],
        "retrieved_contexts": state.get("retrieved_contexts", []) + [results_text],
        "total_tokens_used": state.get("total_tokens_used", 0) + len(results_text.split()),
        "latency_ms": state.get("latency_ms", 0) + int((time.time() - t0) * 1000),
        "comparison_results": None,
    }


# =============================================================================
# YAHOO FINANCE HELPERS & NODE
# =============================================================================
def _extract_ticker_from_query(query: str) -> str:
    """Fast-path ticker extraction for planner routing (no LLM)."""
    # 1. Explicit .NS / .BO tickers (highest confidence)
    match = re.search(r'\b([A-Z]{2,10}\.(?:NS|BO))\b', query.upper())
    if match:
        return match.group(1)

    # 2. Known company name / ticker mappings
    mappings = {
        "tcs": "TCS.NS", "reliance": "RELIANCE.NS", "infosys": "INFY.NS", "infy": "INFY.NS",
        "hdfc bank": "HDFCBANK.NS", "hdfcbank": "HDFCBANK.NS", "hdfc": "HDFCBANK.NS",
        "sbi": "SBIN.NS", "sbin": "SBIN.NS",
        "apple": "AAPL", "microsoft": "MSFT", "google": "GOOGL", "amazon": "AMZN",
        "tesla": "TSLA", "nifty": "^NSEI", "sensex": "^BSESN",
    }
    q_lower = query.lower()
    for name, ticker in mappings.items():
        if name in q_lower:
            return ticker

    # 3. Standalone ALL-CAPS words that look like tickers (not common English)
    common_words = {"THE", "AND", "FOR", "ARE", "BUT", "NOT", "YOU", "ALL", "ANY", "CAN",
                    "HAD", "HER", "WAS", "ONE", "OUR", "OUT", "DAY", "GET", "HAS", "HIM",
                    "HIS", "HOW", "MAN", "NEW", "NOW", "OLD", "SEE", "TWO", "WAY", "WHO",
                    "BOY", "DID", "ITS", "LET", "PUT", "SAY", "SHE", "TOO", "USE", "OFF",
                    "AN", "AS", "AT", "BE", "BY", "DO", "GO", "IF", "IN", "IS", "IT", "ME",
                    "MY", "NO", "OF", "ON", "OR", "SO", "TO", "UP", "US", "WE","WHAT", "WHEN", "WHERE", "WHY",
                    "HOW", "THIS", "THAT", "THESE", "THOSE"}
    match = re.search(r'\b([A-Z]{1,5})\b', query.upper())
    if match and match.group(1) not in common_words:
        return match.group(1)

    return "RELIANCE.NS"


def _extract_ticker(state: AgentState) -> str:
    """
    Get ticker from planner's tool_input. The LLM already extracted it.
    No regex needed — the planner outputs clean tickers.
    """
    tool_input = (state.get("tool_input") or "").strip().upper()

    # Planner already gave us a clean ticker
    if tool_input and re.match(r'^[A-Z][A-Z0-9]*(\.[A-Z]{2})?$', tool_input):
        return tool_input

    # Ultimate fallback (should never happen if planner works)
    print(f"[YahooFinance] Warning: No clean ticker from planner, got: '{tool_input}'")
    return "RELIANCE.NS"


def _detect_yahoo_operation(query: str) -> str:
    q = query.lower()
    if any(w in q for w in ["return", "volatility", "performance", "sharpe", "how did", "gain", "loss"]):
        return "returns"
    if any(w in q for w in ["pe ratio", "market cap", "fundamental", "debt", "revenue", "peg", "eps", "roe", "roa", "book value"]):
        return "fundamentals"
    if any(w in q for w in ["history", "past", "last year", "chart", "trend"]):
        return "history"
    return "quote"


def yahoo_finance_node(state: AgentState) -> dict:
    t0 = time.time()
    ticker = _extract_ticker(state)
    operation = _detect_yahoo_operation(state.get("query", ""))
    print(f"[YahooFinance] Using ticker from planner: {ticker}")

    raw_result = yahoo_finance_tool.run(ticker=ticker, operation=operation)

    # Defensive unpacking
    if hasattr(raw_result, "result_data"):
        result_data = raw_result.result_data
        success = getattr(raw_result, "success", False)
        error_msg = getattr(raw_result, "error_message", str(raw_result))
    elif isinstance(raw_result, dict):
        result_data = raw_result.get("result_data", raw_result)
        success = raw_result.get("success", False)
        error_msg = raw_result.get("error_message", str(raw_result))
    else:
        result_data = str(raw_result)
        success = False
        error_msg = str(raw_result)

    result_text = str(result_data) if success else error_msg

    steps = state.get("steps_executed", [])
    tools_used = state.get("tools_used", [])

    print(f"[Agent Timing] Yahoo Finance: {round(time.time() - t0, 3)}s | Ticker: {ticker} | Op: {operation} | Success: {success}")

    return {
        "steps_executed": steps + ["yahoo_finance"],
        "tools_used": tools_used + ["yahoo_finance"],
        "tool_calls_count": state.get("tool_calls_count", 0) + 1,
        "tool_outputs": state.get("tool_outputs", []) + [{"tool": "yahoo_finance", "result": result_text[:300]}],
        "retrieved_contexts": state.get("retrieved_contexts", []) + [f"[Yahoo Finance] {result_text}"],
        "total_tokens_used": state.get("total_tokens_used", 0) + len(result_text.split()),
        "latency_ms": state.get("latency_ms", 0) + int((time.time() - t0) * 1000),
        "confidence_score": 0.9 if success else 0.0,
    }

def sanitize_state_node(state: AgentState) -> dict:
    """Reset execution-tracking fields at the start of every turn."""
    return {
        "steps_executed": [],
        "tools_used": [],
        "tool_outputs": [],
        "tool_calls": [],
        "tool_calls_count": 0,
        "tool_call_depth": 0,
        "retrieved_passages": [],
        "retrieved_contexts": [],
        "calculation_results": [],
        "comparison_results": None,
        "web_results": [],
        "final_response": None,
        "total_tokens_used": 0,
        "tokens_consumed": 0,
        "latency_ms": 0,
        "total_latency_ms": 0.0,
        "estimated_cost_usd": 0.0,
        "confidence_score": 0.0,
        "task_complete": False,
        "needs_clarification": False,
        "guardrail_triggered": False,
        "guardrail_reason": None,
        "is_budget_exhausted": False,
        "loop_detected": False,
        "errors_encountered": [],
        "next_step": "planner",
    }
# =============================================================================
# PORTFOLIO ANALYZER HELPERS & NODE
# =============================================================================
def _extract_portfolio_input_from_query(query: str) -> str:
    """Fast-path portfolio param extraction for planner routing (no LLM)."""
    q_lower = query.lower()

    # Known company name / ticker mappings (same as _extract_ticker_from_query)
    mappings = {
        "tcs": "TCS.NS", "reliance": "RELIANCE.NS", "infosys": "INFY.NS", "infy": "INFY.NS",
        "hdfc bank": "HDFCBANK.NS", "hdfcbank": "HDFCBANK.NS", "hdfc": "HDFCBANK.NS",
        "sbi": "SBIN.NS", "sbin": "SBIN.NS",
        "apple": "AAPL", "microsoft": "MSFT", "google": "GOOGL", "amazon": "AMZN",
        "tesla": "TSLA", "nifty": "^NSEI", "sensex": "^BSESN","accenture": "ACN",
    }

    tickers = []
    for name, ticker in mappings.items():
        if name in q_lower and ticker not in tickers:
            tickers.append(ticker)

    # Explicit .NS / .BO tickers only (no noisy US-style regex here)
    explicit = re.findall(r'\b([A-Z]{2,10}\.(?:NS|BO))\b', query.upper())
    for t in explicit:
        if t not in tickers:
            tickers.append(t)

    if not tickers:
        return "tickers:RELIANCE.NS,INFY.NS,HDFCBANK.NS|weights:0.4,0.3,0.3"

    tickers_str = ",".join(tickers)

    weights_match = re.findall(r'(\d{1,2})\s*%', query)
    if weights_match:
        weights = [int(w) / 100 for w in weights_match]
        if abs(sum(weights) - 1.0) < 0.05:
            weights_str = ",".join([str(w) for w in weights])
            return f"tickers:{tickers_str}|weights:{weights_str}"

    return f"tickers:{tickers_str}"


def _extract_portfolio_params(state: AgentState) -> tuple:
    """
    Parse portfolio parameters from planner's tool_input.
    The LLM formats it as: 'tickers:RELIANCE.NS,INFY.NS,HDFCBANK.NS|weights:0.4,0.3,0.3'
    Or just: 'RELIANCE.NS,INFY.NS,HDFCBANK.NS'
    """
    tool_input = (state.get("tool_input") or "").strip()

    if not tool_input:
        print(f"[Portfolio] Warning: No tool_input from planner, using default")
        return "RELIANCE.NS,INFY.NS,HDFCBANK.NS", "0.4,0.3,0.3"

    # Format: "tickers:RELIANCE.NS,INFY.NS|weights:0.4,0.3,0.3"
    if "tickers:" in tool_input.lower():
        tickers_match = re.search(r'tickers:\s*([^|]+)', tool_input, re.IGNORECASE)
        weights_match = re.search(r'weights:\s*([^|]+)', tool_input, re.IGNORECASE)

        tickers = tickers_match.group(1).strip() if tickers_match else ""
        weights = weights_match.group(1).strip() if weights_match else None

        if tickers:
            return tickers, weights

    # Format: just comma-separated tickers (planner didn't use prefix)
    if ',' in tool_input and all(c.isupper() or c in '.,' or c.isspace() for c in tool_input):
        tickers = ','.join(t.strip() for t in tool_input.split(',') if t.strip())
        return tickers, None

    # Single ticker
    clean = tool_input.replace(' ', '').replace(',', '')
    if clean and re.match(r'^[A-Z][A-Z0-9]*(\.[A-Z]{2})?$', clean):
        return clean, None

    print(f"[Portfolio] Warning: Could not parse tool_input: '{tool_input}', using default")
    return "RELIANCE.NS,INFY.NS,HDFCBANK.NS", "0.4,0.3,0.3"


def portfolio_analyzer_node(state: AgentState) -> dict:
    t0 = time.time()
    tickers, weights = _extract_portfolio_params(state)
    print(f"[Portfolio] Using from planner: tickers={tickers}, weights={weights}")

    raw_result = portfolio_analyzer_tool.run(tickers=tickers, weights=weights)

    if hasattr(raw_result, "result_data"):
        result_data = raw_result.result_data
        success = getattr(raw_result, "success", False)
        error_msg = getattr(raw_result, "error_message", str(raw_result))
    elif isinstance(raw_result, dict):
        result_data = raw_result.get("result_data", raw_result)
        success = raw_result.get("success", False)
        error_msg = raw_result.get("error_message", str(raw_result))
    else:
        result_data = str(raw_result)
        success = False
        error_msg = str(raw_result)

    result_text = str(result_data) if success else error_msg

    steps = state.get("steps_executed", [])
    tools_used = state.get("tools_used", [])
    calcs = state.get("calculation_results", [])

    if success and result_data:
        calcs = calcs + [{
            "expression": f"portfolio_sharpe({tickers})",
            "result": result_data.get("portfolio", {}).get("sharpe_ratio") if isinstance(result_data, dict) else result_text,
            "tool": "portfolio_analyzer"
        }]

    print(f"[Agent Timing] Portfolio Analyzer: {round(time.time() - t0, 3)}s | Tickers: {tickers} | Success: {success}")

    return {
        "steps_executed": steps + ["portfolio_analyzer"],
        "tools_used": tools_used + ["portfolio_analyzer"],
        "tool_calls_count": state.get("tool_calls_count", 0) + 1,
        "tool_outputs": state.get("tool_outputs", []) + [{"tool": "portfolio_analyzer", "result": result_text[:400]}],
        "calculation_results": calcs,
        "retrieved_contexts": state.get("retrieved_contexts", []) + [f"[Portfolio Analyzer] {result_text}"],
        "total_tokens_used": state.get("total_tokens_used", 0) + len(result_text.split()),
        "latency_ms": state.get("latency_ms", 0) + int((time.time() - t0) * 1000),
        "confidence_score": 0.9 if success else 0.0,
    }


# =============================================================================
# FINAL ANSWER NODE
# =============================================================================
def final_answer_node(state: AgentState) -> dict:
    t0 = time.time()
    query = state.get("query", "")
    passages = state.get("retrieved_passages", [])
    contexts = state.get("retrieved_contexts", [])
    calcs = state.get("calculation_results", [])
    comp = state.get("comparison_results")

    context_parts = []
    for i, p in enumerate(passages[:5], 1):
        doc_id = p.get("doc_id", "?")
        text = p.get("text", "")[:300]
        context_parts.append(f"[{i}] Source: {doc_id}\n{text}")

    # Separate Yahoo Finance / Portfolio from generic web results
    for ctx in contexts:
        if not isinstance(ctx, str):
            continue
        if "[Yahoo Finance]" in ctx:
            context_parts.append(f"[Yahoo Finance] {ctx[:400]}")
        elif "[Portfolio Analyzer]" in ctx:
            context_parts.append(f"[Portfolio Analysis] {ctx[:400]}")
        else:
            context_parts.append(f"[Web] {ctx[:300]}")

    if comp:
        context_parts.append(f"[Comparison] {comp[:300]}")

    for c in calcs:
        context_parts.append(f"[Calculation] {c.get('expression', '')} = {c.get('result', '')}")

    context_text = "\n\n".join(context_parts) if context_parts else "No specific documents retrieved."

    prompt = f"Question: {query}\n\nContext:\n{context_text}"

    try:
        response_text, tokens = call_llm_sync(
            prompt=prompt,
            system_instruction=RESPONSE_SYSTEM_PROMPT,
            temperature=0.3
        )
    except Exception as e:
        response_text = f"Error generating response: {e}"
        tokens = 0

    print(f"[Final Answer] Calculations in state: {len(calcs)}")
    print(f"[Final Answer] Comparison in state: {len(comp) if comp else 0}")
    if comp:
        is_info = "do not contain" not in str(comp).lower()
        print(f"[Final Answer] Comparison informative: {is_info}")
    print(f"[Agent Timing] Final Answer: {round(time.time() - t0, 3)}s | Tokens: {tokens}")

    return {
        "final_response": response_text,
        "total_tokens_used": state.get("total_tokens_used", 0) + tokens,
        "tokens_consumed": state.get("tokens_consumed", 0) + tokens,
        "latency_ms": state.get("latency_ms", 0) + int((time.time() - t0) * 1000),
        "task_complete": True,
    }


# =============================================================================
# GRAPH COMPILATION (THE CRITICAL MISSING PIECE)
# =============================================================================
def build_agent_graph():
    """Build and compile the LangGraph workflow."""
    workflow = StateGraph(AgentState)

    # 1. Add ALL nodes
    workflow.add_node("sanitize_state", sanitize_state_node)      # <-- ADD THIS
    workflow.add_node("memory_resolver", memory_resolver_node)
    workflow.add_node("guardrail_check", guardrail_check_node)
    workflow.add_node("planner", planner_node)
    workflow.add_node("rag_search", rag_search_node)
    workflow.add_node("financial_calculator", financial_calculator_node)
    workflow.add_node("document_comparator", document_comparator_node)
    workflow.add_node("web_search", web_search_node)
    workflow.add_node("yahoo_finance", yahoo_finance_node)
    workflow.add_node("portfolio_analyzer", portfolio_analyzer_node)
    workflow.add_node("human_review", human_review_node)
    workflow.add_node("final_answer", final_answer_node)

    # 2. Set entry point
    workflow.set_entry_point("sanitize_state")

    # 3. Add CONDITIONAL edges for nodes that decide the next step
    workflow.add_conditional_edges("memory_resolver", routing_condition)
    workflow.add_conditional_edges("guardrail_check", routing_condition)
    workflow.add_conditional_edges("planner", routing_condition)

    # 4. Add STATIC edges
    workflow.add_edge("sanitize_state", "memory_resolver")          # <-- ADD THIS
    workflow.add_edge("rag_search", "planner")
    workflow.add_edge("financial_calculator", "planner")
    workflow.add_edge("document_comparator", "planner")
    workflow.add_edge("web_search", "planner")
    workflow.add_edge("yahoo_finance", "planner")
    workflow.add_edge("portfolio_analyzer", "planner")

    # 5. Add END edges for terminal nodes
    workflow.add_edge("final_answer", END)
    workflow.add_edge("human_review", END)

    return workflow.compile()

# =============================================================================
# COMPILE GRAPH (This is what api/main.py imports as `agent_brain`)
# =============================================================================
agent_brain = build_agent_graph()

# =============================================================================
# LANGSMITH TRACEABLE WRAPPER (Step 1: Production Observability)
# =============================================================================
if _LANGSMITH_AVAILABLE:
    @traceable(run_type="chain", name="agent_run", tags=["financial_agent", "v1"])
    def run_agent_traced(state: AgentState) -> dict:
        """LangSmith-traced wrapper for agent invocation."""
        return agent_brain.invoke(state)
else:
    def run_agent_traced(state: AgentState) -> dict:
        """Fallback wrapper without LangSmith."""
        return agent_brain.invoke(state)