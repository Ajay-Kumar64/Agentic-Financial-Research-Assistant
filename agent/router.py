from typing import Literal
from agent.state import AgentState


def routing_condition(state: AgentState) -> Literal[
    "planner", "guardrail_check", "rag_search", "financial_calculator",
    "document_comparator", "web_search", "yahoo_finance", "portfolio_analyzer",
    "human_review", "final_answer"]:
    """Route to the next node based on planner output.

    Handles ALL possible next_step values from the planner and other nodes.
    Unknown values safely fall back to final_answer.
    """
    next_step = state.get("next_step")

    if not next_step:
        return "final_answer"

    next_step_clean = next_step.strip().lower()

    # Planner routing
    if next_step_clean in ("planner", "plan"):
        return "planner"

    # Guardrail routing
    elif next_step_clean in ("guardrail_check", "guardrail", "check"):
        return "guardrail_check"

    # RAG retrieval
    elif next_step_clean in ("rag_search", "retrieve", "rag"):
        return "rag_search"

    # Calculator
    elif next_step_clean in ("financial_calculator", "calculator", "calc", "compute", "calculate"):
        return "financial_calculator"

    # Document comparator
    elif next_step_clean in ("document_comparator", "comparator", "compare"):
        return "document_comparator"

    # Web search fallback
    elif next_step_clean in ("web_search", "web", "fallback", "search"):
        return "web_search"

    # Yahoo Finance (stock data)
    elif next_step_clean in ("yahoo_finance", "yahoo", "stock", "finance"):
        return "yahoo_finance"

    # Portfolio analyzer
    elif next_step_clean in ("portfolio_analyzer", "portfolio", "allocation"):
        return "portfolio_analyzer"

    # Human review
    elif next_step_clean in ("human_review", "human", "review"):
        return "human_review"

    # Final answer / terminal
    elif next_step_clean in ("final_answer", "answer", "respond", "end", "done"):
        return "final_answer"

    # Safety fallback: unknown next_step -> final_answer to prevent graph loops
    print(f"[Router] WARNING: Unknown next_step '{next_step}', falling back to final_answer")
    return "final_answer"