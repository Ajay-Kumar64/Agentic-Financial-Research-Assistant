from typing import Literal
from agent.state import AgentState


def routing_condition(state: AgentState) -> Literal[
    "rag_search", "financial_calculator", "document_comparator", "web_search", "final_answer"]:
    next_step = state.get("next_step")

    if not next_step:
        return "final_answer"

    next_step_clean = next_step.strip().lower()

    if next_step_clean in ("rag_search", "retrieve", "rag"):
        return "rag_search"
    elif next_step_clean in ("financial_calculator", "calculator", "calc", "compute", "calculate"):
        return "financial_calculator"
    elif next_step_clean in ("document_comparator", "comparator", "compare"):
        return "document_comparator"
    elif next_step_clean in ("web_search", "web", "fallback", "search"):
        return "web_search"
    elif next_step_clean in ("final_answer", "answer", "respond"):
        return "final_answer"

    return "final_answer"