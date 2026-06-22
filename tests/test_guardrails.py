# tests/test_guardrails.py
import pytest
from agent.guardrails import check_guardrails, _detect_loop, GUARDRAIL_CONFIG
from agent.state import initialize_agent_state


def test_no_guardrails_triggered():
    """Fresh state should pass all guardrails."""
    state = initialize_agent_state("What is the repo rate?")
    decision, reason = check_guardrails(state)
    assert decision == "continue"
    assert reason is None


def test_loop_detection_same_tool():
    """A→A pattern should trigger loop guardrail."""
    state = initialize_agent_state("test")
    state["tools_used"] = ["rag_search", "rag_search"]
    decision, reason = check_guardrails(state)
    assert decision == "force_respond"
    assert reason == "loop_detected"


def test_loop_detection_oscillation():
    """A→B→A pattern should trigger loop guardrail."""
    state = initialize_agent_state("test")
    state["tools_used"] = ["rag_search", "financial_calculator", "rag_search"]
    assert _detect_loop(state["tools_used"]) is True


def test_max_tool_calls():
    """Exceeding max_tool_calls should force respond (no loop in data)."""
    state = initialize_agent_state("test")
    # 5 distinct tool calls — no loop, so max_tool_calls should trigger
    state["tools_used"] = [
        "rag_search",
        "financial_calculator",
        "document_comparator",
        "web_search",
        "rag_search"
    ]
    decision, reason = check_guardrails(state)
    assert decision == "force_respond"
    assert reason == "max_tool_calls_reached"

def test_token_budget():
    """Exceeding token budget should force respond."""
    state = initialize_agent_state("test")
    state["total_tokens_used"] = 5000
    decision, reason = check_guardrails(state)
    assert decision == "force_respond"
    assert reason == "token_budget_exceeded"


def test_latency_budget():
    """Exceeding latency budget should force respond."""
    state = initialize_agent_state("test")
    state["latency_ms"] = 9000
    decision, reason = check_guardrails(state)
    assert decision == "force_respond"
    assert reason == "latency_budget_exceeded"


def test_low_confidence_hint():
    """Low confidence with no web_search should suggest continue."""
    state = initialize_agent_state("test")
    state["confidence_score"] = 0.5
    state["tools_used"] = ["rag_search"]
    state["tool_calls_count"] = 1
    decision, reason = check_guardrails(state)
    assert decision == "continue"
    assert reason == "low_confidence_hint"


def test_low_confidence_already_web_searched():
    """Low confidence but web_search already used — no hint."""
    state = initialize_agent_state("test")
    state["confidence_score"] = 0.5
    state["tools_used"] = ["rag_search", "web_search"]
    state["tool_calls_count"] = 2
    decision, _ = check_guardrails(state)
    assert decision == "continue"  # No guardrail, just continue


def test_task_complete():
    """Task complete flag should route to respond."""
    state = initialize_agent_state("test")
    state["task_complete"] = True
    decision, reason = check_guardrails(state)
    assert decision == "respond"
    assert reason == "task_complete"


def test_guardrail_config_loaded():
    assert GUARDRAIL_CONFIG["max_tool_calls"] == 5
    assert GUARDRAIL_CONFIG["max_tokens"] == 4000
    assert GUARDRAIL_CONFIG["max_latency_ms"] == 8000
    assert GUARDRAIL_CONFIG["confidence_threshold"] == 0.6