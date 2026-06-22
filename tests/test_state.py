# tests/test_state.py
import pytest
from agent.state import initialize_agent_state, AgentState


def test_state_initialization():
    state = initialize_agent_state("What is the repo rate?", conversation_id="conv_123")

    assert state["query"] == "What is the repo rate?"
    assert state["original_query"] == "What is the repo rate?"
    assert state["conversation_id"] == "conv_123"
    assert state["turn_number"] == 1
    assert state["plan"] == ""
    assert state["planned_tools"] == []
    assert state["tool_calls"] == []
    assert state["tool_calls_count"] == 0
    assert state["current_step"] == 0
    assert state["estimated_cost_usd"] == 0.0
    assert state["confidence_score"] == 0.0
    assert state["task_complete"] is False
    assert state["needs_clarification"] is False
    assert state["loop_detected"] is False
    assert state["resolved_references"] == {}
    assert state["max_depth"] == 4
    assert state["max_token_budget"] == 50000


def test_state_with_history():
    history = [{"turn": 1, "query": "prev", "response": "ok"}]
    state = initialize_agent_state("Follow up?", conversation_history=history)
    assert state["conversation_history"] == history


def test_state_all_fields_present():
    """Verify every spec field exists in the initialized state."""
    state = initialize_agent_state("test")
    required_keys = [
        "query", "input_query", "current_query", "original_query",
        "conversation_id", "turn_number", "current_plan", "plan",
        "planned_tools", "next_step", "current_step", "steps_executed",
        "tool_sequence", "tools_used", "tool_outputs", "tool_input",
        "tool_calls", "tool_calls_count", "tool_call_depth", "last_tool_output",
        "retrieved_contexts", "retrieved_passages", "calculation_results",
        "comparison_results", "web_results", "final_response",
        "total_tokens_used", "tokens_consumed", "latency_ms",
        "total_latency_ms", "estimated_cost_usd", "confidence_score",
        "task_complete", "needs_clarification", "guardrail_triggered",
        "guardrail_reason", "is_budget_exhausted", "loop_detected",
        "errors_encountered", "max_depth", "max_token_budget",
        "year_filter", "conversation_history", "resolved_references"
    ]
    for key in required_keys:
        assert key in state, f"Missing key: {key}"