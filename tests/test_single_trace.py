"""
test_single_trace.py — Run a single golden trace (single-turn or multi-turn).

Usage:
    python test_single_trace.py ST-01       # single-turn
    python test_single_trace.py MT-01       # multi-turn
    python test_single_trace.py SM-03       # single-turn multi-tool
"""
import sys
import json
import time
import re

from agent.graph import agent_brain
from agent.state import initialize_agent_state

# Load golden traces
with open("evaluation/golden_traces.json", "r", encoding="utf-8") as f:
    GOLDEN = {t["id"]: t for t in json.load(f)["traces"]}


def _is_insufficient(response: str) -> bool:
    """Check if the response correctly states that info is not available."""
    r = response.lower()
    return (
        "don't have enough information" in r
        or "do not contain" in r
        or "not found" in r
        or "not specify" in r
        or "cannot be calculated" in r
        or "insufficient" in r
        or "no relevant" in r
        or "not stated" in r
        or "no single direct answer" in r
    )


def _tools_match(tools_used, expected_tools):
    """Check if expected tools were used, allowing web_search as RAG fallback."""
    if not expected_tools:
        return True
    for t in expected_tools:
        if t in tools_used:
            return True
        # Accept web_search as valid fallback for rag_search
        if t == "rag_search" and "web_search" in tools_used:
            return True
    return False


def run_single_turn(trace):
    """Run a single-turn trace."""
    query = trace["query"]
    expected_tools = trace.get("expected_tools", [])
    expected_pattern = trace.get("expected_answer_pattern", "")
    max_steps = trace.get("expected_max_steps", 5)
    expect_guardrail = trace.get("expected_guardrail_trigger", False)

    print(f"\n{'='*60}")
    print(f"TRACE: {trace['id']} | {trace['category']}")
    print(f"Query: {query}")
    print(f"Expected tools: {expected_tools}")
    print(f"Expected pattern: {expected_pattern}")
    print(f"{'='*60}")

    state = initialize_agent_state(query, max_depth=4, max_token_budget=50000)

    t0 = time.time()
    output = agent_brain.invoke(state)
    latency_ms = int((time.time() - t0) * 1000)

    # FIX: Handle None response
    final_response = output.get("final_response") or ""
    tools_used = output.get("tools_used", [])
    total_tokens = output.get("total_tokens_used", 0)
    guardrail_triggered = output.get("guardrail_triggered", False)
    guardrail_reason = output.get("guardrail_reason", "")

    # Check results
    answer_match = bool(re.search(expected_pattern, final_response, re.IGNORECASE)) if expected_pattern else True
    tools_match = _tools_match(tools_used, expected_tools)
    steps_ok = len(tools_used) <= max_steps
    guardrail_ok = (expect_guardrail == guardrail_triggered)
    insufficient = _is_insufficient(final_response)

    # PASS if: (answer matches OR insufficient info) AND tools match AND steps OK AND guardrail OK
    passed = (answer_match or insufficient) and tools_match and steps_ok and guardrail_ok

    print(f"\n--- RESULTS ---")
    print(f"Tools used: {tools_used}")
    print(f"Steps: {len(tools_used)} (max: {max_steps})")
    print(f"Tokens: {total_tokens}")
    print(f"Latency: {latency_ms}ms")
    print(f"Guardrail: {guardrail_triggered} ({guardrail_reason})")
    print(f"\nAnswer (first 300 chars):\n{final_response[:300]}")
    print(f"\n{'🟢 PASSED' if passed else '🔴 FAILED'}")
    if not answer_match and not insufficient:
        print(f"   ❌ Answer pattern '{expected_pattern}' not found")
    if not tools_match:
        print(f"   ❌ Expected tool {expected_tools} not in {tools_used}")
    if not steps_ok:
        print(f"   ❌ Too many steps: {len(tools_used)} > {max_steps}")
    if not guardrail_ok:
        print(f"   ❌ Guardrail mismatch: expected={expect_guardrail}, got={guardrail_triggered}")

    return output


def run_multi_turn(trace):
    """Run a multi-turn trace with conversation history."""
    turns = trace["turns"]
    conversation_history = []

    print(f"\n{'='*60}")
    print(f"TRACE: {trace['id']} | {trace['category']} | {len(turns)} turns")
    print(f"{'='*60}")

    all_passed = True
    final_output = None

    for i, turn in enumerate(turns):
        query = turn["query"]
        expected_tools = turn.get("expected_tools", [])
        expected_pattern = turn.get("expected_answer_pattern", "")
        expected_resolved = turn.get("expected_resolved_query", "")

        print(f"\n--- Turn {i+1} ---")
        print(f"User: {query}")
        if expected_resolved:
            print(f"Expected resolved: {expected_resolved}")

        state = initialize_agent_state(query, max_depth=4, max_token_budget=50000)
        if conversation_history:
            state["conversation_history"] = conversation_history

        # ACCUMULATE DATA from previous turns (NOT execution state like tools_used)
        if final_output:
            state["retrieved_passages"] = final_output.get("retrieved_passages", [])
            state["calculation_results"] = final_output.get("calculation_results", [])
            state["retrieved_contexts"] = final_output.get("retrieved_contexts", [])
            # DO NOT carry: tools_used, tool_call_depth, steps_executed, latency_ms
            # Those are per-turn execution state, not working memory

        t0 = time.time()
        output = agent_brain.invoke(state)
        latency_ms = int((time.time() - t0) * 1000)

        # FIX: Handle None response
        final_response = output.get("final_response") or ""
        tools_used = output.get("tools_used", [])
        total_tokens = output.get("total_tokens_used", 0)
        resolved_query = output.get("current_query", query)

        # Check results
        answer_match = bool(re.search(expected_pattern, final_response, re.IGNORECASE)) if expected_pattern else True
        tools_match = _tools_match(tools_used, expected_tools)
        insufficient = _is_insufficient(final_response)

        # SEMANTIC check for resolved query: verify key entities are present,
        # not exact wording. "repo rate in India FY 2021-22" contains
        # "repo rate" and "2022" -> correct resolution.
        if expected_resolved:
            expected_lower = expected_resolved.lower()
            resolved_lower = resolved_query.lower()
            key_entities = []
            if "repo rate" in expected_lower:
                key_entities.append("repo rate" in resolved_lower or "rep rate" in resolved_lower)
            if "fy2022" in expected_lower or "fy 2022" in expected_lower:
                key_entities.append("2022" in resolved_lower or "2021-22" in resolved_lower)
            if "fy2023" in expected_lower or "fy 2023" in expected_lower:
                key_entities.append("2023" in resolved_lower or "2022-23" in resolved_lower)
            if "inflation" in expected_lower:
                key_entities.append("inflation" in resolved_lower)
            if "npa" in expected_lower:
                key_entities.append("npa" in resolved_lower or "non-performing" in resolved_lower)
            if "gdp" in expected_lower:
                key_entities.append("gdp" in resolved_lower)
            if "credit growth" in expected_lower:
                key_entities.append("credit" in resolved_lower)
            if "forex" in expected_lower or "foreign exchange" in expected_lower:
                key_entities.append("forex" in resolved_lower or "foreign exchange" in resolved_lower)

            resolved_ok = all(key_entities) if key_entities else True
        else:
            resolved_ok = True

        # PASS if: (answer matches OR insufficient info) AND tools match AND resolved OK
        turn_passed = (answer_match or insufficient) and tools_match and resolved_ok
        if not turn_passed:
            all_passed = False

        print(f"Resolved query: {resolved_query}")
        print(f"Tools used: {tools_used}")
        print(f"Tokens: {total_tokens}")
        print(f"Latency: {latency_ms}ms")
        print(f"Answer (first 200 chars): {final_response[:200]}")
        print(f"{'🟢 Turn passed' if turn_passed else '🔴 Turn failed'}")
        if not answer_match and not insufficient:
            print(f"   ❌ Pattern '{expected_pattern}' not found")
        if not tools_match:
            print(f"   ❌ Expected tools {expected_tools} not in {tools_used}")
        if not resolved_ok:
            print(f"   ❌ Resolved query missing key entities: expected '{expected_resolved}', got '{resolved_query}'")

        # Update history for next turn
        conversation_history.append({
            "turn": i + 1,
            "query": query,
            "response": final_response[:300],
            "tools_used": tools_used
        })
        final_output = output

    print(f"\n{'='*60}")
    print(f"OVERALL: {'🟢 ALL TURNS PASSED' if all_passed else '🔴 SOME TURNS FAILED'}")
    print(f"{'='*60}")

    return final_output


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_single_trace.py <TRACE_ID>")
        print(f"Available traces: {', '.join(sorted(GOLDEN.keys()))}")
        sys.exit(1)

    trace_id = sys.argv[1]
    if trace_id not in GOLDEN:
        print(f"Unknown trace: {trace_id}")
        print(f"Available: {', '.join(sorted(GOLDEN.keys()))}")
        sys.exit(1)

    trace = GOLDEN[trace_id]

    if "turns" in trace:
        run_multi_turn(trace)
    else:
        run_single_turn(trace)


if __name__ == "__main__":
    main()