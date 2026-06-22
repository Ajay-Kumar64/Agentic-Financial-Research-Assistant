# agent/guardrails.py
from typing import Tuple
from agent.state import AgentState

GUARDRAIL_CONFIG = {
    "max_tool_calls": 5,              # Total tool calls per turn
    "max_tokens": 4000,               # Total tokens per turn
    "max_latency_ms": 8000,           # Total wall-clock time
    "confidence_threshold": 0.6,      # Below this → try fallback
    "loop_detection_window": 3,       # Check last N tool calls for loops
    "max_retries_per_tool": 2,        # Retry limit per individual tool call
}


def check_guardrails(state: AgentState) -> Tuple[str, str | None]:
    """
    Check all guardrails and return routing decision.

    Returns:
        (decision, reason)
        decision: "continue" | "respond" | "force_respond"
        reason: guardrail_reason string or None
    """
    tools_used = state.get("tools_used", [])
    # Filter out non-tool entries for counting
    tool_calls = [t for t in tools_used if t not in ("final_answer", "planner")]

    # 1. LOOP DETECTION (highest priority)
    if _detect_loop(tools_used):
        return ("force_respond", "loop_detected")

    # 2. TOOL CALL DEPTH
    if len(tool_calls) >= GUARDRAIL_CONFIG["max_tool_calls"]:
        return ("force_respond", "max_tool_calls_reached")

    # 3. TOKEN BUDGET
    if state.get("total_tokens_used", 0) >= GUARDRAIL_CONFIG["max_tokens"]:
        return ("force_respond", "token_budget_exceeded")

    # 4. LATENCY BUDGET
    if state.get("latency_ms", 0) >= GUARDRAIL_CONFIG["max_latency_ms"]:
        return ("force_respond", "latency_budget_exceeded")

    # 5. LOW CONFIDENCE → suggest web search fallback
    # (only if web_search hasn't been used yet and we have at least 1 tool call)
    if (state.get("confidence_score", 0.0) < GUARDRAIL_CONFIG["confidence_threshold"]
            and "web_search" not in tools_used
            and len(tool_calls) > 0):
        # Don't force_respond — route back to planner with a hint
        return ("continue", "low_confidence_hint")

    # 6. TASK COMPLETE
    if state.get("task_complete", False):
        return ("respond", "task_complete")

    # Default: continue planning
    return ("continue", None)


def _detect_loop(tools_used: list) -> bool:
    """
    Check if the last tool calls entered a loop.
    Detects: A→A (same tool twice) or A→B→A (oscillation).
    """
    if len(tools_used) < 2:
        return False

    # Same tool twice in a row (excluding final_answer)
    clean = [t for t in tools_used if t not in ("final_answer", "planner")]
    if len(clean) >= 2 and clean[-1] == clean[-2]:
        return True

    # Oscillation pattern A→B→A (name-only, excludes final_answer)
    if len(clean) >= 3:
        if clean[-1] == clean[-3] and clean[-1] != clean[-2]:
            return True

    return False