from typing import TypedDict, List, Dict, Any, Optional


class ToolCallRecord(TypedDict):
    """Record of a single tool call for trace logging."""
    tool_name: str
    tool_input: Dict[str, Any]
    tool_output: Dict[str, Any]
    success: bool
    latency_ms: float
    tokens_used: int
    confidence: float
    timestamp: str
    error: Optional[str]


class AgentState(TypedDict):
    """Complete agent state passed through the LangGraph state machine."""

    # Conversation
    query: str
    input_query: str
    current_query: str
    original_query: str
    conversation_id: str
    turn_number: int

    # Planning
    current_plan: Optional[str]
    plan: str
    planned_tools: List[str]
    next_step: Optional[str]
    current_step: int

    # Tool execution tracking
    steps_executed: List[str]
    tool_sequence: List[str]
    tools_used: List[str]
    tool_outputs: List[Dict[str, Any]]
    tool_input: Optional[str]
    tool_calls: List[ToolCallRecord]
    tool_calls_count: int
    tool_call_depth: int
    last_tool_output: Dict[str, Any]

    # Accumulated results
    retrieved_contexts: List[str]
    retrieved_passages: List[Dict[str, Any]]
    calculation_results: List[Dict[str, Any]]
    comparison_results: Optional[str]
    web_results: List[Dict[str, Any]]
    final_response: Optional[str]

    # Budget tracking
    total_tokens_used: int
    tokens_consumed: int
    latency_ms: int
    total_latency_ms: float
    estimated_cost_usd: float

    # Quality signals
    confidence_score: float
    task_complete: bool
    needs_clarification: bool

    # Guardrail flags
    guardrail_triggered: bool
    guardrail_reason: Optional[str]
    is_budget_exhausted: bool
    loop_detected: bool
    errors_encountered: List[str]

    # Limits
    max_depth: int
    max_token_budget: int

    # Memory + Recency
    year_filter: Optional[str]
    conversation_history: Optional[List[Dict[str, Any]]]
    resolved_references: Dict[str, Any]


def initialize_agent_state(
    query: str,
    conversation_id: str = "",
    turn_number: int = 1,
    max_depth: int = 4,
    max_token_budget: int = 50000,
    conversation_history: Optional[List[Dict[str, Any]]] = None
) -> AgentState:
    """Create initial state for a new agent turn."""
    return {
        # Conversation
        "query": query,
        "input_query": query,
        "current_query": query,
        "original_query": query,
        "conversation_id": conversation_id or "",
        "turn_number": turn_number,

        # Planning
        "current_plan": None,
        "plan": "",
        "planned_tools": [],
        "next_step": "planner",
        "current_step": 0,

        # Tool execution
        "steps_executed": [],
        "tool_sequence": [],
        "tools_used": [],
        "tool_outputs": [],
        "tool_input": None,
        "tool_calls": [],
        "tool_calls_count": 0,
        "tool_call_depth": 0,
        "last_tool_output": {},

        # Results
        "retrieved_contexts": [],
        "retrieved_passages": [],
        "calculation_results": [],
        "comparison_results": None,
        "web_results": [],
        "final_response": None,

        # Budget
        "total_tokens_used": 0,
        "tokens_consumed": 0,
        "latency_ms": 0,
        "total_latency_ms": 0.0,
        "estimated_cost_usd": 0.0,

        # Quality
        "confidence_score": 0.0,
        "task_complete": False,
        "needs_clarification": False,

        # Guardrails
        "guardrail_triggered": False,
        "guardrail_reason": None,
        "is_budget_exhausted": False,
        "loop_detected": False,
        "errors_encountered": [],

        # Limits
        "max_depth": max_depth,
        "max_token_budget": max_token_budget,

        # Memory
        "year_filter": None,
        "conversation_history": conversation_history,
        "resolved_references": {},
    }