# File: api/models.py
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Literal


class ChatRequest(BaseModel):
    """
    Incoming chat request from the UI.
    """
    message: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The query string or message from the user."
    )
    conversation_id: Optional[str] = Field(
        None,
        description="Session token for continuing a multi-turn conversation."
    )


class Citation(BaseModel):
    """
    Source citation for a claim in the agent's response.
    """
    source: Literal["rag", "web", "calc", "compare"] = Field(
        ...,
        description="Type of source: rag, web, calc, or compare."
    )
    reference: str = Field(
        ...,
        description="Document ID, URL, or formula string."
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score for this citation."
    )


class TraceStep(BaseModel):
    """
    Single step in the agent's execution trace.
    """
    step_number: int = Field(..., description="Chronological step index.")
    node_name: str = Field(..., description="Node or tool that executed.")
    action_taken: str = Field(..., description="What was done at this step.")
    telemetry_metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Latency, tokens, and other per-step metrics."
    )


class Trace(BaseModel):
    """
    Full execution trace for a single turn.
    """
    steps: List[TraceStep] = Field(default_factory=list, description="All steps taken.")
    total_steps: int = Field(0, description="Total tool calls / steps.")
    total_latency_ms: float = Field(0.0, description="End-to-end latency in ms.")
    total_tokens: int = Field(0, description="Total tokens consumed.")
    estimated_cost_usd: float = Field(0.0, description="Estimated API cost.")
    guardrail_triggered: bool = Field(
        False,
        description="Whether a guardrail forced early termination."
    )
    guardrail_reason: Optional[str] = Field(
        None,
        description="Which guardrail fired and why."
    )


class ChatResponse(BaseModel):
    """
    Complete response envelope with answer, citations, and trace.
    """
    response: str = Field(..., description="The agent's final answer.")
    conversation_id: str = Field(..., description="Session identifier.")
    turn_number: int = Field(1, description="Turn number in this conversation.")
    citations: List[Citation] = Field(
        default_factory=list,
        description="Sources backing the response."
    )
    trace: Trace = Field(
        default_factory=Trace,
        description="Full execution trace for observability."
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Extra metadata: model name, timestamp, etc."
    )