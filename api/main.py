# File: api/main.py
import os
import uuid
import json
import time
import hashlib
import traceback
import asyncio
from datetime import datetime
from typing import Dict, Any, List
import multiprocessing
import uvicorn

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from langsmith import traceable
multiprocessing.freeze_support()
from agent.graph import agent_brain
from api.models import (
    ChatRequest,
    ChatResponse,
    TraceStep,
    Citation,
    Trace,
)
from api.middleware import RequestLoggingMiddleware, ErrorHandlingMiddleware
from agent.graph import agent_brain as workflow_graph
from agent.state import initialize_agent_state
from agent.llm_provider import warmup_llm_async

# =============================================================================
# REDIS CONVERSATION STORE (NEW)
# =============================================================================
try:
    from agent.redis_store import (
        get_conversation,
        save_conversation,
        append_turn,
        delete_conversation,
        list_conversation_ids,
        get_cached_response as redis_get_cached,
        set_cached_response as redis_set_cached,
        get_store_metrics as redis_store_metrics,
        health_check as redis_health,
    )
    _REDIS_CONVERSATION_AVAILABLE = True
except Exception as e:
    print(f"[API] Redis conversation store import failed: {e}")
    _REDIS_CONVERSATION_AVAILABLE = False

# =============================================================================
# RAGAS EVALUATION (NEW)
# =============================================================================
try:
    from eval.ragas_eval import RagasEvaluator, evaluate_golden_traces
    _RAGAS_AVAILABLE = True
except Exception as e:
    print(f"[API] RAGAS evaluation import failed: {e}")
    _RAGAS_AVAILABLE = False

# =============================================================================
# LEGACY EVALUATION
# =============================================================================
try:
    from evaluation.run_eval import load_golden_traces, run_single_trace
    _EVAL_AVAILABLE = True
except Exception as e:
    print(f"[API] Legacy evaluation import failed: {e}")
    _EVAL_AVAILABLE = False

# =============================================================================
# LEGACY REDIS CACHE (for response caching)
# =============================================================================
try:
    from rag.cache import get_response as redis_get, put_response as redis_put, norm
    _REDIS_CACHE_AVAILABLE = True
except Exception:
    _REDIS_CACHE_AVAILABLE = False
    norm = lambda x: x.lower().strip()

# =============================================================================
# APP SETUP
# =============================================================================
app = FastAPI(
    title="Agentic Financial Research Assistant API",
    version="2.0.0",
    description="Production backend with Redis conversation store, RAGAS evaluation, caching, and trace logging."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(ErrorHandlingMiddleware)

# =============================================================================
# CONFIGURATION
# =============================================================================
ENABLE_RAGAS = os.getenv("ENABLE_RAGAS_EVAL", "true").lower() == "true"
LOGS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
os.makedirs(LOGS_DIR, exist_ok=True)

# =============================================================================
# IN-MEMORY FALLBACK (used when Redis conversation store is unavailable)
# =============================================================================
CONVERSATION_STORE: Dict[str, Dict[str, Any]] = {}
_local_response_cache: Dict[str, Any] = {}
_MAX_LOCAL_CACHE = 100


# =============================================================================
# HELPERS
# =============================================================================
def _cache_key(query: str, conversation_id: str = "") -> str:
    return hashlib.sha256(f"{conversation_id}:{norm(query)}".encode()).hexdigest()


def _get_local_cached_response(query: str, conversation_id: str = "") -> str | None:
    """Get from local in-memory cache."""
    key = _cache_key(query, conversation_id)
    if key in _local_response_cache:
        entry = _local_response_cache[key]
        if time.time() - entry.get("ts", 0) < 300:
            return entry["response"]
        del _local_response_cache[key]
    return None


def _set_local_cached_response(query: str, response: str, conversation_id: str = ""):
    """Set local in-memory cache."""
    key = _cache_key(query, conversation_id)
    _local_response_cache[key] = {"response": response, "ts": time.time()}
    if len(_local_response_cache) > _MAX_LOCAL_CACHE:
        oldest = next(iter(_local_response_cache))
        del _local_response_cache[oldest]


def _get_cached_response(query: str, conversation_id: str = "") -> str | None:
    """Try Redis cache first, fall back to local cache."""
    # Try new Redis cache
    if _REDIS_CONVERSATION_AVAILABLE:
        cached = redis_get_cached(query, conversation_id)
        if cached:
            return cached
    # Fallback to local
    return _get_local_cached_response(query, conversation_id)


def _set_cached_response(query: str, response: str, conversation_id: str = ""):
    """Set both Redis and local cache."""
    if _REDIS_CONVERSATION_AVAILABLE:
        redis_set_cached(query, response, conversation_id)
    _set_local_cached_response(query, response, conversation_id)


def _cleanup_expired_conversations():
    """Clean up expired in-memory conversations (fallback only)."""
    now = time.time()
    expired = [cid for cid, data in CONVERSATION_STORE.items() if now - data.get("last_access", 0) > 1800]
    for cid in expired:
        del CONVERSATION_STORE[cid]


def _get_or_create_conversation(conv_id: str) -> tuple[str, Dict[str, Any]]:
    """Get conversation from Redis or create new. Returns (conv_id, session)."""
    # Try Redis first
    if _REDIS_CONVERSATION_AVAILABLE and conv_id:
        conv = get_conversation(conv_id)
        if conv:
            return conv_id, conv

    # Fallback to in-memory
    if conv_id and conv_id in CONVERSATION_STORE:
        return conv_id, CONVERSATION_STORE[conv_id]

    # Create new
    new_id = conv_id or str(uuid.uuid4())
    session = {
        "history": [],
        "created_at": time.time(),
        "last_access": time.time(),
        "last_state": {},
    }

    # Store in appropriate backend
    if _REDIS_CONVERSATION_AVAILABLE:
        save_conversation(new_id, [])
    else:
        CONVERSATION_STORE[new_id] = session

    return new_id, session


def _update_conversation(conv_id: str, query: str, response: str, tools_used: list, state_snapshot: dict):
    """Persist conversation turn to Redis or memory."""
    if _REDIS_CONVERSATION_AVAILABLE:
        append_turn(
            conversation_id=conv_id,
            query=query,
            response=response,
            tools_used=tools_used,
            state_snapshot=state_snapshot,
        )
    else:
        session = CONVERSATION_STORE.get(conv_id)
        if session:
            session["history"].append({
                "turn": len(session["history"]) + 1,
                "query": query,
                "response": response[:300],
                "tools_used": tools_used,
            })
            session["last_state"] = state_snapshot
            session["last_access"] = time.time()
            if len(session["history"]) > 5:
                session["history"] = session["history"][-5:]


def _estimate_cost_usd(tokens: int) -> float:
    return tokens * 0.00000025


def _build_citations(state: dict) -> List[Citation]:
    citations = []
    try:
        for p in (state.get("retrieved_passages") or [])[:5]:
            if isinstance(p, dict):
                citations.append(Citation(
                    source="rag",
                    reference=f"{p.get('doc_id', 'unknown')}:{p.get('chunk_id', 'unknown')}",
                    confidence=float(p.get("score", 0.0))
                ))
    except Exception as e:
        print(f"[API] Citation passages error: {e}")

    try:
        for c in (state.get("calculation_results") or []):
            if isinstance(c, dict):
                citations.append(Citation(
                    source="calc",
                    reference=str(c.get("expression", "calculation")),
                    confidence=1.0
                ))
    except Exception as e:
        print(f"[API] Citation calcs error: {e}")

    try:
        for ctx in (state.get("retrieved_contexts") or []):
            if isinstance(ctx, str) and len(ctx) > 10 and not ctx.startswith("[RAG"):
                citations.append(Citation(source="web", reference=ctx[:100], confidence=0.7))
    except Exception as e:
        print(f"[API] Citation web error: {e}")

    try:
        comp = state.get("comparison_results")
        if isinstance(comp, str) and len(comp) > 5:
            citations.append(Citation(source="compare", reference=comp[:100], confidence=0.85))
        elif isinstance(comp, dict):
            citations.append(Citation(source="compare", reference=str(comp.get("summary", "comparison"))[:100], confidence=0.85))
    except Exception as e:
        print(f"[API] Citation compare error: {e}")

    return citations if citations else [Citation(source="rag", reference="no_citations", confidence=0.0)]


def _build_trace(state: dict) -> Trace:
    steps = []
    try:
        for idx, step in enumerate((state.get("steps_executed") or [])):
            steps.append(TraceStep(
                step_number=idx + 1,
                node_name="agent",
                action_taken=str(step),
                telemetry_metadata={"timestamp": datetime.utcnow().isoformat()}
            ))
    except Exception as e:
        print(f"[API] Trace steps error: {e}")

    total_tokens = 0
    try:
        total_tokens = int(state.get("total_tokens_used", 0) or 0)
    except Exception:
        pass

    latency = 0.0
    try:
        latency = float(state.get("latency_ms", 0) or 0)
    except Exception:
        pass

    return Trace(
        steps=steps,
        total_steps=len(steps),
        total_latency_ms=latency,
        total_tokens=total_tokens,
        estimated_cost_usd=_estimate_cost_usd(total_tokens),
        guardrail_triggered=bool(state.get("guardrail_triggered", False)),
        guardrail_reason=str(state.get("guardrail_reason")) if state.get("guardrail_reason") else None
    )


def commit_trace_log(session_id: str, final_state: dict) -> None:
    try:
        log_payload = {
            "timestamp": datetime.utcnow().isoformat(),
            "session_id": session_id,
            "query": final_state.get("query", ""),
            "tool_call_depth": final_state.get("tool_call_depth", 0),
            "total_tokens_used": final_state.get("total_tokens_used", 0),
            "steps_executed": list(final_state.get("steps_executed", [])),
            "final_response": str(final_state.get("final_response", ""))[:500],
            "guardrail_triggered": final_state.get("guardrail_triggered", False),
            "guardrail_reason": final_state.get("guardrail_reason")
        }
        log_path = os.path.join(LOGS_DIR, f"trace_{session_id}_{int(time.time())}.json")
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(log_payload, f, indent=2)
    except Exception as e:
        print(f"[API] Trace log commit failed: {e}")


# =============================================================================
# AGENT RUNNER — Intentionally sync; always executed inside asyncio.to_thread
# =============================================================================
def _run_agent(state):
    """
    Synchronous agent invocation. This function is INTENTIONALLY sync
    because it is always executed inside asyncio.to_thread() from the endpoint.
    """
    return workflow_graph.invoke(state)


# =============================================================================
# STARTUP EVENT
# =============================================================================
@app.on_event("startup")
async def startup_event():
    """Non-blocking LLM warm-up on startup."""
    asyncio.create_task(warmup_llm_async())


# =============================================================================
# ENDPOINTS
# =============================================================================
@app.post("/api/v1/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest, background_tasks: BackgroundTasks):
    try:
        await asyncio.to_thread(_cleanup_expired_conversations)

        # Get or create conversation (Redis-backed)
        conv_id, session = await asyncio.to_thread(
            _get_or_create_conversation, request.conversation_id or ""
        )

        # Cache check
        cached = await asyncio.to_thread(_get_cached_response, request.message, conv_id)
        if cached:
            return ChatResponse(
                conversation_id=conv_id,
                response=cached,
                turn_number=len(session.get("history", [])) + 1,
                citations=[],
                trace=Trace(),
                metadata={"cache": "hit", "store": "redis" if _REDIS_CONVERSATION_AVAILABLE else "memory"}
            )

        # Build state
        initial_state = initialize_agent_state(
            query=request.message,
            max_depth=4,
            max_token_budget=50000
        )

        # Load conversation history
        history = session.get("history", [])
        if history:
            initial_state["conversation_history"] = history

        # # Load previous state for context continuity
        # last_state = session.get("last_state", {})
        # if last_state:
        #     initial_state["retrieved_passages"] = last_state.get("retrieved_passages", [])
        #     initial_state["calculation_results"] = last_state.get("calculation_results", [])
        #     initial_state["retrieved_contexts"] = last_state.get("retrieved_contexts", [])
        #     initial_state["tools_used"] = last_state.get("tools_used", [])

        # RUN AGENT — Entire graph runs in a worker thread
        print(f"[API] Starting agent for query: {request.message[:60]}...")

        @traceable(run_type="chain", name="agent_chat", tags=["financial_agent", "v2"])
        def _run_agent_traced(state):
            return workflow_graph.invoke(state)

        output_state = await asyncio.to_thread(_run_agent_traced, initial_state)

        print(f"[API] Agent completed. Keys in output: {list(output_state.keys())}")

        # Extract response safely
        final_response = ""
        if isinstance(output_state, dict):
            final_response = str(output_state.get("final_response", "") or "")
        print(f"[API] Final response length: {len(final_response)}")

        # Update conversation in Redis/memory
        if final_response:
            tools_used = output_state.get("tools_used", []) if isinstance(output_state, dict) else []
            state_snapshot = {
                "retrieved_passages": output_state.get("retrieved_passages", []) if isinstance(output_state, dict) else [],
                "calculation_results": output_state.get("calculation_results", []) if isinstance(output_state, dict) else [],
                "retrieved_contexts": output_state.get("retrieved_contexts", []) if isinstance(output_state, dict) else [],
                "tools_used": tools_used,
            }
            await asyncio.to_thread(
                _update_conversation, conv_id, request.message, final_response, tools_used, state_snapshot
            )

        # Cache response
        await asyncio.to_thread(_set_cached_response, request.message, final_response, conv_id)

        # Build response
        citations = _build_citations(output_state) if isinstance(output_state, dict) else []
        trace = _build_trace(output_state) if isinstance(output_state, dict) else Trace()

        background_tasks.add_task(commit_trace_log, conv_id, output_state if isinstance(output_state, dict) else {})

        return ChatResponse(
            response=final_response or "No response generated.",
            conversation_id=conv_id,
            turn_number=len(session.get("history", [])) + 1,
            citations=citations,
            trace=trace,
            metadata={
                "model": "gemini-3.1-flash-lite",
                "timestamp": datetime.utcnow().isoformat(),
                "confidence": float(output_state.get("confidence_score", 0.0)) if isinstance(output_state, dict) else 0.0,
                "store": "redis" if _REDIS_CONVERSATION_AVAILABLE else "memory",
            }
        )

    except Exception as e:
        print(f"\n{'='*60}")
        print(f"[API] CRITICAL ERROR in chat_endpoint: {str(e)}")
        print(traceback.format_exc())
        print(f"{'='*60}\n")
        raise HTTPException(status_code=500, detail=f"Agent execution failed: {str(e)}")


@app.get("/api/v1/health")
async def health_check():
    """Health check with Redis and RAGAS availability."""
    redis_status = {"available": False, "mode": "memory"}
    if _REDIS_CONVERSATION_AVAILABLE:
        try:
            redis_status = await asyncio.to_thread(redis_health)
        except Exception as e:
            redis_status["error"] = str(e)

    return {
        "status": "healthy",
        "version": "2.0.0",
        "timestamp": datetime.utcnow().isoformat(),
        "graph_loaded": workflow_graph is not None,
        "store": {
            "redis_conversation_available": _REDIS_CONVERSATION_AVAILABLE,
            "redis_cache_available": _REDIS_CACHE_AVAILABLE,
            "redis_status": redis_status,
        },
        "evaluation": {
            "legacy_eval_available": _EVAL_AVAILABLE,
            "ragas_available": _RAGAS_AVAILABLE and ENABLE_RAGAS,
        },
        "cache_size_local": len(_local_response_cache),
    }


@app.get("/api/v1/trace/{conversation_id}")
async def get_trace(conversation_id: str):
    """Get conversation trace from Redis or memory."""
    conv = None

    # Try Redis first
    if _REDIS_CONVERSATION_AVAILABLE:
        conv = await asyncio.to_thread(get_conversation, conversation_id)

    # Fallback to memory
    if conv is None:
        conv = CONVERSATION_STORE.get(conversation_id)

    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    return {
        "conversation_id": conversation_id,
        "history": conv.get("history", []),
        "turn_count": len(conv.get("history", [])),
        "store": "redis" if _REDIS_CONVERSATION_AVAILABLE else "memory",
    }


@app.delete("/api/v1/conversation/{conversation_id}")
async def delete_conversation_endpoint(conversation_id: str):
    """Delete a conversation from Redis and memory."""
    success = False

    if _REDIS_CONVERSATION_AVAILABLE:
        success = await asyncio.to_thread(delete_conversation, conversation_id)

    if conversation_id in CONVERSATION_STORE:
        del CONVERSATION_STORE[conversation_id]
        success = True

    if not success:
        raise HTTPException(status_code=404, detail="Conversation not found")

    return {"deleted": True, "conversation_id": conversation_id}


@app.get("/api/v1/conversations")
async def list_conversations(limit: int = 100):
    """List recent conversation IDs."""
    if _REDIS_CONVERSATION_AVAILABLE:
        ids = await asyncio.to_thread(list_conversation_ids, limit=limit)
    else:
        ids = list(CONVERSATION_STORE.keys())[:limit]

    return {
        "conversation_ids": ids,
        "count": len(ids),
        "store": "redis" if _REDIS_CONVERSATION_AVAILABLE else "memory",
    }


@app.get("/api/v1/metrics")
async def get_metrics():
    """Get store and system metrics."""
    metrics = {
        "timestamp": datetime.utcnow().isoformat(),
        "store": {
            "mode": "redis" if _REDIS_CONVERSATION_AVAILABLE else "memory",
            "local_conversations": len(CONVERSATION_STORE),
            "local_cache_size": len(_local_response_cache),
        },
        "features": {
            "redis_conversation": _REDIS_CONVERSATION_AVAILABLE,
            "redis_cache": _REDIS_CACHE_AVAILABLE,
            "legacy_eval": _EVAL_AVAILABLE,
            "ragas_eval": _RAGAS_AVAILABLE and ENABLE_RAGAS,
        },
    }

    if _REDIS_CONVERSATION_AVAILABLE:
        try:
            redis_metrics = await asyncio.to_thread(redis_store_metrics)
            metrics["store"]["redis"] = redis_metrics
        except Exception as e:
            metrics["store"]["redis_error"] = str(e)

    return metrics


# =============================================================================
# EVALUATION ENDPOINTS
# =============================================================================

@app.post("/api/v1/evaluate")
async def run_evaluation():
    """Run legacy golden trace evaluation."""
    if not _EVAL_AVAILABLE:
        raise HTTPException(status_code=503, detail="Evaluation module not available")
    traces = await asyncio.to_thread(load_golden_traces)
    results = []
    for trace in traces:
        result = await asyncio.to_thread(run_single_trace, trace)
        results.append(result)
    passed = sum(1 for r in results if r.get("status") == "PASSED")
    total = len(results)
    return {
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "pass_rate": passed / total if total else 0,
        "timestamp": datetime.utcnow().isoformat()
    }


@app.post("/api/v1/evaluate/ragas")
async def run_ragas_evaluation(
    trace_id: str = None,
    background_tasks: BackgroundTasks = None,
):
    """
    Run RAGAS evaluation on golden traces.

    Args:
        trace_id: Optional specific trace ID to evaluate. If omitted, evaluates all.

    Returns:
        RAGAS evaluation results with faithfulness, relevancy, precision, recall scores.
    """
    if not _RAGAS_AVAILABLE or not ENABLE_RAGAS:
        raise HTTPException(
            status_code=503,
            detail="RAGAS evaluation not available. Install: pip install ragas datasets"
        )

    output_path = os.path.join(os.path.dirname(__file__), "..", "eval", "results", "ragas_results.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if trace_id:
        summary = await asyncio.to_thread(
            evaluate_golden_traces,
            trace_ids=[trace_id],
            output_path=output_path,
        )
    else:
        summary = await asyncio.to_thread(evaluate_golden_traces, output_path=output_path)

    return {
        "ragas_available": _RAGAS_AVAILABLE,
        "timestamp": datetime.utcnow().isoformat(),
        **summary,
    }


@app.post("/api/v1/evaluate/ragas/single")
async def run_ragas_single_evaluation(
    query: str,
    answer: str,
    contexts: List[str] = None,
):
    """
    Run RAGAS evaluation on a single query-answer-context triplet.

    Args:
        query: The user query
        answer: The agent's answer
        contexts: Retrieved context strings

    Returns:
        RAGAS scores for the triplet.
    """
    if not _RAGAS_AVAILABLE or not ENABLE_RAGAS:
        raise HTTPException(
            status_code=503,
            detail="RAGAS evaluation not available. Install: pip install ragas datasets"
        )

    contexts = contexts or []
    evaluator = RagasEvaluator()
    result = await asyncio.to_thread(
        evaluator.evaluate_single,
        query=query,
        answer=answer,
        contexts=contexts,
        trace_id="api_single",
    )

    return result.to_dict()


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    multiprocessing.freeze_support()
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # <-- disable reload on Windows
        workers=1,  # <-- use single worker on Windows (or omit workers)
        loop="asyncio",
    )