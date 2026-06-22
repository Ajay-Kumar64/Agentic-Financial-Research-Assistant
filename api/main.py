# File: api/main.py
import os
import uuid
import json
import time
import hashlib
import traceback
from datetime import datetime
from typing import Dict, Any, List

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

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

# Evaluation imports
try:
    from evaluation.run_eval import load_golden_traces, run_single_trace
    _EVAL_AVAILABLE = True
except Exception as e:
    print(f"[API] Evaluation import failed: {e}")
    _EVAL_AVAILABLE = False

# Cache imports
try:
    from rag.cache import get_response as redis_get, put_response as redis_put, norm
    _REDIS_AVAILABLE = True
except Exception:
    _REDIS_AVAILABLE = False
    norm = lambda x: x.lower().strip()

# =============================================================================
# APP SETUP
# =============================================================================
app = FastAPI(
    title="Agentic Financial Research Assistant API",
    version="1.0.0",
    description="Production backend with caching, trace logging, and evaluation."
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
# STORES
# =============================================================================
CONVERSATION_STORE: Dict[str, Dict[str, Any]] = {}
LOGS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
os.makedirs(LOGS_DIR, exist_ok=True)

_local_response_cache: Dict[str, Any] = {}
_MAX_LOCAL_CACHE = 100


# =============================================================================
# HELPERS
# =============================================================================
def _cache_key(query: str, conversation_id: str = "") -> str:
    return hashlib.sha256(f"{conversation_id}:{norm(query)}".encode()).hexdigest()


def _get_cached_response(query: str, conversation_id: str = "") -> str | None:
    key = _cache_key(query, conversation_id)
    if key in _local_response_cache:
        entry = _local_response_cache[key]
        if time.time() - entry.get("ts", 0) < 300:
            return entry["response"]
        del _local_response_cache[key]
    return None


def _set_cached_response(query: str, response: str, conversation_id: str = "", ttl: int = 300):
    key = _cache_key(query, conversation_id)
    _local_response_cache[key] = {"response": response, "ts": time.time()}
    if len(_local_response_cache) > _MAX_LOCAL_CACHE:
        oldest = next(iter(_local_response_cache))
        del _local_response_cache[oldest]


def _cleanup_expired_conversations():
    now = time.time()
    expired = [cid for cid, data in CONVERSATION_STORE.items() if now - data.get("last_access", 0) > 1800]
    for cid in expired:
        del CONVERSATION_STORE[cid]


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
# ENDPOINTS
# =============================================================================
@app.post("/api/v1/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest, background_tasks: BackgroundTasks):
    try:
        _cleanup_expired_conversations()

        conv_id = request.conversation_id
        if not conv_id or conv_id not in CONVERSATION_STORE:
            conv_id = str(uuid.uuid4())
            CONVERSATION_STORE[conv_id] = {
                "history": [],
                "created_at": time.time(),
                "last_access": time.time()
            }

        session = CONVERSATION_STORE[conv_id]
        session["last_access"] = time.time()

        # Cache check
        cached = _get_cached_response(request.message, conv_id)
        if cached:
            return ChatResponse(
                conversation_id=conv_id,
                response=cached,
                turn_number=len(session.get("history", [])) + 1,
                citations=[],
                trace=Trace(),
                metadata={"cache": "hit"}
            )

        # Build state
        initial_state = initialize_agent_state(
            query=request.message,
            max_depth=4,
            max_token_budget=50000
        )
        if session.get("history"):
            initial_state["conversation_history"] = session["history"]

        if session.get("last_state"):
            prev = session["last_state"]
            initial_state["retrieved_passages"] = prev.get("retrieved_passages", [])
            initial_state["calculation_results"] = prev.get("calculation_results", [])
            initial_state["retrieved_contexts"] = prev.get("retrieved_contexts", [])
            initial_state["tools_used"] = prev.get("tools_used", [])

        # RUN AGENT
        print(f"[API] Starting agent for query: {request.message[:60]}...")
        output_state = workflow_graph.invoke(initial_state)
        print(f"[API] Agent completed. Keys in output: {list(output_state.keys())}")

        # Extract response safely
        final_response = ""
        if isinstance(output_state, dict):
            final_response = str(output_state.get("final_response", "") or "")
        print(f"[API] Final response length: {len(final_response)}")

        # Update session
        if final_response:
            session["history"].append({
                "turn": len(session["history"]) + 1,
                "query": request.message,
                "response": final_response[:300],
                "tools_used": output_state.get("tools_used", []) if isinstance(output_state, dict) else []
            })
            if isinstance(output_state, dict):
                session["last_state"] = {
                    "retrieved_passages": output_state.get("retrieved_passages", []),
                    "calculation_results": output_state.get("calculation_results", []),
                    "retrieved_contexts": output_state.get("retrieved_contexts", []),
                    "tools_used": output_state.get("tools_used", []),
                }
            if len(session["history"]) > 5:
                session["history"] = session["history"][-5:]

        _set_cached_response(request.message, final_response, conv_id)

        # Build response safely
        citations = _build_citations(output_state) if isinstance(output_state, dict) else []
        trace = _build_trace(output_state) if isinstance(output_state, dict) else Trace()

        background_tasks.add_task(commit_trace_log, conv_id, output_state if isinstance(output_state, dict) else {})

        return ChatResponse(
            response=final_response or "No response generated.",
            conversation_id=conv_id,
            turn_number=len(session.get("history", [])),
            citations=citations,
            trace=trace,
            metadata={
                "model": "gemini-2.0-flash",
                "timestamp": datetime.utcnow().isoformat(),
                "confidence": float(output_state.get("confidence_score", 0.0)) if isinstance(output_state, dict) else 0.0
            }
        )

    except Exception as e:
        # CRITICAL: Log full traceback to server console
        print(f"\n{'='*60}")
        print(f"[API] CRITICAL ERROR in chat_endpoint: {str(e)}")
        print(traceback.format_exc())
        print(f"{'='*60}\n")
        raise HTTPException(status_code=500, detail=f"Agent execution failed: {str(e)}")


@app.get("/api/v1/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "graph_loaded": workflow_graph is not None,
        "redis_available": _REDIS_AVAILABLE,
        "cache_size_local": len(_local_response_cache),
        "eval_available": _EVAL_AVAILABLE
    }


@app.get("/api/v1/trace/{conversation_id}")
async def get_trace(conversation_id: str):
    if conversation_id not in CONVERSATION_STORE:
        raise HTTPException(status_code=404, detail="Conversation not found")
    session = CONVERSATION_STORE[conversation_id]
    return {
        "conversation_id": conversation_id,
        "history": session.get("history", []),
        "turn_count": len(session.get("history", []))
    }


@app.post("/api/v1/evaluate")
async def run_evaluation():
    if not _EVAL_AVAILABLE:
        raise HTTPException(status_code=503, detail="Evaluation module not available")
    traces = load_golden_traces()
    results = []
    for trace in traces:
        result = run_single_trace(trace)
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)