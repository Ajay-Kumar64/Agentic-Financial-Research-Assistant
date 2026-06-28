
# =============================================================================
# REDIS HEALTH & ADMIN ENDPOINTS (Add these to api/main.py)
# =============================================================================

from fastapi import Query

class RedisHealthResponse(BaseModel):
    status: str
    redis_available: bool
    redis_mode: str
    redis_latency_ms: Optional[float]
    redis_version: Optional[str]
    memory_conversations: int
    memory_cache_size: int
    uptime_seconds: Optional[int]
    connected_clients: Optional[int]
    used_memory_human: Optional[str]
    timestamp: str


@app.get("/api/v1/health/redis", response_model=RedisHealthResponse)
async def redis_health_check():
    """Detailed Redis health check with metrics."""
    redis_status = {
        "available": False,
        "mode": "unavailable",
        "latency_ms": None,
        "version": None,
        "uptime_seconds": None,
        "connected_clients": None,
        "used_memory_human": None,
    }

    if _REDIS_CONVERSATION_AVAILABLE:
        try:
            health = redis_health()
            redis_status["available"] = health.get("redis_available", False)
            redis_status["mode"] = health.get("mode", "unknown")
            redis_status["latency_ms"] = health.get("latency_ms")

            # Try to get extended metrics
            try:
                metrics = redis_store_metrics()
                redis_status["version"] = metrics.get("redis_version")
                redis_status["uptime_seconds"] = metrics.get("redis_uptime_in_seconds")
                redis_status["connected_clients"] = metrics.get("redis_connected_clients")
                redis_status["used_memory_human"] = metrics.get("redis_used_memory_human")
            except Exception:
                pass
        except Exception as e:
            redis_status["error"] = str(e)

    return RedisHealthResponse(
        status="healthy" if redis_status["available"] else "degraded",
        redis_available=redis_status["available"],
        redis_mode=redis_status["mode"],
        redis_latency_ms=redis_status["latency_ms"],
        redis_version=redis_status["version"],
        memory_conversations=len(CONVERSATION_STORE),
        memory_cache_size=len(_local_response_cache),
        uptime_seconds=redis_status["uptime_seconds"],
        connected_clients=redis_status["connected_clients"],
        used_memory_human=redis_status["used_memory_human"],
        timestamp=datetime.utcnow().isoformat(),
    )


@app.post("/api/v1/admin/redis/test")
async def redis_test_write():
    """Test Redis write/read/delete cycle."""
    if not _REDIS_CONVERSATION_AVAILABLE:
        raise HTTPException(status_code=503, detail="Redis not available")

    import uuid
    test_id = f"health-test-{uuid.uuid4().hex[:8]}"
    test_data = {"test": True, "timestamp": datetime.utcnow().isoformat()}

    try:
        # Write
        save_conversation(test_id, [{"turn": 1, "query": "health check", "response": "ok"}])

        # Read
        conv = get_conversation(test_id)
        if not conv:
            raise Exception("Read failed: conversation not found")

        # Delete
        delete_conversation(test_id)

        # Verify delete
        conv_after = get_conversation(test_id)
        if conv_after:
            raise Exception("Delete failed: conversation still exists")

        return {
            "status": "success",
            "test_id": test_id,
            "write": True,
            "read": True,
            "delete": True,
            "latency_ms": None,  # Could measure if needed
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Redis test failed: {str(e)}")


@app.get("/api/v1/admin/redis/stats")
async def redis_detailed_stats():
    """Get detailed Redis statistics."""
    if not _REDIS_CONVERSATION_AVAILABLE:
        raise HTTPException(status_code=503, detail="Redis not available")

    try:
        stats = redis_store_metrics()
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "redis": stats,
            "local_fallback": {
                "conversations": len(CONVERSATION_STORE),
                "cache_entries": len(_local_response_cache),
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")