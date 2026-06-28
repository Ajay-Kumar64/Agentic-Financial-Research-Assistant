"""
agent/redis_store.py — Redis-backed conversation store with connection pooling,
retry logic, and in-memory fallback.

Features:
- Connection pooling (max 20 connections)
- Exponential backoff retry (3 attempts)
- Persistent storage with TTL (default 72h)
- Auto-fallback to in-memory dict if Redis is unavailable
- Health check with latency measurement
- Sliding window history management
- Thread-safe operations
"""

import os
import json
import time
import hashlib
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from functools import wraps
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================
REDIS_URL = os.getenv("REDIS_URL", "")
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB = int(os.getenv("REDIS_DB", 0))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)
CONVERSATION_TTL_SECONDS = int(os.getenv("CONVERSATION_TTL_SECONDS", 259200))  # 72h
REDIS_MAX_CONNECTIONS = int(os.getenv("REDIS_MAX_CONNECTIONS", 20))

# =============================================================================
# CONNECTION POOL (singleton)
# =============================================================================
_pool = None
_redis_available = False


def _get_pool():
    """Lazy initializer for Redis connection pool."""
    global _pool
    if _pool is not None:
        return _pool

    try:
        import redis
        from redis.connection import ConnectionPool

        if REDIS_URL:
            _pool = ConnectionPool.from_url(
                REDIS_URL,
                max_connections=REDIS_MAX_CONNECTIONS,
                socket_connect_timeout=5,
                socket_timeout=5,
                health_check_interval=30,
                decode_responses=True,
            )
        else:
            _pool = ConnectionPool(
                host=REDIS_HOST,
                port=REDIS_PORT,
                db=REDIS_DB,
                password=REDIS_PASSWORD,
                max_connections=REDIS_MAX_CONNECTIONS,
                socket_connect_timeout=5,
                socket_timeout=5,
                health_check_interval=30,
                decode_responses=True,
            )
        logger.info(f"[RedisStore] Connection pool created: max={REDIS_MAX_CONNECTIONS}")
    except Exception as e:
        logger.error(f"[RedisStore] Failed to create connection pool: {e}")
        _pool = None

    return _pool


def _get_redis_client():
    """Get Redis client from pool with health check."""
    global _redis_available

    pool = _get_pool()
    if pool is None:
        _redis_available = False
        return None

    try:
        import redis
        client = redis.Redis(connection_pool=pool)
        client.ping()
        _redis_available = True
        return client
    except Exception as e:
        _redis_available = False
        logger.warning(f"[RedisStore] Redis ping failed: {e}")
        return None


# =============================================================================
# RETRY DECORATOR
# =============================================================================
def _redis_retry(max_retries=3, backoff=0.5):
    """Decorator: retry Redis operations with exponential backoff."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    if attempt < max_retries - 1:
                        sleep_time = backoff * (2 ** attempt)
                        logger.warning(f"[RedisStore] Retry {attempt + 1}/{max_retries} after {sleep_time}s: {e}")
                        time.sleep(sleep_time)
                        # Force reconnect on next attempt
                        global _redis_available
                        _redis_available = False
            logger.error(f"[RedisStore] All retries exhausted: {last_error}")
            raise last_error
        return wrapper
    return decorator


# =============================================================================
# IN-MEMORY FALLBACK STORE
# =============================================================================
_in_memory_store: Dict[str, Dict[str, Any]] = {}
_in_memory_access_times: Dict[str, float] = {}
_MAX_MEMORY_STORE = 500


def _memory_cleanup():
    """Evict oldest conversations if memory store exceeds limit."""
    global _in_memory_store, _in_memory_access_times
    if len(_in_memory_store) <= _MAX_MEMORY_STORE:
        return

    sorted_keys = sorted(
        _in_memory_access_times.keys(),
        key=lambda k: _in_memory_access_times.get(k, 0)
    )
    evict_count = max(1, int(_MAX_MEMORY_STORE * 0.2))
    for key in sorted_keys[:evict_count]:
        _in_memory_store.pop(key, None)
        _in_memory_access_times.pop(key, None)
    logger.info(f"[RedisStore] Memory cleanup: evicted {evict_count} conversations")


# =============================================================================
# CORE OPERATIONS
# =============================================================================

def _key(conversation_id: str) -> str:
    return f"conv:{conversation_id}"


def _hash_query(query: str) -> str:
    return hashlib.sha256(query.lower().strip().encode()).hexdigest()[:16]


def health_check() -> Dict[str, Any]:
    """Check Redis health and return status."""
    result = {
        "redis_available": False,
        "redis_host": REDIS_HOST,
        "redis_port": REDIS_PORT,
        "latency_ms": None,
        "mode": "memory",
        "timestamp": datetime.utcnow().isoformat(),
    }

    client = _get_redis_client()
    if client:
        try:
            t0 = time.time()
            client.ping()
            latency_ms = round((time.time() - t0) * 1000, 2)
            result["redis_available"] = True
            result["latency_ms"] = latency_ms
            result["mode"] = "redis"
            info = client.info("server")
            result["redis_version"] = info.get("redis_version", "unknown")
            result["redis_mode"] = info.get("redis_mode", "unknown")
        except Exception as e:
            result["error"] = str(e)

    return result


@_redis_retry(max_retries=3)
def get_conversation(conversation_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve a conversation by ID. Returns None if not found."""
    global _redis_available

    client = _get_redis_client()
    if client:
        try:
            data = client.get(_key(conversation_id))
            if data:
                client.expire(_key(conversation_id), CONVERSATION_TTL_SECONDS)
                return json.loads(data)
        except Exception as e:
            logger.warning(f"[RedisStore] Redis get failed, falling back to memory: {e}")
            _redis_available = False

    # Fallback to in-memory
    conv = _in_memory_store.get(conversation_id)
    if conv:
        _in_memory_access_times[conversation_id] = time.time()
    return conv


@_redis_retry(max_retries=3)
def save_conversation(
    conversation_id: str,
    history: List[Dict[str, Any]],
    last_state: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> bool:
    """Save or update a conversation. Returns True if successful."""
    global _redis_available

    payload = {
        "conversation_id": conversation_id,
        "history": history,
        "last_state": last_state or {},
        "metadata": metadata or {},
        "updated_at": datetime.utcnow().isoformat(),
        "version": 2,
    }

    client = _get_redis_client()
    if client:
        try:
            client.setex(
                _key(conversation_id),
                CONVERSATION_TTL_SECONDS,
                json.dumps(payload, default=str)
            )
            return True
        except Exception as e:
            logger.warning(f"[RedisStore] Redis save failed, falling back to memory: {e}")
            _redis_available = False

    # Fallback to in-memory
    _in_memory_store[conversation_id] = payload
    _in_memory_access_times[conversation_id] = time.time()
    _memory_cleanup()
    return True


@_redis_retry(max_retries=3)
def delete_conversation(conversation_id: str) -> bool:
    """Delete a conversation by ID."""
    global _redis_available

    success = False
    client = _get_redis_client()
    if client:
        try:
            client.delete(_key(conversation_id))
            success = True
        except Exception as e:
            logger.warning(f"[RedisStore] Redis delete failed: {e}")

    _in_memory_store.pop(conversation_id, None)
    _in_memory_access_times.pop(conversation_id, None)
    return success


@_redis_retry(max_retries=3)
def list_conversation_ids(limit: int = 100) -> List[str]:
    """List recent conversation IDs."""
    client = _get_redis_client()
    if client:
        try:
            ids = []
            for key in client.scan_iter(match="conv:*", count=100):
                ids.append(key.replace("conv:", ""))
                if len(ids) >= limit:
                    break
            return ids[:limit]
        except Exception as e:
            logger.warning(f"[RedisStore] Redis scan failed: {e}")

    # Fallback: return in-memory IDs sorted by access time
    sorted_ids = sorted(
        _in_memory_access_times.keys(),
        key=lambda k: _in_memory_access_times.get(k, 0),
        reverse=True
    )
    return sorted_ids[:limit]


def append_turn(
    conversation_id: str,
    query: str,
    response: str,
    tools_used: List[str] = None,
    state_snapshot: Optional[Dict[str, Any]] = None,
) -> bool:
    """Append a new turn to an existing conversation."""
    conv = get_conversation(conversation_id)

    if conv is None:
        conv = {
            "conversation_id": conversation_id,
            "history": [],
            "last_state": {},
            "metadata": {"created_at": datetime.utcnow().isoformat()},
        }

    history = conv.get("history", [])
    turn_number = len(history) + 1

    history.append({
        "turn": turn_number,
        "query": query,
        "response": response[:500],
        "tools_used": tools_used or [],
        "timestamp": datetime.utcnow().isoformat(),
    })

    # Sliding window: keep last 10 turns
    if len(history) > 10:
        history = history[-10:]

    return save_conversation(
        conversation_id=conversation_id,
        history=history,
        last_state=state_snapshot,
        metadata=conv.get("metadata", {}),
    )


# =============================================================================
# RESPONSE CACHE
# =============================================================================

@_redis_retry(max_retries=2)
def get_cached_response(query: str, conversation_id: str = "") -> Optional[str]:
    """Get a cached response for a query."""
    cache_key = f"cache:{conversation_id}:{_hash_query(query)}"

    client = _get_redis_client()
    if client:
        try:
            data = client.get(cache_key)
            if data:
                entry = json.loads(data)
                if time.time() - entry.get("ts", 0) < 300:
                    return entry.get("response")
                client.delete(cache_key)
        except Exception:
            pass

    return None


@_redis_retry(max_retries=2)
def set_cached_response(
    query: str,
    response: str,
    conversation_id: str = "",
    ttl_seconds: int = 300
) -> bool:
    """Cache a response for a query."""
    cache_key = f"cache:{conversation_id}:{_hash_query(query)}"
    payload = json.dumps({"response": response, "ts": time.time()})

    client = _get_redis_client()
    if client:
        try:
            client.setex(cache_key, ttl_seconds, payload)
            return True
        except Exception:
            pass

    return False


# =============================================================================
# STATISTICS / METRICS
# =============================================================================

def get_store_metrics() -> Dict[str, Any]:
    """Get store statistics for monitoring."""
    metrics = {
        "mode": "redis" if _redis_available else "memory",
        "memory_conversations": len(_in_memory_store),
        "ttl_seconds": CONVERSATION_TTL_SECONDS,
        "pool_max_connections": REDIS_MAX_CONNECTIONS,
    }

    client = _get_redis_client()
    if client:
        try:
            info = client.info()
            metrics["redis_used_memory_human"] = info.get("used_memory_human", "unknown")
            metrics["redis_connected_clients"] = info.get("connected_clients", 0)
            metrics["redis_uptime_in_seconds"] = info.get("uptime_in_seconds", 0)
            pool = _get_pool()
            if pool:
                metrics["pool_in_use"] = len(pool._in_use_connections) if hasattr(pool, '_in_use_connections') else 'unknown'
                metrics["pool_available"] = len(pool._available_connections) if hasattr(pool, '_available_connections') else 'unknown'
        except Exception as e:
            metrics["redis_error"] = str(e)

    return metrics


# =============================================================================
# INITIALIZATION
# =============================================================================
def initialize():
    """Initialize the Redis store on startup."""
    health = health_check()
    logger.info(f"[RedisStore] Initialized — mode={health['mode']}, available={health['redis_available']}")
    return health