# rag/cache.py — SAFE REPLACEMENT
# Uses JSON serialization instead of unsafe eval()
# Falls back to in-memory dict if Redis is unavailable

import os
import json
import hashlib
import logging
from typing import Optional, Any

logger = logging.getLogger(__name__)

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))

# =============================================================================
# REDIS CLIENT (with safe fallback)
# =============================================================================
_r = None
_redis_available = False

try:
    import redis
    _r = redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        decode_responses=True,
        socket_connect_timeout=2,
        socket_timeout=2,
    )
    _r.ping()
    _redis_available = True
    logger.info(f"[Cache] Connected to Redis at {REDIS_HOST}:{REDIS_PORT}")
except Exception as e:
    logger.warning(f"[Cache] Redis unavailable, using in-memory fallback: {e}")
    _redis_available = False
    _r = None

# =============================================================================
# IN-MEMORY FALLBACK
# =============================================================================
_memory_cache: dict = {}
_MEMORY_MAX = 100

def _memory_cleanup():
    """Evict oldest entries if over limit."""
    global _memory_cache
    if len(_memory_cache) <= _MEMORY_MAX:
        return
    # Remove oldest 20%
    sorted_keys = sorted(_memory_cache.keys(), key=lambda k: _memory_cache[k].get("_ts", 0))
    for key in sorted_keys[:int(_MEMORY_MAX * 0.2)]:
        del _memory_cache[key]

# =============================================================================
# PUBLIC API
# =============================================================================

def norm(text: str) -> str:
    """Normalize text for cache key generation."""
    return text.lower().strip()


def hash_key(s: str) -> str:
    """Generate a hash key for a string."""
    return hashlib.sha256(s.encode()).hexdigest()


def get_response(query_norm: str) -> Optional[Any]:
    """
    Retrieve a cached response for a normalized query.
    Returns None if not found or expired.
    """
    key = f"resp:{hash_key(query_norm)}"

    # Try Redis first
    if _redis_available and _r:
        try:
            data = _r.get(key)
            if data:
                # SAFE: Use json.loads instead of eval()
                entry = json.loads(data)
                return entry.get("value")
        except Exception as e:
            logger.warning(f"[Cache] Redis get failed: {e}")

    # Fallback to memory
    entry = _memory_cache.get(key)
    if entry:
        if entry.get("expires", 0) > __import__("time").time():
            return entry.get("value")
        # Expired — clean up
        del _memory_cache[key]

    return None


def put_response(query_norm: str, value: Any, ttl: int = 259200) -> bool:
    """
    Cache a response for a normalized query.
    Returns True if successful.
    """
    key = f"resp:{hash_key(query_norm)}"

    # SAFE: Use json.dumps instead of str()
    payload = json.dumps({"value": value}, default=str)

    # Try Redis first
    if _redis_available and _r:
        try:
            _r.setex(key, ttl, payload)
            return True
        except Exception as e:
            logger.warning(f"[Cache] Redis set failed: {e}")

    # Fallback to memory
    _memory_cache[key] = {
        "value": value,
        "expires": __import__("time").time() + ttl,
        "_ts": __import__("time").time(),
    }
    _memory_cleanup()
    return True


def delete_response(query_norm: str) -> bool:
    """Delete a cached response."""
    key = f"resp:{hash_key(query_norm)}"

    success = False
    if _redis_available and _r:
        try:
            _r.delete(key)
            success = True
        except Exception as e:
            logger.warning(f"[Cache] Redis delete failed: {e}")

    if key in _memory_cache:
        del _memory_cache[key]
        success = True

    return success


def cache_status() -> dict:
    """Get cache status for health checks."""
    return {
        "redis_available": _redis_available,
        "redis_host": REDIS_HOST,
        "redis_port": REDIS_PORT,
        "memory_entries": len(_memory_cache),
    }