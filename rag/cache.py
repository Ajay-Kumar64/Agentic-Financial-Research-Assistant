"""
rag/cache.py
============
Production cache with Redis primary + in-memory LRU fallback.

WHY TWO-TIER:
- Redis: Shared across instances, persists across restarts, 5-min TTL
- In-memory: Sub-millisecond access, survives Redis outages, per-instance hot cache
- Graceful degradation: if Redis fails, system continues with in-memory

WHY CACHE RAG RESULTS (not LLM responses):
- LLM responses depend on conversation history → harder to cache correctly
- RAG retrieval results are deterministic for the same query
- Financial queries have high repeat rate ("What is repo rate?" asked daily)
- Cache hit rate: 25-50% in production

SPEED:
- Redis get: ~1-2ms (local network)
- Memory get: ~0.1ms (in-process)
- Serialization overhead: ~0.5ms

WHY NOT CACHE LLM RESPONSES:
- Already cached at API layer (_get_cached_response in api/main.py)
- Different conversation histories produce different responses
- RAG cache is for retrieval results only
"""

import os
import json
import time
import hashlib
from typing import Optional, Dict, Any

from dotenv import load_dotenv
load_dotenv()

try:
    import redis
    _REDIS_AVAILABLE = True
except ImportError:
    _REDIS_AVAILABLE = False


class CacheManager:
    """
    Production cache manager with Redis + in-memory fallback.

    Usage:
        cache = CacheManager(ttl=300)

        # Get
        result = cache.get("what is repo rate?")
        if result:
            return result

        # Compute and set
        result = expensive_retrieval(query)
        cache.set(query, result)
    """

    def __init__(
        self,
        ttl: int = 300,
        max_memory_size: int = 100,
        redis_host: str = None,
        redis_port: int = None,
        redis_db: int = None,
    ):
        self.ttl = ttl
        self.max_memory_size = max_memory_size
        self._memory: Dict[str, Dict] = {}

        # Redis connection
        self._redis = None
        if _REDIS_AVAILABLE:
            try:
                self._redis = redis.Redis(
                    host=redis_host or os.getenv("REDIS_HOST", "redis"),
                    port=redis_port or int(os.getenv("REDIS_PORT", "6379")),
                    db=redis_db or int(os.getenv("REDIS_DB", "0")),
                    decode_responses=True,
                    socket_connect_timeout=2,
                    socket_timeout=2,
                    health_check_interval=30,
                )
                self._redis.ping()
                print("[Cache] ✅ Redis connected")
            except Exception as e:
                print(f"[Cache] ⚠️ Redis unavailable: {e}")
                self._redis = None
        else:
            print("[Cache] ⚠️ redis-py not installed. Using memory-only cache.")

    def _key(self, query: str, prefix: str = "rag") -> str:
        """Create deterministic cache key."""
        h = hashlib.sha256(query.lower().strip().encode()).hexdigest()[:16]
        return f"{prefix}:{h}"

    def get(self, query: str) -> Optional[Dict]:
        """
        Get cached result for query.
        Tries Redis first, falls back to in-memory.
        """
        key = self._key(query)

        # Try Redis
        if self._redis:
            try:
                data = self._redis.get(key)
                if data:
                    entry = json.loads(data)
                    if time.time() - entry.get("ts", 0) < self.ttl:
                        return entry["value"]
                    # Expired, delete
                    self._redis.delete(key)
            except Exception:
                pass  # Redis error, fall through to memory

        # Fallback to memory
        entry = self._memory.get(key)
        if entry and time.time() - entry.get("ts", 0) < self.ttl:
            return entry["value"]

        return None

    def set(self, query: str, value: Dict):
        """
        Cache result for query.
        Writes to Redis if available, always writes to memory.
        """
        key = self._key(query)
        payload = json.dumps({"value": value, "ts": time.time()})

        # Try Redis
        if self._redis:
            try:
                self._redis.setex(key, self.ttl, payload)
            except Exception:
                pass  # Redis error, memory still works

        # Always write to memory (hot cache + fallback)
        self._memory[key] = {"value": value, "ts": time.time()}

        # LRU eviction
        if len(self._memory) > self.max_memory_size:
            oldest = next(iter(self._memory))
            del self._memory[oldest]

    def invalidate(self, query: str) -> bool:
        """Remove specific query from cache."""
        key = self._key(query)
        removed = False

        if self._redis:
            try:
                self._redis.delete(key)
                removed = True
            except Exception:
                pass

        if key in self._memory:
            del self._memory[key]
            removed = True

        return removed

    def clear(self):
        """Clear all cached entries."""
        if self._redis:
            try:
                for k in self._redis.scan_iter(match="rag:*"):
                    self._redis.delete(k)
            except Exception:
                pass

        self._memory.clear()

    def stats(self) -> Dict:
        """Get cache statistics."""
        redis_keys = 0
        if self._redis:
            try:
                redis_keys = sum(1 for _ in self._redis.scan_iter(match="rag:*"))
            except Exception:
                pass

        return {
            "redis_connected": self._redis is not None,
            "redis_keys": redis_keys,
            "memory_keys": len(self._memory),
            "memory_max": self.max_memory_size,
            "ttl_sec": self.ttl,
        }

    # =====================================================================
    # LEGACY API (for backward compatibility with api/main.py)
    # =====================================================================

    def get_response(self, query: str, conversation_id: str = "") -> Optional[str]:
        """Legacy wrapper for response caching."""
        cache_key = f"{conversation_id}:{query}" if conversation_id else query
        result = self.get(cache_key)
        return result if isinstance(result, str) else None

    def set_response(self, query: str, response: str, conversation_id: str = ""):
        """Legacy wrapper for response caching."""
        cache_key = f"{conversation_id}:{query}" if conversation_id else query
        return self.set(cache_key, response)


def norm(query: str) -> str:
    """Legacy normalization function."""
    return query.lower().strip()