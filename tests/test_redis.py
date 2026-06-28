#!/usr/bin/env python3
"""
Redis Diagnostic Suite for Agentic Financial Research Assistant
Checks: connectivity, module imports, docker-compose wiring, and fallbacks.
"""

import os
import sys
import json
import time
import traceback

print("=" * 70)
print("REDIS DIAGNOSTIC SUITE")
print("=" * 70)

# =============================================================================
# TEST 1: Environment Variables
# =============================================================================
print("\n[TEST 1] Environment Variables")
print("-" * 50)

env_vars = {
    "REDIS_URL": os.getenv("REDIS_URL", "NOT SET"),
    "REDIS_HOST": os.getenv("REDIS_HOST", "NOT SET (default: localhost)"),
    "REDIS_PORT": os.getenv("REDIS_PORT", "NOT SET (default: 6379)"),
    "REDIS_DB": os.getenv("REDIS_DB", "NOT SET (default: 0)"),
    "REDIS_PASSWORD": "***SET***" if os.getenv("REDIS_PASSWORD") else "NOT SET",
    "CONVERSATION_TTL_SECONDS": os.getenv("CONVERSATION_TTL_SECONDS", "NOT SET (default: 259200 = 72h)"),
}

for k, v in env_vars.items():
    print(f"  {k}: {v}")

# =============================================================================
# TEST 2: Redis Package Installation
# =============================================================================
print("\n[TEST 2] Redis Package")
print("-" * 50)

try:
    import redis
    print(f"  ✅ redis package installed: {redis.__version__}")
except ImportError as e:
    print(f"  ❌ redis package NOT installed: {e}")
    print("     Fix: pip install redis>=5.0.0")
    sys.exit(1)

# =============================================================================
# TEST 3: Direct Redis Connection
# =============================================================================
print("\n[TEST 3] Direct Redis Connection")
print("-" * 50)

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_URL = os.getenv("REDIS_URL", "")

direct_client = None
try:
    if REDIS_URL:
        direct_client = redis.from_url(
            REDIS_URL,
            decode_responses=True,
            socket_connect_timeout=3,
            socket_timeout=3,
        )
    else:
        direct_client = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            decode_responses=True,
            socket_connect_timeout=3,
            socket_timeout=3,
        )

    t0 = time.time()
    direct_client.ping()
    latency = round((time.time() - t0) * 1000, 2)
    print(f"  ✅ PING successful")
    print(f"     Host: {REDIS_HOST}:{REDIS_PORT}")
    print(f"     Latency: {latency}ms")

    info = direct_client.info("server")
    print(f"     Redis Version: {info.get('redis_version', 'unknown')}")
    print(f"     Mode: {info.get('redis_mode', 'unknown')}")

except Exception as e:
    print(f"  ❌ Connection FAILED: {e}")
    print(f"     Host attempted: {REDIS_HOST}:{REDIS_PORT}")
    print(f"     If using Docker, ensure Redis service is running and accessible")
    direct_client = None

# =============================================================================
# TEST 4: agent/redis_store.py Module
# =============================================================================
print("\n[TEST 4] agent/redis_store.py Module")
print("-" * 50)

store_health = None
try:
    # Change to project root if needed
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if os.path.basename(script_dir) != "Agentic-Financial-Research-Assistant":
        # Try to find project root
        for parent in [script_dir, os.path.dirname(script_dir), os.path.dirname(os.path.dirname(script_dir))]:
            if os.path.exists(os.path.join(parent, "agent", "redis_store.py")):
                sys.path.insert(0, parent)
                print(f"  📁 Added to path: {parent}")
                break

    from agent.redis_store import (
        health_check,
        save_conversation,
        get_conversation,
        delete_conversation,
        list_conversation_ids,
        get_cached_response,
        set_cached_response,
        get_store_metrics,
        _redis_available,
        _redis_client,
    )

    print("  ✅ Module imports successfully")

    # Run health check
    store_health = health_check()
    print(f"  ✅ health_check() returned:")
    for k, v in store_health.items():
        print(f"     {k}: {v}")

    # Check internal state
    print(f"  📊 Internal _redis_available: {_redis_available}")
    print(f"  📊 Internal _redis_client: {'SET' if _redis_client else 'None'}")

except Exception as e:
    print(f"  ❌ Module import/execution FAILED")
    print(f"     Error: {e}")
    traceback.print_exc()

# =============================================================================
# TEST 5: CRUD Operations on redis_store.py
# =============================================================================
print("\n[TEST 5] CRUD Operations via redis_store.py")
print("-" * 50)

if store_health and store_health.get("redis_available"):
    test_conv_id = f"test-diag-{int(time.time())}"

    try:
        # CREATE
        save_result = save_conversation(
            conversation_id=test_conv_id,
            history=[
                {"turn": 1, "query": "What is repo rate?", "response": "6.5%", "tools_used": ["rag_search"]}
            ],
            last_state={"test": True},
            metadata={"source": "diagnostic"}
        )
        print(f"  ✅ CREATE: save_conversation returned {save_result}")

        # READ
        conv = get_conversation(test_conv_id)
        if conv:
            print(f"  ✅ READ: get_conversation found conv with {len(conv.get('history', []))} turns")
        else:
            print(f"  ❌ READ: get_conversation returned None")

        # UPDATE (append_turn)
        from agent.redis_store import append_turn
        append_result = append_turn(
            conversation_id=test_conv_id,
            query="And previous year?",
            response="4.0%",
            tools_used=["rag_search"],
            state_snapshot={"test": True, "updated": True}
        )
        print(f"  ✅ UPDATE: append_turn returned {append_result}")

        # Verify update
        conv2 = get_conversation(test_conv_id)
        if conv2:
            print(f"  ✅ VERIFY: conv now has {len(conv2.get('history', []))} turns")

        # LIST
        ids = list_conversation_ids(limit=10)
        print(f"  ✅ LIST: found {len(ids)} conversation IDs")

        # DELETE
        del_result = delete_conversation(test_conv_id)
        print(f"  ✅ DELETE: delete_conversation returned {del_result}")

        # Verify delete
        conv3 = get_conversation(test_conv_id)
        if conv3 is None:
            print(f"  ✅ VERIFY: conversation properly deleted")
        else:
            print(f"  ⚠️  VERIFY: conversation still exists after delete (TTL may apply)")

    except Exception as e:
        print(f"  ❌ CRUD test FAILED: {e}")
        traceback.print_exc()
else:
    print("  ⏭️  SKIPPED: Redis not available")

# =============================================================================
# TEST 6: Response Cache Operations
# =============================================================================
print("\n[TEST 6] Response Cache via redis_store.py")
print("-" * 50)

if store_health and store_health.get("redis_available"):
    test_query = "test query for cache diagnostic"
    test_response = "This is a cached test response"

    try:
        # SET cache
        cache_set = set_cached_response(test_query, test_response, conversation_id="diag")
        print(f"  ✅ SET: set_cached_response returned {cache_set}")

        # GET cache
        cached = get_cached_response(test_query, conversation_id="diag")
        if cached == test_response:
            print(f"  ✅ GET: cache hit, response matches")
        elif cached:
            print(f"  ⚠️  GET: cache hit but response mismatch: {cached[:50]}")
        else:
            print(f"  ❌ GET: cache miss (returned None)")

        # GET with wrong conversation_id (should miss)
        cached_wrong = get_cached_response(test_query, conversation_id="wrong")
        if cached_wrong is None:
            print(f"  ✅ ISOLATION: wrong conv_id correctly returns None")
        else:
            print(f"  ⚠️  ISOLATION: wrong conv_id returned data (cache key collision?)")

    except Exception as e:
        print(f"  ❌ Cache test FAILED: {e}")
        traceback.print_exc()
else:
    print("  ⏭️  SKIPPED: Redis not available")

# =============================================================================
# TEST 7: api/main.py Redis Integration
# =============================================================================
print("\n[TEST 7] api/main.py Integration Check")
print("-" * 50)

try:
    # We can't fully import FastAPI app here without all deps, but we can check the import logic
    print("  Checking if api/main.py would successfully import redis_store...")

    # Simulate the import block from api/main.py
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
        print("  ✅ api/main.py import block would SUCCEED")
    except Exception as e:
        _REDIS_CONVERSATION_AVAILABLE = False
        print(f"  ❌ api/main.py import block would FAIL: {e}")

    # Check legacy rag/cache.py
    print("\n  Checking legacy rag/cache.py...")
    try:
        from rag.cache import get_response as redis_get, put_response as redis_put, norm
        print("  ⚠️  rag/cache.py imports successfully (BUT uses unsafe eval()!)")
        print("     Recommendation: Remove this module, use agent/redis_store.py instead")
    except Exception as e:
        print(f"  ℹ️  rag/cache.py import failed (expected if Redis down): {e}")

except Exception as e:
    print(f"  ❌ Integration check FAILED: {e}")

# =============================================================================
# TEST 8: Docker Compose Configuration Check
# =============================================================================
print("\n[TEST 8] Docker Compose Configuration")
print("-" * 50)

docker_compose_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docker-compose.yml")
if not os.path.exists(docker_compose_path):
    # Try parent directories
    for parent in [os.path.dirname(os.path.abspath(__file__)),
                   os.path.dirname(os.path.dirname(os.path.abspath(__file__)))]:
        candidate = os.path.join(parent, "docker-compose.yml")
        if os.path.exists(candidate):
            docker_compose_path = candidate
            break

if os.path.exists(docker_compose_path):
    print(f"  📁 Found: {docker_compose_path}")
    with open(docker_compose_path, "r") as f:
        content = f.read()

    has_redis_service = "redis:" in content or "redis" in content
    has_redis_url = "REDIS_URL" in content
    has_redis_depends = "depends_on" in content and "redis" in content

    if has_redis_service:
        print("  ✅ Redis service defined in docker-compose.yml")
    else:
        print("  ❌ Redis service NOT found in docker-compose.yml")
        print("     Fix: Add redis service and REDIS_URL env var to agent service")

    if has_redis_url:
        print("  ✅ REDIS_URL environment variable configured")
    else:
        print("  ❌ REDIS_URL not found in docker-compose.yml")

    if has_redis_depends:
        print("  ✅ Agent service has depends_on for Redis")
    else:
        print("  ⚠️  Agent service may not depend on Redis (startup order issues possible)")
else:
    print(f"  ❌ docker-compose.yml not found")

# =============================================================================
# TEST 9: requirements.txt Check
# =============================================================================
print("\n[TEST 9] requirements.txt")
print("-" * 50)

req_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "requirements.txt")
if not os.path.exists(req_path):
    for parent in [os.path.dirname(os.path.abspath(__file__)),
                   os.path.dirname(os.path.dirname(os.path.abspath(__file__)))]:
        candidate = os.path.join(parent, "requirements.txt")
        if os.path.exists(candidate):
            req_path = candidate
            break

if os.path.exists(req_path):
    with open(req_path, "r") as f:
        reqs = f.read()
    if "redis" in reqs.lower():
        print("  ✅ redis package listed in requirements.txt")
    else:
        print("  ❌ redis package NOT in requirements.txt")
else:
    print("  ⚠️  requirements.txt not found")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

issues = []
recommendations = []

if not direct_client:
    issues.append("Redis server is not reachable")
    recommendations.append("Start Redis: docker run -d -p 6379:6379 redis:7-alpine")
    recommendations.append("Or add 'redis' service to docker-compose.yml")

if store_health and not store_health.get("redis_available"):
    issues.append("agent/redis_store.py cannot connect to Redis")
    recommendations.append("Check REDIS_HOST/REDIS_PORT env vars")
    recommendations.append("If in Docker, use REDIS_URL=redis://redis:6379/0")

try:
    from rag.cache import get_response
    issues.append("Legacy rag/cache.py still exists (uses unsafe eval())")
    recommendations.append("Delete rag/cache.py and migrate to agent/redis_store.py")
except:
    pass

print(f"\nIssues found: {len(issues)}")
for i, issue in enumerate(issues, 1):
    print(f"  {i}. ❌ {issue}")

if not issues:
    print("\n  ✅ All checks passed! Redis is properly configured.")
else:
    print(f"\nRecommendations:")
    for rec in recommendations:
        print(f"  → {rec}")

print("\n" + "=" * 70)