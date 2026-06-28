#!/usr/bin/env python3
"""async_test.py — Definitive async concurrency test."""
import asyncio
import time
import httpx

BASE_URL = "http://localhost:8000"
QUERIES = [
    ("What is the current repo rate?", "conc-0"),
    ("Calculate the percentage increase from 4.0 to 6.5", "conc-1"),
    ("What is GDP growth outlook?", "conc-2"),
]

async def send_one(client: httpx.AsyncClient, query: str, cid: str):
    t0 = time.perf_counter()
    r = await client.post(
        f"{BASE_URL}/api/v1/chat",
        json={"message": query, "conversation_id": cid},
        timeout=60
    )
    elapsed = time.perf_counter() - t0
    data = r.json()
    return {
        "cid": cid,
        "status": r.status_code,
        "elapsed": round(elapsed, 2),
        "preview": data.get("response", "")[:80],
        "cache_hit": data.get("metadata", {}).get("cache") == "hit"
    }

async def main():
    print("=" * 60)
    print("ASYNC CONCURRENCY TEST")
    print("=" * 60)

    async with httpx.AsyncClient() as client:
        # 1. Single baseline
        print("\n[1] Single request baseline...")
        single = await send_one(client, QUERIES[0][0], "baseline-0")
        print(f"    {single['elapsed']}s | {single['preview']}")

        # 2. Concurrent burst (fire all at once, wait for all to finish)
        print(f"\n[2] Firing {len(QUERIES)} requests CONCURRENTLY...")
        wall_t0 = time.perf_counter()
        results = await asyncio.gather(*[
            send_one(client, q, cid) for q, cid in QUERIES
        ])
        wall_total = time.perf_counter() - wall_t0

    # Print results
    for r in results:
        cache_str = "[CACHE]" if r["cache_hit"] else "[LIVE]"
        print(f"    {r['elapsed']}s {cache_str} | {r['preview']}")

    # Analysis
    max_single = max(r["elapsed"] for r in results)
    ratio = round(wall_total / single["elapsed"], 2) if single["elapsed"] > 0 else 0

    print("\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60)
    print(f"Single request time:  {single['elapsed']}s")
    print(f"Total wall time:        {round(wall_total, 2)}s")
    print(f"Slowest request:      {max_single}s")
    print(f"Concurrency ratio:    {ratio}x")

    if ratio < 1.5:
        print("\n✅ PASS: Requests are CONCURRENT. Event loop is NOT blocked.")
    elif ratio < 2.5:
        print("\n⚠️  WARNING: Partial concurrency. Check thread pool limits.")
    else:
        print("\n❌ FAIL: Requests are SEQUENTIAL. Event loop IS blocked.")
        print("   -> Check if uvicorn is running with enough workers")
        print("   -> Try: docker exec financial-agent ps aux | grep uvicorn")

if __name__ == "__main__":
    asyncio.run(main())