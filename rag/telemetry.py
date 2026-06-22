# rag/telemetry.py
import time, os, json
from datetime import datetime

TELEMETRY_FILE = os.getenv("TELEMETRY_FILE", "telemetry.log")

def log(query, topk, answer):
    entry = {
        "ts": datetime.utcnow().isoformat(),
        "query": query,
        "topk_chunks": [c["chunk_id"] for c in topk],
        "answer": answer["text"],
        "tokens_in": len(query.split()),
        "tokens_out": len(answer["text"].split()),
        "status": "ok"
    }
    with open(TELEMETRY_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")
