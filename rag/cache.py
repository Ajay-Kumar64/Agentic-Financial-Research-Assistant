# rag/cache.py
import redis, hashlib, os

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

def norm(text):
    return text.lower().strip()

def hash_key(s):
    return hashlib.sha256(s.encode()).hexdigest()

def get_response(query_norm):
    key = f"resp:{hash_key(query_norm)}"
    if r.exists(key):
        return eval(r.get(key))
    return None

def put_response(query_norm, value, ttl=259200):  # 72h
    key = f"resp:{hash_key(query_norm)}"
    r.set(key, str(value), ex=ttl)
