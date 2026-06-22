# rag/reranker_loader.py
import pickle
from FlagEmbedding import FlagReranker

CONFIG_PATH = "artifacts/bge_reranker/config.pkl"

with open(CONFIG_PATH, "rb") as f:
    cfg = pickle.load(f)

reranker = FlagReranker(
    cfg["model_name"],
    use_fp16=True
)

def rerank(query: str, docs: list):
    """
    Rerank retrieved documents
    """
    pairs = [[query, d["text"]] for d in docs]
    scores = reranker.compute_score(pairs)

    for d, s in zip(docs, scores):
        d["rerank_score"] = float(s)

    return sorted(docs, key=lambda x: x["rerank_score"], reverse=True)
