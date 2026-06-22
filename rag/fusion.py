# rag/fusion.py
from collections import defaultdict

def rrf(ranks, k=60):
    """
    Reciprocal Rank Fusion
    ranks: list of dicts {chunk_id: rank_index}
    """
    score = defaultdict(float)
    for rank in ranks:
        for doc, r in rank.items():
            score[doc] += 1.0 / (k + r + 1)
    # Sort descending by score
    return sorted(score.items(), key=lambda x: x[1], reverse=True)
