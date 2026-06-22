# rag/reranker.py
from sentence_transformers import CrossEncoder

class BGEReranker:
    def __init__(self, model_name="BAAI/bge-reranker-large"):
        self.model_name = model_name
        self.model = None

    def load(self):
        if self.model is None:
            print(f"[Reranker] Loading model: {self.model_name}")
            self.model = CrossEncoder(self.model_name)

    def cross_encode(self, query, docs, topn=5):
        """
        docs: list of dicts with 'text'
        returns topn docs sorted by reranker score
        """
        self.load()
        if not docs:
            return []

        texts = [d["text"] for d in docs]
        pairs = [[query, t] for t in texts]
        scores = self.model.predict(pairs)

        # Sort docs by score descending
        sorted_docs = [d for _, d in sorted(zip(scores, docs), key=lambda x: -x[0])]
        return sorted_docs[:topn]
