#------------------------
#Using Colab
#---------------------
import pickle, faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ---- Load chunks ----
with open("/content/colab_chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

texts = [c["text"] for c in chunks]

# ---- Dense embeddings ----
encoder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
emb = encoder.encode(
    texts,
    normalize_embeddings=True,
    show_progress_bar=True
).astype("float32")

# ---- FAISS index ----
index = faiss.IndexHNSWFlat(emb.shape[1], 32)
index.hnsw.efSearch = 64
index.add(emb)

faiss.write_index(index, f"{ART}/faiss_index/index.faiss")

with open(f"{ART}/faiss_index/meta.pkl", "wb") as f:
    pickle.dump(chunks, f)

# ---- Reranker config ----
pickle.dump(
    {"model_name": "BAAI/bge-reranker-large"},
    open(f"{ART}/bge_reranker/config.pkl", "wb")
)

# ---- LLaMA config ----
pickle.dump(
    {"model_name": "TheBloke/Llama-2-7B-Chat-GGUF"},
    open(f"{ART}/llama_model/config.pkl", "wb")
)

print("Artifacts generated from colab_chunks.pkl")
