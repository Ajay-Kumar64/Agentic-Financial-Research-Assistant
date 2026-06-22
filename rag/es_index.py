
# rag/es_index.py
from elasticsearch import Elasticsearch
import os

ES_HOST = os.getenv("ES_HOST", "http://localhost:9200")
INDEX_NAME = "rag_chunks"
es = Elasticsearch(ES_HOST,
    request_timeout=120,   # 2 minutes per request
    max_retries=5,
    retry_on_timeout=True,
    headers={"Accept": "application/vnd.elasticsearch+json; compatible-with=8"}
                   )

def create_index():
    if es.indices.exists(INDEX_NAME):
        es.indices.delete(index=INDEX_NAME)
    es.indices.create(index=INDEX_NAME, body={
        "settings": {
            "analysis": {"analyzer": {"default": {"type": "english"}}}
        },
        "mappings": {
            "properties": {
                "text": {"type": "text"},
                "doc_id": {"type": "keyword"},
                "chunk_id": {"type": "keyword"},
                "page": {"type": "integer"},
                "section": {"type": "keyword"}
            }
        }
    })

def index_chunks(chunks):
    for chunk in chunks:
        es.index(index=INDEX_NAME, id=chunk["chunk_id"], document=chunk)

def bm25_search(query, k=10):
    res = es.search(index=INDEX_NAME, query={"match": {"text": query}}, size=k)
    results = []
    for hit in res["hits"]["hits"]:
        source = hit["_source"]
        source["score"] = float(hit.get("_score", 0.0))
        results.append(source)
    return results
