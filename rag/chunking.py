# rag/chunking.py
import re
from typing import Dict, List

def chunk_text(
    text: str,
    meta: Dict,
    tokens: int = 512,
    overlap: int = 128
) -> List[Dict]:
    """
    Returns list of chunks:
    {
      text,
      doc_id,
      source,
      chunk_id
    }
    """
    words = re.findall(r'\S+\s*', text)
    step = tokens - overlap
    chunks = []

    chunk_idx = 0
    for i in range(0, len(words), step):
        chunk_words = words[i:i + tokens]
        if not chunk_words:
            continue

        chunk_text = ''.join(chunk_words).strip()

        chunks.append({
            "text": chunk_text,
            "doc_id": meta["doc_id"],
            "source": meta["source"],
            "chunk_id": f"{meta['doc_id']}_{chunk_idx}"
        })

        chunk_idx += 1

    return chunks
