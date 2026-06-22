# rag/ingest.py
import os
import fitz  # PyMuPDF
import pdfplumber

def load_pdfs(folder="data/raw_pdfs"):
    """
    Reads all PDFs in the folder and returns:
    [{"doc_id": "...", "text": "...", "source": "..."}]
    """
    docs = []

    for filename in os.listdir(folder):
        if not filename.lower().endswith(".pdf"):
            continue

        path = os.path.join(folder, filename)
        text = ""

        try:
            with pdfplumber.open(path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        except Exception:
            with fitz.open(path) as doc:
                for page in doc:
                    text += page.get_text() + "\n"

        docs.append({
            "doc_id": filename,
            "source": filename,
            "text": text
        })

    return docs
