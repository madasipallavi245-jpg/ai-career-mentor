import os
import fitz
import docx
import numpy as np
from io import BytesIO
from typing import List
import requests

HF_TOKEN = os.environ.get("HF_TOKEN", "")
EMB_API_URL = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"

# Simple in-memory vector store without FAISS
_chunks = []
_embeddings_cache = []
_is_indexed = False

def get_embedding(texts: List[str]) -> List[List[float]]:
    try:
        headers = {"Authorization": f"Bearer {HF_TOKEN}"}
        response = requests.post(
            EMB_API_URL,
            headers=headers,
            json={"inputs": texts, "options": {"wait_for_model": True}},
            timeout=30
        )
        if response.status_code == 200:
            result = response.json()
            embeddings = []
            for emb in result:
                arr = np.array(emb)
                norm = np.linalg.norm(arr)
                if norm > 0:
                    arr = arr / norm
                embeddings.append(arr.tolist())
            return embeddings
        return [[0.0] * 384 for _ in texts]
    except Exception:
        return [[0.0] * 384 for _ in texts]

def extract_text_from_file(uploaded_file) -> str:
    ext = uploaded_file.name.split(".")[-1].lower()
    if ext == "pdf":
        pdf = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        text = "".join([pdf.load_page(i).get_text() for i in range(len(pdf))])
        pdf.close()
        return text
    elif ext == "txt":
        return uploaded_file.read().decode("utf-8", errors="ignore")
    elif ext == "docx":
        doc = docx.Document(BytesIO(uploaded_file.read()))
        return "\n".join([p.text for p in doc.paragraphs])
    else:
        raise ValueError(f"Unsupported: .{ext}")

def split_into_chunks(text: str) -> list:
    words = text.split()
    chunks = []
    chunk_size = 100  # words per chunk
    overlap = 20
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks

def process_uploaded_file(uploaded_file) -> str:
    global _chunks, _embeddings_cache, _is_indexed
    try:
        text = extract_text_from_file(uploaded_file)
        if not text.strip():
            return "❌ Could not extract text."
        _chunks = split_into_chunks(text)
        _embeddings_cache = get_embedding(_chunks)
        _is_indexed = True
        return f"✅ Resume processed! {len(_chunks)} sections indexed."
    except Exception as e:
        return f"❌ Error: {str(e)}"

def get_relevant_context(question: str, k: int = 3) -> str:
    global _chunks, _embeddings_cache
    if not _chunks or not _embeddings_cache:
        return ""
    try:
        q_emb = np.array(get_embedding([question])[0])
        similarities = []
        for i, chunk_emb in enumerate(_embeddings_cache):
            c_emb = np.array(chunk_emb)
            sim = np.dot(q_emb, c_emb)
            similarities.append((sim, i))
        similarities.sort(reverse=True)
        top_chunks = [_chunks[i] for _, i in similarities[:k]]
        return "\n\n".join(top_chunks)
    except Exception:
        return "\n\n".join(_chunks[:k])

def has_document() -> bool:
    return _is_indexed

def reset_vector_store():
    global _chunks, _embeddings_cache, _is_indexed
    _chunks = []
    _embeddings_cache = []
    _is_indexed = False
