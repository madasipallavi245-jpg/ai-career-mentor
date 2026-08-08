import os
import fitz
import docx
import faiss
import numpy as np
from io import BytesIO
from typing import List
import requests

HF_TOKEN = os.environ.get("HF_TOKEN", "")
EMB_API_URL = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384  # all-MiniLM-L6-v2 output size

# Real FAISS vector store
_chunks = []
_faiss_index = None
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
    global _chunks, _faiss_index, _is_indexed
    try:
        text = extract_text_from_file(uploaded_file)
        if not text.strip():
            return "❌ Could not extract text."

        _chunks = split_into_chunks(text)
        embeddings = get_embedding(_chunks)
        emb_matrix = np.array(embeddings, dtype="float32")

        # Embeddings from get_embedding() are already L2-normalized,
        # so inner product = cosine similarity. IndexFlatIP is FAISS's
        # exact (brute-force) index — fine at this chunk-count scale.
        _faiss_index = faiss.IndexFlatIP(EMBEDDING_DIM)
        _faiss_index.add(emb_matrix)

        _is_indexed = True
        return f"✅ Resume processed! {len(_chunks)} sections indexed with FAISS."
    except Exception as e:
        return f"❌ Error: {str(e)}"

def get_relevant_context(question: str, k: int = 3) -> str:
    global _chunks, _faiss_index
    if not _chunks or _faiss_index is None:
        return ""
    try:
        q_emb = np.array(get_embedding([question]), dtype="float32")
        k = min(k, len(_chunks))
        distances, indices = _faiss_index.search(q_emb, k)
        top_chunks = [_chunks[i] for i in indices[0] if i != -1]
        return "\n\n".join(top_chunks)
    except Exception:
        return "\n\n".join(_chunks[:k])

def has_document() -> bool:
    return _is_indexed

def reset_vector_store():
    global _chunks, _faiss_index, _is_indexed
    _chunks = []
    _faiss_index = None
    _is_indexed = False