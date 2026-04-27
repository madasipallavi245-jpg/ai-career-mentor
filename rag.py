import os
import fitz
import docx
import numpy as np
from io import BytesIO
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
import requests

HF_TOKEN = os.environ.get("HF_TOKEN", "")
EMB_API_URL = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"

class HFAPIEmbeddings(Embeddings):
    def __init__(self):
        self.headers = {"Authorization": f"Bearer {HF_TOKEN}"}

    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        try:
            response = requests.post(
                EMB_API_URL,
                headers=self.headers,
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
            else:
                # Return zero vectors on error
                return [[0.0] * 384 for _ in texts]
        except Exception:
            return [[0.0] * 384 for _ in texts]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        all_embeddings = []
        for i in range(0, len(texts), 10):
            batch = texts[i:i+10]
            all_embeddings.extend(self._get_embeddings(batch))
        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        return self._get_embeddings([text])[0]

_vector_store = None
_embeddings = None

def load_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = HFAPIEmbeddings()
    return _embeddings

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
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=100,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = splitter.split_text(text)
    return chunks

def build_vector_store(chunks: list):
    global _vector_store
    emb = load_embeddings()
    _vector_store = FAISS.from_texts(chunks, emb)

def process_uploaded_file(uploaded_file) -> str:
    try:
        text = extract_text_from_file(uploaded_file)
        if not text.strip():
            return "❌ Could not extract text."
        chunks = split_into_chunks(text)
        build_vector_store(chunks)
        return f"✅ Resume processed! {len(chunks)} sections indexed."
    except Exception as e:
        return f"❌ Error: {str(e)}"

def get_relevant_context(question: str, k: int = 3) -> str:
    if _vector_store is None:
        return ""
    docs = _vector_store.similarity_search(question, k=k)
    return "\n\n".join([d.page_content for d in docs])

def has_document() -> bool:
    return _vector_store is not None

def reset_vector_store():
    global _vector_store
    _vector_store = None