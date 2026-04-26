import os, fitz, docx, torch, numpy as np
from io import BytesIO
from typing import List
from transformers import AutoTokenizer, AutoModel
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings.base import Embeddings

class PureTransformerEmbeddings(Embeddings):
    def __init__(self):
        print("⏳ Loading embedding model...")
        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        self.model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        self.model.eval()
        print("✅ Embedding model loaded!")
    def _embed(self, texts):
        enc = self.tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
        with torch.no_grad():
            out = self.model(**enc)
        emb = out.last_hidden_state.mean(dim=1)
        return torch.nn.functional.normalize(emb, p=2, dim=1).numpy()
    def embed_documents(self, texts):
        all_e = []
        for i in range(0, len(texts), 32):
            all_e.extend(self._embed(texts[i:i+32]).tolist())
        return all_e
    def embed_query(self, text):
        return self._embed([text])[0].tolist()

_vector_store = None
_embeddings = None

def load_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = PureTransformerEmbeddings()
    return _embeddings

def extract_text_from_file(uploaded_file):
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

def split_into_chunks(text):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    s = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100, separators=["\n\n","\n","."," ",""])
    chunks = s.split_text(text)
    print(f"✅ {len(chunks)} chunks")
    return chunks

def build_vector_store(chunks):
    global _vector_store
    _vector_store = FAISS.from_texts(chunks, load_embeddings())
    print("✅ FAISS built!")

def process_uploaded_file(uploaded_file):
    text = extract_text_from_file(uploaded_file)
    if not text.strip():
        return "❌ Could not extract text."
    chunks = split_into_chunks(text)
    build_vector_store(chunks)
    return f"✅ Resume processed! {len(chunks)} sections indexed."

def get_relevant_context(question, k=3):
    if _vector_store is None:
        return ""
    docs = _vector_store.similarity_search(question, k=k)
    return "\n\n".join([d.page_content for d in docs])

def has_document():
    return _vector_store is not None

def reset_vector_store():
    global _vector_store
    _vector_store = None
