"""RAG pipeline: document chunking, embeddings, vector storage, and retrieval.

Supports ChromaDB when installed and compatible, and includes a persistent
SQLite + NumPy cosine vector store in vyraq.db as a Python 3.13/3.15 native backend.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from sklearn.feature_extraction.text import HashingVectorizer

from ai.gemini_provider import create_gemini_embeddings, resolve_gemini_api_key
from ai.ollama_provider import check_ollama_health, create_ollama_embeddings
from config import get_config
from database.db import get_connection, init_db


@dataclass
class RetrievedChunk:
    """A retrieved document chunk with similarity score and source metadata."""

    doc_type: str
    chunk_index: int
    text: str
    score: float
    source_name: str = ""
    section_hint: str = ""


class LocalTfidfEmbeddings(Embeddings):
    """Deterministic local dense embeddings built with scikit-learn HashingVectorizer.

    Ensures the RAG pipeline always produces L2-normalized semantic/lexical vectors
    even when external embedding APIs or local Ollama embedding models are unavailable.
    """

    def __init__(self, n_features: int = 384) -> None:
        self.vectorizer = HashingVectorizer(
            n_features=n_features,
            alternate_sign=False,
            norm="l2",
            ngram_range=(1, 2),
            stop_words="english",
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        matrix = self.vectorizer.transform(texts).toarray()
        return [[float(x) for x in row] for row in matrix]

    def embed_query(self, text: str) -> list[float]:
        vec = self.vectorizer.transform([text]).toarray()[0]
        return [float(x) for x in vec]


def split_text_into_documents(
    text: str,
    doc_type: str,
    user_id: int,
    source_name: str = "",
    chunk_size: int = 700,
    chunk_overlap: int = 120,
) -> list[Document]:
    """Split raw document text into overlapping LangChain Document chunks."""
    cleaned = text.strip()
    if not cleaned:
        return []

    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter  # type: ignore

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        raw_chunks = splitter.split_text(cleaned)
    except ImportError:
        raw_chunks = _recursive_split(cleaned, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    docs: list[Document] = []
    for idx, chunk in enumerate(raw_chunks):
        chunk_text = chunk.strip()
        if not chunk_text:
            continue
        section_hint = _infer_section_hint(chunk_text)
        docs.append(
            Document(
                page_content=chunk_text,
                metadata={
                    "user_id": user_id,
                    "doc_type": doc_type,
                    "source_name": source_name,
                    "chunk_index": idx,
                    "section_hint": section_hint,
                },
            )
        )
    return docs


def _recursive_split(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Paragraph- and sentence-aware recursive text splitter."""
    if len(text) <= chunk_size:
        return [text]

    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    units: list[str] = []
    for para in paragraphs:
        if len(para) <= chunk_size:
            units.append(para)
        else:
            lines = [ln.strip() for ln in para.splitlines() if ln.strip()]
            for ln in lines:
                if len(ln) <= chunk_size:
                    units.append(ln)
                else:
                    words = ln.split()
                    buf: list[str] = []
                    for word in words:
                        if sum(len(w) + 1 for w in buf) + len(word) > chunk_size and buf:
                            units.append(" ".join(buf))
                            buf = [word]
                        else:
                            buf.append(word)
                    if buf:
                        units.append(" ".join(buf))

    chunks: list[str] = []
    current = ""
    for unit in units:
        candidate = f"{current}\n{unit}".strip() if current else unit
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current)
            overlap_tail = current[-chunk_overlap:].strip() if chunk_overlap > 0 and current else ""
            current = f"{overlap_tail}\n{unit}".strip() if overlap_tail else unit
    if current:
        chunks.append(current)
    return chunks


def _infer_section_hint(chunk_text: str) -> str:
    """Detect likely resume/JD section from chunk header keywords."""
    lower = chunk_text[:160].lower()
    for label, keywords in (
        ("Experience", ("experience", "employment", "work history", "responsibilities")),
        ("Skills", ("skills", "technologies", "tech stack", "competencies", "requirements")),
        ("Education", ("education", "university", "bachelor", "master", "degree", "certification")),
        ("Projects", ("projects", "portfolio", "built", "developed")),
        ("Summary", ("summary", "objective", "profile", "about")),
    ):
        if any(kw in lower for kw in keywords):
            return label
    return "General"


def select_embeddings_backend(
    provider: str = "ollama",
    api_key: str | None = None,
    ollama_base_url: str | None = None,
    embedding_model: str | None = None,
) -> tuple[Embeddings, str]:
    """Select the best working LangChain Embeddings implementation with safe local fallback."""
    cfg = get_config()
    chosen = (provider or cfg.default_provider).strip().lower()

    if chosen == "gemini":
        resolved_key = resolve_gemini_api_key(api_key or cfg.gemini_api_key)
        if resolved_key:
            try:
                emb = create_gemini_embeddings(
                    model_name=embedding_model or cfg.gemini_embedding_model,
                    api_key=resolved_key,
                )
                return emb, f"gemini:{embedding_model or cfg.gemini_embedding_model}"
            except Exception:
                pass

    if chosen == "ollama":
        url = (ollama_base_url or cfg.ollama_base_url).strip()
        status = check_ollama_health(base_url=url, timeout=1.5)
        target_model = embedding_model or cfg.ollama_embedding_model
        if status.available and any(
            m.split(":")[0] == target_model.split(":")[0] for m in status.models
        ):
            try:
                emb = create_ollama_embeddings(model_name=target_model, base_url=url)
                return emb, f"ollama:{target_model}"
            except Exception:
                pass

    return LocalTfidfEmbeddings(), "local-tfidf-dense-384"


class VyraqRAGPipeline:
    """User-scoped RAG pipeline indexing resume and JD chunks into a vector store."""

    def __init__(
        self,
        user_id: int,
        collection_name: str,
        db_path: str | Path | None = None,
        vector_store_dir: str | Path | None = None,
        embeddings: Embeddings | None = None,
        provider: str = "ollama",
        api_key: str | None = None,
        ollama_base_url: str | None = None,
    ) -> None:
        cfg = get_config()
        self.user_id = int(user_id)
        self.collection_name = collection_name
        self.db_path = Path(db_path) if db_path else cfg.db_path
        self.vector_store_dir = Path(vector_store_dir) if vector_store_dir else cfg.vector_store_dir
        init_db(self.db_path)

        if embeddings is not None:
            self.embeddings = embeddings
            self.embedding_backend_name = type(embeddings).__name__
        else:
            self.embeddings, self.embedding_backend_name = select_embeddings_backend(
                provider=provider,
                api_key=api_key,
                ollama_base_url=ollama_base_url,
            )

        self._chroma_collection = self._try_init_chroma()
        self.vector_backend_name = "chromadb" if self._chroma_collection is not None else "sqlite-numpy-vector"

    def _try_init_chroma(self) -> Any | None:
        """Use ChromaDB if installed and compatible with the current Python runtime."""
        try:
            import chromadb  # type: ignore

            self.vector_store_dir.mkdir(parents=True, exist_ok=True)
            client = chromadb.PersistentClient(path=str(self.vector_store_dir))
            safe_col = re.sub(r"[^A-Za-z0-9_-]", "_", f"u{self.user_id}_{self.collection_name}")[:60]
            return client.get_or_create_collection(name=safe_col)
        except Exception:
            return None

    def _embed_with_fallback(self, texts: list[str]) -> list[list[float]]:
        """Embed texts using configured backend, falling back to LocalTfidfEmbeddings if API fails."""
        try:
            vectors = self.embeddings.embed_documents(texts)
            if vectors and len(vectors) == len(texts):
                return vectors
        except Exception:
            pass
        fallback = LocalTfidfEmbeddings()
        self.embeddings = fallback
        self.embedding_backend_name = "local-tfidf-dense-384"
        return fallback.embed_documents(texts)

    def _embed_query_with_fallback(self, query: str) -> list[float]:
        try:
            vec = self.embeddings.embed_query(query)
            if vec:
                return vec
        except Exception:
            pass
        fallback = LocalTfidfEmbeddings()
        self.embeddings = fallback
        self.embedding_backend_name = "local-tfidf-dense-384"
        return fallback.embed_query(query)

    def index_documents(
        self,
        resume_text: str,
        job_text: str,
        resume_name: str = "resume",
        job_name: str = "job_description",
        chunk_size: int = 700,
        chunk_overlap: int = 120,
    ) -> dict[str, int]:
        """Chunk, embed, and store resume and job description chunks for this user."""
        resume_docs = split_text_into_documents(
            resume_text,
            doc_type="resume",
            user_id=self.user_id,
            source_name=resume_name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        job_docs = split_text_into_documents(
            job_text,
            doc_type="job_description",
            user_id=self.user_id,
            source_name=job_name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        all_docs = resume_docs + job_docs
        if not all_docs:
            return {"resume_chunks": 0, "job_chunks": 0, "total_chunks": 0}

        texts = [d.page_content for d in all_docs]
        vectors = self._embed_with_fallback(texts)

        # Persist in SQLite vector_chunks scoped by user_id & collection_name
        with get_connection(self.db_path) as conn:
            conn.execute(
                "DELETE FROM vector_chunks WHERE user_id = ? AND collection_name = ?",
                (self.user_id, self.collection_name),
            )
            for doc, vec in zip(all_docs, vectors):
                conn.execute(
                    """
                    INSERT INTO vector_chunks (
                        user_id, collection_name, doc_type, source_name,
                        chunk_index, chunk_text, metadata_json, embedding_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        self.user_id,
                        self.collection_name,
                        str(doc.metadata.get("doc_type", "")),
                        str(doc.metadata.get("source_name", "")),
                        int(doc.metadata.get("chunk_index", 0)),
                        doc.page_content,
                        json.dumps(doc.metadata),
                        json.dumps(vec),
                    ),
                )

        # Also mirror into ChromaDB if available
        if self._chroma_collection is not None:
            try:
                ids = [f"{d.metadata['doc_type']}_{d.metadata['chunk_index']}" for d in all_docs]
                self._chroma_collection.upsert(
                    ids=ids,
                    documents=texts,
                    embeddings=vectors,
                    metadatas=[dict(d.metadata) for d in all_docs],
                )
            except Exception:
                pass

        return {
            "resume_chunks": len(resume_docs),
            "job_chunks": len(job_docs),
            "total_chunks": len(all_docs),
        }

    def retrieve(
        self,
        query: str,
        doc_type: str | None = None,
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        """Retrieve top-k chunks matching query via cosine similarity, scoped to user_id."""
        query_vec = np.asarray(self._embed_query_with_fallback(query), dtype=np.float32)
        query_norm = float(np.linalg.norm(query_vec))
        if query_norm == 0.0:
            query_norm = 1.0

        with get_connection(self.db_path) as conn:
            if doc_type:
                rows = conn.execute(
                    """
                    SELECT doc_type, source_name, chunk_index, chunk_text, metadata_json, embedding_json
                    FROM vector_chunks
                    WHERE user_id = ? AND collection_name = ? AND doc_type = ?
                    """,
                    (self.user_id, self.collection_name, doc_type),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT doc_type, source_name, chunk_index, chunk_text, metadata_json, embedding_json
                    FROM vector_chunks
                    WHERE user_id = ? AND collection_name = ?
                    """,
                    (self.user_id, self.collection_name),
                ).fetchall()

        scored: list[RetrievedChunk] = []
        for row in rows:
            vec = np.asarray(json.loads(row["embedding_json"]), dtype=np.float32)
            if vec.shape != query_vec.shape:
                continue
            denom = float(np.linalg.norm(vec)) * query_norm
            sim = float(np.dot(query_vec, vec) / denom) if denom > 0.0 else 0.0
            meta = json.loads(row["metadata_json"])
            scored.append(
                RetrievedChunk(
                    doc_type=str(row["doc_type"]),
                    chunk_index=int(row["chunk_index"]),
                    text=str(row["chunk_text"]),
                    score=round(sim, 4),
                    source_name=str(row["source_name"]),
                    section_hint=str(meta.get("section_hint", "")),
                )
            )

        scored.sort(key=lambda c: c.score, reverse=True)
        return scored[:top_k]

    def build_grounded_context(
        self,
        query: str,
        top_k_resume: int = 4,
        top_k_job: int = 4,
    ) -> str:
        """Retrieve relevant resume and JD chunks and format them as cited context."""
        resume_hits = self.retrieve(query, doc_type="resume", top_k=top_k_resume)
        job_hits = self.retrieve(query, doc_type="job_description", top_k=top_k_job)

        sections: list[str] = []
        if resume_hits:
            sections.append("### Retrieved Resume Evidence")
            for hit in resume_hits:
                sections.append(
                    f"[Resume Chunk #{hit.chunk_index + 1} | {hit.section_hint} | sim={hit.score:.2f}]\n{hit.text}"
                )
        if job_hits:
            sections.append("### Retrieved Job Description Evidence")
            for hit in job_hits:
                sections.append(
                    f"[JD Chunk #{hit.chunk_index + 1} | {hit.section_hint} | sim={hit.score:.2f}]\n{hit.text}"
                )
        return "\n\n".join(sections)
