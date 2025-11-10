# backend/rag_utils.py
"""
RAG helpers for Retrieval-Augmented Generation.

Features:
- Build or load a FAISS index of embeddings for a collection of documents (local store).
- Support two embedding backends:
    1) Gemini embeddings via gemini_client.embed_texts (if available)
    2) Local sentence-transformers fallback (all-MiniLM-L6-v2)
- Persist index to disk under settings.MODELS_DIR/embeddings and store metadata (JSON) alongside.
- Retrieve top-k documents for a query and return assembled context strings.

Usage:
    from rag_utils import RAGStore
    store = RAGStore()
    store.build(documents)                # documents: list[{"id":..., "text":..., "meta":{...}}]
    hits = store.retrieve("what is the login endpoint?", k=5)
    context = store.build_context(hits)
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np

from backend.config import settings

# Try to use gemini_client embeddings
try:
    from backend.gemini_client import get_gemini_client, GeminiClientError  # type: ignore
    _HAS_GEMINI_EMBED = True
except Exception:
    _HAS_GEMINI_EMBED = False

# Local embedding fallback
try:
    from sentence_transformers import SentenceTransformer
    _HAS_LOCAL_EMBED = True
except Exception:
    SentenceTransformer = None
    _HAS_LOCAL_EMBED = False

# FAISS
try:
    import faiss
    _HAS_FAISS = True
except Exception:
    faiss = None
    _HAS_FAISS = False

# Defaults
EMBED_MODEL_NAME = settings.RAG_EMBEDDING_MODEL or "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM = 384  # typical for all-MiniLM-L6-v2


class RAGError(RuntimeError):
    pass


class RAGStore:
    """
    Simple FAISS-backed RAG store with metadata persistence.

    Documents are dicts: {"id": "<unique-id>", "text": "<full text>", "meta": { ... }}
    """

    def __init__(self, index_dir: Optional[str] = None, embedding_model: Optional[str] = None):
        self.index_dir = Path(index_dir) if index_dir else Path(settings.MODELS_DIR) / "embeddings"
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.embedding_model = embedding_model or EMBED_MODEL_NAME
        self._index = None  # FAISS index
        self._metadatas: List[Dict[str, Any]] = []  # aligned with vectors in index
        self._id_to_pos: Dict[str, int] = {}
        self._dim = EMBED_DIM

        # local embedder init lazily
        self._local_embedder = None
        if _HAS_LOCAL_EMBED:
            try:
                self._local_embedder = SentenceTransformer(self.embedding_model)
                # override dim if model provides
                if hasattr(self._local_embedder, "get_sentence_embedding_dimension"):
                    self._dim = int(self._local_embedder.get_sentence_embedding_dimension())
            except Exception:
                self._local_embedder = None

        # try load existing index/metadata
        try:
            self._load_index_and_meta()
        except Exception:
            # start fresh if anything fails
            self._index = None
            self._metadatas = []
            self._id_to_pos = {}

    # -------------------------
    # Embedding helpers
    # -------------------------
    def _embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Produce embeddings for a list of texts using Gemini (if available) or local sentence-transformers.
        Returns list of vectors (lists of floats).
        """
        # Try Gemini first if available and configured
        if _HAS_GEMINI_EMBED:
            try:
                client = get_gemini_client()
                # Gemini embed API may expect a single call for multiple inputs
                vecs = client.embed_texts(texts, model="embed-text-v1")
                # ensure numpy-compatible floats
                return [list(map(float, v)) for v in vecs]
            except Exception:
                # fallback to local
                pass

        # Local fallback
        if self._local_embedder:
            try:
                arr = self._local_embedder.encode(texts, show_progress_bar=False)
                # convert to list of lists
                return [list(map(float, vec)) for vec in np.array(arr)]
            except Exception as exc:
                raise RAGError(f"Local embedding failed: {exc}")

        raise RAGError("No embedding backend available. Install google-generativeai or sentence-transformers.")

    # -------------------------
    # FAISS index helpers
    # -------------------------
    def _make_index(self, dim: int):
        if not _HAS_FAISS:
            raise RAGError("FAISS is not installed. Install faiss-cpu or faiss-gpu to use RAG features.")
        # simple index: IndexFlatIP (cosine via normalized vectors) or IndexFlatL2
        # we'll use IndexFlatIP with normalized vectors for cosine similarity
        index = faiss.IndexFlatIP(dim)
        return index

    def _save_index_and_meta(self):
        """
        Persist FAISS index and metadata JSON to index_dir.
        """
        if self._index is None:
            raise RAGError("No index to save.")
        idx_path = str(self.index_dir / "index.faiss")
        meta_path = self.index_dir / "metadata.json"
        # write faiss
        faiss.write_index(self._index, idx_path)
        # write metadata
        meta = {"docs": self._metadatas}
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def _load_index_and_meta(self):
        """
        Load persisted FAISS index and metadata if present.
        """
        idx_path = self.index_dir / "index.faiss"
        meta_path = self.index_dir / "metadata.json"
        if idx_path.exists() and meta_path.exists() and _HAS_FAISS:
            self._index = faiss.read_index(str(idx_path))
            raw = json.loads(meta_path.read_text(encoding="utf-8"))
            self._metadatas = raw.get("docs", [])
            # rebuild id->pos
            self._id_to_pos = {m["id"]: i for i, m in enumerate(self._metadatas)}
            # set dim from index
            if hasattr(self._index, "d"):
                self._dim = int(self._index.d)
        else:
            # no saved index present
            self._index = None
            self._metadatas = []
            self._id_to_pos = {}

    # -------------------------
    # Public API
    # -------------------------
    def build(self, documents: List[Dict[str, Any]], overwrite: bool = True):
        """
        Build a FAISS index from `documents`, which are dicts: {"id":str, "text":str, "meta": {...}}
        If overwrite=True, replace any existing index. Otherwise append.
        """
        if not documents:
            return

        texts = [doc["text"] for doc in documents]
        ids = [doc["id"] for doc in documents]
        metas = [doc.get("meta", {}) for doc in documents]

        vectors = self._embed_texts(texts)
        arr = np.array(vectors).astype("float32")

        # normalize for cosine similarity (inner product)
        faiss.normalize_L2(arr)

        if overwrite or self._index is None:
            # create new index
            self._index = self._make_index(arr.shape[1])
            self._metadatas = []
            self._id_to_pos = {}

        # append vectors to index
        start_pos = len(self._metadatas)
        self._index.add(arr)
        # store metadatas aligned with index vectors
        for i, doc_id in enumerate(ids):
            pos = start_pos + i
            md = {"id": doc_id, "meta": metas[i], "text": documents[i]["text"]}
            self._metadatas.append(md)
            self._id_to_pos[doc_id] = pos

        # persist
        self._save_index_and_meta()

    def save_documents(self, documents: List[Dict[str, Any]]):
        """
        Convenience wrapper identical to build(..., overwrite=True)
        """
        self.build(documents, overwrite=True)

    def retrieve(self, query: str, k: int = 4) -> List[Dict[str, Any]]:
        """
        Retrieve top-k documents for the query. Returns list of metadata dicts with keys:
        {"id","text","meta","score"}
        """
        if self._index is None or len(self._metadatas) == 0:
            return []

        q_vecs = self._embed_texts([query])
        q_arr = np.array(q_vecs).astype("float32")
        faiss.normalize_L2(q_arr)

        # perform search
        D, I = self._index.search(q_arr, k)
        scores = D[0].tolist()
        idxs = I[0].tolist()
        results = []
        for score, idx in zip(scores, idxs):
            if idx < 0 or idx >= len(self._metadatas):
                continue
            md = dict(self._metadatas[idx])
            md["score"] = float(score)
            results.append(md)
        return results

    def load(self):
        """Reload index & metadata from disk (if present)."""
        self._load_index_and_meta()

    def build_context(self, hits: List[Dict[str, Any]], max_chars: int = 2000) -> str:
        """
        Build a concatenated context string from retrieved hits.
        Truncates to `max_chars` characters while preserving hit order.
        """
        parts = []
        total = 0
        for h in hits:
            snippet = h.get("text", "")
            meta = h.get("meta", {})
            entry = f"---\nsource_id: {h.get('id')}\nmeta: {json.dumps(meta)}\ntext:\n{snippet}\n"
            if total + len(entry) > max_chars:
                # truncate snippet to fit
                remaining = max_chars - total - len("---\nsource_id:\nmeta:\ntext:\n")
                if remaining > 0:
                    entry = entry[:remaining] + "\n...[truncated]\n"
                    parts.append(entry)
                break
            parts.append(entry)
            total += len(entry)
        return "\n".join(parts)

    def add_document(self, doc: Dict[str, Any]):
        """Add a single document to existing index (append)."""
        self.build([doc], overwrite=False)

    def list_docs(self) -> List[Dict[str, Any]]:
        """Return stored metadata list (id, meta)"""
        return [{"id": md["id"], "meta": md.get("meta", {})} for md in self._metadatas]

