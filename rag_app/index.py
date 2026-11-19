"""Index manager: builds, saves, loads and atomically swaps the search index.

Holds the vector index (FAISS), the BM25 corpus and the chunk list. A single
re-entrant lock guards against concurrent rebuild/search, and snapshots are
written to ``data/index/<timestamp>/`` so startup can load instead of rebuild.
"""

from __future__ import annotations

import hashlib
import json
import threading
from datetime import datetime
from pathlib import Path

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from .chunking import Chunk, chunk_documents
from .config import Settings
from .embeddings import load_embedder
from .loaders import LoadedDoc
from .normalize import tokenize


def corpus_hash(chunks: list[dict]) -> str:
    payload = "\n".join(c["text"] for c in chunks).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class IndexManager:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.lock = threading.RLock()

        self.chunks: list[dict] = []
        self.vector_index: faiss.Index | None = None
        self.bm25: BM25Okapi | None = None
        self.embedder = None
        self.embedding_model: str = ""
        self.chunk_size: int = settings.chunk_size
        self.chunk_overlap: int = settings.chunk_overlap
        self.load_error: str | None = None
        self.status: str = "loading"  # loading|building|ready|error|empty

    # ------------------------------------------------------------------
    # building
    # ------------------------------------------------------------------
    def build(self, docs: list[LoadedDoc], embedding_model: str | None = None,
              chunk_size: int | None = None, chunk_overlap: int | None = None,
              embedder=None) -> None:
        with self.lock:
            self.status = "building"
            self.load_error = None
            cs = chunk_size or self.settings.chunk_size
            co = chunk_overlap if chunk_overlap is not None else self.settings.chunk_overlap
            emb_model = embedding_model or self.settings.resolved_embedding

            chunks = chunk_documents(
                docs, chunk_size=cs, chunk_overlap=co,
                min_chunk_size=self.settings.min_chunk_size,
            )
            self.chunks = [c.to_dict() for c in chunks]

            if embedder is None:
                embedder = load_embedder(emb_model, hf_token=self.settings.hf_token)
            texts = [c["text"] for c in self.chunks] or [""]
            emb = embedder.encode(texts)
            idx = faiss.IndexFlatIP(emb.shape[1])
            idx.add(emb)

            corpus = [tokenize(c["text"]) or [""] for c in self.chunks]
            bm25 = BM25Okapi(corpus)

            # atomic swap
            self.embedder = embedder
            self.vector_index = idx
            self.bm25 = bm25
            self.embedding_model = emb_model
            self.chunk_size = cs
            self.chunk_overlap = co
            self.status = "ready"

    def is_ready(self) -> bool:
        return self.vector_index is not None and bool(self.chunks)

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------
    def fingerprint(self) -> str:
        return corpus_hash(self.chunks)

    def save(self, target: Path | None = None) -> tuple[Path, dict]:
        with self.lock:
            if not self.is_ready():
                raise RuntimeError("Индекс ещё не построен")
            target = target or (self.settings.models_path / datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
            target.mkdir(parents=True, exist_ok=True)

            faiss.write_index(self.vector_index, str(target / "index.faiss"))
            (target / "chunks.json").write_text(
                json.dumps(self.chunks, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            meta = {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "embedding_model": self.embedding_model,
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "chunks_total": len(self.chunks),
                "vectors_total": int(self.vector_index.ntotal),
                "vector_dim": int(self.vector_index.d),
                "corpus_sha256": self.fingerprint(),
            }
            (target / "meta.json").write_text(
                json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            return target, meta

    def latest_snapshot(self) -> Path | None:
        base = self.settings.models_path
        if not base.is_dir():
            return None
        folders = sorted(p for p in base.iterdir() if p.is_dir() and (p / "index.faiss").is_file())
        return folders[-1] if folders else None

    def list_snapshots(self) -> list[dict]:
        """Metadata of all saved index snapshots (newest first)."""
        base = self.settings.models_path
        out: list[dict] = []
        if not base.is_dir():
            return out
        folders = sorted(
            (p for p in base.iterdir() if p.is_dir() and (p / "index.faiss").is_file()),
            reverse=True,
        )
        for p in folders:
            meta = {}
            try:
                meta = json.loads((p / "meta.json").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                pass
            out.append({
                "name": p.name,
                "created_at": meta.get("created_at"),
                "embedding_model": meta.get("embedding_model"),
                "chunk_size": meta.get("chunk_size"),
                "chunk_overlap": meta.get("chunk_overlap"),
                "chunks_total": meta.get("chunks_total"),
                "active": p == self.latest_snapshot() or meta.get("embedding_model") == self.embedding_model,
            })
        return out

    def load_snapshot(self, folder: Path, expected_fingerprint: str | None = None) -> bool:
        """Load a saved snapshot if its embedding model (and corpus) match."""
        with self.lock:
            idx_path = folder / "index.faiss"
            chunks_path = folder / "chunks.json"
            meta_path = folder / "meta.json"
            if not (idx_path.is_file() and chunks_path.is_file()):
                return False
            meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
            want = self.settings.resolved_embedding
            if meta.get("embedding_model") and meta["embedding_model"] != want:
                return False
            if expected_fingerprint is not None and meta.get("corpus_sha256") != expected_fingerprint:
                return False

            chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
            idx = faiss.read_index(str(idx_path))
            embedder = load_embedder(want, hf_token=self.settings.hf_token)
            corpus = [tokenize(c["text"]) or [""] for c in chunks]

            self.chunks = chunks
            self.vector_index = idx
            self.bm25 = BM25Okapi(corpus)
            self.embedder = embedder
            self.embedding_model = want
            self.chunk_size = meta.get("chunk_size", self.settings.chunk_size)
            self.chunk_overlap = meta.get("chunk_overlap", self.settings.chunk_overlap)
            self.status = "ready"
            return True
