"""Retrieval modes: vector, bm25, hybrid (RRF), multi-query, multi-query+rerank."""

from __future__ import annotations

import numpy as np

from .config import Settings
from .index import IndexManager
from .normalize import tokenize
from .query_transform import build_query_variants, hyde_with_llm, normalize_query

ALL_MODES = ["vector", "bm25", "hybrid", "multi_query", "multi_query_rerank"]


class Retriever:
    def __init__(self, index: IndexManager, settings: Settings):
        self.index = index
        self.settings = settings
        self.client = None       # openai-compatible client (set by app)
        self.reranker = None     # sentence_transformers CrossEncoder
        self.rerank_error: str | None = None

    # ------------------------------------------------------------------
    def _result(self, mode: str, chunk: dict, score: float) -> dict:
        return {
            "mode": mode,
            "chunk_id": chunk["chunk_id"],
            "score": round(float(score), 4),
            "file_name": chunk["file_name"],
            "section": chunk.get("section", ""),
            "category": chunk.get("category", ""),
            "doc_type": chunk.get("doc_type", ""),
            "text": chunk["text"],
        }

    def retrieve_vector(self, query: str, k: int = 4, pool: int = 8) -> list[dict]:
        if not self.index.is_ready():
            return []
        emb = self.index.embedder.encode([query])
        scores, ids = self.index.vector_index.search(emb, max(pool, k))
        out, seen = [], set()
        for score, idx in zip(scores[0], ids[0]):
            if idx < 0 or int(idx) in seen:
                continue
            seen.add(int(idx))
            out.append(self._result("vector", self.index.chunks[int(idx)], score))
            if len(out) >= k:
                break
        return out

    def retrieve_bm25(self, query: str, k: int = 4, pool: int = 8) -> list[dict]:
        if not self.index.is_ready():
            return []
        scores = self.index.bm25.get_scores(tokenize(query))
        ranked = np.argsort(scores)[::-1][: max(pool, k)]
        return [self._result("bm25", self.index.chunks[int(i)], scores[int(i)]) for i in ranked[:k]]

    def _rrf(self, result_sets: dict[str, list[dict]], k: int) -> list[dict]:
        rrf_k = self.settings.rrf_k
        weights = {"vector": self.settings.vector_weight, "bm25": self.settings.bm25_weight}
        fused: dict[int, float] = {}
        owners: dict[int, dict] = {}
        for name, items in result_sets.items():
            w = weights.get(name, 1.0)
            for rank, item in enumerate(items):
                cid = item["chunk_id"]
                fused[cid] = fused.get(cid, 0.0) + w / (rrf_k + rank + 1)
                owners.setdefault(cid, item)
        ordered = sorted(fused.items(), key=lambda x: x[1], reverse=True)
        return [self._result("hybrid", self.index.chunks[cid], score) for cid, score in ordered[:k]]

    def retrieve_hybrid(self, query: str, k: int = 4, pool: int = 8) -> list[dict]:
        sets = {
            "vector": self.retrieve_vector(query, k=pool, pool=pool),
            "bm25": self.retrieve_bm25(query, k=pool, pool=pool),
        }
        return self._rrf(sets, k)

    def retrieve_multi_query(self, query: str, k: int = 4, pool: int = 8) -> list[dict]:
        variants = build_query_variants(query, self.settings, self.client)
        sets: dict[str, list[dict]] = {}
        for i, v in enumerate(variants):
            sets[f"q{i}"] = self.retrieve_hybrid(v, k=pool, pool=pool)
        # HyDE: embed a hypothetical answer and add it as an extra variant
        if self.settings.enable_hyde:
            hypothetical = hyde_with_llm(query, self.client, self.settings.llm_model)
            if hypothetical:
                sets["hyde"] = self.retrieve_vector(hypothetical, k=pool, pool=pool)
        return self._rrf(sets, k)

    def retrieve_multi_query_rerank(self, query: str, k: int = 4, pool: int = 8) -> list[dict]:
        candidates = self.retrieve_multi_query(query, k=pool, pool=pool)
        if not candidates or self.reranker is None:
            return candidates[:k]
        pairs = [(query, c["text"]) for c in candidates]
        try:
            scores = self.reranker.predict(pairs)
        except Exception as e:  # pragma: no cover
            self.rerank_error = f"{type(e).__name__}: {e}"
            return candidates[:k]
        reranked = []
        for item, score in zip(candidates, scores):
            updated = dict(item)
            updated["mode"] = "multi_query_rerank"
            updated["score"] = round(float(score), 4)
            reranked.append(updated)
        reranked.sort(key=lambda x: x["score"], reverse=True)
        return reranked[:k]

    def retrieve(self, query: str, mode: str = "hybrid", k: int = 4) -> list[dict]:
        query = normalize_query(query)
        if mode == "vector":
            return self.retrieve_vector(query, k=k)
        if mode == "bm25":
            return self.retrieve_bm25(query, k=k)
        if mode == "hybrid":
            return self.retrieve_hybrid(query, k=k)
        if mode == "multi_query":
            return self.retrieve_multi_query(query, k=k)
        if mode == "multi_query_rerank":
            return self.retrieve_multi_query_rerank(query, k=k)
        raise ValueError(f"Unknown mode: {mode}")
