"""Configuration sweep: find the best retrieval settings for THIS knowledge base.

Iterates (embedding model x chunk size x retrieval mode [+ multi-query
temperature]), runs the evaluation questions for each combination and ranks the
results by a chosen metric. The winner is turned into a concrete set of
settings the user can apply with one click.
"""

from __future__ import annotations

import dataclasses
from datetime import datetime

from .config import Settings, resolve_embedding_model
from .embeddings import load_embedder
from .eval import METRIC_COLS, load_questions, run_eval, summarize
from .index import IndexManager
from .retrieval import ALL_MODES, Retriever

RANK_METRICS = ["score_avg"] + METRIC_COLS


def _score_avg(metrics: dict) -> float:
    keys = ("hit@k", "doc_acc@k", "recall@k", "mrr@k")
    return round(sum(metrics.get(k, 0.0) for k in keys) / len(keys), 3)


def run_sweep(ctx, options: dict | None = None, progress=None) -> dict:
    """Run the sweep synchronously. ``progress`` is called as progress(msg, done, total)."""
    options = options or {}
    base: Settings = ctx.settings

    emb_ids = options.get("embedding_ids") or [base.embedding_id]
    chunk_sizes = options.get("chunk_sizes") or [base.chunk_size]
    modes = [m for m in (options.get("modes") or ALL_MODES) if m in ALL_MODES]
    metric = options.get("metric") if options.get("metric") in RANK_METRICS else "score_avg"
    mq_temps = options.get("temperature_mq")

    questions = load_questions(base.eval_questions_path)
    docs = ctx.load_documents()
    k = base.default_k

    def emit(msg, done, total):
        if progress:
            progress(msg, done, total)

    # approximate total number of evaluated configs
    total = 0
    for m in modes:
        total += len(emb_ids) * len(chunk_sizes) * (
            len(mq_temps) if (m.startswith("multi_query") and mq_temps) else 1
        )

    embedders: dict[str, object] = {}
    configs: list[dict] = []
    done = 0

    for emb_id in emb_ids:
        emb_name = resolve_embedding_model(emb_id)
        for cs in chunk_sizes:
            emit(f"Индексирую: {emb_id}, chunk={cs}", done, total)
            s = dataclasses.replace(base, embedding_id=emb_id, embedding_name="", chunk_size=cs)
            idx = IndexManager(s)
            try:
                emb = embedders.get(emb_name)
                if emb is None:
                    emb = load_embedder(emb_name, hf_token=base.hf_token)
                    embedders[emb_name] = emb
                idx.build(docs, embedding_model=emb_name, chunk_size=cs,
                          chunk_overlap=base.chunk_overlap, embedder=emb)
            except Exception as e:  # pragma: no cover
                for m in modes:
                    configs.append(_empty(emb_id, emb_name, cs, m, None, f"{type(e).__name__}: {e}"))
                    done += 1
                continue

            retr = Retriever(idx, s)
            retr.client = ctx.retriever.client
            retr.reranker = ctx.retriever.reranker

            for mode in modes:
                temps = mq_temps if (mode.startswith("multi_query") and mq_temps) else [None]
                for temp in temps:
                    emit(f"Оцениваю: {emb_id} · chunk={cs} · {mode}"
                         + (f" · t={temp}" if temp is not None else ""), done, total)
                    if temp is not None:
                        s.default_temp_mq = temp
                    try:
                        rows = run_eval(retr, k=k, modes=[mode], questions=questions)
                        summ = summarize(rows).get(mode, {})
                        cfg = {
                            "embedding_id": emb_id,
                            "embedding_model": emb_name,
                            "chunk_size": cs,
                            "mode": mode,
                            "temperature_mq": temp,
                            "chunks_total": len(idx.chunks),
                        }
                        cfg.update({m: summ.get(m, 0.0) for m in METRIC_COLS})
                        cfg["score_avg"] = _score_avg(cfg)
                        cfg["error"] = None
                    except Exception as e:  # pragma: no cover
                        cfg = _empty(emb_id, emb_name, cs, mode, temp, f"{type(e).__name__}: {e}")
                        cfg["chunks_total"] = len(idx.chunks)
                    configs.append(cfg)
                    done += 1

    ranked = sorted(configs, key=lambda c: (c.get(metric, 0.0), c.get("mrr@k", 0.0)), reverse=True)
    best = ranked[0] if ranked else None

    best_per_embedding = {}
    for c in ranked:
        key = c["embedding_id"]
        if key not in best_per_embedding and not c.get("error"):
            best_per_embedding[key] = c

    best_per_mode = {}
    for c in ranked:
        if c.get("error"):
            continue
        key = c["mode"]
        if key not in best_per_mode or c.get(metric, 0.0) > best_per_mode[key].get(metric, 0.0):
            best_per_mode[key] = c

    recommended = None
    if best and not best.get("error"):
        recommended = {
            "embedding_id": best["embedding_id"],
            "embedding_model": best["embedding_model"],
            "chunk_size": best["chunk_size"],
            "chunk_overlap": base.chunk_overlap,
            "default_mode": best["mode"],
            "default_temp_mq": best["temperature_mq"] if best["temperature_mq"] is not None else base.default_temp_mq,
            "enable_rerank": base.enable_rerank,
            "rrf_k": base.rrf_k,
            "vector_weight": base.vector_weight,
            "bm25_weight": base.bm25_weight,
            "default_temp": base.default_temp,
            "default_k": base.default_k,
        }

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "metric": metric,
        "questions": len(questions),
        "configs": ranked,
        "best": best,
        "recommended": recommended,
        "best_per_embedding": best_per_embedding,
        "best_per_mode": best_per_mode,
    }


def _empty(emb_id, emb_name, cs, mode, temp, error) -> dict:
    cfg = {
        "embedding_id": emb_id,
        "embedding_model": emb_name,
        "chunk_size": cs,
        "mode": mode,
        "temperature_mq": temp,
        "chunks_total": 0,
        "error": error,
    }
    cfg.update({m: 0.0 for m in METRIC_COLS})
    cfg["score_avg"] = 0.0
    return cfg
