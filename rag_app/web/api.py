"""JSON API blueprint."""

from __future__ import annotations

from pathlib import Path

from flask import Blueprint, current_app, jsonify, request, send_file

from .. import get_ctx
from ..config import EMBEDDING_MODELS, LLM_MODELS, RERANK_MODELS, save_settings
from ..generation import generate_answer
from ..loaders import load_folder, load_knowledge_base_md
from ..normalize import shorten
from ..retrieval import ALL_MODES
from ..eval import (plot_heatmap, run_eval, summarize, load_questions,
                    save_questions, evaluate_detail, suggest_questions, METRIC_COLS)

api_bp = Blueprint("api", __name__, url_prefix="/api")


def _ctx():
    return get_ctx(current_app)


# --------------------------------------------------------------------- meta
@api_bp.get("/modes")
def modes():
    return jsonify({"modes": ALL_MODES})


@api_bp.get("/models")
def models():
    ctx = _ctx()
    return jsonify({
        "embedding": EMBEDDING_MODELS,
        "llm": LLM_MODELS,
        "rerank": RERANK_MODELS,
        "current": ctx.settings.to_public(),
        "title": ctx.title,
        "index_status": ctx.index.status,
        "index_chunks": len(ctx.index.chunks),
        "index_embedding": ctx.index.embedding_model,
        "index_error": ctx.index.load_error,
        "rerank_available": ctx.retriever.reranker is not None,
        "rerank_error": ctx.retriever.rerank_error,
    })


@api_bp.get("/config")
def get_config():
    return jsonify({"config": _ctx().settings.to_public()})


@api_bp.post("/config")
def set_config():
    ctx = _ctx()
    data = request.get_json(silent=True) or {}
    s = ctx.settings
    from dataclasses import fields as _fields
    from ..config import Settings as _Settings
    allowed = {f.name for f in _fields(_Settings)} - {"hf_token", "gigachat_key_id", "gigachat_key_secret"}
    for key, value in data.items():
        if key not in allowed:
            continue
        current = getattr(s, key)
        try:
            if isinstance(current, bool):
                value = bool(value)
            elif isinstance(current, int) and not isinstance(current, bool):
                value = int(value)
            elif isinstance(current, float):
                value = float(value)
        except (TypeError, ValueError):
            continue
        setattr(s, key, value)
    save_settings(s)
    # keep LLM client in sync
    ctx.retriever.client = ctx.llm_client()
    return jsonify({"status": "ok", "config": s.to_public()})


# ------------------------------------------------------------------ search
@api_bp.post("/search")
def search():
    ctx = _ctx()
    data = request.get_json(silent=True) or {}
    query = (data.get("query") or "").strip()
    if not query:
        return jsonify({"error": "Введите запрос"}), 400
    mode = data.get("mode", ctx.settings.default_mode)
    k = int(data.get("k", ctx.settings.default_k))
    results = ctx.retriever.retrieve(query, mode=mode, k=k)
    return jsonify({"query": query, "mode": mode, "results": results,
                    "embedding_model": ctx.index.embedding_model or ctx.settings.resolved_embedding,
                    "temperature_mq": ctx.settings.default_temp_mq})


@api_bp.post("/answer")
def answer():
    ctx = _ctx()
    data = request.get_json(silent=True) or {}
    query = (data.get("query") or "").strip()
    if not query:
        return jsonify({"error": "Введите запрос"}), 400
    mode = data.get("mode", ctx.settings.default_mode)
    k = int(data.get("k", ctx.settings.default_k))
    temperature = float(data.get("temperature", ctx.settings.default_temp))
    temperature_mq = float(data.get("temperature_mq", ctx.settings.default_temp_mq))
    result = generate_answer(ctx.retriever, query, mode=mode, k=k, temperature=temperature)
    return jsonify({
        "query": query, "mode": mode,
        "llm_model": ctx.settings.llm_model,
        "temperature": temperature,
        "temperature_mq": temperature_mq,
        **result,
    })


@api_bp.post("/compare")
def compare():
    ctx = _ctx()
    data = request.get_json(silent=True) or {}
    query = (data.get("query") or "").strip()
    if not query:
        return jsonify({"error": "Введите запрос"}), 400
    k = int(data.get("k", ctx.settings.default_k))
    temp = ctx.settings.default_temp
    temp_mq = ctx.settings.default_temp_mq
    report = []
    for mode in ALL_MODES:
        try:
            results = ctx.retriever.retrieve(query, mode=mode, k=k)
            error = None
        except Exception as e:  # pragma: no cover
            results, error = [], f"{type(e).__name__}: {e}"
        entry = {
            "mode": mode,
            "error": error,
            "temperature": temp,
            "results": [
                {
                    "rank": rank,
                    "chunk_id": r["chunk_id"],
                    "score": r["score"],
                    "file_name": r["file_name"],
                    "section": r.get("section", ""),
                    "category": r.get("category", ""),
                    "preview": shorten(r["text"], 220),
                }
                for rank, r in enumerate(results, start=1)
            ],
        }
        if mode.startswith("multi_query"):
            entry["temperature_mq"] = temp_mq
        report.append(entry)
    return jsonify({"query": query, "k": k, "modes": report})


# ------------------------------------------------------------------ rebuild
@api_bp.post("/rebuild")
def rebuild():
    ctx = _ctx()
    data = request.get_json(silent=True) or {}
    s = ctx.settings
    if "chunk_size" in data:
        s.chunk_size = int(data["chunk_size"])
    if "chunk_overlap" in data:
        s.chunk_overlap = int(data["chunk_overlap"])
    if "embedding_id" in data:
        s.embedding_id = data["embedding_id"]
        s.embedding_name = ""
    if "source" in data:
        s.source = data["source"]
    save_settings(s)
    try:
        info = ctx.rebuild()
    except Exception as e:
        return jsonify({"status": "error", "error": f"{type(e).__name__}: {e}"}), 500
    return jsonify({"status": "ok", "message": "Индекс перестроен", **info})


@api_bp.post("/save_index")
def save_index():
    ctx = _ctx()
    try:
        target, meta = ctx.index.save()
    except Exception as e:
        return jsonify({"status": "error", "error": f"{type(e).__name__}: {e}"}), 500
    return jsonify({"status": "ok", "path": target.name, "meta": meta})


@api_bp.get("/recommend/options")
def recommend_options():
    ctx = _ctx()
    return jsonify({
        "embeddings": EMBEDDING_MODELS,
        "modes": ALL_MODES,
        "metrics": ["score_avg"] + METRIC_COLS,
        "current": ctx.settings.to_public(),
        "questions": len(load_questions(ctx.settings.eval_questions_path)),
    })


@api_bp.post("/recommend/run")
def recommend_run():
    ctx = _ctx()
    data = request.get_json(silent=True) or {}
    started = ctx.start_sweep(data)
    if not started:
        return jsonify({"status": "busy", "message": "Перебор уже запущен"}), 409
    return jsonify({"status": "started"})


@api_bp.get("/recommend/status")
def recommend_status():
    st = _ctx().sweep
    return jsonify({
        "status": st.get("status"),
        "message": st.get("message"),
        "done": st.get("done"),
        "total": st.get("total"),
        "error": st.get("error"),
        "result": st.get("result"),
    })


@api_bp.post("/recommend/apply")
def recommend_apply():
    ctx = _ctx()
    data = request.get_json(silent=True) or {}
    rec = data.get("recommended") or {}
    if not rec:
        return jsonify({"error": "Не переданы рекомендуемые настройки"}), 400
    try:
        info = ctx.apply_recommended(rec)
    except Exception as e:
        return jsonify({"status": "error", "error": f"{type(e).__name__}: {e}"}), 500
    return jsonify({"status": "ok", **info})


@api_bp.post("/recommend/save")
def recommend_save():
    """Persist the best config + the full sweep table to disk."""
    import csv
    import json
    from datetime import datetime

    ctx = _ctx()
    result = ctx.sweep.get("result")
    if not result:
        return jsonify({"error": "Сначала выполните подбор"}), 400

    target = ctx.settings.data_path / "recommendations" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    target.mkdir(parents=True, exist_ok=True)
    written = []

    (target / "recommendations.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    written.append("recommendations.json")

    if result.get("recommended"):
        (target / "best.json").write_text(
            json.dumps(result["recommended"], ensure_ascii=False, indent=2), encoding="utf-8"
        )
        written.append("best.json")

    configs = result.get("configs") or []
    if configs:
        cols = list(configs[0].keys())
        csv_path = target / "sweep.csv"
        with csv_path.open("w", newline="", encoding="utf-8-sig") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            w.writerows(configs)
        written.append("sweep.csv")

    return jsonify({
        "status": "ok",
        "path": target.relative_to(ctx.settings.data_path).as_posix(),
        "files": written,
    })


@api_bp.get("/snapshots")
def list_snapshots():
    ctx = _ctx()
    return jsonify({
        "snapshots": ctx.index.list_snapshots(),
        "active_embedding": ctx.index.embedding_model,
        "chunks": len(ctx.index.chunks),
    })


@api_bp.post("/snapshots/load")
def load_snapshot():
    ctx = _ctx()
    data = request.get_json(silent=True) or {}
    name = data.get("name", "")
    folder = ctx.settings.models_path / name
    if not name or not (folder / "index.faiss").is_file():
        return jsonify({"error": "Снэпшот не найден"}), 404
    try:
        ok = ctx.index.load_snapshot(folder)
    except Exception as e:
        return jsonify({"status": "error", "error": f"{type(e).__name__}: {e}"}), 500
    if not ok:
        return jsonify({"status": "error", "error": "Не удалось загрузить снэпшот"}), 400
    ctx.load_reranker()
    return jsonify({"status": "ok", "chunks": len(ctx.index.chunks),
                    "embedding_model": ctx.index.embedding_model})


@api_bp.delete("/snapshots/<path:name>")
def delete_snapshot(name):
    import shutil
    ctx = _ctx()
    folder = ctx.settings.models_path / name
    if not folder.is_dir():
        return jsonify({"error": "Снэпшот не найден"}), 404
    shutil.rmtree(folder, ignore_errors=True)
    return jsonify({"status": "ok"})


# --------------------------------------------------------------------- eval
@api_bp.get("/eval/questions")
def eval_questions_get():
    ctx = _ctx()
    docs = [d["name"] for d in ctx.ingestion.list_documents()]
    return jsonify({
        "questions": load_questions(ctx.settings.eval_questions_path),
        "documents": docs,
    })


@api_bp.post("/eval/questions")
def eval_questions_save():
    ctx = _ctx()
    data = request.get_json(silent=True) or {}
    questions = data.get("questions")
    if not isinstance(questions, list):
        return jsonify({"error": "Ожидается список вопросов"}), 400
    cleaned = []
    for i, q in enumerate(questions, start=1):
        query = (q.get("query") or "").strip()
        if not query:
            continue
        cleaned.append({
            "id": (q.get("id") or f"Q{i}").strip(),
            "query": query,
            "relevant_docs": [d for d in (q.get("relevant_docs") or []) if d],
            "keywords": [k.strip() for k in (q.get("keywords") or []) if k.strip()],
        })
    save_questions(ctx.settings.eval_questions_path, cleaned)
    return jsonify({"status": "ok", "count": len(cleaned)})


@api_bp.post("/eval/detail")
def eval_detail():
    ctx = _ctx()
    data = request.get_json(silent=True) or {}
    questions = load_questions(ctx.settings.eval_questions_path)
    idx = int(data.get("index", 0))
    if idx < 0 or idx >= len(questions):
        return jsonify({"error": "Вопрос не найден"}), 400
    item = questions[idx]
    mode = data.get("mode", ctx.settings.default_mode)
    k = int(data.get("k", ctx.settings.default_k))
    try:
        chunks = evaluate_detail(ctx.retriever, item, mode=mode, k=k)
    except Exception as e:
        return jsonify({"status": "error", "error": f"{type(e).__name__}: {e}"}), 500
    return jsonify({"question": item, "mode": mode, "k": k, "chunks": chunks})


@api_bp.post("/eval/generate")
def eval_generate():
    ctx = _ctx()
    if ctx.retriever.client is None:
        return jsonify({"error": "LLM недоступен: укажите ключ GigaChat в настройках."}), 400
    data = request.get_json(silent=True) or {}
    count = max(1, min(int(data.get("count", 5)), 20))
    docs = ctx.ingestion.documents()
    if not docs:
        return jsonify({"error": "База знаний пуста"}), 400
    try:
        questions = suggest_questions(ctx.retriever.client, ctx.settings.llm_model, docs, count=count)
    except Exception as e:
        return jsonify({"status": "error", "error": f"{type(e).__name__}: {e}"}), 500
    return jsonify({"questions": questions})


@api_bp.post("/eval/run")
def eval_run():
    ctx = _ctx()
    data = request.get_json(silent=True) or {}
    k = int(data.get("k", ctx.settings.default_k))
    rows = run_eval(ctx.retriever, k=k, questions=load_questions(ctx.settings.eval_questions_path))
    summary = summarize(rows)
    png = plot_heatmap(rows, save_path=ctx.settings.heatmap_path)
    return jsonify({
        "k": k,
        "rows": rows,
        "summary": summary,
        "heatmap": bool(png),
    })


@api_bp.post("/eval/save")
def eval_save():
    """Persist eval metrics + heatmap (+ optional per-query comparison) to disk."""
    import csv
    import json
    from datetime import datetime

    ctx = _ctx()
    data = request.get_json(silent=True) or {}
    k = int(data.get("k", ctx.settings.default_k))
    query = (data.get("query") or "").strip()

    rows = run_eval(ctx.retriever, k=k, questions=load_questions(ctx.settings.eval_questions_path))
    summary = summarize(rows)
    target = ctx.settings.data_path / "eval" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    target.mkdir(parents=True, exist_ok=True)

    written = []
    per_q = target / "eval_per_question.csv"
    with per_q.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    written.append(per_q.name)

    sum_csv = target / "eval_summary.csv"
    cols = list(next(iter(summary.values())).keys())
    with sum_csv.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["mode"] + cols)
        for mode, metrics in summary.items():
            w.writerow([mode] + [metrics[c] for c in cols])
    written.append(sum_csv.name)

    png = plot_heatmap(rows, save_path=target / "heatmap.png")
    if png:
        written.append(png.name)
        import shutil
        shutil.copy2(str(png), str(ctx.settings.heatmap_path))

    if query:
        cmp = _build_compare(query, k)
        (target / "compare.json").write_text(
            json.dumps(cmp, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        written.append("compare.json")

    return jsonify({
        "status": "ok",
        "path": target.relative_to(ctx.settings.data_path).as_posix(),
        "files": written,
        "summary": summary,
        "heatmap": bool(png),
    })


def _build_compare(query: str, k: int) -> dict:
    ctx = _ctx()
    report = []
    for mode in ALL_MODES:
        try:
            results = ctx.retriever.retrieve(query, mode=mode, k=k)
            error = None
        except Exception as e:  # pragma: no cover
            results, error = [], f"{type(e).__name__}: {e}"
        report.append({
            "mode": mode,
            "error": error,
            "results": [
                {"rank": i, "chunk_id": r["chunk_id"], "score": r["score"],
                 "file_name": r["file_name"], "section": r.get("section", ""),
                 "preview": shorten(r["text"], 220)}
                for i, r in enumerate(results, start=1)
            ],
        })
    return {"query": query, "k": k, "modes": report}


# ---------------------------------------------------------------- documents
@api_bp.get("/docs")
def list_docs():
    ctx = _ctx()
    return jsonify({
        "kb_file": "knowledge_base.md",
        "files": ctx.ingestion.list_documents(),
    })


@api_bp.post("/documents")
def upload_document():
    ctx = _ctx()
    if "file" not in request.files:
        return jsonify({"error": "Файл не передан"}), 400
    try:
        info = ctx.ingestion.save_upload(request.files["file"])
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    try:
        info["rebuilt"] = ctx.rebuild()
    except Exception as e:
        return jsonify({"status": "uploaded", "document": info,
                        "warning": f"Файл загружен, но переиндексация не удалась: {e}"}), 200
    return jsonify({"status": "ok", "document": info, "rebuilt": info["rebuilt"]})


@api_bp.delete("/documents/<path:name>")
def delete_document(name):
    from urllib.parse import unquote
    ctx = _ctx()
    if not ctx.ingestion.delete_document(unquote(name)):
        return jsonify({"error": "Документ не найден"}), 404
    try:
        ctx.rebuild()
    except Exception as e:
        return jsonify({"status": "deleted", "warning": str(e)}), 200
    return jsonify({"status": "ok"})


@api_bp.get("/docs/<path:name>")
def download_doc(name):
    from io import BytesIO
    from urllib.parse import unquote

    result = _ctx().ingestion.document_bytes(unquote(name))
    if result is None:
        return jsonify({"error": "Документ не найден"}), 404
    filename, payload = result
    return send_file(BytesIO(payload), mimetype="text/markdown",
                     as_attachment=True, download_name=filename)


# ------------------------------------------------------------------ secrets
SECRET_KEYS = ("GIGACHAT_API_KEY_ID", "GIGACHAT_API_KEY_SECRET", "HF_TOKEN")


def _mask(value: str) -> str:
    if not value:
        return ""
    if len(value) <= 6:
        return "•" * len(value)
    return value[:3] + "•" * (len(value) - 5) + value[-2:]


@api_bp.get("/secrets")
def read_secrets():
    ctx = _ctx()
    s = ctx.settings
    raw = {
        "GIGACHAT_API_KEY_ID": s.gigachat_key_id,
        "GIGACHAT_API_KEY_SECRET": s.gigachat_key_secret,
        "HF_TOKEN": s.hf_token,
    }
    return jsonify({
        "secrets_masked": {k: _mask(v) for k, v in raw.items()},
        "present": {k: bool(v) for k, v in raw.items()},
    })


@api_bp.post("/secrets")
def write_secrets():
    ctx = _ctx()
    data = request.get_json(silent=True) or {}
    secrets = data.get("secrets", {})
    env_path = Path(ctx.settings.env_file)

    mapping = {
        "GIGACHAT_API_KEY_ID": "gigachat_key_id",
        "GIGACHAT_API_KEY_SECRET": "gigachat_key_secret",
        "HF_TOKEN": "hf_token",
    }
    for key, value in secrets.items():
        if key in mapping and value:
            setattr(ctx.settings, mapping[key], value.strip())

    # persist to .env
    lines = []
    if env_path.is_file():
        lines = env_path.read_text(encoding="utf-8").splitlines()
    for env_key, attr in mapping.items():
        val = getattr(ctx.settings, attr, "")
        if not val:
            continue
        replaced = False
        for i, line in enumerate(lines):
            if line.strip().startswith(env_key + "="):
                lines[i] = f"{env_key}={val}"
                replaced = True
                break
        if not replaced:
            lines.append(f"{env_key}={val}")
    env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    ctx.retriever.client = ctx.llm_client()
    return jsonify({"status": "ok", "message": "Секреты сохранены. При смене embeddings перестройте индекс."})


# --------------------------------------------------------------------- chat
@api_bp.get("/chat/sessions")
def chat_sessions():
    return jsonify({"sessions": _ctx().chat.list_sessions()})


@api_bp.post("/chat/sessions")
def chat_new_session():
    data = request.get_json(silent=True) or {}
    return jsonify(_ctx().chat.create_session(data.get("title") or "Новый диалог"))


@api_bp.get("/chat/sessions/<int:sid>")
def chat_get(sid):
    return jsonify({"messages": _ctx().chat.get_messages(sid)})


@api_bp.delete("/chat/sessions/<int:sid>")
def chat_delete(sid):
    _ctx().chat.delete_session(sid)
    return jsonify({"status": "ok"})


@api_bp.post("/chat/message")
def chat_message():
    ctx = _ctx()
    data = request.get_json(silent=True) or {}
    query = (data.get("query") or "").strip()
    if not query:
        return jsonify({"error": "Введите сообщение"}), 400
    sid = data.get("session_id")
    if not sid:
        sid = ctx.chat.create_session()["id"]
    mode = data.get("mode", ctx.settings.default_mode)
    k = int(data.get("k", ctx.settings.default_k))
    temperature = float(data.get("temperature", ctx.settings.default_temp))

    history = [{"role": m["role"], "text": m["text"]} for m in ctx.chat.get_messages(sid)]
    ctx.chat.add_message(sid, "user", query)
    result = generate_answer(ctx.retriever, query, mode=mode, k=k,
                             temperature=temperature, history=history)
    ctx.chat.add_message(sid, "assistant", result["answer"], result["citations"])
    return jsonify({"session_id": sid, "query": query, "mode": mode, **result})
