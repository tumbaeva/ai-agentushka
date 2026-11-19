"""«Создать базу знаний» blueprint: HTML page + JSON API.

Standalone page that assembles ``docs/knowledge_base.md`` from documents
uploaded into ``source_docs/``. Kept separate from the existing pages/api
blueprints so nothing else changes.
"""

from __future__ import annotations

from io import BytesIO
from urllib.parse import unquote

from flask import Blueprint, current_app, jsonify, render_template, request, send_file

from .. import get_ctx
from ..ocr import ocr_available

kb_bp = Blueprint("kb_builder", __name__)


def _ctx():
    return get_ctx(current_app)


# --------------------------------------------------------------------- page
@kb_bp.get("/kb-builder")
def kb_builder_page():
    ctx = _ctx()
    return render_template("kb_builder.html", title=ctx.title or "Создать базу знаний")


# ---------------------------------------------------------------------- api
@kb_bp.get("/api/kb/files")
def kb_list_files():
    ctx = _ctx()
    return jsonify({
        "files": ctx.builder.list_files(),
        "source_dir": str(ctx.builder.source_dir),
        "kb_file": "knowledge_base.md",
        "llm_available": ctx.llm_client() is not None,
        "ocr_available": ocr_available(),
    })


@kb_bp.post("/api/kb/upload")
def kb_upload():
    ctx = _ctx()
    files = request.files.getlist("file") or ([request.files["file"]] if "file" in request.files else [])
    if not files:
        return jsonify({"error": "Файл не передан"}), 400
    saved, errors = [], []
    for storage in files:
        try:
            saved.append(ctx.builder.save_upload(storage))
        except ValueError as e:
            errors.append({"name": storage.filename, "error": str(e)})
    if not saved and errors:
        return jsonify({"error": errors[0]["error"], "errors": errors}), 400
    return jsonify({"status": "ok", "saved": saved, "errors": errors,
                    "files": ctx.builder.list_files()})


@kb_bp.delete("/api/kb/files/<path:name>")
def kb_delete_file(name):
    ctx = _ctx()
    if not ctx.builder.delete_file(unquote(name)):
        return jsonify({"error": "Файл не найден"}), 404
    return jsonify({"status": "ok", "files": ctx.builder.list_files()})


@kb_bp.post("/api/kb/order")
def kb_set_order():
    ctx = _ctx()
    data = request.get_json(silent=True) or {}
    order = data.get("order")
    if not isinstance(order, list):
        return jsonify({"error": "Ожидается список order"}), 400
    ctx.builder.set_order([str(n) for n in order])
    return jsonify({"status": "ok", "files": ctx.builder.list_files()})


@kb_bp.post("/api/kb/selection")
def kb_set_selection():
    ctx = _ctx()
    data = request.get_json(silent=True) or {}
    selected = data.get("selected")
    if not isinstance(selected, list):
        return jsonify({"error": "Ожидается список selected"}), 400
    ctx.builder.set_selection([str(n) for n in selected])
    return jsonify({"status": "ok", "files": ctx.builder.list_files()})


@kb_bp.post("/api/kb/preview")
def kb_preview():
    ctx = _ctx()
    data = request.get_json(silent=True) or {}
    use_llm = bool(data.get("use_llm"))
    use_ocr = bool(data.get("use_ocr"))
    use_headings = bool(data.get("use_headings"))
    client = ctx.llm_client() if (use_llm or use_headings) else None
    try:
        result = ctx.builder.build(
            use_llm=use_llm, llm_client=client, llm_model=ctx.settings.llm_model,
            use_ocr=use_ocr, use_headings=use_headings,
        )
    except Exception as e:  # pragma: no cover
        return jsonify({"status": "error", "error": f"{type(e).__name__}: {e}"}), 500
    result["status"] = "ok"
    result["llm_requested"] = use_llm
    return jsonify(result)


@kb_bp.post("/api/kb/save_draft")
def kb_save_draft():
    ctx = _ctx()
    data = request.get_json(silent=True) or {}
    text = data.get("text")
    if not isinstance(text, str):
        return jsonify({"error": "Ожидается поле text"}), 400
    try:
        info = ctx.builder.save_draft(text)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    return jsonify({"status": "ok", **info})


@kb_bp.post("/api/kb/promote")
def kb_promote():
    ctx = _ctx()
    data = request.get_json(silent=True) or {}
    text = data.get("text")
    if isinstance(text, str):
        try:
            ctx.builder.save_draft(text)
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
    try:
        info = ctx.builder.promote()
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    return jsonify({"status": "ok", "document": info})


@kb_bp.get("/api/kb/download")
def kb_download():
    payload = _ctx().builder.draft_bytes()
    if payload is None:
        return jsonify({"error": "Черновик ещё не собран"}), 404
    return send_file(BytesIO(payload), mimetype="text/markdown",
                     as_attachment=True, download_name="knowledge_base.md")
