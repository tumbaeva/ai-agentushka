"""«AI-агент» blueprint: constructor page + JSON API + standalone export.

Customizes the branding/theme/answer-style of an AI agent and exports a
standalone chat-only agent application (zip). Follows the conventions of
``web/kb_builder.py`` (own blueprint, ``_ctx()`` helper, HTML page + JSON API).
"""

from __future__ import annotations

import re
from io import BytesIO
from urllib.parse import unquote

from flask import (Blueprint, current_app, jsonify, render_template, request,
                   send_file)

from .. import get_ctx
from ..agent_profile import (PALETTE_KEYS, delete_palette, load_palettes,
                             save_agent_profile, save_palette)
from ..agent_styles import (ANSWER_STYLES, STYLE_IDS, compose_system_prompt,
                            fallback_message, tokens_for_length)
from ..generation import SYSTEM_PROMPT, generate_answer, unique_sources

agent_bp = Blueprint("agent", __name__)

ASSET_EXT = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".ico", ".svg"}
_UNSAFE_RE = re.compile(r"[^0-9A-Za-zА-Яа-яЁё._\- ]+")

FONT_SIZES = {"S", "M", "L"}
BUTTON_SIZES = {"small", "normal", "large"}
LENGTHS = {"short", "medium", "long"}
LANGUAGES = {"ru", "en"}
LOGO_TYPES = {"emoji", "initials", "file"}
_COLOR_RE = re.compile(r"^#[0-9A-Fa-f]{3,8}$")


def _ctx():
    return get_ctx(current_app)


# --------------------------------------------------------------------- page
def _fill_subtitle(ctx) -> None:
    """If the agent has no subtitle, inherit the dashboard title (contest name)."""
    if not (ctx.profile.subtitle or "").strip() and (ctx.title or "").strip():
        ctx.profile.subtitle = ctx.title


def render_agent():
    """Agent page: full constructor in the dashboard, chat-only in standalone."""
    ctx = _ctx()
    _fill_subtitle(ctx)
    standalone = bool(current_app.config.get("AGENT_STANDALONE"))
    return render_template(
        "agent_chat.html" if standalone else "agent.html",
        title=ctx.profile.agent_name or "AI-агент",
        profile=ctx.profile.to_dict(),
        starter_text="\n".join(ctx.profile.starter_questions),
        subtitle_default=ctx.title or "",
        styles=ANSWER_STYLES,
        lang=ctx.profile.ui_language,
    )


@agent_bp.get("/agent")
def agent_page():
    return render_agent()


# ------------------------------------------------------------------ profile
def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _apply_updates(profile, data: dict) -> None:
    """Merge a JSON body into the profile with validation/clamping."""
    if not isinstance(data, dict):
        return

    for key in ("agent_name", "subtitle", "logo_emoji", "logo_initials",
                "favicon_initials", "welcome_message", "contact_phone"):
        value = data.get(key)
        if isinstance(value, str):
            setattr(profile, key, value.strip())
    if isinstance(data.get("custom_prompt"), str):
        profile.custom_prompt = data["custom_prompt"]
    if isinstance(data.get("system_prompt"), str):
        profile.system_prompt = data["system_prompt"]

    for key, allowed in (("font_size", FONT_SIZES),
                         ("button_size", BUTTON_SIZES), ("length", LENGTHS),
                         ("ui_language", LANGUAGES), ("logo_type", LOGO_TYPES),
                         ("favicon_type", LOGO_TYPES)):
        if data.get(key) in allowed:
            setattr(profile, key, data[key])

    if data.get("style_id") in STYLE_IDS:
        profile.style_id = data["style_id"]

    if _is_number(data.get("temperature")):
        profile.temperature = _clamp(float(data["temperature"]), 0.0, 1.0)
    if _is_number(data.get("radius")):
        profile.radius = int(_clamp(int(data["radius"]), 0, 40))
    if _is_number(data.get("sidebar_width")):
        profile.sidebar_width = int(_clamp(int(data["sidebar_width"]), 260, 720))

    if "compact" in data:
        profile.compact = bool(data["compact"])
    if "show_sources" in data:
        profile.show_sources = bool(data["show_sources"])

    if isinstance(data.get("starter_questions"), list):
        profile.starter_questions = [str(q).strip() for q in data["starter_questions"]
                                     if str(q).strip()]
    if isinstance(data.get("palette"), dict):
        merged = dict(profile.palette)
        for key in PALETTE_KEYS:
            value = data["palette"].get(key)
            if isinstance(value, str) and _COLOR_RE.match(value):
                merged[key] = value
        profile.palette = merged


@agent_bp.get("/api/agent/profile")
def agent_get_profile():
    ctx = _ctx()
    return jsonify({
        "profile": ctx.profile.to_dict(),
        "styles": ANSWER_STYLES,
        "default_prompt": SYSTEM_PROMPT,
        "palettes": load_palettes(ctx.settings),
    })


@agent_bp.post("/api/agent/profile")
def agent_save_profile():
    ctx = _ctx()
    data = request.get_json(silent=True) or {}
    _apply_updates(ctx.profile, data)
    _fill_subtitle(ctx)
    save_agent_profile(ctx.settings, ctx.profile)
    return jsonify({"status": "ok", "profile": ctx.profile.to_dict()})


@agent_bp.get("/api/agent/styles")
def agent_styles():
    return jsonify({"styles": ANSWER_STYLES})


@agent_bp.get("/api/agent/status")
def agent_status():
    """Index/readiness info for the chat-only agent UI."""
    ctx = _ctx()
    return jsonify({
        "status": ctx.index.status,
        "chunks": len(ctx.index.chunks),
        "error": ctx.index.load_error,
        "embedding_model": ctx.index.embedding_model or ctx.settings.resolved_embedding,
        "rerank_available": ctx.retriever.reranker is not None,
    })


@agent_bp.post("/api/agent/prompt/preview")
def agent_prompt_preview():
    data = request.get_json(silent=True) or {}
    if (data.get("system_prompt") or "").strip():
        return jsonify({"system_prompt": data["system_prompt"]})
    prompt = compose_system_prompt(
        SYSTEM_PROMPT,
        data.get("style_id", _ctx().profile.style_id),
        data.get("custom_prompt", ""),
        data.get("length", _ctx().profile.length),
    )
    return jsonify({"system_prompt": prompt})


# ------------------------------------------------------------------- assets
def _assets_dir():
    path = _ctx().settings.agent_assets_path
    path.mkdir(parents=True, exist_ok=True)
    return path


@agent_bp.post("/api/agent/asset")
def agent_upload_asset():
    if "file" not in request.files:
        return jsonify({"error": "Файл не передан"}), 400
    storage = request.files["file"]
    raw = storage.filename or ""
    ext = "." + raw.rsplit(".", 1)[-1].lower() if "." in raw else ""
    if ext not in ASSET_EXT:
        return jsonify({"error": f"Формат {ext or '?'} не поддерживается. "
                                 f"Разрешены: {', '.join(sorted(ASSET_EXT))}."}), 400
    stem = _UNSAFE_RE.sub("_", raw.rsplit(".", 1)[0]).strip() or "asset"
    name = f"{stem}{ext}"
    target = _assets_dir() / name
    i = 1
    while target.exists():
        name = f"{stem}_{i}{ext}"
        target = _assets_dir() / name
        i += 1
    target.write_bytes(storage.read())
    return jsonify({"status": "ok", "name": name})


@agent_bp.get("/api/agent/palettes")
def agent_list_palettes():
    return jsonify({"palettes": load_palettes(_ctx().settings)})


@agent_bp.post("/api/agent/palettes")
def agent_save_palette():
    ctx = _ctx()
    data = request.get_json(silent=True) or {}
    name = (data.get("name") or "").strip()
    if not name:
        return jsonify({"error": "Укажите название палитры"}), 400
    colors = data.get("palette")
    if not isinstance(colors, dict):
        colors = ctx.profile.palette
    try:
        items = save_palette(ctx.settings, name, colors)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    return jsonify({"status": "ok", "palettes": items})


@agent_bp.delete("/api/agent/palettes/<path:name>")
def agent_delete_palette(name):
    items = delete_palette(_ctx().settings, unquote(name))
    return jsonify({"status": "ok", "palettes": items})


@agent_bp.get("/api/agent/asset/<path:name>")
def agent_get_asset(name):
    safe = _UNSAFE_RE.sub("_", unquote(name))
    path = _ctx().settings.agent_assets_path / safe
    if not path.is_file():
        return jsonify({"error": "Ассет не найден"}), 404
    return send_file(str(path))


@agent_bp.get("/api/agent/favicon")
def agent_favicon():
    ctx = _ctx()
    prof = ctx.profile
    if prof.favicon_type == "file" and prof.favicon_file:
        safe = _UNSAFE_RE.sub("_", prof.favicon_file)
        path = ctx.settings.agent_assets_path / safe
        if path.is_file():
            return send_file(str(path))
    glyph = prof.favicon_initials or prof.logo_initials or "AI"
    color = prof.palette.get("agent_bg") or "#7c3aed"
    if prof.favicon_type == "emoji":
        glyph = prof.logo_emoji or glyph
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="64" height="64">'
        f'<rect width="64" height="64" rx="14" fill="{color}"/>'
        '<text x="32" y="41" font-size="30" text-anchor="middle" '
        f'fill="#ffffff" font-family="Segoe UI, sans-serif">{_xml(glyph)}</text></svg>'
    )
    return send_file(BytesIO(svg.encode("utf-8")), mimetype="image/svg+xml")


def _xml(text: str) -> str:
    return (text.replace("&", "&amp;").replace("<", "&lt;")
                .replace(">", "&gt;").replace('"', "&quot;"))


# ----------------------------------------------------------------- document
@agent_bp.post("/api/agent/document")
def agent_upload_document():
    ctx = _ctx()
    if "file" not in request.files:
        return jsonify({"error": "Файл не передан"}), 400
    try:
        info = ctx.ingestion.save_upload(request.files["file"])
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    try:
        info["rebuilt"] = ctx.rebuild()
    except Exception as e:  # pragma: no cover - rebuild failure
        return jsonify({"status": "uploaded", "document": info,
                        "warning": f"Файл загружен, но переиндексация не удалась: {e}"}), 200
    return jsonify({"status": "ok", "document": info, "rebuilt": info["rebuilt"]})


# --------------------------------------------------------------------- chat
@agent_bp.post("/api/agent/chat/message")
def agent_chat_message():
    ctx = _ctx()
    data = request.get_json(silent=True) or {}
    query = (data.get("query") or "").strip()
    if not query:
        return jsonify({"error": "Введите сообщение"}), 400

    prof = ctx.profile
    sid = data.get("session_id")
    if not sid:
        sid = ctx.chat.create_session()["id"]
    mode = data.get("mode", ctx.settings.default_mode)
    k = int(data.get("k", ctx.settings.default_k))

    system_prompt = (
        (prof.system_prompt or "").strip()
        or compose_system_prompt(SYSTEM_PROMPT, prof.style_id, prof.custom_prompt, prof.length)
    )
    history = [{"role": m["role"], "text": m["text"]} for m in ctx.chat.get_messages(sid)]
    ctx.chat.add_message(sid, "user", query)
    result = generate_answer(
        ctx.retriever, query, mode=mode, k=k, temperature=prof.temperature,
        history=history, system_prompt=system_prompt,
        max_tokens=tokens_for_length(prof.length),
    )
    # Never surface technical wording — answer with a friendly fallback + contacts.
    answer = result.get("answer", "")
    if not result.get("results"):
        result["answer"] = fallback_message(prof, "no_context")
    elif ctx.retriever.client is None:
        result["answer"] = fallback_message(prof, "no_llm")
    elif answer.startswith("Ошибка LLM"):
        result["answer"] = fallback_message(prof, "error")
    # One entry per document: the agent lists document names without repeats.
    result["citations"] = unique_sources(result.get("citations", []))
    if not prof.show_sources:
        result["citations"] = []
    ctx.chat.add_message(sid, "assistant", result["answer"], result.get("citations", []))
    return jsonify({"session_id": sid, "query": query, "mode": mode,
                    "show_sources": prof.show_sources, **result})


# ------------------------------------------------------------------- export
@agent_bp.post("/api/agent/export")
def agent_export():
    # Imported lazily: the exporter ships only with the dashboard, not with the
    # delivered agent (where this route is never used).
    from ..agent_export import build_standalone_zip, slugify

    ctx = _ctx()
    data = request.get_json(silent=True) or {}
    _apply_updates(ctx.profile, data)
    _fill_subtitle(ctx)
    save_agent_profile(ctx.settings, ctx.profile)
    payload = build_standalone_zip(ctx.settings, ctx.profile)
    filename = slugify(ctx.profile.agent_name) + "-agent.zip"
    resp = send_file(BytesIO(payload), mimetype="application/zip",
                     as_attachment=True, download_name=filename)
    # Force a plain ASCII filename (no RFC5987 ``filename*``), otherwise some
    # browsers save the archive with a mangled ``...zip; filename_=...`` name.
    resp.headers["Content-Disposition"] = f'attachment; filename="{filename}"'
    return resp
