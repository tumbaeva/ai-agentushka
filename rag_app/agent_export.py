"""Build a standalone AI-agent application zip from the current project.

The export is intentionally lean: it ships only what the chat-only agent needs —
the agent code subset, the knowledge base, the tuned model settings and the
prebuilt index snapshot (so retrieval works immediately, without a rebuild).
Dashboard-only modules, tests artefacts, chat history and evaluation reports are
left out, as is anything not required to run the agent.
"""

from __future__ import annotations

import io
import json
import re
import zipfile
from dataclasses import fields
from pathlib import Path

from .agent_profile import AgentProfile
from .config import Settings

PACKAGE_DIR = Path(__file__).resolve().parent           # .../rag_app
TEMPLATE_DIR = PACKAGE_DIR / "app_template"
SKIP_PARTS = {"__pycache__", "app_template"}
KB_FILENAME = "knowledge_base.md"

# Dashboard-only modules/templates/scripts that the chat-only agent never loads,
# plus the exporter itself (not needed by the delivered agent).
EXCLUDE_FILES = {
    # dashboard blueprints and their helpers
    "web/api.py", "web/kb_builder.py", "web/pages.py",
    "eval.py", "sweep.py", "markdown.py",
    # KB-builder + OCR: only reached through the dashboard builder (lazy), so the
    # delivered chat-only agent never imports them (and needs no OCR deps).
    "kb_builder.py", "headings.py", "ocr.py",
    # dashboard UI
    "templates/index.html", "templates/kb_builder.html",
    "templates/recommendations.html", "templates/help.html", "templates/agent.html",
    "static/app.js", "static/tooltip.js", "static/kb_builder.js",
    "static/recommendations.js", "static/agent.js",
    # the exporter is only used by the dashboard, not by the delivered agent
    "agent_export.py",
}

# Skeleton files that are not required in the final archive.
SKELETON_SKIP = {".env.example", "run.bat"}

# Paths pointing at the source machine must not leak into the exported app;
# the standalone copy re-derives them from its own location.
_MACHINE_PATH_FIELDS = {"kb_dir", "source_docs_dir", "data_dir", "env_file"}
_SECRET_FIELDS = {"hf_token", "gigachat_key_id", "gigachat_key_secret"}


_TRANSLIT = str.maketrans({
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "e",
    "ж": "zh", "з": "z", "и": "i", "й": "y", "к": "k", "л": "l", "м": "m",
    "н": "n", "о": "o", "п": "p", "р": "r", "с": "s", "т": "t", "у": "u",
    "ф": "f", "х": "kh", "ц": "ts", "ч": "ch", "ш": "sh", "щ": "shch",
    "ъ": "", "ы": "y", "ь": "", "э": "e", "ю": "yu", "я": "ya",
})


def slugify(name: str) -> str:
    """ASCII-safe file stem: transliterate Cyrillic, keep it a clean filename.

    Kept ASCII on purpose: non-ASCII download names make some browsers save the
    file (and later the extracted folder) with a mangled ``filename*`` tail.
    """
    text = (name or "agent").strip().lower().translate(_TRANSLIT)
    text = re.sub(r"[^0-9a-z._-]+", "-", text)
    text = re.sub(r"-{2,}", "-", text).strip("-._")
    return text or "agent"


def _safe_filename(name: str) -> str:
    """A Windows-safe base name for the shortcut.

    Parentheses are dropped too: the launcher passes the name to ``cmd``, where
    ``)`` would break block parsing.
    """
    safe = re.sub(r'[\\/:*?"<>|()\r\n]+', "", name or "").strip().strip(".")
    return safe or "AI-агент"


def _render_skeleton(text: str, profile: AgentProfile) -> str:
    replacements = {
        "{{AGENT_NAME}}": profile.agent_name or "AI-агент",
        "{{AGENT_FILE}}": _safe_filename(profile.agent_name),
        "{{SUBTITLE}}": profile.subtitle or "",
        "{{AGENT_BG}}": (profile.palette or {}).get("agent_bg", "#7c3aed"),
        "{{LANG}}": profile.ui_language or "ru",
    }
    for token, value in replacements.items():
        text = text.replace(token, value)
    return text


def _public_settings(settings: Settings) -> dict:
    data = {f.name: getattr(settings, f.name) for f in fields(Settings)}
    for key in _SECRET_FIELDS | _MACHINE_PATH_FIELDS:
        data.pop(key, None)
    # The exported app bundles docs/knowledge_base.md, so it must index that file.
    data["source"] = "kb_md"
    return data


def _latest_snapshot(settings: Settings) -> Path | None:
    base = settings.models_path
    if not base.is_dir():
        return None
    folders = sorted(p for p in base.iterdir()
                     if p.is_dir() and (p / "index.faiss").is_file())
    return folders[-1] if folders else None


def build_standalone_zip(settings: Settings, profile: AgentProfile) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        # 1) the agent code subset (domain modules, web/agent.py, chat-only UI)
        for path in sorted(PACKAGE_DIR.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(PACKAGE_DIR)
            if any(part in SKIP_PARTS for part in rel.parts):
                continue
            rel_posix = rel.as_posix()
            if rel_posix in EXCLUDE_FILES:
                continue
            zf.writestr((Path("rag_app") / rel).as_posix(), path.read_bytes())

        # 2) entrypoint skeleton (token substitution), written at the root
        if TEMPLATE_DIR.is_dir():
            for path in sorted(TEMPLATE_DIR.iterdir()):
                if not path.is_file() or path.name in SKELETON_SKIP:
                    continue
                text = _render_skeleton(path.read_text(encoding="utf-8"), profile)
                # Windows batch files require CRLF; LF breaks %~dp0 parsing.
                if path.suffix.lower() in {".bat", ".cmd"}:
                    text = text.replace("\r\n", "\n").replace("\n", "\r\n")
                zf.writestr(path.name, text)

        # 3) agent profile (branding, theme, answer style)
        zf.writestr(
            "data/agent_profile.json",
            json.dumps(profile.to_dict(), ensure_ascii=False, indent=2),
        )

        # 4) tuned model settings (no secrets, no machine-specific paths)
        zf.writestr(
            "data/settings.json",
            json.dumps(_public_settings(settings), ensure_ascii=False, indent=2),
        )

        # 5) knowledge base
        kb_file = settings.kb_path / KB_FILENAME
        if kb_file.is_file():
            zf.writestr(f"docs/{KB_FILENAME}", kb_file.read_bytes())

        # 6) prebuilt index snapshot (chunks + vectors + model meta)
        snapshot = _latest_snapshot(settings)
        if snapshot is not None:
            for name in ("index.faiss", "chunks.json", "meta.json"):
                member = snapshot / name
                if member.is_file():
                    zf.writestr(f"data/index/{snapshot.name}/{name}", member.read_bytes())

        # 7) only the branding assets actually referenced by the profile
        assets = settings.agent_assets_path
        for fname in {profile.logo_file, profile.favicon_file}:
            if not fname:
                continue
            asset = assets / Path(fname).name
            if asset.is_file():
                zf.writestr(f"data/agent_assets/{asset.name}", asset.read_bytes())

        # 8) .env with just the LLM credential; no HuggingFace token (the shipped
        # embedding model is public and needs no auth).
        env_lines = []
        if settings.gigachat_key_id:
            env_lines.append(f"GIGACHAT_API_KEY_ID={settings.gigachat_key_id}")
        if settings.gigachat_key_secret:
            env_lines.append(f"GIGACHAT_API_KEY_SECRET={settings.gigachat_key_secret}")
        if env_lines:
            zf.writestr(".env", "\n".join(env_lines) + "\n")

    return buf.getvalue()
