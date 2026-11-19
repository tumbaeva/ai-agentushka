"""AI-agent profile: branding, theme, sizing, layout and answer style.

The profile is UI-owned and persisted separately from ``Settings`` (a flat
dataclass) so its nested structure does not complicate ``settings.json``.
Stored in ``data/agent_profile.json`` and read both by the dashboard
«AI-агент» tab and by an exported standalone agent application.

Colors are intentionally simple: three groups (user / agent / interface), each
with a background ("поле") and a text color, plus one page background and a
subtitle color — 8 colors total. Named colour sets ("palettes") are stored in
``data/palettes.json``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, fields, asdict
from pathlib import Path

from .config import Settings

PROFILE_FILENAME = "agent_profile.json"
PALETTES_FILENAME = "palettes.json"

# The 8 colors the agent is themed with.
DEFAULT_PALETTE = {
    "user_bg": "#0ea5e9",     # Пользователь — цвет поля (фон сообщений)
    "user_text": "#ffffff",   # Пользователь — цвет шрифта
    "agent_bg": "#7c3aed",    # Агент — цвет поля (ответы + эмблема/лого)
    "agent_text": "#ffffff",  # Агент — цвет шрифта
    "ui_bg": "#0f172a",       # Интерфейс — цвет фона (страница и панели)
    "ui_field": "#0ea5e9",    # Интерфейс — цвет поля (кнопки и поля ввода)
    "ui_text": "#e2e8f0",     # Интерфейс — цвет шрифта
    "ui_subtitle": "#94a3b8", # Интерфейс — цвет подписи (подзаголовка) под названием агента
}

PALETTE_KEYS = tuple(DEFAULT_PALETTE.keys())

# Old palette keys (accent / accent-2 / brand / corporate era) -> new keys.
_LEGACY_PALETTE_MAP = {
    "accent": "ui_field",
    "accent2": "ui_text",
    "bg": "ui_bg",
    "panel": "ui_bg",
    "text": "ui_text",
    "user_color": "user_bg",
    "agent_color": "agent_bg",
    "chip_color": "ui_field",
}


def _migrate_palette(value) -> dict:
    """Merge a stored palette onto the defaults, mapping legacy keys."""
    merged = dict(DEFAULT_PALETTE)
    if not isinstance(value, dict):
        return merged
    for key, val in value.items():  # legacy keys first
        if isinstance(val, str) and key in _LEGACY_PALETTE_MAP:
            merged[_LEGACY_PALETTE_MAP[key]] = val
    for key, val in value.items():  # new keys win
        if isinstance(val, str) and key in DEFAULT_PALETTE:
            merged[key] = val
    return merged


def profile_path(settings: Settings) -> Path:
    return settings.data_path / PROFILE_FILENAME


@dataclass
class AgentProfile:
    # branding
    agent_name: str = "AI-агент"
    subtitle: str = ""
    logo_type: str = "emoji"          # emoji | initials | file
    logo_emoji: str = "🤖"
    logo_initials: str = "AI"
    logo_file: str = ""               # filename under data/agent_assets/
    favicon_type: str = "emoji"       # emoji | initials | file
    favicon_file: str = ""            # filename under data/agent_assets/
    favicon_initials: str = "AI"
    # colors
    palette: dict = field(default_factory=lambda: dict(DEFAULT_PALETTE))
    font_size: str = "M"              # S | M | L
    button_size: str = "normal"       # small | normal | large
    radius: int = 12
    compact: bool = False
    sidebar_width: int = 480
    # layout / chat
    welcome_message: str = ""
    starter_questions: list = field(default_factory=list)
    show_sources: bool = True
    ui_language: str = "ru"           # ru | en
    contact_phone: str = ""           # shown when the agent cannot answer
    # answer style
    style_id: str = "business"
    custom_prompt: str = ""
    system_prompt: str = ""         # full override; empty = built from preset
    temperature: float = 0.2
    length: str = "medium"            # short | medium | long

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict | None) -> "AgentProfile":
        prof = cls()
        if not isinstance(data, dict):
            return prof
        allowed = {f.name for f in fields(cls)}
        for key, value in data.items():
            # ``palette`` is handled below; legacy theme keys are dropped.
            if key in ("palette", "brand_color", "theme_preset", "theme_mode"):
                continue
            if key not in allowed:
                continue
            if key == "starter_questions":
                prof.starter_questions = (
                    [str(x) for x in value if str(x).strip()] if isinstance(value, list) else []
                )
            else:
                setattr(prof, key, value)
        raw = data.get("palette")
        prof.palette = _migrate_palette(raw)
        # legacy top-level brand_color -> agent colour (if nothing set it)
        brand = data.get("brand_color")
        if isinstance(brand, str) and brand and not _raw_has(raw, "agent_color", "agent_bg"):
            prof.palette["agent_bg"] = brand
        return prof


def _raw_has(raw, *keys) -> bool:
    return isinstance(raw, dict) and any(isinstance(raw.get(k), str) for k in keys)


def load_agent_profile(settings: Settings) -> AgentProfile:
    path = profile_path(settings)
    if path.is_file():
        try:
            return AgentProfile.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            pass
    return AgentProfile()


def save_agent_profile(settings: Settings, profile: AgentProfile) -> None:
    path = profile_path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(profile.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )


# ------------------------------------------------------------------ palettes
def palettes_path(settings: Settings) -> Path:
    return settings.data_path / PALETTES_FILENAME


def load_palettes(settings: Settings) -> list[dict]:
    path = palettes_path(settings)
    if path.is_file():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []
        if isinstance(data, list):
            out = []
            for item in data:
                if isinstance(item, dict) and str(item.get("name", "")).strip():
                    out.append({"name": str(item["name"]), "colors": _migrate_palette(item.get("colors"))})
            return out
    return []


def _write_palettes(settings: Settings, items: list[dict]) -> None:
    path = palettes_path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")


def save_palette(settings: Settings, name: str, colors) -> list[dict]:
    name = (name or "").strip()
    if not name:
        raise ValueError("Укажите название палитры")
    items = [p for p in load_palettes(settings) if p["name"].lower() != name.lower()]
    items.append({"name": name, "colors": _migrate_palette(colors)})
    _write_palettes(settings, items)
    return items


def delete_palette(settings: Settings, name: str) -> list[dict]:
    target = (name or "").strip().lower()
    items = [p for p in load_palettes(settings) if p["name"].lower() != target]
    _write_palettes(settings, items)
    return items
