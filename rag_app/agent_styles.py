"""Answer-style presets for the AI agent.

Each preset is a short instruction appended to the grounding system prompt.
:func:`compose_system_prompt` always keeps the anti-hallucination base rules, so
a custom prompt can change the tone but never disables fact-grounding.
"""

from __future__ import annotations

from .generation import SYSTEM_PROMPT

# ``medium`` maps to "no explicit limit" so endpoints that reject ``max_tokens``
# keep working.
LENGTH_TOKENS = {"short": 300, "medium": None, "long": 1600}

LENGTH_HINTS = {
    "short": "Отвечай очень кратко: 1–2 предложения по существу.",
    "medium": "Отвечай умеренно подробно, без лишней воды.",
    "long": "Отвечай подробно, со структурой и перечислением деталей.",
}

ANSWER_STYLES = [
    {"id": "business", "label_ru": "Официальный (деловой)", "label_en": "Formal (business)",
     "instruction": "Тон строго официальный и деловой: нейтрально, вежливо, без разговорных выражений и эмодзи.",
     "temperature": 0.2, "length": "medium"},
    {"id": "friendly", "label_ru": "Дружелюбный", "label_en": "Friendly",
     "instruction": "Тон дружелюбный и тёплый: приветливо, простыми словами, можно мягко обратиться к пользователю.",
     "temperature": 0.4, "length": "medium"},
    {"id": "humorous", "label_ru": "Шуточный (с юмором)", "label_en": "Humorous",
     "instruction": "Допусти лёгкий уместный юмор и остроумную формулировку, но факты из материалов передавай точно и не искажай.",
     "temperature": 0.7, "length": "medium"},
    {"id": "concise", "label_ru": "Краткий (телеграфный)", "label_en": "Concise",
     "instruction": "Отвечай предельно кратко: тезисы и списки, без воды и вступлений.",
     "temperature": 0.1, "length": "short"},
    {"id": "expert", "label_ru": "Развёрнутый (экспертный)", "label_en": "Detailed (expert)",
     "instruction": "Отвечай как эксперт: развёрнуто, со структурой (разделы, списки), поясняя детали и нюансы.",
     "temperature": 0.3, "length": "long"},
    {"id": "explain_simple", "label_ru": "Объясни просто", "label_en": "Explain simply",
     "instruction": "Объясняй максимально просто, как для новичка: короткие фразы, бытовые аналогии, без терминов.",
     "temperature": 0.4, "length": "medium"},
    {"id": "legal", "label_ru": "Формально-юридический", "label_en": "Legal",
     "instruction": "Используй строго юридический стиль: точные формулировки, ссылки на пункты и разделы документов, без оценок.",
     "temperature": 0.1, "length": "long"},
    {"id": "informal_emoji", "label_ru": "Неформальный (с эмодзи)", "label_en": "Informal (emoji)",
     "instruction": "Общайся непринуждённо, добавляй уместные эмодзи и живой тон, но факты передавай точно.",
     "temperature": 0.6, "length": "medium"},
]

DEFAULT_STYLE_ID = "business"
STYLE_IDS = {s["id"] for s in ANSWER_STYLES}
_STYLE_BY_ID = {s["id"]: s for s in ANSWER_STYLES}


def get_style(style_id: str) -> dict:
    return _STYLE_BY_ID.get(style_id or "", _STYLE_BY_ID[DEFAULT_STYLE_ID])


def tokens_for_length(length: str) -> int | None:
    return LENGTH_TOKENS.get((length or "").strip())


# Friendly fallbacks used when the agent cannot answer; the organizers' phone is
# appended so the user always has a way to reach a human.
FALLBACK_BASE = {
    "no_context": "К сожалению, я не нашёл ответ на этот вопрос в материалах конкурса.",
    "no_llm": "К сожалению, сейчас я не могу сформировать ответ.",
    "error": "К сожалению, не удалось получить ответ. Попробуйте, пожалуйста, ещё раз позже.",
}


def fallback_message(profile, reason: str = "no_context") -> str:
    base = FALLBACK_BASE.get(reason, FALLBACK_BASE["no_context"])
    phone = (getattr(profile, "contact_phone", "") or "").strip()
    if phone:
        base += f" Свяжитесь с организаторами: {phone}."
    return base


def compose_system_prompt(base: str, style_id: str, custom_prompt: str = "",
                          length: str = "medium") -> str:
    """Grounding base + tone instruction (custom wins) + length clause."""
    style = get_style(style_id)
    tone = (custom_prompt or "").strip() or style["instruction"]
    hint = LENGTH_HINTS.get(length or "", LENGTH_HINTS["medium"])
    parts = [(base or SYSTEM_PROMPT or "").strip(), tone, hint]
    return "\n".join(p for p in parts if p)
