"""Automatic section headings for the assembled knowledge base.

The heading-aware chunker (:mod:`rag_app.chunking`) derives each chunk's
``section`` from markdown headings inside the document body. When a source
document has no structure (e.g. a scanned PDF flattened by OCR), every chunk
ends up under a single section — the file name. This module inserts ``##``
headings so the knowledge base gains meaningful sections.

Two passes, cheapest first:
  1. :func:`add_headings` — offline heuristic on formal cues.
  2. an optional LLM pass (in the builder) that re-labels by meaning.
"""

from __future__ import annotations

import re

_LIST_RE = re.compile(r"^\s*[-*•]\s+")
_APPENDIX_RE = re.compile(r"^\s*(приложение\s*(№\s*)?\d+|раздел\s*\d+)\b", re.IGNORECASE)

HEADINGS_PROMPT = """Ты расставляешь заголовки в тексте документа для базы знаний.
Ты НЕ пишешь оглавление и НЕ пересказываешь текст.
Верни ИСХОДНЫЙ текст целиком, слово в слово, только добавив перед началом смысловой части 
строку-заголовок второго уровня (##). При необходимости — ### для подпунктов.
Правила:
- НИЧЕГО не удаляй, не сокращай, не перефразируй и не переставляй. Каждое предложение исходного текста должно остаться.
- Заголовки — короткие (3–7 слов), по смыслу, без точки в конце.
- Не добавляй заголовок первого уровня (#) — это название документа, он уже есть.
- Если в тексте уже есть заголовки ## / ### — сохрани их.
- Не добавляй вступлений, пояснений и выводов. Только готовый Markdown, без обрамляющих ```.

ЗАПРЕЩЕНО возвращать только список заголовков или оглавление.

Название документа: {name}

Текст:
{text}"""


def looks_like_heading(line: str) -> bool:
    """Conservative formal test: only strong, low-risk heading cues."""
    s = line.strip()
    if not s or len(s) > 90 or s.startswith("#"):
        return False
    if _LIST_RE.match(s):
        return False
    words = s.split()
    if not (2 <= len(words) <= 12):
        return False
    if s.endswith(":"):
        return True
    if _APPENDIX_RE.match(s):
        return True
    # ALL-CAPS short line (e.g. "МИНИСТЕРСТВО ПРОСВЕЩЕНИЯ")
    if s.upper() == s and sum(c.isalpha() for c in s) >= 3:
        return True
    return False


def add_headings(text: str) -> str:
    """Promote heading-like lines to ``##`` sections (heuristic, offline)."""
    lines = text.split("\n")
    out: list[str] = []
    for i, line in enumerate(lines):
        s = line.strip()
        prev_is_heading = bool(out) and out[-1].startswith("#")
        if s and not prev_is_heading and looks_like_heading(s):
            has_body = any(ln.strip() for ln in lines[i + 1: i + 5])
            if has_body:
                out.append("## " + s.rstrip(":").strip())
                continue
        out.append(line)
    return "\n".join(out)
