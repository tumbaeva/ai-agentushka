"""RAG prompt building, answer generation and source citations."""

from __future__ import annotations

import re

from .normalize import highlight_fragment

SYSTEM_PROMPT = """Вы — помощник по базе знаний конкурса детского рисунка.
Отвечайте по приведённым материалам: данным в блоке «СПРАВКА О ДОКУМЕНТАХ» и найденным фрагментам.
Если точного ответа нет — честно скажите об этом.
Не выдумывайте правила, сроки и цифры, которых нет в материалах.
Отвечайте спокойно, коротко и по делу.
В конце ответа перечислите использованные источники в формате: [doc_name — раздел]."""


def build_context(retrieved: list[dict]) -> str:
    blocks = []
    for item in retrieved:
        header = f"[{item['file_name']} — {item.get('section') or '—'}]"
        blocks.append(f"{header}\n{item['text']}")
    return "\n\n".join(blocks)


def doc_overview(chunks: list[dict], per_doc_chars: int = 500) -> str:
    """Short per-document preamble (titles/intro), always given to the LLM.

    Guarantees stable facts like the contest name are in context even when
    retrieval did not surface the intro chunk.
    """
    best: dict[str, dict] = {}
    order: list[str] = []
    for c in chunks or []:
        fn = c.get("file_name", "")
        if not fn:
            continue
        if fn not in best:
            best[fn] = c
            order.append(fn)
        elif c.get("chunk_index", 0) < best[fn].get("chunk_index", 0):
            best[fn] = c
    blocks = []
    for fn in order:
        text = (best[fn].get("text") or "")[:per_doc_chars].strip()
        if text:
            blocks.append(f"### {fn}\n{text}")
    return "\n\n".join(blocks)


def build_prompt(query: str, retrieved: list[dict], history: list[dict] | None = None,
                 overview: str | None = None, system_prompt: str | None = None) -> list[dict]:
    """Return the chat messages array for the LLM."""
    messages = [{"role": "system", "content": system_prompt or SYSTEM_PROMPT}]
    if history:
        for turn in history[-6:]:
            role = "user" if turn.get("role") == "user" else "assistant"
            messages.append({"role": role, "content": turn.get("text", "")})
    parts = []
    if overview:
        parts.append(f"СПРАВКА О ДОКУМЕНТАХ (общие сведения, можно использовать):\n{overview}")
    parts.append(f"НАЙДЕННЫЕ ФРАГМЕНТЫ:\n{build_context(retrieved)}")
    parts.append(f"ВОПРОС:\n{query}")
    messages.append({"role": "user", "content": "\n\n".join(parts)})
    return messages


def generate_answer(retriever, query: str, mode: str = "hybrid", k: int = 4,
                    temperature: float = 0.2, history: list[dict] | None = None,
                    system_prompt: str | None = None,
                    max_tokens: int | None = None) -> dict:
    retrieved = retriever.retrieve(query, mode=mode, k=k)
    if not retrieved:
        return {"answer": "Недостаточно релевантного контекста в базе знаний.",
                "results": [], "citations": []}

    citations = build_citations(retrieved, query)
    client = retriever.client
    if client is None:
        return {
            "answer": "LLM не настроен: укажите ключ GigaChat в разделе «Настройки». "
                      "Ниже приведены наиболее релевантные фрагменты.",
            "results": retrieved,
            "citations": citations,
        }
    overview = doc_overview(getattr(getattr(retriever, "index", None), "chunks", []))
    messages = build_prompt(query, retrieved, history=history, overview=overview,
                            system_prompt=system_prompt)
    kwargs = {
        "model": retriever.settings.llm_model,
        "messages": messages,
        "temperature": temperature,
    }
    if max_tokens:
        kwargs["max_tokens"] = max_tokens
    try:
        resp = client.chat.completions.create(**kwargs)
        answer = (resp.choices[0].message.content or "").strip()
    except Exception as e:
        answer = f"Ошибка LLM: {type(e).__name__}: {e}"
    return {"answer": answer, "results": retrieved, "citations": citations}


def build_citations(retrieved: list[dict], query: str) -> list[dict]:
    citations = []
    for i, item in enumerate(retrieved, start=1):
        frag = highlight_fragment(item["text"], query)
        citations.append({
            "rank": i,
            "chunk_id": item["chunk_id"],
            "file_name": item["file_name"],
            "section": item.get("section", ""),
            "label": source_label(item["file_name"], item.get("section", "")),
            "category": item.get("category", ""),
            "score": item["score"],
            "snippet": frag["snippet"],
            "highlights": frag["highlights"],
        })
    return citations


def unique_sources(citations: list[dict]) -> list[dict]:
    """Collapse citations so each document name appears only once.

    Keeps the first (best-ranked) chunk per document and renumbers ``rank`` so
    the list stays ``1..N``. Items without a usable name are kept as-is.
    """
    seen: set[str] = set()
    out: list[dict] = []
    for c in citations or []:
        key = (c.get("label") or c.get("file_name") or "").strip().lower()
        if key:
            if key in seen:
                continue
            seen.add(key)
        out.append(c)
    for i, c in enumerate(out, start=1):
        c["rank"] = i
    return out


_EXT_RE = re.compile(r"\.(md|markdown|txt|docx|doc|pdf)$", re.IGNORECASE)
_KB_NAME_RE = re.compile(r"^(knowledge[\s_-]*base|база[\s_-]*знаний)[\s_-]*", re.IGNORECASE)


def clean_source_name(name: str) -> str:
    """Strip file extensions and the knowledge-base file token from a name."""
    text = (name or "").strip()
    text = _EXT_RE.sub("", text)
    text = _KB_NAME_RE.sub("", text)
    return text.strip(" _-.").strip()


def source_label(file_name: str, section: str = "") -> str:
    """User-facing source label: the document name, never the knowledge base."""
    name = clean_source_name(file_name)
    if name:
        return name
    section = clean_source_name(section)
    return section or "Документ"


def strip_inline_sources(answer: str) -> str:
    return re.sub(r"\n?Источники:.*$", "", answer, flags=re.IGNORECASE | re.DOTALL).strip()
