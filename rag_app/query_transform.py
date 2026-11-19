"""Query understanding: normalization, synonyms, rewriting and HyDE.

All optional transforms degrade gracefully when the LLM client is unavailable.
"""

from __future__ import annotations

import re

# Niche synonym map (education / competitions). Keys are matched as whole words.
SYNONYMS = {
    "конкурс": ["соревнование", "мероприятие"],
    "положение": ["правила", "регламент", "условия"],
    "номинация": ["категория", "направление"],
    "работа": ["рисунок", "заявка", "поделка"],
    "срок": ["дедлайн", "дата", "период"],
    "требования": ["условия", "критерии", "формат"],
    "награда": ["приз", "диплом", "грамота"],
    "участник": ["ребёнок", "школьник", "воспитанник"],
    "заявка": ["регистрация", "форма"],
    "жюри": ["эксперты", "оценка"],
}


def normalize_query(query: str) -> str:
    q = query.replace("ё", "е").replace("Ё", "Е")
    q = re.sub(r"\s+", " ", q).strip()
    return q


def expand_synonyms(query: str, max_extra: int = 2) -> list[str]:
    """Return query variants where a few words are swapped for synonyms."""
    low = query.lower()
    variants: list[str] = []
    for word, syns in SYNONYMS.items():
        if re.search(rf"\b{re.escape(word)}\w*", low):
            for syn in syns[:max_extra]:
                variants.append(re.sub(rf"\b{re.escape(word)}\w*", syn, query, count=1, flags=re.IGNORECASE))
    return variants


def rewrite_with_llm(query: str, client, model: str, n: int = 3, temperature: float = 0.7) -> list[str]:
    if client is None:
        return []
    try:
        prompt = (
            f"Переформулируй поисковый запрос {n} разными способами для поиска "
            f"по документам конкурса. Верни РОВНО {n} строк, без нумерации.\n\n"
            f"Исходный запрос: {query}"
        )
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        text = (resp.choices[0].message.content or "").strip()
        out = []
        for line in text.splitlines():
            line = line.strip(" \t-•*0123456789.)").strip()
            if line and line.lower() != query.lower():
                out.append(line)
        return out[:n]
    except Exception:
        return []


def hyde_with_llm(query: str, client, model: str, temperature: float = 0.3) -> str | None:
    """Hypothetical Document Embeddings: ask the LLM for a plausible answer."""
    if client is None:
        return None
    try:
        prompt = (
            "Напиши короткий (2-4 предложения) гипотетический ответ на вопрос так, "
            "как он мог бы звучать в официальном положении о конкурсе. "
            "Не выдумывай конкретных дат и цифр.\n\n"
            f"Вопрос: {query}"
        )
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        return (resp.choices[0].message.content or "").strip() or None
    except Exception:
        return None


def build_query_variants(query: str, settings, client=None) -> list[str]:
    """Assemble the full variant list used by multi-query retrieval."""
    base = normalize_query(query)
    variants = [base]
    if settings.enable_query_rewrite:
        variants.extend(expand_synonyms(base))
        variants.extend(rewrite_with_llm(base, client, settings.llm_model,
                                         temperature=settings.default_temp_mq))
    # de-dup, keep order
    seen = set()
    unique = []
    for v in variants:
        key = v.lower().strip()
        if key and key not in seen:
            seen.add(key)
            unique.append(v)
    return unique[:5]
