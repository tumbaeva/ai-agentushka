"""Retrieval evaluation: per-question metrics and a summary heatmap.

Test questions are editable and stored in ``data/eval_questions.json`` (falling
back to :data:`DEFAULT_QUESTIONS`). Each question defines the query, the relevant
document names and keyword markers; ground truth by keywords survives
re-chunking and model changes.
"""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path

from .normalize import highlight_fragment

METRIC_COLS = ["hit@k", "doc_acc@k", "recall@k", "precision@k", "doc_precision@k", "mrr@k"]

DEFAULT_QUESTIONS = [
    {
        "id": "Q1_nominations",
        "query": "Какие номинации предусмотрены Конкурсом?",
        "relevant_docs": ["Информационное письмо", "Положение о конкурсе рисунков_АлтГПУ_2026"],
        "keywords": ["номинац"],
    },
    {
        "id": "Q2_dates",
        "query": "Какие сроки приема заявок и работ на Конкурс?",
        "relevant_docs": ["Информационное письмо", "Положение о конкурсе рисунков_АлтГПУ_2026"],
        "keywords": ["срок", "прием", "этап", "дата"],
    },
    {
        "id": "Q3_file_requirements",
        "query": "Какие требования к загружаемым работам по размеру и типу файла?",
        "relevant_docs": ["Информационное письмо", "Положение о конкурсе рисунков_АлтГПУ_2026"],
        "keywords": ["требован", "формат", "размер", "мбайт", "мб"],
    },
    {
        "id": "Q4_organizer_contact",
        "query": "Кто ответственный за организацию конкурса и как задать вопросы?",
        "relevant_docs": ["Информационное письмо"],
        "keywords": ["ответственн", "контакт", "вопрос", "телефон", "почт"],
    },
]


def load_questions(path: Path | None) -> list[dict]:
    if path is not None and Path(path).is_file():
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            if isinstance(data, list) and data:
                return data
        except (OSError, ValueError):
            pass
    return copy.deepcopy(DEFAULT_QUESTIONS)


def save_questions(path: Path, questions: list[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(questions, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _norm_name(name: str) -> str:
    """Document identity ignoring extension and case."""
    return name.rsplit(".", 1)[0].strip().lower()


def _is_relevant(chunk: dict, item: dict) -> bool:
    doc_set = {_norm_name(d) for d in item.get("relevant_docs", [])}
    if _norm_name(chunk["file_name"]) in doc_set:
        low = chunk["text"].lower()
        if any(kw.lower() in low for kw in item.get("keywords", [])):
            return True
    return False


def evaluate_one(retriever, item: dict, mode: str, k: int = 4) -> dict:
    retrieved = retriever.retrieve(item["query"], mode=mode, k=k)
    ids = [r["chunk_id"] for r in retrieved]
    docs = [r["file_name"] for r in retrieved]
    relevant_ids = [r["chunk_id"] for r in retrieved if _is_relevant(r, item)]
    doc_set = {_norm_name(d) for d in item.get("relevant_docs", [])}
    doc_hits = [d for d in docs if _norm_name(d) in doc_set]

    mrr = 0.0
    for rank, r in enumerate(retrieved, start=1):
        if _is_relevant(r, item):
            mrr = 1.0 / rank
            break

    expected = max(len([c for c in retriever.index.chunks
                        if _norm_name(c["file_name"]) in doc_set
                        and any(kw.lower() in c["text"].lower() for kw in item.get("keywords", []))]), 1)

    return {
        "hit@k": 1.0 if relevant_ids else 0.0,
        "doc_acc@k": 1.0 if doc_hits else 0.0,
        "recall@k": round(len(set(relevant_ids)) / expected, 3),
        "precision@k": round(len(set(relevant_ids)) / max(len(ids), 1), 3),
        "doc_precision@k": round(len(set(doc_hits)) / max(len(set(docs)), 1), 3),
        "mrr@k": round(mrr, 3),
    }


def evaluate_detail(retriever, item: dict, mode: str, k: int = 4) -> list[dict]:
    """Retrieved chunks for one question+mode, each flagged as relevant or not."""
    retrieved = retriever.retrieve(item["query"], mode=mode, k=k)
    keywords = item.get("keywords", [])
    out = []
    for rank, r in enumerate(retrieved, start=1):
        low = r["text"].lower()
        matched = [kw for kw in keywords if kw.lower() in low]
        frag = highlight_fragment(r["text"], " ".join(keywords) or item["query"])
        out.append({
            "rank": rank,
            "chunk_id": r["chunk_id"],
            "file_name": r["file_name"],
            "section": r.get("section", ""),
            "score": r["score"],
            "relevant": _is_relevant(r, item),
            "matched_keywords": matched,
            "snippet": frag["snippet"],
            "highlights": frag["highlights"],
        })
    return out


def run_eval(retriever, k: int = 4, modes: list[str] | None = None,
             questions: list[dict] | None = None) -> list[dict]:
    from .retrieval import ALL_MODES

    modes = modes or list(ALL_MODES)
    questions = questions or DEFAULT_QUESTIONS
    rows = []
    for item in questions:
        for mode in modes:
            try:
                metrics = evaluate_one(retriever, item, mode=mode, k=k)
            except Exception as e:  # pragma: no cover
                metrics = {m: 0.0 for m in METRIC_COLS}
                metrics["error"] = f"{type(e).__name__}: {e}"
            rows.append({"qid": item.get("id", item.get("query", "" )[:20]), "mode": mode, **metrics})
    return rows


def summarize(rows: list[dict]) -> dict[str, dict]:
    from .retrieval import ALL_MODES

    by_mode: dict[str, list[dict]] = {}
    for row in rows:
        by_mode.setdefault(row["mode"], []).append(row)
    summary = {}
    for mode in ALL_MODES:
        items = by_mode.get(mode, [])
        if not items:
            continue
        summary[mode] = {
            m: round(sum(i.get(m, 0.0) for i in items) / len(items), 3) for m in METRIC_COLS
        }
    return summary


def plot_heatmap(rows: list[dict], save_path: Path | None = None) -> Path | None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    summary = summarize(rows)
    if not summary:
        return None
    modes = list(summary.keys())
    data = [[summary[m][c] for c in METRIC_COLS] for m in modes]
    n_questions = len({r.get("qid") for r in rows})

    plt.figure(figsize=(10, max(3, len(modes) * 0.6 + 2)))
    sns.set_theme(style="whitegrid")
    ax = sns.heatmap(data, annot=True, fmt=".3f", cmap="YlGnBu", vmin=0.0, vmax=1.0,
                     linewidths=0.5, linecolor="gray",
                     xticklabels=METRIC_COLS, yticklabels=modes,
                     cbar_kws={"shrink": 0.8, "label": "Score"})
    ax.set_xlabel("")
    ax.set_ylabel("Mode")
    plt.xticks(rotation=45, ha="right")
    plt.title(f"Метрики поиска: среднее по режимам (k=4, {n_questions} вопросов)",
              fontsize=13, pad=15)
    plt.tight_layout()
    if save_path:
        plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
        plt.close()
        return save_path
    plt.close()
    return None


# ---------------------------------------------------------------------------
# LLM-assisted generation of test questions
# ---------------------------------------------------------------------------
_RU_STOP = {
    "какой", "какая", "какие", "каком", "который", "которые", "чтобы", "нужно", "можно",
    "должен", "должны", "должно", "кого", "кому", "чем", "что", "где", "когда",
    "сколько", "почему", "зачем", "этого", "этом", "этой", "есть", "самый", "каким",
    "конкурс", "конкурса", "конкурсе",
}


def _parse_json_array(text: str) -> list:
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    try:
        data = json.loads(text)
        return data if isinstance(data, list) else []
    except ValueError:
        pass
    start, end = text.find("["), text.rfind("]")
    if start != -1 and end != -1 and end > start:
        try:
            data = json.loads(text[start:end + 1])
            return data if isinstance(data, list) else []
        except ValueError:
            return []
    return []


def _derive_keywords(query: str, docs) -> list[str]:
    tokens = re.findall(r"[A-Za-z\u0400-\u04FF0-9]+", query.lower())
    cands = [t for t in tokens if len(t) >= 5 and t not in _RU_STOP]
    corpus = "\n".join(d.text.lower() for d in docs)
    kws: list[str] = []
    for t in cands:
        stem = t[:6] if len(t) > 6 else t
        if stem in corpus and stem not in kws:
            kws.append(stem)
        if len(kws) >= 3:
            break
    if not kws:
        kws = [c[:6] for c in cands[:2]]
    return kws


def suggest_questions(client, model: str, documents: list, count: int = 5,
                     temperature: float = 0.7) -> list[dict]:
    """Ask the LLM to draft evaluation questions grounded in the KB documents."""
    if client is None:
        raise RuntimeError("LLM недоступен: укажите ключ GigaChat в настройках")
    if not documents:
        return []
    doc_names = [d.name for d in documents]
    context = "\n\n".join(f"### {d.name}\n{d.text[:600]}" for d in documents)
    prompt = (
        "Ты помогаешь составить тестовые вопросы для оценки качества поиска по базе знаний.\n"
        f"Документы базы (имя и фрагмент):\n{context}\n\n"
        f"Сгенерируй {count} разнообразных вопросов на русском, ответы на которые есть в этих документах.\n"
        "Для каждого вопроса укажи, к каким документам он относится — выбери имена строго из списка:\n"
        f"{doc_names}\n\n"
        "Верни СТРОГО JSON-массив без пояснений:\n"
        '[{"query": "текст вопроса", "docs": ["Имя документа"]}]'
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    parsed = _parse_json_array(resp.choices[0].message.content or "")
    out = []
    for i, item in enumerate(parsed, start=1):
        query = (item.get("query") or "").strip()
        if not query:
            continue
        docs = [d for d in (item.get("docs") or []) if d in doc_names] or doc_names
        chosen = [d for d in documents if d.name in docs]
        out.append({
            "id": f"G{i}",
            "query": query,
            "relevant_docs": docs,
            "keywords": _derive_keywords(query, chosen),
        })
    return out
