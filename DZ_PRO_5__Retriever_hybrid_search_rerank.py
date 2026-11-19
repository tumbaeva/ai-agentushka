#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Retriever-инженерия: hybrid search и rerank
RAG-демо на базе знаний Всероссийского конкурса детского рисунка
«Мир науки глазами детей – 2026» (АлтГПУ).
Standalone Python script — no Jupyter required.

Run:
    python DZ_PRO_5__Retriever_hybrid_search_rerank.py

Env vars (read via python-dotenv, so a local .env also works):
    DOCS_DIR                  — path to knowledge base folder with .md files
                                (default: ./docs)
    GIGACHAT_API_KEY_ID       — GigaChat API key ID (Cloud.ru)
    GIGACHAT_API_KEY_SECRET   — GigaChat API key secret (Cloud.ru)
                                At least one of the two keys is required:
                                without them the script exits in section 8.
"""

# ============================================================
# CONTENTS
# ============================================================
#  0.  Dependencies (Зависимости: автоустановка недостающих пакетов)
#  1.  Helper: replace Jupyter's display() (Вспомогательный вывод в терминале)
#  2.  Knowledge base folder (Папка базы знаний)
#  3.  Read markdown knowledge base (Чтение .md и metadata)
#  4.  Chunking (Разбиение на чанки)
#  5.  Vector search — baseline (Векторный поиск: sentence-transformers + FAISS)
#  6.  BM25 + hybrid search with RRF (BM25 и гибридный поиск через RRF)
#  7.  Compare retrieval (Сравнение режимов поиска)
#  8.  GigaChat API (Подключение к GigaChat)
#  9.  Rerank (Реранк кандидатов через cross-encoder)
# 10.  Multi-query retrieval (Многозапросный поиск: переформулировки + RRF + rerank)
# 11.  RAG: retrieval + generation (Генерация ответа по найденному контексту)
# 12.  Full comparison of all modes (Полное сравнение всех режимов поиска и ответов)
# 13.  Retrieval metrics: hit@k, recall@k, precision@k,
#      doc_acc@k, doc_precision@k, mrr@k (Мини-оценка качества поиска)
# 14.  Heatmap of retrieval metrics (Тепловая карта метрик поиска)
# ============================================================

import os
import sys
import re
import textwrap
import uuid
import getpass
from pathlib import Path
from typing import Dict, List, Any

from dotenv import load_dotenv
load_dotenv()


# ============================================================
# 0. Dependencies (Зависимости)
# ============================================================
def _ensure_deps() -> None:
    try:
        import faiss  # noqa: F401
        import sentence_transformers  # noqa: F401
        import langchain_text_splitters  # noqa: F401
        import langchain_core  # noqa: F401
        import openai  # noqa: F401
        import rank_bm25  # noqa: F401
        import requests  # noqa: F401
        import pandas  # noqa: F401
        import matplotlib  # noqa: F401
        import seaborn  # noqa: F401
    except ImportError:
        import subprocess
        print("[setup] Installing dependencies...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-q",
            "langchain-text-splitters",
            "sentence-transformers",
            "faiss-cpu",
            "openai",
            "pandas",
            "rank_bm25",
            "requests",
            "matplotlib",
            "seaborn",
        ])


_ensure_deps()


import faiss
import numpy as np
import pandas as pd
import requests
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

from rank_bm25 import BM25Okapi
from openai import OpenAI
from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ============================================================
# 1. Helper: replace Jupyter's display() (Вспомогательный вывод в терминале)
# ============================================================
def show(obj) -> None:
    """Print a pandas DataFrame (or anything else) cleanly in a terminal."""
    if isinstance(obj, pd.DataFrame):
        with pd.option_context("display.max_colwidth", 160, "display.width", 280):
            print(obj.to_string())
    else:
        print(obj)


# ============================================================
# 2. Knowledge base folder (Папка базы знаний)
# ============================================================
KB_DIR = Path(os.environ.get("DOCS_DIR", "./docs"))
KB_DIR.mkdir(parents=True, exist_ok=True)

allowed_ext = {".md"}

kb_files = sorted(p.name for p in KB_DIR.iterdir() if p.suffix.lower() in allowed_ext)
skipped = sorted(
    p.name for p in KB_DIR.iterdir() if p.suffix.lower() not in allowed_ext
)

if not kb_files:
    print(
        f"Папка {KB_DIR} пока пустая. "
        f"Положите в неё документы ({', '.join(sorted(allowed_ext))}) "
        f"и перезапустите скрипт."
    )

print("Файлы в KB:", kb_files)
if skipped:
    print("Проигнорированы (не .md):", skipped)


# ============================================================
# 3. Read markdown knowledge base (Чтение .md и metadata)
# ============================================================
# One .md file may hold several source documents: each starts with a level-2
# heading "## <source document name>". Text before the first heading is a file
# title, so fragments shorter than MIN_DOC_CHARS are dropped as markup noise.
DOC_HEADING_RE = re.compile(r"(?m)^## +(.+?)\s*$")
MIN_DOC_CHARS = 200


def normalize_whitespace(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_markdown(text: str) -> str:
    """Removes markdown markup ('---' rules, '#' heading markers), keeps the text."""
    text = re.sub(r"(?m)^\s*-{3,}\s*$", "\n", text)
    text = re.sub(r"(?m)^#{1,6} +", "", text)
    return text


def read_md_text(path: Path) -> str:
    encodings = ["utf-8", "utf-8-sig", "cp1251", "latin1"]
    for enc in encodings:
        try:
            return path.read_text(encoding=enc)
        except Exception:
            continue
    return path.read_text(encoding="utf-8", errors="ignore")


# Тема чанка/документа определяется по ключевым словам текста корпуса.
# Порядок важен: при равном числе совпадений выигрывает правило, стоящее выше.
CATEGORY_RULES = [
    ("Номинации и направления", [
        "номинации конкурса", "следующим номинациям", "тематические направления",
        "тематическое направление", "тематическому направлению",
        "тематическим направлением",
    ]),
    ("Сроки и этапы", [
        "сроки проведения", "этап", "голосование", "жюри", "порядок рассмотрения",
    ]),
    ("Требования к работам", [
        "требования к работам", "размер файла", "типы файлов", "должны соответствовать",
        "графических редакторах", "фотоколлажи", "самостоятельно выполненные",
        "должна быть оригинальной", "ненормативная лексика",
    ]),
    ("Заявка и участие", [
        "заявк", "персональн", "согласие", "подпис", "бесплатное", "допускаются",
    ]),
    ("Награждение и выставка", [
        "диплом", "сертификат", "награжд", "выставк", "благодарност",
        "призовых мест", "наградного материала",
    ]),
    ("Организаторы и контакты", [
        "ответствен", "электронной почты", "тел.", "организатором конкурса",
        "оргкомитет", "conference.toi",
    ]),
]
DEFAULT_CATEGORY = "Общие положения"


def infer_category(text: str) -> str:
    """
    Простая тематическая разметка по ключевым словам.
    Считает вхождения маркеров каждой темы и возвращает тему-победителя,
    поэтому работает и для короткого чанка, и для whole документа
    (там получается его доминирующая тема).
    """
    low = text.lower()
    best_category, best_hits = DEFAULT_CATEGORY, 0
    for category, keywords in CATEGORY_RULES:
        hits = sum(low.count(keyword) for keyword in keywords)
        if hits > best_hits:
            best_category, best_hits = category, hits
    return best_category


def split_markdown_documents(md_text: str, file_name: str) -> List[Document]:
    """
    Разбивает один .md файл на логические документы по заголовкам '## <имя документа>'.
    Имя из заголовка становится metadata.file_name, физический тип файла — 'md'.
    """
    parts = DOC_HEADING_RE.split(md_text)
    pieces = [(file_name, parts[0])] + [
        (parts[i].strip(), parts[i + 1]) for i in range(1, len(parts), 2)
    ]

    docs: List[Document] = []
    for name, body in pieces:
        text = normalize_whitespace(clean_markdown(body))
        if len(text) < MIN_DOC_CHARS:
            continue
        docs.append(
            Document(
                page_content=text,
                metadata={
                    "file_name": name,
                    "file_type": "md",
                    "category": infer_category(text),
                },
            )
        )
    return docs


def load_documents_from_folder(folder: Path) -> List[Document]:
    docs: List[Document] = []
    for path in sorted(folder.iterdir()):
        if path.suffix.lower() not in allowed_ext:
            continue
        docs.extend(split_markdown_documents(read_md_text(path), path.name))
    return docs


raw_docs = load_documents_from_folder(KB_DIR)
print(f"Документов в корпусе: {len(raw_docs)}")

if not raw_docs:
    raise SystemExit(
        f"В папке {KB_DIR} нет документов ({', '.join(sorted(allowed_ext))}). "
        f"Положите их туда и перезапустите скрипт."
    )

raw_docs_preview = (
    pd.DataFrame(
        [
            {
                "file_name": d.metadata["file_name"],
                "file_type": d.metadata["file_type"],
                "category": d.metadata["category"],
                "chars": len(d.page_content),
                "preview": textwrap.shorten(
                    d.page_content.replace("\n", " "),
                    width=150,
                    placeholder="...",
                ),
            }
            for d in raw_docs
        ]
    )
    .sort_values(["category", "file_name"])
    .reset_index(drop=True)
)

show(raw_docs_preview)


# ============================================================
# 4. Chunking (Разбиение на чанки)
# ============================================================
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=120,
    separators=["\n\n", "\n", ". ", " ", ""],
)


def build_chunks(raw_docs: List[Document]) -> List[Dict[str, Any]]:
    split_docs = splitter.split_documents(raw_docs)
    chunks = []
    for doc in split_docs:
        text = normalize_whitespace(doc.page_content)
        if not text:
            continue
        chunks.append(
            {
                # Плотной нумерацией заведует len(chunks): chunk_id всегда
                # совпадает с позицией в списке, даже если чанк ниже отброшен.
                "chunk_id": len(chunks),
                "text": text,
                "file_name": doc.metadata.get("file_name", ""),
                "file_type": doc.metadata.get("file_type", ""),
                # Тема берётся из текста чанка, а не документа: целые документы
                # тематически смешанные, колонка category иначе не различалась бы.
                "category": infer_category(text),
                "chars": len(text),
            }
        )
    return chunks


chunks = build_chunks(raw_docs)
print(f"Всего чанков: {len(chunks)}")

chunk_stats = pd.DataFrame(
    [
        {
            "file_name": c["file_name"],
            "category": c["category"],
            "chars": c["chars"],
            "preview": textwrap.shorten(c["text"], width=140, placeholder="..."),
        }
        for c in chunks[:12]
    ]
)
show(chunk_stats)

stats_df = pd.DataFrame(
    [
        {"metric": "chunks_total", "value": len(chunks)},
        {
            "metric": "avg_chars",
            "value": round(float(np.mean([c["chars"] for c in chunks])), 1),
        },
        {
            "metric": "median_chars",
            "value": int(np.median([c["chars"] for c in chunks])),
        },
    ]
)
show(stats_df)


# ============================================================
# 5. Vector search — baseline (Векторный поиск: sentence-transformers + FAISS)
# ============================================================
EMB_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
embedder = SentenceTransformer(EMB_MODEL_NAME)


def build_vector_index(chunks: List[Dict[str, Any]]):
    texts = [c["text"] for c in chunks]
    emb = embedder.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    emb = np.array(emb, dtype="float32")
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    return index, emb


vector_index, chunk_embeddings = build_vector_index(chunks)
print(f"Векторов в индексе: {vector_index.ntotal}")


def retrieve_vector(
    query: str, k: int = 4, search_k: int = 8
) -> List[Dict[str, Any]]:
    q_emb = embedder.encode([query], normalize_embeddings=True)
    q_emb = np.array(q_emb, dtype="float32")

    scores, ids = vector_index.search(q_emb, search_k)

    results: List[Dict[str, Any]] = []
    seen: set = set()

    for score, idx in zip(scores[0], ids[0]):
        if idx < 0:
            continue
        if int(idx) in seen:
            continue
        seen.add(int(idx))

        chunk = chunks[int(idx)]
        results.append(
            {
                "mode": "vector",
                "chunk_id": int(idx),
                "score": round(float(score), 4),
                "file_name": chunk["file_name"],
                "category": chunk["category"],
                "text": chunk["text"],
            }
        )
        if len(results) >= k:
            break

    return results


# ============================================================
# 6. BM25 + hybrid search with RRF (BM25 и гибридный поиск через RRF)
# ============================================================
def tokenize_for_bm25(text: str) -> List[str]:
    return re.findall(r"[A-Za-zА-Яа-я0-9_\-]+", text.lower())


bm25_corpus = [tokenize_for_bm25(c["text"]) for c in chunks]
bm25 = BM25Okapi(bm25_corpus)


def retrieve_bm25(
    query: str, k: int = 4, search_k: int = 8
) -> List[Dict[str, Any]]:
    query_tokens = tokenize_for_bm25(query)
    scores = bm25.get_scores(query_tokens)
    ranked_ids = np.argsort(scores)[::-1]

    results: List[Dict[str, Any]] = []
    for idx in ranked_ids[:search_k]:
        chunk = chunks[int(idx)]
        results.append(
            {
                "mode": "bm25",
                "chunk_id": int(idx),
                "score": round(float(scores[int(idx)]), 4),
                "file_name": chunk["file_name"],
                "category": chunk["category"],
                "text": chunk["text"],
            }
        )
        if len(results) >= k:
            break

    return results


def reciprocal_rank_fusion(
    result_sets: Dict[str, List[Dict[str, Any]]], rrf_k: int = 60
) -> List[Dict[str, Any]]:
    fused: Dict[int, Dict[str, Any]] = {}

    for source_name, items in result_sets.items():
        for rank, item in enumerate(items, start=1):
            key = item["chunk_id"]
            if key not in fused:
                fused[key] = {
                    "mode": "hybrid",
                    "chunk_id": item["chunk_id"],
                    "file_name": item["file_name"],
                    "category": item["category"],
                    "text": item["text"],
                    "score": 0.0,
                    "sources": [],
                }

            fused[key]["score"] += 1.0 / (rrf_k + rank)
            fused[key]["sources"].append(source_name)

    results = sorted(fused.values(), key=lambda x: x["score"], reverse=True)
    for item in results:
        item["score"] = round(float(item["score"]), 4)

    return results


def retrieve_hybrid(
    query: str, k: int = 4, vector_pool: int = 8, bm25_pool: int = 8
) -> List[Dict[str, Any]]:
    vector_items = retrieve_vector(query, k=vector_pool, search_k=vector_pool)
    bm25_items = retrieve_bm25(query, k=bm25_pool, search_k=bm25_pool)

    fused = reciprocal_rank_fusion(
        {"vector": vector_items, "bm25": bm25_items}
    )
    return fused[:k]


# ============================================================
# 7. Compare retrieval (Сравнение режимов поиска)
# ============================================================
def compare_retrieval(query: str, k: int = 4) -> pd.DataFrame:
    rows = []

    for mode, fn in [
        ("vector", retrieve_vector),
        ("bm25", retrieve_bm25),
        ("hybrid", retrieve_hybrid),
    ]:
        results = fn(query, k=k)
        for rank, item in enumerate(results, start=1):
            rows.append(
                {
                    "mode": mode,
                    "rank": rank,
                    "score": item["score"],
                    "file_name": item["file_name"],
                    "category": item["category"],
                    "preview": textwrap.shorten(
                        item["text"], width=140, placeholder="..."
                    ),
                }
            )

    return pd.DataFrame(rows)


for q in [
    "Какие номинации предусмотрены Конкурсом?",
    "Какие требования к загружаемым работам по размеру и типу файла?",
]:
    print("\n" + "=" * 120)
    print("ВОПРОС:", q)
    print("=" * 120)
    show(compare_retrieval(q, k=4))


# ============================================================
# 8. GigaChat API (Подключение к GigaChat)
# ============================================================
GIGACHAT_BASE_URL = "https://foundation-models.api.cloud.ru/v1"
GIGACHAT_MODEL = "ai-sage/GigaChat3-10B-A1.8B"

GIGACHAT_API_KEY_ID = os.environ.get("GIGACHAT_API_KEY_ID", "").strip()
GIGACHAT_API_KEY_SECRET = os.environ.get("GIGACHAT_API_KEY_SECRET", "").strip()

# Используем SECRET как API key (Cloud.ru формат)
GIGACHAT_CREDENTIALS = GIGACHAT_API_KEY_SECRET or GIGACHAT_API_KEY_ID

if not GIGACHAT_CREDENTIALS:
    raise SystemExit(
        "Set GIGACHAT_API_KEY_ID or GIGACHAT_API_KEY_SECRET environment variables."
    )

client = OpenAI(api_key=GIGACHAT_CREDENTIALS, base_url=GIGACHAT_BASE_URL)


# ============================================================
# 9. Rerank (Реранк кандидатов через cross-encoder)
# ============================================================
RERANK_MODEL_NAME = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
reranker = None


def get_reranker():
    global reranker
    if reranker is None:
        reranker = CrossEncoder(RERANK_MODEL_NAME)
    return reranker


def rerank_candidates(
    query: str, items: List[Dict[str, Any]], top_k: int = 4
) -> List[Dict[str, Any]]:
    if not items:
        return []

    model = get_reranker()
    pairs = [(query, item["text"]) for item in items]
    scores = model.predict(pairs)

    reranked = []
    for item, score in zip(items, scores):
        updated = dict(item)
        updated["mode"] = "hybrid_rerank"
        updated["score"] = round(float(score), 4)
        reranked.append(updated)

    reranked = sorted(reranked, key=lambda x: x["score"], reverse=True)
    return reranked[:top_k]


_query = "Какие требования к загружаемым работам по размеру и типу файла?"
_candidates = retrieve_hybrid(_query, k=8)
_reranked = rerank_candidates(_query, _candidates, top_k=4)

rerank_df = pd.DataFrame(
    [
        {
            "rank": i + 1,
            "score": item["score"],
            "file_name": item["file_name"],
            "category": item["category"],
            "preview": textwrap.shorten(
                item["text"], width=150, placeholder="..."
            ),
        }
        for i, item in enumerate(_reranked)
    ]
)
print("\n" + "=" * 120)
print("ВОПРОС:", _query)
print("=" * 120)
show(rerank_df)

# ============================================================
# 10. Multi-query retrieval (Многозапросный поиск: переформулировки + RRF + rerank)
# ============================================================
def generate_query_variants(query: str, n: int = 3) -> List[str]:
    """
    Генерируем n переформулировок исходного вопроса через LLM.
    Возвращаем [original] + variants.
    Если LLM недоступен — возвращаем только original (graceful degradation).
    """
    prompt = (
        f"Переформулируй поисковый запрос {n} разными способами так, "
        f"чтобы каждая переформулировка сохраняла исходный смысл, но использовала "
        f"другие слова: синонимы, другую грамматическую конструкцию, более узкую "
        f"или более широкую формулировку.\n"
        f"Верни РОВНО {n} строк, по одной на строку, без нумерации и без пояснений.\n\n"
        f"Исходный запрос: {query}\n"
    )
    try:
        resp = client.chat.completions.create(
            model=GIGACHAT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
        )
        text = resp.choices[0].message.content.strip()
        lines = []
        for line in text.splitlines():
            line = line.strip(" \t-•*0123456789.)")
            if line and line.lower() != query.lower():
                lines.append(line)
        return [query] + lines[:n]
    except Exception as e:
        print(f"[multi-query] variant generation failed: {e}")
        return [query]


def retrieve_multi_query(
    query: str,
    k: int = 4,
    n_variants: int = 3,
    pool: int = 8,
    verbose: bool = False,
) -> List[Dict[str, Any]]:
    """
    Multi-query hybrid retrieval:
      1. генерируем n переформулировок,
      2. для каждой запускаем hybrid search,
      3. сливаем результаты через RRF,
      4. возвращаем top-k.
    """
    variants = generate_query_variants(query, n=n_variants)
    if verbose:
        print("[multi-query] variants:")
        for i, v in enumerate(variants):
            print(f"  q{i}: {v}")

    result_sets = {
        f"q{i}": retrieve_hybrid(v, k=pool, vector_pool=pool, bm25_pool=pool)
        for i, v in enumerate(variants)
    }
    fused = reciprocal_rank_fusion(result_sets)
    for item in fused:
        item["mode"] = "multi_query"
    return fused[:k]


def retrieve_multi_query_rerank(
    query: str,
    k: int = 4,
    n_variants: int = 3,
    pool: int = 8,
    rerank_pool: int = 8,
    verbose: bool = False,
) -> List[Dict[str, Any]]:
    """
    Multi-query hybrid retrieval + cross-encoder rerank поверх объединённого пула.
    """
    candidates = retrieve_multi_query(
        query, k=rerank_pool, n_variants=n_variants, pool=pool, verbose=verbose
    )
    reranked = rerank_candidates(query, candidates, top_k=k)
    for item in reranked:
        item["mode"] = "multi_query_rerank"
    return reranked


# --- Multi-query demo (Демонстрация multi-query) ---
_demo_q = "Кто ответственный за организацию конкурса и как задать вопросы?"
print("\n" + "=" * 120)
print("MULTI-QUERY DEMO:", _demo_q)
print("=" * 120)
print("Сгенерированные варианты запроса:")
for i, v in enumerate(generate_query_variants(_demo_q)):
    print(f"  q{i}: {v}")

show(
    pd.DataFrame(
        [
            {
                "rank": i + 1,
                "score": item["score"],
                "chunk_id": item["chunk_id"],
                "file_name": item["file_name"],
                "category": item["category"],
                "preview": textwrap.shorten(
                    item["text"], width=150, placeholder="..."
                ),
            }
            for i, item in enumerate(retrieve_multi_query(_demo_q, k=4))
        ]
    )
)

# ============================================================
# 11. RAG: retrieval + generation (Генерация ответа по найденному контексту)
# ============================================================
def build_rag_prompt(query: str, retrieved: List[Dict[str, Any]]) -> str:
    blocks = []
    for item in retrieved:
        block = (
            f"[chunk_{item['chunk_id']} | {item['file_name']} | {item['category']}]\n"
            f"{item['text']}"
        )
        blocks.append(block)

    context = "\n\n".join(blocks)

    prompt = f"""Вы — помощник по базе знаний Всероссийского конкурса детского рисунка
«Мир науки глазами детей – 2026» (ФГБОУ ВО «АлтГПУ»).

Отвечайте только по контексту ниже.
Если точного ответа в контексте нет, скажите об этом прямо.
Не выдумывайте правила, сроки, номинации и ограничения.
Отвечайте спокойно, коротко и по делу.

КОНТЕКСТ:
{context}

ВОПРОС:
{query}

В конце ответа добавьте строку:
Источники: file_name / chunk_id
"""
    return prompt


def llm_openai(prompt: str) -> str:
    resp = client.chat.completions.create(
        model=GIGACHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.6,
    )
    return resp.choices[0].message.content.strip()


def retrieve_by_mode(
    query: str, mode: str, k: int = 4
) -> List[Dict[str, Any]]:
    if mode == "vector":
        return retrieve_vector(query, k=k)
    if mode == "bm25":
        return retrieve_bm25(query, k=k)
    if mode == "hybrid":
        return retrieve_hybrid(query, k=k)
    if mode == "multi_query":
        return retrieve_multi_query(query, k=k)
    if mode == "multi_query_rerank":
        return retrieve_multi_query_rerank(query, k=k)
    raise ValueError(f"Неизвестный режим retrieval: {mode}")


def answer_with_mode(query: str, mode: str, k: int = 4):
    retrieved = retrieve_by_mode(query, mode=mode, k=k)
    if not retrieved:
        return "Недостаточно релевантного контекста.", retrieved

    prompt = build_rag_prompt(query, retrieved)
    answer = llm_openai(prompt)
    return answer, retrieved


def compare_answers(
    query: str, modes=("vector", "hybrid"), k: int = 4
) -> None:
    print("=" * 120)
    print("ВОПРОС:", query)
    print("=" * 120)

    for mode in modes:
        print(f"\n--- {mode} ---")
        answer, retrieved = answer_with_mode(query, mode=mode, k=k)
        print(answer)

        print("\nПоднятые чанки:")
        for item in retrieved:
            print(
                f"  - chunk_{item['chunk_id']} | "
                f"{item['file_name']} | score={item['score']}"
            )
        print("-" * 100)


compare_answers(
    "Какие сроки приема заявок и работ на Конкурс?",
    modes=("vector", "hybrid"),
    k=4,
)

compare_answers(
    "В каких графических редакторах можно выполнять конкурсные работы?",
    modes=("vector", "hybrid"),
    k=4,
)

print("\nГотово.")

# ============================================================
# 12. Full comparison of all modes: vector, bm25, hybrid,
#     multi-query, multi-query + rerank (Полное сравнение всех режимов)
# ============================================================
ALL_MODES = ("vector", "bm25", "hybrid", "multi_query", "multi_query_rerank")


def compare_all_modes_retrieval(query: str, k: int = 4) -> pd.DataFrame:
    rows = []
    for mode in ALL_MODES:
        for rank, item in enumerate(
            retrieve_by_mode(query, mode=mode, k=k), start=1
        ):
            rows.append(
                {
                    "mode": mode,
                    "rank": rank,
                    "score": item["score"],
                    "chunk_id": item["chunk_id"],
                    "file_name": item["file_name"],
                    "category": item["category"],
                    "preview": textwrap.shorten(
                        item["text"], width=120, placeholder="..."
                    ),
                }
            )
    return pd.DataFrame(rows)


for q in [
    "Какие номинации предусмотрены Конкурсом?",
    "Из каких этапов состоит рассмотрение конкурсных работ?",
    "Чем награждают победителей Конкурса?",
]:
    print("\n" + "=" * 120)
    print("ПОЛНОЕ СРАВНЕНИЕ РЕЖИМОВ, ВОПРОС:", q)
    print("=" * 120)
    show(compare_all_modes_retrieval(q, k=4))


# Compare RAG answers with multi-query and rerank
# (Сравнение ответов RAG с multi-query и rerank)
compare_answers(
    "Какие требования к загружаемым работам по размеру и типу файла?",
    modes=("vector", "hybrid", "multi_query", "multi_query_rerank"),
    k=4,
)

compare_answers(
    "Кто ответственный за организацию конкурса и как задать вопросы?",
    modes=("hybrid", "multi_query", "multi_query_rerank"),
    k=4,
)

print("\nГотово (с multi-query).")

# ============================================================
# 13. Retrieval metrics: hit@k, recall@k, precision@k,
#     doc_acc@k, doc_precision@k, mrr@k (Мини-оценка качества поиска)
# ============================================================
"""
Метрики для оценки качества поиска (retrieval) в задачах RAG (Retrieval-Augmented Generation).
Буква k в названии означает, что метрика считается не по всей выдаче, а только по первым k
результатам (например, по топ-3 или топ-5).

hit@k (или accuracy@k) — попал ли правильный ответ в топ-k. Если хотя бы один из k найденных
документов содержит верный ответ, метрика равна 1, иначе 0. Это самая простая метрика
«нашли или нет».

doc_acc@k (Document Accuracy) — доля запросов, где правильный (золотой) документ стоит строго
на первом месте в топ-k. Более строгая версия hit@k.

recall@k (Полнота) — сколько из всех релевантных документов для запроса попало в топ-k.
Показывает, не упустили ли мы важную информацию. Если всего есть 5 правильных документов,
а в топ-5 попало 3, recall@5 = 0.6.

precision@k (Точность) — сколько из выданных системой k документов на самом деле релевантны
запросу. Показывает чистоту выдачи. Если из 5 выданных документов только 2 подходят,
precision@5 = 0.4.

doc_precision@k — специфическая метрика для RAG. Показывает, сколько из k документов
действительно содержат правильный ответ (или его фрагмент), а не просто тематически близки.
Помогает отсеять «шумные» документы, которые кажутся релевантными, но не дают модели фактуры
для ответа.

mrr@k (Mean Reciprocal Rank) — средний обратный ранг. Показывает, насколько высоко в топ-k
обычно поднимается первый правильный документ. Если первый верный ответ найден на 1-м месте,
вклад в метрику — 1.0; на 2-м месте — 0.5; на 3-м — 0.33. Чем выше значение (ближе к 1),
тем быстрее система находит правильный ответ.

Короткий пример для топ-3 (k=3):
Запрос: «Столица Австралии». Система выдала: [1. Сидней, 2. Мельбурн, 3. Канберра].
Правильный ответ — Канберра.

  hit@3 = 1 (нашли в топ-3)
  doc_acc@3 = 0 (правильный документ не на первом месте)
  recall@3 = 1 (если других правильных документов нет)
  precision@3 = 0.33 (только 1 из 3 документов верен)
  mrr@3 = 0.33 (так как первый верный ответ на 3-м месте, 1/3)
"""

# Ground truth: ключевые опорные чанки и профильные документы.
# Собрано вручную по содержимому knowledge_base.md (номера чанков из блока 4).
RETRIEVAL_EVAL = [
    {
        "qid": "Q1_nominations",
        "query": "Какие номинации предусмотрены Конкурсом?",
        "relevant_chunks": {2, 17},
        "relevant_docs": {"Информационное письмо.docx", "Положение о конкурсе рисунков_АлтГПУ_2026.docx"},
    },
    {
        "qid": "Q2_dates",
        "query": "Какие сроки приема заявок и работ на Конкурс?",
        "relevant_chunks": {3, 18},
        "relevant_docs": {"Информационное письмо.docx", "Положение о конкурсе рисунков_АлтГПУ_2026.docx"},
    },
    {
        "qid": "Q3_file_requirements",
        "query": "Какие требования к загружаемым работам по размеру и типу файла?",
        "relevant_chunks": {6, 22},
        "relevant_docs": {"Информационное письмо.docx", "Положение о конкурсе рисунков_АлтГПУ_2026.docx"},
    },
    {
        "qid": "Q4_organizer_contact",
        "query": "Кто ответственный за организацию конкурса и как задать вопросы?",
        "relevant_chunks": {10, 11},
        "relevant_docs": {"Информационное письмо.docx"},
    },
]


def evaluate_one(
    query: str,
    relevant_chunks: set,
    relevant_docs: set,
    mode: str,
    k: int = 4,
) -> Dict[str, float]:
    """Retrieval-метрики для одного вопроса в одном режиме."""
    retrieved = retrieve_by_mode(query, mode=mode, k=k)
    ids = [r["chunk_id"] for r in retrieved]
    docs = [r["file_name"] for r in retrieved]

    chunk_hits = [cid for cid in ids if cid in relevant_chunks]
    doc_hits = [d for d in docs if d in relevant_docs]

    mrr_at_k = 0.0
    for rank, cid in enumerate(ids, start=1):
        if cid in relevant_chunks:
            mrr_at_k = 1.0 / rank
            break

    return {
        "hit@k": 1.0 if chunk_hits else 0.0,
        "doc_acc@k": 1.0 if doc_hits else 0.0,
        "recall@k": round(len(set(chunk_hits)) / max(len(relevant_chunks), 1), 3),
        "precision@k": round(len(set(chunk_hits)) / max(len(ids), 1), 3),
        "doc_precision@k": round(len(doc_hits) / max(len(docs), 1), 3),
        "mrr@k": round(mrr_at_k, 3),
    }


def run_retrieval_eval(k: int = 4, modes=ALL_MODES) -> pd.DataFrame:
    rows = []
    for item in RETRIEVAL_EVAL:
        for mode in modes:
            rows.append({
                "qid": item["qid"],
                "mode": mode,
                **evaluate_one(
                    item["query"],
                    item["relevant_chunks"],
                    item["relevant_docs"],
                    mode=mode,
                    k=k,
                ),
            })
    return pd.DataFrame(rows)


def summarize_retrieval_eval(df: pd.DataFrame) -> pd.DataFrame:
    """Средние метрики по всем вопросам, режимы — строки."""
    cols = ["hit@k", "doc_acc@k", "recall@k", "precision@k", "doc_precision@k", "mrr@k"]
    return (
        df.groupby("mode")[cols]
        .mean()
        .round(3)
        .reindex(list(ALL_MODES))
    )


print("\n" + "=" * 120)
print("МИНИ-ОЦЕНКА RETRIEVAL (k=4): per-question")
print("=" * 120)
eval_df = run_retrieval_eval(k=4)
show(eval_df)

print("\n" + "=" * 120)
print("МИНИ-ОЦЕНКА RETRIEVAL (k=4): средние по 4 вопросам")
print("=" * 120)
show(summarize_retrieval_eval(eval_df).reset_index())


# ============================================================
# 14. Heatmap of retrieval metrics (Тепловая карта метрик поиска)
# ============================================================
def plot_retrieval_heatmap(
    eval_df: pd.DataFrame,
    modes: tuple = ALL_MODES,
    metrics: tuple = ("hit@k", "doc_acc@k", "recall@k", "precision@k", "doc_precision@k", "mrr@k"),
    save_path: str = "retrieval_heatmap.png",
) -> None:
    """
    Строит тепловую карту по данным run_retrieval_eval(eval_df):
      - строки: mode
      - столбцы: метрики
    Функция сама агрегирует eval_df в средние значения.
    """
    # (a) Агрегируем данные внутри функции
    summary = summarize_retrieval_eval(eval_df)

    # (b) Защита от пустых данных
    if summary.empty:
        print("[heatmap] No data to plot.")
        return

    # (c) Оставляем только нужные метрики, которые реально есть в колонках
    available_metrics = [m for m in metrics if m in summary.columns]
    if not available_metrics:
        print("[heatmap] None of the requested metrics found in columns.")
        return

    heatmap_data = summary[available_metrics]

    plt.figure(figsize=(10, 5))
    sns.set_theme(style="whitegrid")

    ax = sns.heatmap(
        heatmap_data,
        annot=True,
        fmt=".3f",
        cmap="YlGnBu",
        vmin=0.0,
        vmax=1.0,
        linewidths=0.5,
        linecolor="gray",
        cbar_kws={"shrink": 0.8, "label": "Score"},
    )

    ax.set_xlabel("")
    ax.set_ylabel("Mode")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    plt.title(
        f"Retrieval Metrics: Average by Mode (k=4, {len(eval_df['qid'].unique())} questions)",
        fontsize=14,
        pad=15,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nHeatmap saved to: {save_path}")


# --- Generate heatmap (Построение тепловой карты) ---
plot_retrieval_heatmap(eval_df, modes=ALL_MODES)

print("\nГотово (с multi-query и heatmaps).")