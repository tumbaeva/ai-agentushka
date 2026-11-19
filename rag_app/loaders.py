"""Document loaders: .docx / .pdf / .md / .txt -> Document(text, section, meta).

Each loader returns a list of ``LoadedDoc`` objects where ``text`` keeps
paragraph boundaries (single ``\n``) and heading structure so the chunker can
split semantically.
"""

from __future__ import annotations

import collections
import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class LoadedDoc:
    name: str
    text: str
    doc_type: str
    category: str = "Общие материалы"
    source: str = ""  # original path or upload id


CATEGORY_RULES = [
    (["положение", "правило", "условие", "критерий"], "Положения и правила"),
    (["информационное", "письмо", "объявление"], "Информационные материалы"),
    (["номинац", "тематич", "направлен"], "Номинации и направления"),
    (["срок", "дата", "календар", "этап"], "Сроки и этапы"),
    (["требование", "формат", "размер", "файл"], "Требования к работам"),
    (["награ", "приз", "балл", "оцен"], "Награждение и оценка"),
    (["конкурс", "фестивал", "мероприят"], "Конкурсные материалы"),
]


def infer_category(file_name: str) -> str:
    name = file_name.lower()
    for keys, label in CATEGORY_RULES:
        if any(k in name for k in keys):
            return label
    return "Общие материалы"


def _read_text_any_encoding(path: Path) -> str:
    for enc in ("utf-8", "cp1251", "utf-8-sig", "latin1"):
        try:
            return path.read_text(encoding=enc)
        except (UnicodeDecodeError, LookupError):
            continue
    return path.read_text(encoding="utf-8", errors="ignore")


def _docx_paragraphs(source) -> list[str]:
    from docx import Document as DocxDocument

    doc = DocxDocument(str(source) if isinstance(source, Path) else source)
    out: list[str] = []
    for para in doc.paragraphs:
        text = para.text.strip()
        style = (para.style.name or "").lower() if para.style is not None else ""
        if not text:
            continue
        if style.startswith("heading"):
            # Encode heading level as markdown so the chunker can see it.
            level = "".join(ch for ch in style if ch.isdigit()) or "1"
            out.append("#" * int(level) + " " + text)
        else:
            out.append(text)
    # tables
    for table in getattr(doc, "tables", []):
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells if c.text.strip()]
            if cells:
                out.append(" | ".join(cells))
    return out


def _pdf_text(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError:  # pragma: no cover
        raise RuntimeError("PDF support requires the 'pypdf' package")

    reader = PdfReader(str(path))
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n".join(pages)


def load_file(path: Path) -> LoadedDoc | None:
    suffix = path.suffix.lower()
    if suffix == ".docx":
        text = "\n".join(_docx_paragraphs(path))
    elif suffix == ".pdf":
        text = _pdf_text(path)
    elif suffix in {".md", ".txt", ".markdown"}:
        text = _read_text_any_encoding(path)
    else:
        return None

    text = text.replace("\u00a0", " ").strip()
    if not text:
        return None
    return LoadedDoc(
        name=path.name,
        text=text,
        doc_type=suffix.lstrip("."),
        category=infer_category(path.name),
        source=str(path),
    )


def load_folder(folder: Path, allowed: set[str] | None = None) -> list[LoadedDoc]:
    allowed = allowed or {".docx", ".pdf", ".md", ".txt", ".markdown"}
    docs: list[LoadedDoc] = []
    if not folder.is_dir():
        return docs
    for path in sorted(folder.iterdir()):
        if path.is_file() and path.suffix.lower() in allowed:
            # knowledge_base.md is the concatenation of the source documents; it
            # is used in 'kb_md' mode and must not be double-counted in 'files'.
            if path.name == "knowledge_base.md":
                continue
            doc = load_file(path)
            if doc is not None:
                docs.append(doc)
    return docs


_TITLE_RE = re.compile(r"«([^»]{3,120})»")


def detect_title(docs) -> str:
    """Pick the contest name from the knowledge base.

    Heuristic: the most frequently quoted «…» phrase (contest names repeat
    across documents); tie-break prefers phrases mentioning конкурс/наук/детей.
    """
    counts: collections.Counter = collections.Counter()
    for d in docs:
        for m in _TITLE_RE.findall(d.text):
            phrase = m.strip()
            if any(ch.isalpha() for ch in phrase):
                counts[phrase] += 1
    if not counts:
        return ""

    def rank(item):
        phrase, c = item
        low = phrase.lower()
        bonus = sum(1 for kw in ("конкурс", "наук", "дет", "рисунк") if kw in low)
        return (c, bonus, len(phrase))

    return max(counts.items(), key=rank)[0]


def load_knowledge_base_md(md_path: Path) -> list[LoadedDoc]:
    """Split a concatenated knowledge_base.md into documents.

    Documents are separated by ``# `` (h1) headings; ``## `` are sections
    inside a document and stay in the text so the chunker keeps them.
    """
    if not md_path.is_file():
        return []
    text = md_path.read_text(encoding="utf-8")
    parts = re.split(r"(?m)^# (?!#)", text)
    docs: list[LoadedDoc] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        lines = part.split("\n")
        name = lines[0].strip() or "knowledge_base.md"
        body = "\n".join(lines[1:]).strip()
        docs.append(
            LoadedDoc(
                name=name,
                text=body,
                doc_type="md",
                category=infer_category(name),
                source=str(md_path),
            )
        )
    if not docs and text.strip():
        docs.append(
            LoadedDoc(
                name=md_path.name,
                text=text,
                doc_type="md",
                category=infer_category(md_path.name),
                source=str(md_path),
            )
        )
    return docs
