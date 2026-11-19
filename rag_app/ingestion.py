"""Knowledge-base management.

The single source of truth is ``docs/knowledge_base.md`` — a markdown file
where each document is introduced by an ``# `` (h1) heading. Chunks are built
from it by :func:`rag_app.loaders.load_knowledge_base_md`.

Conversion of ``.docx``/``.pdf`` into this markdown is a separate tool; here we
only manage the markdown itself (list / append / delete document sections).
"""

from __future__ import annotations

import re
from io import BytesIO
from pathlib import Path

from .config import Settings
from .loaders import LoadedDoc, _docx_paragraphs, infer_category, load_knowledge_base_md

KB_FILENAME = "knowledge_base.md"
TEXT_EXT = {".md", ".txt", ".markdown"}
DOCX_EXT = {".docx"}
UPLOAD_EXT = TEXT_EXT | DOCX_EXT
H1_SPLIT = re.compile(r"(?m)^# (?!#)")


class IngestionService:
    def __init__(self, settings: Settings):
        self.settings = settings

    @property
    def kb_path(self) -> Path:
        return self.settings.kb_path / KB_FILENAME

    # ------------------------------------------------------------------
    def _read(self) -> str:
        if not self.kb_path.is_file():
            return ""
        return self.kb_path.read_text(encoding="utf-8")

    def _write(self, text: str) -> None:
        self.settings.kb_path.mkdir(parents=True, exist_ok=True)
        self.kb_path.write_text(text, encoding="utf-8")

    def documents(self) -> list[LoadedDoc]:
        return load_knowledge_base_md(self.kb_path)

    def list_documents(self) -> list[dict]:
        out = []
        for doc in self.documents():
            out.append({
                "name": doc.name,
                "type": doc.doc_type,
                "category": doc.category,
                "sections": len(re.findall(r"(?m)^#{2,6} ", doc.text)),
                "chars": len(doc.text),
            })
        return out

    # ------------------------------------------------------------------
    def save_upload(self, file_storage) -> dict:
        """Append an uploaded document as a new KB section (or replace the KB)."""
        raw_name = file_storage.filename or ""
        ext = Path(raw_name).suffix.lower()
        if ext not in UPLOAD_EXT:
            raise ValueError(
                f"Формат {ext or '?'} не поддерживается. "
                f"Разрешены: {', '.join(sorted(UPLOAD_EXT))}."
            )
        raw = file_storage.read()
        if ext in DOCX_EXT:
            try:
                text = "\n".join(_docx_paragraphs(BytesIO(raw)))
            except Exception as e:  # pragma: no cover - corrupted file
                raise ValueError(f"Не удалось прочитать .docx: {type(e).__name__}: {e}")
            text = text.replace("\u00a0", " ").strip()
        else:
            text = raw.decode("utf-8", errors="ignore").replace("\u00a0", " ").strip()
        if not text:
            raise ValueError("Файл пустой")

        # Загрузка самого файла КБ заменяет базу знаний целиком
        if Path(raw_name).stem.lower() == Path(KB_FILENAME).stem.lower():
            self._write(text.rstrip() + "\n")
            return {"name": KB_FILENAME, "chars": len(text),
                    "category": "База знаний", "replaced": True}

        name = Path(raw_name).stem + ".md"
        # strip an existing top-level heading so we don't double it
        text = re.sub(r"(?m)^# (?!#).*\n?", "", text, count=1).strip()

        existing = {doc.name.lower() for doc in self.documents()}
        if name.lower() in existing:
            raise ValueError(f"Документ «{name}» уже есть в базе знаний")

        body = self._read().rstrip()
        block = f"\n\n# {name}\n\n{text}\n"
        self._write((body + block).lstrip("\n"))

        return {"name": name, "chars": len(text), "category": infer_category(name)}

    def delete_document(self, name: str) -> bool:
        text = self._read()
        if not text:
            return False
        parts = H1_SPLIT.split(text)
        kept = []
        removed = False
        for part in parts:
            stripped = part.strip()
            if not stripped:
                continue
            first = stripped.split("\n", 1)[0].strip()
            heading = re.sub(r"^#\s*", "", first)
            if heading == name or heading == Path(name).name:
                removed = True
                continue
            kept.append(part.rstrip())
        if not removed:
            return False
        self._write("\n\n".join(_ensure_h1(p) for p in kept) + "\n")
        return True

    def document_bytes(self, name: str) -> tuple[str, bytes] | None:
        """Return (filename, bytes) for a KB document, streamed from memory."""
        for doc in self.documents():
            if doc.name == name or Path(doc.name).stem == Path(name).stem:
                filename = Path(doc.name).stem + ".md"
                return filename, f"# {doc.name}\n\n{doc.text}\n".encode("utf-8")
        return None


def _ensure_h1(part: str) -> str:
    part = part.strip()
    if not part.startswith("# "):
        part = "# " + part
    return part
