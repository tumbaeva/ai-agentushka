"""Knowledge-base builder: assemble ``knowledge_base.md`` from source documents.

A standalone helper behind the «Создать базу знаний» page. Uploaded documents
live in ``Settings.source_docs_path`` (``source_docs/``); the user orders and
selects them, previews the assembled markdown, and only then promotes it into
the live ``docs/knowledge_base.md``.

The assembly is intentionally simple: each selected document becomes one
``# <name>`` section, so the result matches what :mod:`rag_app.loaders` expects
when it splits the knowledge base back into documents. LLM structuring is
optional and best-effort (a missing client or an API error falls back to the
plain extracted text).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from .config import Settings
from .headings import HEADINGS_PROMPT, add_headings
from .loaders import _docx_paragraphs, _pdf_text, _read_text_any_encoding
from .ocr import ocr_available, ocr_file

KB_FILENAME = "knowledge_base.md"
ALLOWED_EXT = {".docx", ".pdf", ".txt", ".md", ".markdown"}
MIN_TEXT_CHARS = 20  # below this a document is treated as "without extractable text"

_HEADING_RE = re.compile(r"(?m)^(#{1,5}) ")
_UNSAFE_RE = re.compile(r"[^0-9A-Za-zА-Яа-яЁё._\- ()]+")

LLM_PROMPT = """Ты — редактор базы знаний. Приведи текст документа к аккуратному \
Markdown, сохранив ВСЁ содержимое без сокращений:
- сохрани структуру: разделы становятся заголовками (##, ###), списки, таблицы, нумерацию;
- убери мусор распознавания (лишние пустые строки, повторяющиеся пробелы, \u00a0, переносы слов);
- НЕ добавляй заголовок первого уровня (#) — он будет добавлен автоматически;
- НЕ добавляй ничего от себя: никаких вступлений, пояснений и выводов.

Верни только готовый Markdown без обрамляющих ```.

Имя документа: {name}

Текст:
{text}"""


class KnowledgeBaseBuilder:
    """List / order / select source documents and assemble the knowledge base."""

    def __init__(self, settings: Settings):
        self.settings = settings

    # ------------------------------------------------------------------ paths
    @property
    def source_dir(self) -> Path:
        return self.settings.source_docs_path

    @property
    def build_dir(self) -> Path:
        return self.settings.data_path / "kb_build"

    @property
    def draft_path(self) -> Path:
        return self.build_dir / "knowledge_base.md"

    @property
    def state_path(self) -> Path:
        return self.build_dir / "state.json"

    # ------------------------------------------------------------------ state
    def _load_state(self) -> dict:
        if self.state_path.is_file():
            try:
                data = json.loads(self.state_path.read_text(encoding="utf-8"))
                if isinstance(data, dict) and isinstance(data.get("files"), dict):
                    return data
            except (OSError, ValueError):
                pass
        return {"files": {}}

    def _save_state(self, state: dict) -> None:
        self.build_dir.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(
            json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @staticmethod
    def _next_order(files: dict) -> int:
        return 1 + max((int(v.get("order", 0)) for v in files.values()), default=-1)

    # ------------------------------------------------------------------ files
    def list_files(self) -> list[dict]:
        state = self._load_state()
        files = state.setdefault("files", {})
        result: list[dict] = []
        present: set[str] = set()
        if self.source_dir.is_dir():
            for path in sorted(self.source_dir.iterdir()):
                if not path.is_file() or path.suffix.lower() not in ALLOWED_EXT:
                    continue
                name = path.name
                present.add(name)
                entry = files.get(name)
                if not isinstance(entry, dict):
                    entry = {"order": self._next_order(files), "selected": True}
                    files[name] = entry
                entry.setdefault("order", self._next_order(files))
                entry.setdefault("selected", True)
                try:
                    chars = len(self.extract(path).strip())
                except Exception:  # pragma: no cover - corrupted file
                    chars = 0
                result.append({
                    "name": name,
                    "ext": path.suffix.lower().lstrip("."),
                    "size": path.stat().st_size,
                    "order": int(entry["order"]),
                    "selected": bool(entry["selected"]),
                    "chars": chars,
                    "empty": chars < MIN_TEXT_CHARS,
                })
        for name in [n for n in files if n not in present]:
            files.pop(name, None)
        self._save_state(state)
        result.sort(key=lambda f: (f["order"], f["name"].lower()))
        return result

    def save_upload(self, file_storage) -> dict:
        raw_name = file_storage.filename or ""
        ext = Path(raw_name).suffix.lower()
        if ext not in ALLOWED_EXT:
            raise ValueError(
                f"Формат {ext or '?'} не поддерживается. "
                f"Разрешены: {', '.join(sorted(ALLOWED_EXT))}."
            )
        data = file_storage.read()
        if not data:
            raise ValueError("Файл пустой")
        name = self._unique_name(Path(raw_name).name)
        self.source_dir.mkdir(parents=True, exist_ok=True)
        (self.source_dir / name).write_bytes(data)

        state = self._load_state()
        files = state.setdefault("files", {})
        files[name] = {"order": self._next_order(files), "selected": True}
        self._save_state(state)
        return {"name": name, "ext": ext.lstrip("."), "size": len(data)}

    def delete_file(self, name: str) -> bool:
        safe = Path(name).name
        path = self.source_dir / safe
        if not path.is_file() or path.suffix.lower() not in ALLOWED_EXT:
            return False
        path.unlink()
        state = self._load_state()
        state.setdefault("files", {}).pop(safe, None)
        self._save_state(state)
        return True

    def set_order(self, order: list[str]) -> None:
        order = [Path(n).name for n in order]
        state = self._load_state()
        files = state.setdefault("files", {})
        remaining = [n for n in files if n not in order]
        for i, name in enumerate(order + remaining):
            if name in files:
                files[name]["order"] = i
        self._save_state(state)

    def set_selection(self, selected: list[str]) -> None:
        chosen = {Path(n).name for n in selected}
        state = self._load_state()
        files = state.setdefault("files", {})
        for name in files:
            files[name]["selected"] = name in chosen
        self._save_state(state)

    def _unique_name(self, raw: str) -> str:
        base = _UNSAFE_RE.sub("_", raw).strip() or "document"
        base = Path(base).name
        stem, ext = Path(base).stem, Path(base).suffix
        candidate, i = base, 1
        while (self.source_dir / candidate).exists():
            candidate = f"{stem}_{i}{ext}"
            i += 1
        return candidate

    # --------------------------------------------------------------- assembly
    def extract(self, path: Path, use_ocr: bool = False) -> str:
        ext = path.suffix.lower()
        if ext == ".docx":
            text = "\n".join(_docx_paragraphs(path))
        elif ext == ".pdf":
            text = _pdf_text(path)
        elif ext in {".md", ".txt", ".markdown"}:
            text = _read_text_any_encoding(path)
        else:
            raise ValueError(f"Формат {ext} не поддерживается")
        text = self._demote_headings(self._normalize(text))
        # OCR is a best-effort fallback for scans with no text layer.
        if use_ocr and len(text.strip()) < MIN_TEXT_CHARS:
            ocr_text = self._ocr(path)
            if ocr_text.strip():
                text = self._demote_headings(self._normalize(ocr_text))
        return text

    @staticmethod
    def _ocr(path: Path) -> str:
        try:
            return ocr_file(path)
        except Exception:
            return ""

    def assemble(self, use_llm: bool = False, llm_client=None,
                 llm_model: str = "", use_ocr: bool = False,
                 use_headings: bool = False) -> tuple[str, list[dict], bool, bool, bool, list[str]]:
        files = [f for f in self.list_files() if f["selected"]]
        blocks: list[str] = []
        meta: list[dict] = []
        warnings: list[str] = []
        llm_used = False
        ocr_used = False
        headings_used = False
        labels: set[str] = set()
        for f in files:
            path = self.source_dir / f["name"]
            try:
                body = self.extract(path, use_ocr=use_ocr)
            except Exception as e:  # pragma: no cover - corrupted file
                body = f"> Не удалось прочитать документ: {type(e).__name__}: {e}"
            if use_ocr and f.get("empty") and len(body.strip()) >= MIN_TEXT_CHARS:
                ocr_used = True
            if use_headings and body.strip():
                # 1) heuristic sections (offline), 2) LLM refines by meaning.
                structured = add_headings(body)
                if structured != body:
                    headings_used = True
                if llm_client is not None:
                    refined = self.llm_headings(llm_client, llm_model, structured, f["name"])
                    if refined.strip() and self._keeps_content(structured, refined):
                        if refined != structured:
                            headings_used = True
                        structured = refined
                    elif refined.strip():
                        warnings.append(
                            f"Файл «{f['name']}»: ответ LLM оказался короче исходного текста "
                            f"(похоже на оглавление) — оставлены разделы по эвристике, текст сохранён."
                        )
                body = structured
            elif use_llm and llm_client is not None and body.strip():
                structured = self.llm_structure(llm_client, llm_model, body, f["name"])
                if structured.strip() and self._keeps_content(body, structured):
                    body = structured
                    llm_used = True
                elif structured.strip():
                    warnings.append(
                        f"Файл «{f['name']}»: ответ LLM оказался короче исходного текста — "
                        f"оставлен исходный текст без обработки."
                    )
            # One H1 per document; disambiguate identical base names so every
            # uploaded file stays a distinct section in the knowledge base.
            label = self._unique_label(Path(f["name"]).stem, f["ext"], labels)
            labels.add(label)
            blocks.append(f"# {label}\n\n{body.strip()}")
            meta.append({"name": f["name"], "section": label,
                         "chars": len(body.strip()),
                         "empty": len(body.strip()) < MIN_TEXT_CHARS})
        text = ("\n\n".join(blocks).strip() + "\n") if blocks else ""
        return text, meta, llm_used, ocr_used, headings_used, warnings

    @staticmethod
    def _keeps_content(original: str, result: str) -> bool:
        """True if ``result`` plausibly still holds the document body.

        Guards against an LLM collapsing the text into a table of contents or
        summary: such output is far shorter than the source.
        """
        if not original.strip():
            return True
        return len(result.strip()) >= 0.5 * len(original.strip())

    @staticmethod
    def _unique_label(stem: str, ext: str, used: set[str]) -> str:
        stem = stem.strip() or "document"
        if stem not in used:
            return stem
        label = f"{stem} ({ext})"
        n = 2
        while label in used:
            label = f"{stem} ({ext} {n})"
            n += 1
        return label

    def build(self, use_llm: bool = False, llm_client=None,
              llm_model: str = "", use_ocr: bool = False,
              use_headings: bool = False) -> dict:
        text, files, llm_used, ocr_used, headings_used, warnings = self.assemble(
            use_llm=use_llm, llm_client=llm_client, llm_model=llm_model,
            use_ocr=use_ocr, use_headings=use_headings,
        )
        self.build_dir.mkdir(parents=True, exist_ok=True)
        self.draft_path.write_text(text, encoding="utf-8")
        warnings = warnings + [
            f"Файл «{m['name']}»: извлечено мало текста ({m['chars']} симв.). "
            f"Возможно, это сканированный PDF/DOCX без текстового слоя — он попадёт в базу пустым."
            for m in files if m.get("empty")
        ]
        return {
            "text": text,
            "files": files,
            "chars": len(text),
            "selected": len(files),
            "documents": len(files),
            "warnings": warnings,
            "path": str(self.draft_path),
            "llm_used": llm_used,
            "ocr_requested": use_ocr,
            "ocr_used": ocr_used,
            "headings_requested": use_headings,
            "headings_used": headings_used,
        }

    def save_draft(self, text: str) -> dict:
        """Persist manually edited preview text as the current draft."""
        if not isinstance(text, str):
            raise ValueError("Ожидается текст черновика")
        self.build_dir.mkdir(parents=True, exist_ok=True)
        self.draft_path.write_text(text, encoding="utf-8")
        return {"chars": len(text), "path": str(self.draft_path)}

    def promote(self) -> dict:
        if not self.draft_path.is_file():
            raise ValueError("Сначала соберите базу знаний (нажмите «Собрать и предпросмотреть»).")
        text = self.draft_path.read_text(encoding="utf-8")
        if not text.strip():
            raise ValueError("Собранная база знаний пуста")
        self.settings.kb_path.mkdir(parents=True, exist_ok=True)
        target = self.settings.kb_path / KB_FILENAME
        target.write_text(text, encoding="utf-8")
        return {"path": str(target), "chars": len(text)}

    def draft_bytes(self) -> bytes | None:
        if not self.draft_path.is_file():
            return None
        return self.draft_path.read_bytes()

    # ---------------------------------------------------------------- helpers
    def llm_structure(self, client, model: str, text: str, name: str) -> str:
        if client is None or not text.strip():
            return text
        prompt = LLM_PROMPT.format(name=name, text=text)
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            out = (resp.choices[0].message.content or "").strip()
            out = re.sub(r"^```[a-zA-Z]*\n?|\n?```$", "", out).strip()
            return out or text
        except Exception:
            return text

    def llm_headings(self, client, model: str, text: str, name: str) -> str:
        """LLM pass: relabel/insert section headings by meaning."""
        if client is None or not text.strip():
            return text
        prompt = HEADINGS_PROMPT.format(name=name, text=text)
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            out = (resp.choices[0].message.content or "").strip()
            out = re.sub(r"^```[a-zA-Z]*\n?|\n?```$", "", out).strip()
            # never let the model introduce a doc-level H1
            out = re.sub(r"(?m)^#\s+", "## ", out)
            return out or text
        except Exception:
            return text

    @staticmethod
    def _normalize(text: str) -> str:
        text = text.replace("\u00a0", " ")
        out: list[str] = []
        blank = 0
        for line in text.splitlines():
            line = line.rstrip()
            if line.strip():
                out.append(line)
                blank = 0
            else:
                blank += 1
                if blank <= 1:
                    out.append("")
        return "\n".join(out).strip()

    @staticmethod
    def _demote_headings(text: str) -> str:
        """Shift headings down one level so the document name stays the only H1."""
        return _HEADING_RE.sub(lambda m: "#" * (len(m.group(1)) + 1) + " ", text)
