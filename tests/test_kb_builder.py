from io import BytesIO
from pathlib import Path

import pytest

from rag_app import create_app
from rag_app.config import Settings


@pytest.fixture()
def kbclient(tmp_path: Path):
    settings = Settings(
        kb_dir=str(tmp_path / "docs"),
        source_docs_dir=str(tmp_path / "source_docs"),
        data_dir=str(tmp_path / "data"),
        env_file=str(tmp_path / ".env"),
        source="files",
        enable_rerank=False,
    )
    settings.ensure_dirs()
    app = create_app(settings)
    app.config["TESTING"] = True
    return app.test_client()


def _upload(client, name, content=b"hello"):
    return client.post(
        "/api/kb/upload",
        data={"file": (BytesIO(content), name)},
        content_type="multipart/form-data",
    )


def test_page_renders(kbclient):
    r = kbclient.get("/kb-builder")
    assert r.status_code == 200
    assert "kb_builder.js" in r.get_data(as_text=True)


def test_index_has_builder_link(kbclient):
    body = kbclient.get("/").get_data(as_text=True)
    assert "/kb-builder" in body


def test_upload_rejects_bad_extension(kbclient):
    r = _upload(kbclient, "evil.exe")
    assert r.status_code == 400


def test_upload_accepts_txt_and_lists(kbclient):
    r = _upload(kbclient, "doc1.txt", "Привет\n\nМир".encode("utf-8"))
    assert r.status_code == 200
    files = kbclient.get("/api/kb/files").get_json()["files"]
    assert [f["name"] for f in files] == ["doc1.txt"]
    assert files[0]["selected"] is True
    assert files[0]["ext"] == "txt"


def test_preview_assembles_single_doc(kbclient):
    _upload(kbclient, "doc1.txt", "# Свой заголовок\n\nтекст".encode("utf-8"))
    d = kbclient.post("/api/kb/preview", json={}).get_json()
    assert d["status"] == "ok"
    assert d["selected"] == 1
    h1 = [ln for ln in d["text"].splitlines() if ln.startswith("# ")]
    assert h1 == ["# doc1"]                       # source H1 demoted, doc name is the only H1
    assert "\n## Свой заголовок" in d["text"]


def test_order_affects_assembly(kbclient):
    _upload(kbclient, "a.txt", b"AAA")
    _upload(kbclient, "b.txt", b"BBB")
    kbclient.post("/api/kb/order", json={"order": ["b.txt", "a.txt"]})
    d = kbclient.post("/api/kb/preview", json={}).get_json()
    assert d["text"].index("# b") < d["text"].index("# a")


def test_selection_excludes_file(kbclient):
    _upload(kbclient, "a.txt", b"AAA")
    _upload(kbclient, "b.txt", b"BBB")
    kbclient.post("/api/kb/selection", json={"selected": ["a.txt"]})
    d = kbclient.post("/api/kb/preview", json={}).get_json()
    assert d["selected"] == 1
    assert "BBB" not in d["text"]


def test_delete_file(kbclient):
    _upload(kbclient, "a.txt", b"AAA")
    r = kbclient.delete("/api/kb/files/a.txt")
    assert r.status_code == 200
    assert kbclient.get("/api/kb/files").get_json()["files"] == []


def test_promote_requires_preview(kbclient):
    assert kbclient.post("/api/kb/promote").status_code == 400


def test_promote_writes_knowledge_base(kbclient, tmp_path):
    _upload(kbclient, "a.txt", b"AAA")
    assert kbclient.post("/api/kb/preview", json={}).status_code == 200
    r = kbclient.post("/api/kb/promote")
    assert r.status_code == 200
    kb_file = tmp_path / "docs" / "knowledge_base.md"
    assert kb_file.is_file()
    assert "AAA" in kb_file.read_text(encoding="utf-8")


def test_preview_llm_without_client_falls_back(kbclient):
    _upload(kbclient, "a.txt", b"AAA")
    d = kbclient.post("/api/kb/preview", json={"use_llm": True}).get_json()
    assert d["llm_requested"] is True
    assert d["llm_used"] is False        # no GigaChat key -> graceful fallback
    assert "AAA" in d["text"]


def test_save_draft_requires_text(kbclient):
    assert kbclient.post("/api/kb/save_draft", json={}).status_code == 400


def test_save_draft_persists_edits(kbclient):
    _upload(kbclient, "a.txt", b"AAA")
    assert kbclient.post("/api/kb/preview", json={}).status_code == 200
    r = kbclient.post("/api/kb/save_draft", json={"text": "# a\n\nEDITED TEXT\n"})
    assert r.status_code == 200
    dl = kbclient.get("/api/kb/download").get_data(as_text=True)
    assert "EDITED TEXT" in dl


def test_add_headings_heuristic_unit():
    from rag_app.headings import add_headings
    out = add_headings("Сроки конкурса:\n11 марта 2026.\n\nТребования к работам:\nРазмер файла 10 МБ.")
    assert out.count("## ") == 2
    assert "## Сроки конкурса" in out
    assert "## Требования к работам" in out


def test_headings_create_chunk_sections():
    from rag_app.headings import add_headings
    from rag_app.loaders import LoadedDoc
    from rag_app.chunking import chunk_document

    body = add_headings("Общие положения:\nТекст про общие положения.\n\nСроки проведения:\nПриём заявок до 30 октября.")
    sections = {c.section for c in chunk_document(LoadedDoc(name="Doc", text=body, doc_type="md"))}
    assert "Общие положения" in sections
    assert "Сроки проведения" in sections


def test_headings_flag_adds_sections(kbclient):
    text = "Общие положения:\nКонкурс проводится ежегодно.\n\nСроки проведения:\nПриём заявок до 30 октября."
    _upload(kbclient, "doc.txt", text.encode("utf-8"))
    d = kbclient.post("/api/kb/preview", json={"use_headings": True}).get_json()
    assert d["headings_requested"] is True
    assert d["headings_used"] is True
    assert "## Общие положения" in d["text"]
    assert "## Сроки проведения" in d["text"]
    h1 = [ln for ln in d["text"].splitlines() if ln.startswith("# ")]
    assert h1 == ["# doc"]           # single doc-level H1 remains


def test_files_expose_ocr_flag(kbclient):
    data = kbclient.get("/api/kb/files").get_json()
    assert "ocr_available" in data


def test_preview_ocr_requested_falls_back_for_non_scans(kbclient):
    _upload(kbclient, "scan.txt", b"   \n   ")
    d = kbclient.post("/api/kb/preview", json={"use_ocr": True}).get_json()
    assert d["ocr_requested"] is True
    assert d["ocr_used"] is False          # .txt is not OCR-able -> graceful
    assert "# scan" in d["text"]


def test_empty_extraction_is_flagged(kbclient):
    _upload(kbclient, "empty.txt", b"   \n  ")
    _upload(kbclient, "ok.txt", b"AAA BBB CCC DDD EEE FFF GGG HHH III JJJ")
    files = {f["name"]: f for f in kbclient.get("/api/kb/files").get_json()["files"]}
    assert files["empty.txt"]["empty"] is True
    assert files["ok.txt"]["empty"] is False
    d = kbclient.post("/api/kb/preview", json={}).get_json()
    assert any("empty.txt" in w for w in d["warnings"])


def test_same_stem_files_get_distinct_sections(kbclient, tmp_path):
    _upload(kbclient, "Док.txt", b"AAA")
    _upload(kbclient, "Док.md", b"BBB")
    d = kbclient.post("/api/kb/preview", json={}).get_json()
    assert d["documents"] == 2
    h1 = [ln for ln in d["text"].splitlines() if ln.startswith("# ")]
    assert len(h1) == 2
    assert len(set(h1)) == 2                 # identical base names disambiguated
    kbclient.post("/api/kb/promote")
    names = [x["name"] for x in kbclient.get("/api/docs").get_json()["files"]]
    assert len(names) == 2 and len(set(names)) == 2


def test_promote_uses_edited_text(kbclient, tmp_path):
    _upload(kbclient, "a.txt", b"AAA")
    kbclient.post("/api/kb/preview", json={})
    r = kbclient.post("/api/kb/promote", json={"text": "# a\n\nEDITED BEFORE PROMOTE\n"})
    assert r.status_code == 200
    kb_file = tmp_path / "docs" / "knowledge_base.md"
    assert "EDITED BEFORE PROMOTE" in kb_file.read_text(encoding="utf-8")


class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content):
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, content):
        self.choices = [_FakeChoice(content)]


class _FakeCompletions:
    def __init__(self, content):
        self._content = content

    def create(self, **kwargs):
        return _FakeResponse(self._content)


class _FakeChat:
    def __init__(self, content):
        self.completions = _FakeCompletions(content)


class FakeLLM:
    def __init__(self, content):
        self.chat = _FakeChat(content)


def test_llm_headings_summary_is_rejected(tmp_path):
    from rag_app.config import Settings
    from rag_app.kb_builder import KnowledgeBaseBuilder

    s = Settings(kb_dir=str(tmp_path / "docs"), source_docs_dir=str(tmp_path / "src"),
                 data_dir=str(tmp_path / "data"))
    s.ensure_dirs()
    body = "Общие положения:\n" + "Длинный содержательный текст. " * 30
    (s.source_docs_path / "doc.txt").write_text(body, encoding="utf-8")
    b = KnowledgeBaseBuilder(s)
    fake = FakeLLM("## Общие положения\n## Сроки")  # a TOC, far shorter than the text
    text, meta, llm_used, ocr_used, headings_used, warnings = b.assemble(
        use_headings=True, llm_client=fake, llm_model="x"
    )
    assert "Длинный содержательный текст" in text        # body preserved
    assert "## Общие положения" in text                    # heuristic heading kept
    assert any("оглавление" in w for w in warnings)
