import io
import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from rag_app import create_app
from rag_app.config import Settings


@pytest.fixture()
def agentclient(tmp_path: Path):
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


@pytest.fixture()
def standalone_client(tmp_path: Path):
    settings = Settings(
        kb_dir=str(tmp_path / "docs"),
        data_dir=str(tmp_path / "data"),
        env_file=str(tmp_path / ".env"),
        source="files",
        enable_rerank=False,
    )
    settings.ensure_dirs()
    app = create_app(settings, standalone=True)
    app.config["TESTING"] = True
    return app.test_client()


def _upload(client, name, content):
    return client.post(
        "/api/agent/document",
        data={"file": (io.BytesIO(content), name)},
        content_type="multipart/form-data",
    )


# ------------------------------------------------------------------ page
def test_agent_page_renders(agentclient):
    r = agentclient.get("/agent")
    assert r.status_code == 200
    assert "agent.js" in r.get_data(as_text=True)


def test_index_has_agent_link(agentclient):
    body = agentclient.get("/").get_data(as_text=True)
    assert "/agent" in body


def test_standalone_is_chat_only(standalone_client):
    r = standalone_client.get("/")
    assert r.status_code == 200
    body = r.get_data(as_text=True)
    assert "agent_chat.js" in body
    assert "chatText" in body
    assert "indexStatus" in body
    # no constructor UI: settings panel / system-prompt preview must be absent
    assert "Настройки агента" not in body
    assert "Итоговый системный промпт" not in body
    assert "agent.js" not in body
    # standalone serves no dashboard API
    assert standalone_client.get("/api/modes").status_code == 404
    assert standalone_client.get("/agent").status_code == 200


def test_agent_status_endpoint(agentclient):
    d = agentclient.get("/api/agent/status").get_json()
    assert set(d) >= {"status", "chunks", "embedding_model", "error"}


def test_profile_stores_editable_system_prompt(agentclient):
    agentclient.post("/api/agent/profile", json={"system_prompt": "МОЙ ПОЛНЫЙ ПРОМПТ"})
    prof = agentclient.get("/api/agent/profile").get_json()["profile"]
    assert prof["system_prompt"] == "МОЙ ПОЛНЫЙ ПРОМПТ"


def test_prompt_preview_returns_custom_system_prompt(agentclient):
    d = agentclient.post("/api/agent/prompt/preview", json={"system_prompt": "ABC"}).get_json()
    assert d["system_prompt"] == "ABC"


def test_source_labels_hide_database_name():
    from rag_app.generation import build_citations, source_label
    assert source_label("knowledge_base.md", "knowledge_base.md") == "Документ"
    assert source_label("knowledge_base_рисунки.md", "x") == "рисунки"
    assert source_label("Положение о конкурсе.docx", "Порядок") == "Положение о конкурсе"
    cits = build_citations(
        [{"file_name": "knowledge_base.md", "section": "knowledge_base.md",
          "text": "x", "chunk_id": 1, "score": 0.5}], "x",
    )
    assert cits[0]["label"] == "Документ"


def test_unique_sources_dedupes_document_names():
    from rag_app.generation import unique_sources
    cits = [
        {"rank": 1, "label": "Информационное письмо", "file_name": "Информационное письмо"},
        {"rank": 2, "label": "Информационное письмо", "file_name": "Информационное письмо"},
        {"rank": 3, "label": "Положение", "file_name": "Положение"},
        {"rank": 4, "label": "информационное письмо", "file_name": "Информационное письмо"},
    ]
    out = unique_sources(cits)
    assert [c["label"] for c in out] == ["Информационное письмо", "Положение"]
    assert [c["rank"] for c in out] == [1, 2]


def test_agent_chat_citations_are_unique_documents(agentclient):
    from types import SimpleNamespace
    ctx = agentclient.application.extensions["rag_ctx"]
    ctx.retriever.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=lambda **kw: SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ок"))]))))
    ctx.retriever.retrieve = lambda query, mode="hybrid", k=4: [
        {"file_name": "Информационное письмо", "section": "Сроки", "text": "a",
         "chunk_id": 1, "score": 0.9, "category": "x"},
        {"file_name": "Информационное письмо", "section": "Номинации", "text": "b",
         "chunk_id": 2, "score": 0.8, "category": "x"},
        {"file_name": "Положение", "section": "Общие", "text": "c",
         "chunk_id": 3, "score": 0.7, "category": "x"},
    ]
    d = agentclient.post("/api/agent/chat/message", json={"query": "сроки"}).get_json()
    assert [c["label"] for c in d["citations"]] == ["Информационное письмо", "Положение"]


def test_chat_fallback_includes_contact_phone(agentclient):
    agentclient.post("/api/agent/profile", json={"contact_phone": "+7 999 000-00-00"})
    d = agentclient.post("/api/agent/chat/message", json={"query": "сроки"}).get_json()
    assert "+7 999 000-00-00" in d["answer"]
    assert "Настройки" not in d["answer"]         # no technical wording leaks
    assert "Недостаточно релевантного" not in d["answer"]


# --------------------------------------------------------------- profile
def test_profile_defaults_and_roundtrip(agentclient):
    d = agentclient.get("/api/agent/profile").get_json()
    assert d["profile"]["agent_name"] == "AI-агент"
    assert len(d["styles"]) == 8

    agentclient.post("/api/agent/profile", json={
        "agent_name": "Bot X", "temperature": 0.9, "length": "long",
        "sidebar_width": 999, "ui_language": "en",
        "contact_phone": "+7 111 222-33-44",
    })
    prof = agentclient.get("/api/agent/profile").get_json()["profile"]
    assert prof["agent_name"] == "Bot X"
    assert abs(prof["temperature"] - 0.9) < 1e-6
    assert prof["length"] == "long"
    assert prof["sidebar_width"] == 720        # clamped to max
    assert prof["ui_language"] == "en"
    assert prof["contact_phone"] == "+7 111 222-33-44"
    assert "theme_mode" not in prof            # theme switch removed


def test_profile_save_is_persisted_to_disk(agentclient, tmp_path):
    agentclient.post("/api/agent/profile", json={"agent_name": "Persisted"})
    path = tmp_path / "data" / "agent_profile.json"
    assert path.is_file()
    assert json.loads(path.read_text(encoding="utf-8"))["agent_name"] == "Persisted"


def test_profile_ignores_unknown_and_invalid(agentclient):
    agentclient.post("/api/agent/profile", json={"bad_key": 1, "length": "gigantic"})
    prof = agentclient.get("/api/agent/profile").get_json()["profile"]
    assert "bad_key" not in prof
    assert prof["length"] == "medium"          # invalid enum ignored


def test_styles_endpoint(agentclient):
    styles = agentclient.get("/api/agent/styles").get_json()["styles"]
    ids = {s["id"] for s in styles}
    assert ids == {"business", "friendly", "humorous", "concise", "expert",
                   "explain_simple", "legal", "informal_emoji"}
    assert all("label_ru" in s and "instruction" in s for s in styles)


# ---------------------------------------------------------------- prompt
def test_compose_system_prompt_unit():
    from rag_app.agent_styles import compose_system_prompt, tokens_for_length
    out = compose_system_prompt("БАЗА-ПРАВИЛА", "humorous", "МОЙ-КАСТОМ", "short")
    assert "БАЗА-ПРАВИЛА" in out
    assert "МОЙ-КАСТОМ" in out
    assert "кратко" in out.lower()
    assert tokens_for_length("short") == 300
    assert tokens_for_length("medium") is None
    assert tokens_for_length("long") == 1600


def test_prompt_preview_endpoint(agentclient):
    d = agentclient.post("/api/agent/prompt/preview",
                         json={"style_id": "legal", "custom_prompt": "", "length": "long"}).get_json()
    assert "юриди" in d["system_prompt"].lower()


def test_style_override_reaches_prompt():
    from rag_app.agent_styles import compose_system_prompt
    from rag_app.generation import generate_answer

    calls = []

    def _create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ответ"))])

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=_create)))

    class _Retriever:
        def __init__(self):
            self.client = client
            self.index = SimpleNamespace(chunks=[])
            self.settings = SimpleNamespace(llm_model="m")

        def retrieve(self, query, mode="hybrid", k=4):
            return [{"file_name": "A.md", "section": "s", "text": "ctx",
                     "chunk_id": 1, "score": 0.5}]

    sp = compose_system_prompt("BASE-RULES", "humorous", "CUSTOM-TONE", "short")
    generate_answer(_Retriever(), "вопрос", system_prompt=sp, max_tokens=300)
    sent = calls[0]
    assert "BASE-RULES" in sent["messages"][0]["content"]
    assert "CUSTOM-TONE" in sent["messages"][0]["content"]
    assert sent["max_tokens"] == 300

    generate_answer(_Retriever(), "вопрос", system_prompt="X", max_tokens=None)
    assert "max_tokens" not in calls[1]


# ------------------------------------------------------------------ export
def test_export_zip_contains_expected_files(agentclient, tmp_path):
    (tmp_path / "docs" / "knowledge_base.md").write_text("# Док\n\nтекст\n", encoding="utf-8")
    r = agentclient.post("/api/agent/export", json={})
    assert r.status_code == 200
    zf = zipfile.ZipFile(io.BytesIO(r.data))
    names = set(zf.namelist())
    for member in ("run.py", "Запустить агента.bat", "requirements.txt",
                   "README.md",
                   "docs/knowledge_base.md", "data/agent_profile.json", "data/settings.json",
                   "rag_app/__init__.py", "rag_app/web/agent.py",
                   "rag_app/templates/agent_chat.html", "rag_app/static/agent_chat.js",
                   "rag_app/static/style.css"):
        assert member in names, f"missing {member}"
    # the skeleton is overlaid, not nested inside the package copy
    assert not any(n.startswith("rag_app/app_template/") for n in names)


def test_export_drops_dashboard_and_artifacts(agentclient, tmp_path):
    (tmp_path / "data" / "chat.db").write_text("x", encoding="utf-8")
    r = agentclient.post("/api/agent/export", json={})
    names = set(zipfile.ZipFile(io.BytesIO(r.data)).namelist())
    for junk in ("rag_app/web/api.py", "rag_app/web/pages.py", "rag_app/eval.py",
                 "rag_app/sweep.py", "rag_app/markdown.py", "rag_app/kb_builder.py",
                 "rag_app/headings.py", "rag_app/ocr.py", "rag_app/agent_export.py",
                 "rag_app/templates/index.html", "rag_app/templates/agent.html",
                 "rag_app/static/app.js", "rag_app/static/agent.js",
                 "run.bat", ".env.example", "data/chat.db"):
        assert junk not in names, f"unexpected {junk}"


def test_export_requirements_are_minimal(agentclient, tmp_path):
    r = agentclient.post("/api/agent/export", json={})
    reqs = zipfile.ZipFile(io.BytesIO(r.data)).read("requirements.txt").decode("utf-8")
    for needed in ("flask", "sentence-transformers", "faiss-cpu", "rank-bm25", "openai"):
        assert needed in reqs
    for heavy in ("rapidocr", "matplotlib", "seaborn", "pandas", "pypdf", "opencv"):
        assert heavy not in reqs, f"unnecessary dependency: {heavy}"


def test_export_includes_index_snapshot(agentclient, tmp_path):
    snap = tmp_path / "data" / "index" / "2026-01-01_00-00-00"
    snap.mkdir(parents=True)
    (snap / "index.faiss").write_bytes(b"FAISS")
    (snap / "chunks.json").write_text("[]", encoding="utf-8")
    (snap / "meta.json").write_text('{"embedding_model": "BAAI/bge-m3"}', encoding="utf-8")
    r = agentclient.post("/api/agent/export", json={})
    names = set(zipfile.ZipFile(io.BytesIO(r.data)).namelist())
    assert "data/index/2026-01-01_00-00-00/index.faiss" in names
    assert "data/index/2026-01-01_00-00-00/chunks.json" in names
    assert "data/index/2026-01-01_00-00-00/meta.json" in names


def test_export_excludes_secrets_and_machine_paths(agentclient, tmp_path):
    r = agentclient.post("/api/agent/export", json={})
    zf = zipfile.ZipFile(io.BytesIO(r.data))
    settings_data = json.loads(zf.read("data/settings.json"))
    for key in ("gigachat_key_secret", "gigachat_key_id", "hf_token",
                "kb_dir", "data_dir", "env_file"):
        assert key not in settings_data
    assert settings_data["source"] == "kb_md"   # bundle indexes the shipped KB
    # no HuggingFace token anywhere in the archive
    for name in zf.namelist():
        if name.endswith((".json", ".env", ".md", ".txt", ".example")):
            assert "HF_TOKEN" not in zf.read(name).decode("utf-8", "ignore")


def test_export_copies_env_with_secrets(tmp_path):
    settings = Settings(
        kb_dir=str(tmp_path / "docs"), data_dir=str(tmp_path / "data"),
        env_file=str(tmp_path / ".env"), source="files", enable_rerank=False,
    )
    settings.gigachat_key_id = "id-123"
    settings.gigachat_key_secret = "secret-abc"
    settings.hf_token = "hf_xyz"
    settings.ensure_dirs()
    app = create_app(settings)
    app.config["TESTING"] = True
    r = app.test_client().post("/api/agent/export", json={})
    zf = zipfile.ZipFile(io.BytesIO(r.data))
    assert ".env" in zf.namelist()
    env = zf.read(".env").decode("utf-8")
    assert "GIGACHAT_API_KEY_SECRET=secret-abc" in env
    assert "HF_TOKEN" not in env                # HF is not needed in the delivered agent
    # settings.json still must not leak the secrets
    assert "gigachat_key_secret" not in json.loads(zf.read("data/settings.json"))


def test_export_launcher_creates_shortcut(agentclient, tmp_path):
    agentclient.post("/api/agent/profile", json={"agent_name": "Acme Bot"})
    r = agentclient.post("/api/agent/export", json={})
    zf = zipfile.ZipFile(io.BytesIO(r.data))
    launcher = zf.read("Запустить агента.bat").decode("utf-8")
    assert "Acme Bot" in launcher
    assert "CreateShortcut" in launcher        # desktop shortcut is created on first run
    assert "{{" not in launcher                # all placeholders substituted


def test_export_bat_files_use_crlf(agentclient, tmp_path):
    r = agentclient.post("/api/agent/export", json={})
    zf = zipfile.ZipFile(io.BytesIO(r.data))
    for name in ("Запустить агента.bat",):
        data = zf.read(name)
        assert b"\r\n" in data, f"{name} must use CRLF"
        assert b"\n" not in data.replace(b"\r\n", b""), f"{name} has bare LF"


def test_export_reflects_saved_branding(agentclient, tmp_path):
    agentclient.post("/api/agent/profile", json={"agent_name": "Acme Bot", "subtitle": "Помощник"})
    r = agentclient.post("/api/agent/export", json={})
    zf = zipfile.ZipFile(io.BytesIO(r.data))
    prof = json.loads(zf.read("data/agent_profile.json"))
    assert prof["agent_name"] == "Acme Bot"
    assert "Acme Bot" in zf.read("README.md").decode("utf-8")


def test_profile_palette_colors_roundtrip(agentclient):
    agentclient.post("/api/agent/profile", json={"palette": {
        "user_bg": "#111111", "user_text": "#222222",
        "agent_bg": "#333333", "agent_text": "#444444",
        "ui_bg": "#555555", "ui_field": "#666666", "ui_text": "#777777",
    }})
    pal = agentclient.get("/api/agent/profile").get_json()["profile"]["palette"]
    assert pal == {
        "user_bg": "#111111", "user_text": "#222222",
        "agent_bg": "#333333", "agent_text": "#444444",
        "ui_bg": "#555555", "ui_field": "#666666", "ui_text": "#777777",
        "ui_subtitle": "#94a3b8",
    }


def test_subtitle_color_roundtrip(agentclient):
    agentclient.post("/api/agent/profile", json={"palette": {"ui_subtitle": "#abcdef"}})
    pal = agentclient.get("/api/agent/profile").get_json()["profile"]["palette"]
    assert pal["ui_subtitle"] == "#abcdef"


def test_subtitle_color_defaults(agentclient):
    pal = agentclient.get("/api/agent/profile").get_json()["profile"]["palette"]
    assert pal["ui_subtitle"] == "#94a3b8"      # backfilled by _migrate_palette


def test_subtitle_color_ignores_bad_hex(agentclient):
    agentclient.post("/api/agent/profile", json={"palette": {"ui_subtitle": "nope"}})
    pal = agentclient.get("/api/agent/profile").get_json()["profile"]["palette"]
    assert pal["ui_subtitle"] == "#94a3b8"      # default kept


def test_palette_ignores_bad_hex(agentclient):
    agentclient.post("/api/agent/profile", json={"palette": {"user_bg": "not-a-color"}})
    pal = agentclient.get("/api/agent/profile").get_json()["profile"]["palette"]
    assert pal["user_bg"] == "#0ea5e9"   # default kept


def test_export_keeps_color_settings(agentclient, tmp_path):
    agentclient.post("/api/agent/profile", json={"palette": {
        "user_bg": "#abcdef", "agent_bg": "#fedcba", "ui_field": "#0a0b0c",
        "ui_subtitle": "#0d0e0f",
    }})
    r = agentclient.post("/api/agent/export", json={})
    prof = json.loads(zipfile.ZipFile(io.BytesIO(r.data)).read("data/agent_profile.json"))
    assert prof["palette"]["user_bg"] == "#abcdef"
    assert prof["palette"]["agent_bg"] == "#fedcba"
    assert prof["palette"]["ui_field"] == "#0a0b0c"
    assert prof["palette"]["ui_subtitle"] == "#0d0e0f"


def test_exported_agent_sources_use_agent_text_colour(agentclient, tmp_path):
    rule = ".chat-bubble .cite .file { color: inherit; }"
    base = Path(__file__).resolve().parents[1]
    assert rule in (base / "rag_app" / "static" / "style.css").read_text(encoding="utf-8")
    r = agentclient.post("/api/agent/export", json={})
    zip_css = zipfile.ZipFile(io.BytesIO(r.data)).read("rag_app/static/style.css").decode("utf-8")
    assert rule in zip_css


def test_palettes_save_list_delete(agentclient):
    agentclient.post("/api/agent/profile", json={"palette": {"user_bg": "#010203"}})
    r = agentclient.post("/api/agent/palettes", json={"name": "Brand", "palette": {"user_bg": "#010203", "agent_bg": "#040506"}})
    assert r.status_code == 200
    palettes = agentclient.get("/api/agent/palettes").get_json()["palettes"]
    saved = {p["name"]: p for p in palettes}
    assert "Brand" in saved
    assert saved["Brand"]["colors"]["user_bg"] == "#010203"
    assert set(saved["Brand"]["colors"]) == {
        "user_bg", "user_text", "agent_bg", "agent_text", "ui_bg", "ui_field", "ui_text",
        "ui_subtitle"}
    # save requires a name
    assert agentclient.post("/api/agent/palettes", json={"name": "  "}).status_code == 400
    # delete
    agentclient.delete("/api/agent/palettes/Brand")
    names = [p["name"] for p in agentclient.get("/api/agent/palettes").get_json()["palettes"]]
    assert "Brand" not in names


def test_favicon_uses_agent_bg(agentclient):
    agentclient.post("/api/agent/profile", json={"palette": {"agent_bg": "#123456"}})
    r = agentclient.get("/api/agent/favicon")
    assert r.status_code == 200
    assert "#123456" in r.get_data(as_text=True)


def test_subtitle_defaults_to_dashboard_title(tmp_path):
    settings = Settings(
        kb_dir=str(tmp_path / "docs"), data_dir=str(tmp_path / "data"),
        env_file=str(tmp_path / ".env"), source="files", enable_rerank=False,
        title="Конкурс рисунков — тест",
    )
    settings.ensure_dirs()
    app = create_app(settings)
    app.config["TESTING"] = True
    c = app.test_client()
    r = c.post("/api/agent/export", json={})
    prof = json.loads(zipfile.ZipFile(io.BytesIO(r.data)).read("data/agent_profile.json"))
    assert prof["subtitle"] == "Конкурс рисунков — тест"


def test_slugify_is_ascii():
    from rag_app.agent_export import slugify
    assert slugify("AI-агентушка") == "ai-agentushka"
    assert slugify("Лучший Бот 2026!") == "luchshiy-bot-2026"
    assert slugify("") == "agent"


def test_export_download_name_is_ascii(agentclient):
    agentclient.post("/api/agent/profile", json={"agent_name": "AI-агентушка"})
    r = agentclient.post("/api/agent/export", json={})
    cd = r.headers.get("Content-Disposition", "")
    # plain ASCII filename, no RFC5987 filename* (which mangles the saved name)
    assert "filename*" not in cd
    assert 'filename="ai-agentushka-agent.zip"' in cd


# -------------------------------------------------------------- documents
def test_document_upload_triggers_rebuild(agentclient):
    r = _upload(agentclient, "knowledge_base.md", "# Док\n\nПривет мир\n".encode("utf-8"))
    assert r.status_code == 200
    body = r.get_json()
    assert "document" in body
    # rebuild either succeeds (ok) or degrades gracefully (uploaded + warning)
    assert body["status"] in ("ok", "uploaded")
    if body["status"] == "ok":
        assert "chunks" in body["rebuilt"]


def test_document_upload_rejects_bad_extension(agentclient):
    r = _upload(agentclient, "evil.exe", b"data")
    assert r.status_code == 400


class _FakeUpload:
    def __init__(self, name: str, data: bytes):
        self.filename = name
        self._data = data

    def read(self) -> bytes:
        return self._data


def test_ingestion_accepts_docx(tmp_path):
    docx = pytest.importorskip("docx")
    from rag_app.ingestion import IngestionService

    s = Settings(kb_dir=str(tmp_path / "docs"), data_dir=str(tmp_path / "data"),
                 source_docs_dir=str(tmp_path / "src"))
    s.ensure_dirs()
    buf = io.BytesIO()
    doc = docx.Document()
    doc.add_paragraph("УникальныйТекстDocx")
    doc.save(buf)

    info = IngestionService(s).save_upload(_FakeUpload("Письмо.docx", buf.getvalue()))
    assert info["name"] == "Письмо.md"
    kb = (tmp_path / "docs" / "knowledge_base.md").read_text(encoding="utf-8")
    assert "УникальныйТекстDocx" in kb


# ---------------------------------------------------------------- favicon
def test_favicon_is_svg(agentclient):
    r = agentclient.get("/api/agent/favicon")
    assert r.status_code == 200
    assert "svg" in r.headers.get("Content-Type", "")
