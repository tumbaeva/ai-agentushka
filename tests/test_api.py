from pathlib import Path

import pytest

from rag_app import create_app
from rag_app.config import Settings


@pytest.fixture()
def client(tmp_path: Path):
    settings = Settings(
        kb_dir=str(tmp_path / "docs"),
        data_dir=str(tmp_path / "data"),
        env_file=str(tmp_path / ".env"),
        source="files",
        enable_rerank=False,
    )
    settings.ensure_dirs()
    app = create_app(settings)
    app.config["TESTING"] = True
    return app.test_client()


def test_modes(client):
    data = client.get("/api/modes").get_json()
    assert "hybrid" in data["modes"]


def test_config_public_hides_secrets(client):
    data = client.get("/api/config").get_json()["config"]
    assert "hf_token" not in data
    assert "gigachat_key_secret" not in data
    assert "hf_token_set" in data


def test_secrets_are_masked(client):
    client.post("/api/secrets", json={"secrets": {"GIGACHAT_API_KEY_ID": "abcdef123456"}})
    data = client.get("/api/secrets").get_json()
    masked = data["secrets_masked"]["GIGACHAT_API_KEY_ID"]
    assert masked != "abcdef123456"
    assert "•" in masked
    assert data["present"]["GIGACHAT_API_KEY_ID"] is True


def test_search_requires_query(client):
    resp = client.post("/api/search", json={})
    assert resp.status_code == 400


def test_search_empty_index_returns_results_list(client):
    resp = client.post("/api/search", json={"query": "номинации", "mode": "hybrid"})
    assert resp.status_code == 200
    assert resp.get_json()["results"] == []


def test_upload_rejects_bad_extension(client):
    from io import BytesIO
    resp = client.post(
        "/api/documents",
        data={"file": (BytesIO(b"data"), "evil.exe")},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 400


def test_eval_generate_without_llm(client):
    r = client.post("/api/eval/generate", json={"count": 3})
    assert r.status_code == 400


def test_eval_questions_default(client):
    data = client.get("/api/eval/questions").get_json()
    assert len(data["questions"]) >= 1
    assert "documents" in data


def test_eval_questions_save_and_read(client):
    qs = [{"id": "X", "query": "тестовый вопрос", "relevant_docs": ["a"], "keywords": ["к"]}]
    r = client.post("/api/eval/questions", json={"questions": qs}).get_json()
    assert r["count"] == 1
    data = client.get("/api/eval/questions").get_json()
    assert data["questions"][0]["query"] == "тестовый вопрос"


def test_eval_detail_returns_chunks_list(client):
    r = client.post("/api/eval/detail", json={"index": 0, "mode": "hybrid", "k": 4})
    assert r.status_code == 200
    assert r.get_json()["chunks"] == []


def test_eval_detail_bad_index(client):
    r = client.post("/api/eval/detail", json={"index": 99, "mode": "hybrid"})
    assert r.status_code == 400


def test_help_page(client):
    r = client.get("/help")
    assert r.status_code == 200
    body = r.get_data(as_text=True)
    assert "Справка" in body
    assert "md-table" in body or "<h1" in body


def test_recommendations_page(client):
    r = client.get("/recommendations")
    assert r.status_code == 200
    body = r.get_data(as_text=True)
    assert "recommendations.js" in body
    assert "tooltip.js" in body


def test_index_page_has_recommendations_link(client):
    body = client.get("/").get_data(as_text=True)
    assert "/recommendations" in body
    assert "tooltip.js" in body


def test_models_exposes_title(client):
    d = client.get("/api/models").get_json()
    assert "title" in d


def test_recommend_options(client):
    d = client.get("/api/recommend/options").get_json()
    assert "embeddings" in d and "modes" in d and "metrics" in d
    assert "score_avg" in d["metrics"]


def test_recommend_status_idle(client):
    d = client.get("/api/recommend/status").get_json()
    assert d["status"] in ("idle", "done", "running", "error", None) or d["status"] == "idle"


def test_recommend_save_without_run(client):
    r = client.post("/api/recommend/save")
    assert r.status_code == 400


def test_recommend_apply_requires_payload(client):
    resp = client.post("/api/recommend/apply", json={})
    assert resp.status_code == 400


def test_snapshots_endpoint_empty(client):
    data = client.get("/api/snapshots").get_json()
    assert data["snapshots"] == []


def test_upload_kb_replaces_knowledge_base(client):
    from io import BytesIO
    payload = "# Документ 1\n\nтекст\n\n# Документ 2\n\nещё текст\n".encode("utf-8")
    resp = client.post(
        "/api/documents",
        data={"file": (BytesIO(payload), "knowledge_base.md")},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["document"]["replaced"] is True
    # the KB file now contains both documents
    docs = client.get("/api/docs").get_json()["files"]
    assert {d["name"] for d in docs} == {"Документ 1", "Документ 2"}


def test_chat_message_without_llm_still_answers(client):
    resp = client.post("/api/chat/message", json={"query": "сроки"})
    assert resp.status_code == 200
    body = resp.get_json()
    assert "answer" in body
    assert body["session_id"]
    # history persists
    sessions = client.get("/api/chat/sessions").get_json()["sessions"]
    assert len(sessions) == 1
