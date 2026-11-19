"""Application settings: paths, model registry and tunable retrieval params.

Settings are loaded from environment variables / .env and from
``data/settings.json`` (written by the UI). Everything that used to live in
module-level globals in the monolithic app now flows through here.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict, field, fields
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SETTINGS_FILE = DATA_DIR / "settings.json"

# ---------------------------------------------------------------------------
# Embedding model registry
# ---------------------------------------------------------------------------
# ``type`` drives how the model is loaded: ``local``/``hf`` go through
# sentence-transformers, ``api`` is reserved for remote embedding endpoints.
EMBEDDING_MODELS = [
    {
        "id": "bge-m3",
        "name": "BGE-M3 (мультиязычная, рекомендуется)",
        "type": "hf",
        "model": "BAAI/bge-m3",
        "dim": 1024,
        "note": "Лучшее качество для русского. ~2.3 ГБ, скачивается один раз.",
    },
    {
        "id": "e5-large",
        "name": "multilingual-e5-large",
        "type": "hf",
        "model": "intfloat/multilingual-e5-large",
        "dim": 1024,
        "note": "Сильная мультиязычная модель. ~2.2 ГБ.",
    },
    {
        "id": "frida",
        "name": "FRIDA (Sber, требует HF_TOKEN)",
        "type": "hf",
        "model": "ai-forever/FRIDA",
        "dim": 1536,
        "note": "Gated-модель: нужен HuggingFace token.",
    },
    {
        "id": "local",
        "name": "MiniLM (лёгкая, быстрая)",
        "type": "local",
        "model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "dim": 384,
        "note": "Маленькая и быстрая, но слабее на русском.",
    },
]

RERANK_MODELS = [
    {"id": "bge-reranker-v2-m3", "name": "BGE Reranker v2 m3", "model": "BAAI/bge-reranker-v2-m3"},
    {"id": "mmarco", "name": "mmarco-mMiniLMv2 (лёгкий)", "model": "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"},
]

LLM_MODELS = [
    {"id": "gigachat3-10b", "name": "GigaChat3 10B (Sber)", "model": "ai-sage/GigaChat3-10B-A1.8B", "base_url": "https://foundation-models.api.cloud.ru/v1"},
    {"id": "gigachat3-8b", "name": "GigaChat3 8B (Sber)", "model": "ai-sage/GigaChat3-8B-A1.8B", "base_url": "https://foundation-models.api.cloud.ru/v1"},
]

DEFAULT_LLM_BASE_URL = "https://foundation-models.api.cloud.ru/v1"


def _emb_by_id(model_id: str) -> dict | None:
    return next((m for m in EMBEDDING_MODELS if m["id"] == model_id), None)


def _rerank_by_id(model_id: str) -> dict | None:
    return next((m for m in RERANK_MODELS if m["id"] == model_id), None)


def resolve_embedding_model(embedding_id: str, embedding_name: str | None = None) -> str:
    """Return the concrete HF/ST model name for a registry id."""
    if embedding_name:
        return embedding_name
    entry = _emb_by_id(embedding_id)
    if entry:
        return entry["model"]
    # Legacy ids from the old app.
    legacy = {
        "bge-large": "BAAI/bge-m3",
        "sber": "ai-ru/bge-mularge",
    }
    return legacy.get(embedding_id, EMBEDDING_MODELS[0]["model"])


@dataclass
class Settings:
    # paths
    kb_dir: str = str(PROJECT_ROOT / "docs")
    source_docs_dir: str = str(PROJECT_ROOT / "source_docs")
    data_dir: str = str(DATA_DIR)
    env_file: str = str(PROJECT_ROOT / ".env")
    host: str = "127.0.0.1"
    port: int = 5555

    # generation
    default_temp: float = 0.2
    default_temp_mq: float = 0.7
    llm_model: str = "ai-sage/GigaChat3-10B-A1.8B"
    llm_base_url: str = DEFAULT_LLM_BASE_URL

    # chunking
    chunk_size: int = 700
    chunk_overlap: int = 120
    min_chunk_size: int = 120

    # embedding
    embedding_id: str = "local"
    embedding_name: str = ""  # concrete model; empty = resolve from embedding_id

    # retrieval tuning
    rrf_k: int = 60
    vector_weight: float = 1.0
    bm25_weight: float = 1.0
    default_mode: str = "hybrid"
    default_k: int = 4

    # query transforms
    enable_query_rewrite: bool = True
    enable_hyde: bool = False

    # rerank
    enable_rerank: bool = True
    rerank_id: str = "mmarco"

    # source: 'kb_md' (chunks from knowledge_base.md) or 'files' (raw documents)
    source: str = "kb_md"
    title: str = ""  # contest/product name; empty = auto-detect from KB

    # secrets (not persisted to settings.json)
    hf_token: str = field(default="", repr=False)
    gigachat_key_id: str = field(default="", repr=False)
    gigachat_key_secret: str = field(default="", repr=False)

    # ------------------------------------------------------------------
    @property
    def kb_path(self) -> Path:
        return Path(self.kb_dir)

    @property
    def source_docs_path(self) -> Path:
        return Path(self.source_docs_dir)

    @property
    def data_path(self) -> Path:
        return Path(self.data_dir)

    @property
    def uploads_path(self) -> Path:
        return self.data_path / "uploads"

    @property
    def models_path(self) -> Path:
        return self.data_path / "index"

    @property
    def chat_db_path(self) -> Path:
        return self.data_path / "chat.db"

    @property
    def eval_questions_path(self) -> Path:
        return self.data_path / "eval_questions.json"

    @property
    def heatmap_path(self) -> Path:
        return self.data_path / "retrieval_heatmap.png"

    @property
    def agent_profile_path(self) -> Path:
        return self.data_path / "agent_profile.json"

    @property
    def agent_assets_path(self) -> Path:
        return self.data_path / "agent_assets"

    @property
    def resolved_embedding(self) -> str:
        return resolve_embedding_model(self.embedding_id, self.embedding_name or None)

    @property
    def resolved_reranker(self) -> str:
        entry = _rerank_by_id(self.rerank_id)
        return entry["model"] if entry else RERANK_MODELS[0]["model"]

    @property
    def gigachat_credentials(self) -> str:
        return (self.gigachat_key_secret or self.gigachat_key_id or "").strip()

    def ensure_dirs(self) -> None:
        for p in (self.kb_path, self.source_docs_path, self.data_path,
                  self.uploads_path, self.models_path):
            p.mkdir(parents=True, exist_ok=True)

    def to_public(self) -> dict:
        """Serializable view without secret values."""
        data = asdict(self)
        data.pop("hf_token", None)
        data.pop("gigachat_key_id", None)
        data.pop("gigachat_key_secret", None)
        data["resolved_embedding"] = self.resolved_embedding
        data["hf_token_set"] = bool(self.hf_token)
        data["gigachat_configured"] = bool(self.gigachat_credentials)
        return data


def _env(*names: str, default: str = "") -> str:
    for n in names:
        v = os.environ.get(n)
        if v:
            return v.strip()
    return default


def load_settings() -> Settings:
    """Build Settings from .env + data/settings.json (JSON wins over env)."""
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    s = Settings()

    # secrets / things that only come from env
    s.hf_token = _env("HF_TOKEN")
    if s.hf_token in {"ВАШ_HF_READ_ТОКЕН", "your_token_here"}:
        s.hf_token = ""
    s.gigachat_key_id = _env("GIGACHAT_API_KEY_ID")
    s.gigachat_key_secret = _env("GIGACHAT_API_KEY_SECRET")

    s.kb_dir = _env("RAG_KB_DIR", default=s.kb_dir)
    s.llm_model = _env("LLM_MODEL", default=s.llm_model)
    s.llm_base_url = _env("LLM_BASE_URL", default=s.llm_base_url)
    s.embedding_id = _env("EMB_MODEL_TYPE", default=s.embedding_id)
    s.embedding_name = _env("EMB_MODEL_NAME", default="")
    if os.environ.get("RAG_HOST"):
        s.host = os.environ["RAG_HOST"].strip()
    if os.environ.get("RAG_PORT"):
        try:
            s.port = int(os.environ["RAG_PORT"])
        except ValueError:
            pass

    # persisted UI settings
    if SETTINGS_FILE.is_file():
        try:
            stored = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            stored = {}
        allowed = {f.name for f in fields(Settings)} - {
            "hf_token", "gigachat_key_id", "gigachat_key_secret"
        }
        for key, value in stored.items():
            if key in allowed:
                setattr(s, key, value)

    return s


_SECRET_FIELDS = {"hf_token", "gigachat_key_id", "gigachat_key_secret"}


def save_settings(s: Settings) -> None:
    SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    data = {f.name: getattr(s, f.name) for f in fields(Settings)}
    for key in _SECRET_FIELDS:
        data.pop(key, None)
    SETTINGS_FILE.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
