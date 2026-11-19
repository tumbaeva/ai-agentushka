"""rag_app — application factory and shared context."""

from __future__ import annotations

import logging
import threading

from flask import Flask

from .chat_store import ChatStore
from .config import Settings, load_settings
from .index import IndexManager
from .ingestion import IngestionService
from .loaders import detect_title, load_folder, load_knowledge_base_md
from .agent_profile import load_agent_profile
from .retrieval import Retriever

log = logging.getLogger("rag_app")


class AppContext:
    """Runtime container shared by all blueprints."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.index = IndexManager(settings)
        self.retriever = Retriever(self.index, settings)
        self.chat = ChatStore(settings.chat_db_path)
        self.ingestion = IngestionService(settings)
        self.profile = load_agent_profile(settings)
        self._builder = None
        self.rebuild_lock = threading.Lock()
        self.sweep: dict = {"status": "idle", "message": "", "done": 0, "total": 0,
                            "result": None, "error": None}
        self.title: str = settings.title or ""

    @property
    def builder(self):
        """Lazy: the KB-builder (and its OCR deps) is dashboard-only."""
        if self._builder is None:
            from .kb_builder import KnowledgeBaseBuilder
            self._builder = KnowledgeBaseBuilder(self.settings)
        return self._builder

    # ------------------------------------------------------------------
    def llm_client(self):
        creds = self.settings.gigachat_credentials
        if not creds:
            return None
        from openai import OpenAI
        return OpenAI(api_key=creds, base_url=self.settings.llm_base_url)

    def load_reranker(self) -> None:
        self.retriever.rerank_error = None
        if not self.settings.enable_rerank:
            self.retriever.reranker = None
            return
        try:
            from sentence_transformers import CrossEncoder
            self.retriever.reranker = CrossEncoder(self.settings.resolved_reranker)
            log.info("Reranker loaded: %s", self.settings.resolved_reranker)
        except Exception as e:  # pragma: no cover
            self.retriever.reranker = None
            self.retriever.rerank_error = f"{type(e).__name__}: {e}"
            log.warning("Reranker unavailable: %s", e)

    def load_documents(self):
        if self.settings.source == "kb_md":
            md = self.settings.kb_path / "knowledge_base.md"
            return load_knowledge_base_md(md)
        return load_folder(self.settings.kb_path)

    def compute_title(self, docs=None) -> str:
        self.title = self.settings.title or detect_title(docs if docs is not None else self.load_documents())
        return self.title

    def rebuild(self) -> dict:
        with self.rebuild_lock:
            docs = self.load_documents()
            self.index.build(
                docs,
                embedding_model=self.settings.resolved_embedding,
                chunk_size=self.settings.chunk_size,
                chunk_overlap=self.settings.chunk_overlap,
            )
            self.compute_title(docs)
            self.retriever.client = self.llm_client()
            self.load_reranker()
            return {"chunks": len(self.index.chunks), "documents": len(docs)}

    def bootstrap_async(self) -> threading.Thread:
        """Start indexing in the background so the HTTP server is responsive."""
        t = threading.Thread(target=self.bootstrap, name="rag-bootstrap", daemon=True)
        t.start()
        return t

    # ------------------------------------------------------------------
    def start_sweep(self, options: dict) -> bool:
        """Run a configuration sweep in the background. Returns False if busy."""
        if self.sweep.get("status") == "running":
            return False
        from .sweep import run_sweep

        self.sweep = {"status": "running", "message": "Запуск…", "done": 0, "total": 0,
                      "result": None, "error": None}
        state = self.sweep

        def progress(msg, done, total):
            state["message"] = msg
            state["done"] = done
            state["total"] = total

        def worker():
            try:
                self.sweep["result"] = run_sweep(self, options, progress=progress)
                self.sweep["status"] = "done"
            except Exception as e:  # pragma: no cover
                self.sweep["status"] = "error"
                self.sweep["error"] = f"{type(e).__name__}: {e}"
                log.error("Sweep failed: %s", e)

        threading.Thread(target=worker, name="rag-sweep", daemon=True).start()
        return True

    def apply_recommended(self, rec: dict) -> dict:
        """Apply recommended settings and rebuild the live index."""
        s = self.settings
        if "embedding_id" in rec:
            s.embedding_id = rec["embedding_id"]
            s.embedding_name = ""
        if "chunk_size" in rec:
            s.chunk_size = int(rec["chunk_size"])
        if "chunk_overlap" in rec:
            s.chunk_overlap = int(rec["chunk_overlap"])
        if "default_mode" in rec:
            s.default_mode = rec["default_mode"]
        if "default_temp_mq" in rec:
            s.default_temp_mq = float(rec["default_temp_mq"])
        if "default_temp" in rec:
            s.default_temp = float(rec["default_temp"])
        if "enable_rerank" in rec:
            s.enable_rerank = bool(rec["enable_rerank"])
        if "rrf_k" in rec:
            s.rrf_k = int(rec["rrf_k"])
        if "vector_weight" in rec:
            s.vector_weight = float(rec["vector_weight"])
        if "bm25_weight" in rec:
            s.bm25_weight = float(rec["bm25_weight"])
        from .config import save_settings
        save_settings(s)
        info = self.rebuild()
        return {"config": s.to_public(), **info}

    def bootstrap(self) -> None:
        """Load a snapshot when it matches the current KB, otherwise build fresh."""
        self.settings.ensure_dirs()
        self.retriever.client = self.llm_client()

        docs = self.load_documents()
        self.compute_title(docs)
        if not docs:
            self.index.status = "empty"
            return
        self.index.status = "building"
        from .chunking import chunk_documents
        from .index import corpus_hash
        chunks = [c.to_dict() for c in chunk_documents(
            docs, chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
            min_chunk_size=self.settings.min_chunk_size,
        )]
        expected = corpus_hash(chunks)

        snapshot = self.index.latest_snapshot()
        if snapshot is not None:
            try:
                if self.index.load_snapshot(snapshot, expected_fingerprint=expected):
                    log.info("Loaded snapshot: %s", snapshot.name)
                    self.load_reranker()
                    return
            except Exception as e:  # pragma: no cover
                log.warning("Snapshot load failed (%s), rebuilding", e)

        try:
            self.rebuild()
        except Exception as e:  # pragma: no cover
            self.index.status = "error"
            self.index.load_error = f"{type(e).__name__}: {e}"
            log.error("Bootstrap build failed: %s", e)


def create_app(settings: Settings | None = None, *, standalone: bool = False) -> Flask:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    settings = settings or load_settings()
    app = Flask(__name__, static_folder="static", template_folder="templates")
    app.config["SETTINGS"] = settings
    # Always re-read templates from disk (local product; avoids stale UI after edits)
    app.config["TEMPLATES_AUTO_RELOAD"] = True
    app.jinja_env.auto_reload = True

    ctx = AppContext(settings)
    app.extensions["rag_ctx"] = ctx

    from .web.agent import agent_bp, render_agent
    if standalone:
        # Chat-only agent app: no dashboard chrome/settings, just the agent chat.
        app.config["AGENT_STANDALONE"] = True
        app.register_blueprint(agent_bp)
        app.add_url_rule("/", "agent_home", render_agent)
    else:
        from .web.api import api_bp
        from .web.kb_builder import kb_bp
        from .web.pages import pages_bp
        app.register_blueprint(api_bp)
        app.register_blueprint(kb_bp)
        app.register_blueprint(pages_bp)
        app.register_blueprint(agent_bp)

    ctx.bootstrap_async()
    return app


def get_ctx(app: Flask) -> AppContext:
    return app.extensions["rag_ctx"]
