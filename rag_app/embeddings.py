"""Embedding model loading with a local cache and offline support."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from .config import PROJECT_ROOT

# Reuse the standard HuggingFace cache when it already exists (so an
# existing installation keeps working offline). On a truly fresh machine we
# fall back to a project-local cache to keep the product self-contained.
_HF_CACHE = PROJECT_ROOT / "data" / "models_cache"
_DEFAULT_HF_CACHE = Path(os.path.expanduser("~/.cache/huggingface"))
if not os.environ.get("HF_HOME") and not _DEFAULT_HF_CACHE.is_dir():
    os.environ.setdefault("HF_HOME", str(_HF_CACHE))
    os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(_HF_CACHE))


class Embedder:
    def __init__(self, model_name: str, hf_token: str = ""):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        kwargs = {}
        if hf_token:
            kwargs["token"] = hf_token
        self._model = SentenceTransformer(model_name, **kwargs)

    @property
    def dim(self) -> int:
        return int(self._model.get_sentence_embedding_dimension())

    def encode(self, texts: list[str], normalize: bool = True) -> np.ndarray:
        emb = self._model.encode(
            texts, show_progress_bar=False, normalize_embeddings=normalize
        )
        return np.asarray(emb, dtype="float32")


def load_embedder(model_name: str, hf_token: str = "") -> Embedder:
    return Embedder(model_name, hf_token=hf_token)
