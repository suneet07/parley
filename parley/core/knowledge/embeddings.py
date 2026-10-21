"""Text embedding providers: fastembed (BGE) with deterministic HashEmbedder fallback."""

from __future__ import annotations

import functools
import zlib
from typing import Protocol

import numpy as np
import structlog

logger = structlog.get_logger(__name__)


class Embedder(Protocol):
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray:
        """Shape (n, dim), L2-normalized float32."""
        ...

    def embed_query(self, text: str) -> np.ndarray:
        """Shape (dim,), L2-normalized float32."""
        ...


class FastEmbedder:
    """Wraps fastembed.TextEmbedding with BAAI/bge-small-en-v1.5 (384 dimensions)."""

    dim = 384

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        cache_dir: str = ".cache/fastembed",
    ) -> None:
        from fastembed import TextEmbedding

        self._model = TextEmbedding(model_name=model_name, cache_dir=cache_dir)
        self.dim = 384

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)

        raw = list(self._model.embed(texts))
        arr = np.array(raw, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (arr / norms).astype(np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        raw = list(self._model.query_embed(text))
        if not raw:
            return np.zeros(self.dim, dtype=np.float32)

        arr = np.asarray(raw[0], dtype=np.float32).ravel()
        norm = np.linalg.norm(arr)
        if norm > 0:
            arr = arr / norm
        return arr.astype(np.float32)


class HashEmbedder:
    """Deterministic, dependency-free fallback for tests/offline:
    hashed char 3-grams -> dim 256, L2-normalized.
    """

    dim = 256

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)

        matrix = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            matrix[i] = self._vectorize(text)
        return matrix

    def embed_query(self, text: str) -> np.ndarray:
        return self._vectorize(text)

    def _vectorize(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        cleaned = f"  {text.lower().strip()}  "
        if len(cleaned) < 3:
            vec[0] = 1.0
            return vec

        for idx in range(len(cleaned) - 2):
            gram = cleaned[idx : idx + 3]
            bucket = zlib.crc32(gram.encode("utf-8")) % self.dim
            vec[bucket] += 1.0

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        else:
            vec[0] = 1.0
        return vec.astype(np.float32)


@functools.cache
def default_embedder() -> Embedder:
    """FastEmbedder, or HashEmbedder (with a warning) if the model cannot load."""
    try:
        embedder = FastEmbedder()
        _ = embedder.embed(["warmup"])
        return embedder
    except Exception as exc:
        logger.warning("fastembed_load_failed", error=str(exc), fallback="HashEmbedder")
        return HashEmbedder()
