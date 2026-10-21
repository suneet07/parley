"""In-memory hybrid retriever combining dense vector cosine similarity and BM25 with RRF."""

from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from parley.core.knowledge.base import Passage
from parley.core.knowledge.bm25 import BM25
from parley.core.knowledge.chunking import Chunk, chunk_markdown
from parley.core.knowledge.embeddings import Embedder, default_embedder
from parley.core.knowledge.scoring import fuse_score

if TYPE_CHECKING:
    from parley.config import BusinessConfig


class InMemoryRetriever:
    """In-memory retriever using dense embeddings + BM25 reciprocal rank fusion."""

    def __init__(
        self,
        chunks: list[Chunk],
        embedder: Embedder | None = None,
        vectors: np.ndarray | None = None,
    ) -> None:
        self.chunks = list(chunks)
        self.embedder = embedder if embedder is not None else default_embedder()
        dim = getattr(self.embedder, "dim", 384)

        if vectors is not None:
            self.vectors = np.asarray(vectors, dtype=np.float32)
        elif self.chunks:
            texts = [c.text for c in self.chunks]
            self.vectors = np.asarray(self.embedder.embed(texts), dtype=np.float32)
        else:
            self.vectors = np.empty((0, dim), dtype=np.float32)

        if len(self.vectors) > 0:
            norms = np.linalg.norm(self.vectors, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            self.vectors = (self.vectors / norms).astype(np.float32)

        self.bm25 = BM25([c.text for c in self.chunks])

    @classmethod
    async def from_business(
        cls,
        business: BusinessConfig,
        embedder: Embedder | None = None,
        cache_dir: Path = Path(".cache/knowledge"),
    ) -> InMemoryRetriever:
        """Read every business.knowledge file (utf-8), chunk, embed.

        Cache vectors as .npy keyed by sha256 of (embedder class name + all chunk texts)
        so restarts are instant. Missing file -> raise FileNotFoundError naming it.
        """
        if embedder is None:
            embedder = default_embedder()

        chunks: list[Chunk] = []
        for rel_path in business.knowledge:
            p = business.resolve(rel_path)
            if not p.exists() or not p.is_file():
                raise FileNotFoundError(f"Knowledge file not found: {p}")
            text = p.read_text(encoding="utf-8")
            file_chunks = chunk_markdown(source=Path(rel_path).name, text=text)
            chunks.extend(file_chunks)

        cache_path_dir = Path(cache_dir)
        embedder_name = embedder.__class__.__name__
        key_data = embedder_name + "\n" + "\n".join(c.text for c in chunks)
        key_hash = hashlib.sha256(key_data.encode("utf-8")).hexdigest()
        cache_file = cache_path_dir / f"{key_hash}.npy"

        if cache_file.exists():
            vectors = np.load(cache_file)
        else:
            texts = [c.text for c in chunks]
            if texts:
                vectors = await asyncio.to_thread(embedder.embed, texts)
                vectors = np.asarray(vectors, dtype=np.float32)
            else:
                dim = getattr(embedder, "dim", 384)
                vectors = np.empty((0, dim), dtype=np.float32)
            cache_path_dir.mkdir(parents=True, exist_ok=True)
            np.save(cache_file, vectors)

        return cls(chunks=chunks, embedder=embedder, vectors=vectors)

    async def search(self, query: str, k: int = 4) -> list[Passage]:
        if not query or not query.strip() or not self.chunks or k <= 0:
            return []

        if hasattr(self.embedder, "embed_query"):
            q_vec = await asyncio.to_thread(self.embedder.embed_query, query)
        else:
            q_vecs = await asyncio.to_thread(self.embedder.embed, [query])
            q_vec = q_vecs[0]

        query_vec = np.asarray(q_vec, dtype=np.float32).ravel()
        q_norm = np.linalg.norm(query_vec)
        if q_norm > 0:
            query_vec = query_vec / q_norm

        n = len(self.chunks)
        dense = np.dot(self.vectors, query_vec)
        sparse = self.bm25.scores(query)

        # Reciprocal Rank Fusion: rrf[i] = Σ 1/(60 + rank) (rank from 1)
        rrf = np.zeros(n, dtype=np.float64)

        dense_top_indices = np.argsort(-dense)[: min(20, n)]
        for rank, idx in enumerate(dense_top_indices, start=1):
            rrf[idx] += 1.0 / (60.0 + rank)

        sparse_top_indices = np.argsort(-sparse)[: min(20, n)]
        for rank, idx in enumerate(sparse_top_indices, start=1):
            if sparse[idx] > 0:
                rrf[idx] += 1.0 / (60.0 + rank)

        scores = np.zeros(n, dtype=np.float32)
        for i in range(n):
            dense_cos = float(dense[i])
            bm25_val = float(sparse[i])
            scores[i] = fuse_score(dense_cos, bm25_val, sparse_half=3.0)

        candidate_indices = sorted(range(n), key=lambda i: (rrf[i], scores[i]), reverse=True)
        top_indices = candidate_indices[:k]

        return [
            Passage(
                source=self.chunks[i].source,
                chunk_id=self.chunks[i].chunk_id,
                heading=self.chunks[i].heading,
                text=self.chunks[i].text,
                score=float(scores[i]),
            )
            for i in top_indices
        ]
