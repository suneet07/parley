"""PostgreSQL + pgvector hybrid retriever backend with RRF."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import numpy as np

from parley.core.knowledge.base import Passage
from parley.core.knowledge.chunking import Chunk
from parley.core.knowledge.embeddings import Embedder, default_embedder
from parley.core.knowledge.scoring import fuse_score

if TYPE_CHECKING:
    import asyncpg

SCHEMA_SQL = """\
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS kb_chunks (
    id bigserial PRIMARY KEY,
    tenant_id text NOT NULL,
    source text NOT NULL,
    chunk_id text NOT NULL,
    heading text,
    text text NOT NULL,
    embedding vector(384),
    tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED,
    UNIQUE(tenant_id, chunk_id)
);

CREATE INDEX IF NOT EXISTS idx_kb_chunks_embedding
    ON kb_chunks USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS idx_kb_chunks_tsv
    ON kb_chunks USING gin (tsv);
"""


class PgVectorRetriever:
    """PostgreSQL hybrid backend using pgvector + full-text search."""

    def __init__(
        self,
        dsn: str,
        tenant_id: str,
        embedder: Embedder | None = None,
    ) -> None:
        self.dsn = dsn.replace("+asyncpg", "")
        self.tenant_id = tenant_id
        self.embedder = embedder if embedder is not None else default_embedder()
        self._pool: Any = None

    async def _get_pool(self) -> asyncpg.Pool:
        if self._pool is None:
            import asyncpg
            from pgvector.asyncpg import register_vector

            async def init_connection(conn: asyncpg.Connection) -> None:
                await register_vector(conn)

            self._pool = await asyncpg.create_pool(self.dsn, init=init_connection)
        return self._pool

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    async def ensure_schema(self) -> None:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            await conn.execute(SCHEMA_SQL)

    async def upsert(self, chunks: list[Chunk]) -> int:
        if not chunks:
            return 0

        sources = list({c.source for c in chunks})
        texts = [c.text for c in chunks]
        embeddings = await asyncio.to_thread(self.embedder.embed, texts)

        pool = await self._get_pool()
        async with pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    "DELETE FROM kb_chunks WHERE tenant_id = $1 AND source = ANY($2::text[])",
                    self.tenant_id,
                    sources,
                )
                records = [
                    (
                        self.tenant_id,
                        c.source,
                        c.chunk_id,
                        c.heading,
                        c.text,
                        np.asarray(emb, dtype=np.float32),
                    )
                    for c, emb in zip(chunks, embeddings, strict=True)
                ]
                await conn.executemany(
                    """
                    INSERT INTO kb_chunks (tenant_id, source, chunk_id, heading, text, embedding)
                    VALUES ($1, $2, $3, $4, $5, $6)
                    ON CONFLICT (tenant_id, chunk_id) DO UPDATE
                    SET heading = EXCLUDED.heading,
                        text = EXCLUDED.text,
                        embedding = EXCLUDED.embedding
                    """,
                    records,
                )
        return len(chunks)

    async def search(self, query: str, k: int = 4) -> list[Passage]:
        if not query or not query.strip() or k <= 0:
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

        query_sql = """
        WITH dense AS (
            SELECT
                id,
                source,
                chunk_id,
                heading,
                text,
                (1.0 - (embedding <=> $1)) AS dense_cos,
                ROW_NUMBER() OVER (ORDER BY embedding <=> $1) AS dense_rank
            FROM kb_chunks
            WHERE tenant_id = $2
            ORDER BY embedding <=> $1
            LIMIT 20
        ),
        sparse AS (
            SELECT
                id,
                source,
                chunk_id,
                heading,
                text,
                ts_rank_cd(tsv, websearch_to_tsquery('english', $3)) AS sparse_rank_cd,
                ROW_NUMBER() OVER (
                    ORDER BY ts_rank_cd(tsv, websearch_to_tsquery('english', $3)) DESC
                ) AS sparse_rank
            FROM kb_chunks
            WHERE tenant_id = $2
              AND tsv @@ websearch_to_tsquery('english', $3)
            ORDER BY ts_rank_cd(tsv, websearch_to_tsquery('english', $3)) DESC
            LIMIT 20
        ),
        fused AS (
            SELECT
                COALESCE(d.id, s.id) AS id,
                COALESCE(d.source, s.source) AS source,
                COALESCE(d.chunk_id, s.chunk_id) AS chunk_id,
                COALESCE(d.heading, s.heading) AS heading,
                COALESCE(d.text, s.text) AS text,
                COALESCE(d.dense_cos, 0.0) AS dense_cos,
                COALESCE(s.sparse_rank_cd, 0.0) AS ts_rank,
                (COALESCE(1.0 / (60.0 + d.dense_rank), 0.0) +
                 COALESCE(1.0 / (60.0 + s.sparse_rank), 0.0)) AS rrf_score
            FROM dense d
            FULL OUTER JOIN sparse s ON d.id = s.id
        )
        SELECT
            source,
            chunk_id,
            heading,
            text,
            dense_cos,
            ts_rank,
            rrf_score
        FROM fused
        ORDER BY rrf_score DESC
        LIMIT $4;
        """

        pool = await self._get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(query_sql, query_vec, self.tenant_id, query, k)

        if not rows:
            return []

        results: list[Passage] = []
        for r in rows:
            dense_cos = float(r["dense_cos"])
            ts_rank = float(r["ts_rank"])
            score = fuse_score(dense_cos, ts_rank, sparse_half=0.1)
            results.append(
                Passage(
                    source=r["source"],
                    chunk_id=r["chunk_id"],
                    heading=r["heading"],
                    text=r["text"],
                    score=score,
                )
            )

        return results
