"""Knowledge ingestion and retriever factory."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from parley.core.knowledge.base import Retriever

if TYPE_CHECKING:
    from parley.config import BusinessConfig


async def build_retriever(
    business: BusinessConfig,
    database_url: str | None = None,
    tenant_id: str = "default",
) -> Retriever:
    """Build a retriever for the given business.

    postgresql URL -> PgVectorRetriever (ensure_schema + upsert docs);
    anything else (sqlite or None) -> InMemoryRetriever.from_business.
    """
    if database_url and (
        database_url.startswith("postgresql://")
        or database_url.startswith("postgresql+")
        or database_url.startswith("postgres://")
    ):
        from parley.core.knowledge.chunking import Chunk, chunk_markdown
        from parley.core.knowledge.pgvector import PgVectorRetriever

        retriever = PgVectorRetriever(dsn=database_url, tenant_id=tenant_id)
        await retriever.ensure_schema()

        chunks: list[Chunk] = []
        for rel_path in business.knowledge:
            p = business.resolve(rel_path)
            if not p.exists() or not p.is_file():
                raise FileNotFoundError(f"Knowledge file not found: {p}")
            text = p.read_text(encoding="utf-8")
            chunks.extend(chunk_markdown(source=Path(rel_path).name, text=text))

        await retriever.upsert(chunks)
        return retriever

    from parley.core.knowledge.memory import InMemoryRetriever

    return await InMemoryRetriever.from_business(business=business)


async def main() -> None:
    """CLI entry for `parley ingest`: load settings + business, build retriever,

    print number of chunks per source and a sample search for 'return policy' with scores.
    """
    from parley.config import get_settings, load_business
    from parley.core.knowledge.chunking import chunk_markdown

    settings = get_settings()
    business = load_business(settings.business_config)

    retriever = await build_retriever(
        business=business,
        database_url=settings.database_url,
        tenant_id=settings.tenant_id,
    )

    chunk_counts: dict[str, int] = {}
    for rel_path in business.knowledge:
        p = business.resolve(rel_path)
        if p.exists() and p.is_file():
            chunks = chunk_markdown(source=Path(rel_path).name, text=p.read_text(encoding="utf-8"))
            chunk_counts[Path(rel_path).name] = len(chunks)

    total_chunks = sum(chunk_counts.values())
    print(f"Ingested {total_chunks} chunks from {len(chunk_counts)} sources:")
    for src, count in sorted(chunk_counts.items()):
        print(f"  {src}: {count} chunks")

    results = await retriever.search("return policy", k=4)
    print("\nSample search for 'return policy':")
    for r in results:
        preview = r.text.replace("\n", " ")
        if len(preview) > 80:
            preview = preview[:77] + "..."
        print(f"  [{r.score:.3f}] {r.chunk_id} ({r.heading}): {preview}")
