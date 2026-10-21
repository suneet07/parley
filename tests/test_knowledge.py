"""Tests for knowledge ingestion, chunking, and hybrid retrieval."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from parley.config import load_business
from parley.core.knowledge.bm25 import BM25
from parley.core.knowledge.chunking import Chunk, chunk_markdown
from parley.core.knowledge.embeddings import Embedder, HashEmbedder
from parley.core.knowledge.memory import InMemoryRetriever


def test_chunk_markdown() -> None:
    # 1. Heading attribution, sequential IDs, dropping empty headings
    sample_md = """\
# Welcome to Kettle & Co
We craft premium electric kettles and accessories.

# Empty Section
## Another Empty

## Returns Policy
You can return any undamaged item within 30 days of delivery.
Original packaging and proof of purchase are required.

### Damaged Items
Report damaged items within 48 hours for immediate replacement.

## Shipping Policy
Standard domestic shipping takes 3-5 business days across India.
"""
    chunks = chunk_markdown("test.md", sample_md)
    assert len(chunks) == 4

    # Heading attribution
    assert chunks[0].heading == "Welcome to Kettle & Co"
    assert chunks[1].heading == "Returns Policy"
    assert chunks[2].heading == "Damaged Items"
    assert chunks[3].heading == "Shipping Policy"

    # Sequential IDs
    assert [c.chunk_id for c in chunks] == [f"test.md#{i}" for i in range(4)]

    # Heading prepended to text
    assert chunks[0].text.startswith("Welcome to Kettle & Co: ")
    assert chunks[1].text.startswith("Returns Policy: ")

    # 2. Long section split without mid-word cuts, overlap present, max_chars respected
    sentences = [
        f"This is sentence number {i:02d} discussing warranty conditions and return procedures."
        for i in range(25)
    ]
    long_body = " ".join(sentences)
    long_md = f"## Detailed Policy\n{long_body}"
    max_chars = 300
    overlap_chars = 60

    long_chunks = chunk_markdown(
        "long.md",
        long_md,
        max_chars=max_chars,
        overlap_chars=overlap_chars,
    )
    assert len(long_chunks) > 1

    # Sequential IDs
    assert [c.chunk_id for c in long_chunks] == [f"long.md#{i}" for i in range(len(long_chunks))]
    # All inherit the heading
    assert all(c.heading == "Detailed Policy" for c in long_chunks)

    prefix = "Detailed Policy: "
    for c in long_chunks:
        assert c.text.startswith(prefix)
        body = c.text.removeprefix(prefix)
        # max_chars respected (allowing heading prefix)
        assert len(body) <= max_chars
        # No mid-word cuts at start or end
        assert not body[0].isspace()
        assert not body[-1].isspace()
        # Words must not be sliced
        first_word = body.split()[0]
        assert first_word in long_body

    # Overlap present between consecutive chunks
    for i in range(len(long_chunks) - 1):
        body_curr = long_chunks[i].text.removeprefix(prefix)
        body_next = long_chunks[i + 1].text.removeprefix(prefix)
        words_curr = body_curr.split()
        words_next = body_next.split()
        overlap = set(words_curr[-6:]).intersection(set(words_next[:8]))
        assert len(overlap) > 0


def test_bm25() -> None:
    docs = [
        "Our standard refund window is 30 days from purchase for all eligible items.",
        "A full refund is provided to your original payment method within a few days.",
        "Please look through the store window to see our product line during open hours.",
    ]
    bm25 = BM25(docs)

    # Query 'refund window' ranks doc containing both terms first
    scores = bm25.scores("refund window")
    assert scores[0] > scores[1]
    assert scores[0] > scores[2]

    # Stopwords ignored
    scores_with_stops = bm25.scores("what is the refund window and how")
    assert scores_with_stops[0] > scores_with_stops[1]
    assert scores_with_stops[0] > scores_with_stops[2]

    # Numbers like '30' kept
    scores_num = bm25.scores("30")
    assert scores_num[0] > 0.0
    assert scores_num[1] == 0.0
    assert scores_num[2] == 0.0


@pytest.mark.asyncio
async def test_in_memory_retriever_hash() -> None:
    doc_text = """\
## Returns Policy
You can return any undamaged item within 30 days of delivery for a full refund.

## Shipping Times
Standard domestic shipping takes 3 to 5 business days across India.

## Payment Methods
We accept UPI, credit cards, debit cards, and net banking with zero fees.

## Warranty Details
All electric kettles come with a 2-year manufacturer replacement warranty.
"""
    chunks = chunk_markdown("faq.md", doc_text)
    embedder = HashEmbedder()
    retriever = InMemoryRetriever(chunks=chunks, embedder=embedder)

    queries = [
        ("how many days do I have to return an item", "Returns Policy"),
        ("how fast will my order be shipped", "Shipping Times"),
        ("what is the warranty on electric kettles", "Warranty Details"),
    ]

    for q, expected_heading in queries:
        results = await retriever.search(q, k=4)
        assert len(results) > 0
        assert results[0].heading == expected_heading
        # Scores in [0, 1]
        assert all(0.0 <= p.score <= 1.0 for p in results)
        # Scores sorted descending
        assert all(results[i].score >= results[i + 1].score for i in range(len(results) - 1))

    # Empty query -> []
    assert await retriever.search("") == []
    assert await retriever.search("   ") == []


@pytest.mark.asyncio
async def test_vector_cache(tmp_path: Path) -> None:
    doc_path = tmp_path / "policy.md"
    doc_path.write_text("## Returns\nItems can be returned within 30 days.", encoding="utf-8")

    yaml_path = tmp_path / "business.yaml"
    yaml_path.write_text(
        """\
business: "Test Kettle Co"
persona: "Support Agent"
knowledge:
  - policy.md
""",
        encoding="utf-8",
    )

    business = load_business(yaml_path)

    class CountingEmbedder:
        def __init__(self, inner: Embedder) -> None:
            self.inner = inner
            self.embed_call_count = 0
            self.dim = inner.dim

        def embed(self, texts: list[str]) -> np.ndarray:
            self.embed_call_count += 1
            return self.inner.embed(texts)

        def embed_query(self, text: str) -> np.ndarray:
            if hasattr(self.inner, "embed_query"):
                return self.inner.embed_query(text)
            return self.embed([text])[0]

    counting = CountingEmbedder(HashEmbedder())
    cache_dir = tmp_path / ".cache"

    r1 = await InMemoryRetriever.from_business(business, embedder=counting, cache_dir=cache_dir)
    assert counting.embed_call_count == 1
    assert len(r1.chunks) == 1

    # Second call should load from cache, not calling embed() again
    r2 = await InMemoryRetriever.from_business(business, embedder=counting, cache_dir=cache_dir)
    assert counting.embed_call_count == 1
    assert len(r2.chunks) == 1


@pytest.mark.asyncio
async def test_real_embedder_thresholds() -> None:
    demo_dir = Path("examples/kettle-and-co/docs")
    demo_files = list(demo_dir.glob("*.md")) if demo_dir.exists() else []
    if not demo_files:
        pytest.skip("Demo docs examples/kettle-and-co/docs/*.md do not exist")

    try:
        from parley.core.knowledge.embeddings import FastEmbedder

        embedder = FastEmbedder()
        _ = embedder.embed(["warmup"])
    except Exception as exc:
        pytest.skip(f"FastEmbedder cannot be loaded: {exc}")

    chunks: list[Chunk] = []
    for f in sorted(demo_files):
        chunks.extend(chunk_markdown(source=f.name, text=f.read_text(encoding="utf-8")))

    retriever = InMemoryRetriever(chunks=chunks, embedder=embedder)

    # Direct question: score >= 0.5, returns-policy chunk ranked first
    results_direct = await retriever.search("how many days do I have to return an item", k=4)
    assert len(results_direct) > 0
    assert results_direct[0].score >= 0.5
    top_source_heading = (f"{results_direct[0].source} {results_direct[0].heading or ''}").lower()
    assert "return" in top_source_heading

    # Unrelated queries must score < 0.35
    for unrelated in ["what's the weather in Paris", "tell me a joke"]:
        results_unrelated = await retriever.search(unrelated, k=4)
        if results_unrelated:
            assert results_unrelated[0].score < 0.35


def test_pgvector_schema() -> None:
    import parley.core.knowledge.pgvector as pgv

    assert hasattr(pgv, "SCHEMA_SQL")
    assert "vector(384)" in pgv.SCHEMA_SQL
    assert hasattr(pgv, "PgVectorRetriever")
