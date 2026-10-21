"""Knowledge retrieval contracts."""

from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel


class Passage(BaseModel):
    source: str  # file name, e.g. "returns-policy.md"
    chunk_id: str  # "<source>#<n>"
    heading: str | None = None  # nearest markdown heading
    text: str
    score: float  # fused relevance in [0, 1]; higher is better


class Retriever(Protocol):
    async def search(self, query: str, k: int = 4) -> list[Passage]: ...
