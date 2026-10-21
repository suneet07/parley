"""Markdown chunking with heading attribution and boundary-aware splitting."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    source: str
    chunk_id: str
    heading: str | None
    text: str


def _split_section(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    start = 0
    text_len = len(text)

    while start < text_len:
        if text_len - start <= max_chars:
            remaining = text[start:].strip()
            if remaining:
                chunks.append(remaining)
            break

        end = start + max_chars
        sub = text[start:end]

        # 1. Paragraph boundary: \n\s*\n
        para_idx = -1
        for m in re.finditer(r"\n\s*\n", sub):
            if m.start() >= max(overlap_chars, max_chars // 3):
                para_idx = m.end()

        # 2. Sentence boundary: [.!?] followed by whitespace or at end of sub
        sent_idx = -1
        if para_idx == -1:
            for m in re.finditer(r"[.!?](?:\s+|$)", sub):
                if m.end() >= max(overlap_chars, max_chars // 3):
                    sent_idx = m.end()

        # 3. Word boundary: whitespace
        word_idx = -1
        if para_idx == -1 and sent_idx == -1:
            for m in re.finditer(r"\s+", sub):
                word_idx = m.start()

        if para_idx != -1:
            cut = start + para_idx
        elif sent_idx != -1:
            cut = start + sent_idx
        elif word_idx != -1:
            cut = start + word_idx
        else:
            # Word exceeds max_chars: find the next whitespace after end to avoid mid-word cut
            m = re.search(r"\s+", text[end:])
            cut = end + m.start() if m else text_len

        chunk_text = text[start:cut].strip()
        if chunk_text:
            chunks.append(chunk_text)

        if cut >= text_len:
            break

        target_start = cut - overlap_chars
        if target_start <= start:
            target_start = start + 1

        if target_start < cut:
            if (
                target_start > 0
                and not text[target_start - 1].isspace()
                and not text[target_start].isspace()
            ):
                m = re.search(r"\s+", text[target_start:cut])
                next_start = target_start + m.end() if m else cut
            else:
                next_start = target_start
                while next_start < cut and text[next_start].isspace():
                    next_start += 1
        else:
            next_start = cut

        if next_start <= start:
            next_start = cut

        start = next_start

    return chunks


def chunk_markdown(
    source: str,
    text: str,
    max_chars: int = 600,
    overlap_chars: int = 80,
) -> list[Chunk]:
    """Split markdown text into chunks.

    Rules:
    - Split on markdown headings (#..###).
    - Each chunk carries its nearest heading (without #).
    - Sections longer than max_chars split on paragraph, then sentence boundaries,
      never mid-word, with overlap.
    - Prepend heading as 'Heading: text' to make it searchable.
    - Drop empty chunks.
    - chunk_id = f"{source}#{n}" with n from 0 in document order.
    """
    sections: list[tuple[str | None, list[str]]] = []
    current_heading: str | None = None
    current_lines: list[str] = []

    for line in text.splitlines():
        m = re.match(r"^\s{0,3}(#{1,3})\s+(.+)$", line)
        if m:
            if current_lines or current_heading is not None:
                sections.append((current_heading, current_lines))
            current_heading = m.group(2).strip().rstrip("#").strip().rstrip(":")
            current_lines = []
        else:
            current_lines.append(line)

    if current_lines or current_heading is not None:
        sections.append((current_heading, current_lines))

    chunks: list[Chunk] = []
    n = 0

    for heading, lines in sections:
        body = "\n".join(lines).strip()
        if not body:
            continue

        sub_chunks = _split_section(body, max_chars=max_chars, overlap_chars=overlap_chars)
        for sub in sub_chunks:
            sub_clean = sub.strip()
            if not sub_clean:
                continue

            chunk_text = f"{heading}: {sub_clean}" if heading else sub_clean
            chunks.append(
                Chunk(
                    source=source,
                    chunk_id=f"{source}#{n}",
                    heading=heading,
                    text=chunk_text,
                )
            )
            n += 1

    return chunks
