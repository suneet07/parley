"""Pure-Python BM25 implementation with no external dependencies beyond numpy."""

from __future__ import annotations

import math
import re

import numpy as np

# Small English stopword list (~40 words)
STOPWORDS: frozenset[str] = frozenset({
    "a", "about", "all", "an", "and", "are", "as", "at", "be", "been",
    "but", "by", "can", "do", "for", "from", "had", "has", "have", "he",
    "her", "his", "how", "i", "if", "in", "is", "it", "not", "of",
    "on", "or", "so", "that", "the", "their", "this", "to", "was", "we",
    "what", "when", "where", "which", "will", "with", "you",
})


class BM25:
    """Okapi BM25 implementation for passage retrieval."""

    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.doc_count = len(docs)
        self.doc_tokens: list[list[str]] = [self.tokenize(d) for d in docs]
        self.doc_lens = np.array([len(dt) for dt in self.doc_tokens], dtype=np.float32)
        self.avgdl = float(np.mean(self.doc_lens)) if self.doc_count > 0 else 0.0

        self.df: dict[str, int] = {}
        self.doc_tfs: list[dict[str, int]] = []
        for dt in self.doc_tokens:
            tf: dict[str, int] = {}
            for tok in dt:
                tf[tok] = tf.get(tok, 0) + 1
            self.doc_tfs.append(tf)
            for tok in tf:
                self.df[tok] = self.df.get(tok, 0) + 1

        self.idf: dict[str, float] = {}
        for term, freq in self.df.items():
            self.idf[term] = math.log(1.0 + (self.doc_count - freq + 0.5) / (freq + 0.5))

    @staticmethod
    def tokenize(text: str) -> list[str]:
        tokens = re.findall(r"[a-z0-9₹]+", text.lower())
        result: list[str] = []
        for tok in tokens:
            if tok in STOPWORDS:
                continue
            if len(tok) > 3 and tok.endswith("s"):
                tok = tok[:-1]
            result.append(tok)
        return result

    def scores(self, query: str) -> np.ndarray:
        if self.doc_count == 0:
            return np.zeros(0, dtype=np.float32)

        q_tokens = self.tokenize(query)
        if not q_tokens:
            return np.zeros(self.doc_count, dtype=np.float32)

        scores = np.zeros(self.doc_count, dtype=np.float32)
        for tok in q_tokens:
            if tok not in self.idf:
                continue
            idf = self.idf[tok]
            for i in range(self.doc_count):
                tf = self.doc_tfs[i].get(tok, 0)
                if tf == 0:
                    continue
                if self.avgdl > 0:
                    den = tf + self.k1 * (1.0 - self.b + self.b * (self.doc_lens[i] / self.avgdl))
                else:
                    den = tf + self.k1
                scores[i] += idf * (tf * (self.k1 + 1.0)) / den

        return scores
