"""Absolute relevance score shared by both retrieval backends.

RRF decides the ranking; this score decides confidence. It must be absolute (not relative to the
best hit), otherwise an off-topic query that shares one word with the docs looks confident.
Calibrated on bge-small-en-v1.5: off-topic queries land at cosine 0.47-0.61, on-topic 0.58-0.83.
The brain treats score < 0.35 as "low confidence".
"""

import numpy as np

DENSE_FLOOR = 0.45
DENSE_SPAN = 0.35


def fuse_score(dense_cos: float, sparse: float, sparse_half: float) -> float:
    """sparse_half: the sparse value that counts as half-saturated (BM25 ~3, ts_rank_cd ~0.1)."""
    dense_part = 0.7 * float(np.clip((dense_cos - DENSE_FLOOR) / DENSE_SPAN, 0.0, 1.0))
    sparse_part = 0.3 * (sparse / (sparse + sparse_half) if sparse > 0 else 0.0)
    return float(np.clip(dense_part + sparse_part, 0.0, 1.0))
