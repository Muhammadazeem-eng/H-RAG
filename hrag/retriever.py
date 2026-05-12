"""Hybrid dense-sparse retriever (paper §3.2 + Table 1, alpha=0.7).

  hybrid_score = alpha * dense + (1 - alpha) * sparse

Both score streams are min-max normalized per-query so the weight has the
intended semantic meaning across different score scales.
"""

from __future__ import annotations

from typing import Dict, List

from .bm25_index import BM25Index
from .config import settings
from .vector_store import ChildVectorStore


def _minmax(values: List[float]) -> List[float]:
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi - lo < 1e-12:
        return [0.0 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


class HybridRetriever:
    def __init__(
        self,
        vector_store: ChildVectorStore,
        bm25: BM25Index,
        alpha: float | None = None,
    ):
        self.vector_store = vector_store
        self.bm25 = bm25
        self.alpha = settings.hybrid_alpha if alpha is None else alpha

    def search(self, query: str, top_k: int | None = None) -> List[dict]:
        """Return the top-k child chunks fused across dense + sparse."""
        k = top_k or settings.top_k_children
        # Pull a wider candidate pool from each retriever, then fuse.
        pool = max(k * 2, 50)
        dense_hits = self.vector_store.dense_search(query, top_k=pool)
        sparse_hits = self.bm25.search(query, top_k=pool)

        merged: Dict[str, dict] = {}
        for h in dense_hits:
            merged.setdefault(h["child_id"], {**h})
        for h in sparse_hits:
            cur = merged.setdefault(
                h["child_id"],
                {
                    "child_id": h["child_id"],
                    "parent_id": h["parent_id"],
                    "text": h["text"],
                },
            )
            cur["sparse_score"] = h.get("sparse_score", 0.0)

        ids = list(merged.keys())
        dense_raw = [merged[i].get("dense_score", 0.0) for i in ids]
        sparse_raw = [merged[i].get("sparse_score", 0.0) for i in ids]
        dn = _minmax(dense_raw)
        sn = _minmax(sparse_raw)

        for i, cid in enumerate(ids):
            merged[cid]["dense_norm"] = dn[i]
            merged[cid]["sparse_norm"] = sn[i]
            merged[cid]["hybrid_score"] = self.alpha * dn[i] + (1 - self.alpha) * sn[i]

        ranked = sorted(merged.values(), key=lambda x: x["hybrid_score"], reverse=True)
        return ranked[:k]
