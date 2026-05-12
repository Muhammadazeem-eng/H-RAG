"""Parent-level aggregation (paper §3.3).

Each parent's relevance score is the MAX score of its associated children:
    parent_score(p) = max{ score(c) : c.parent_id == p.id }

This is the 'max child score' strategy in Table 1.

We support both ranking strategies described in §3.3:
  1) child-first: rank child chunks first, then de-duplicate parents
     keeping the best child score per parent (rank_parents=True).
  2) parent-first: identify unique parent candidates earlier and rescore
     them with parent-level content (rank_parents=False).
"""

from __future__ import annotations

from typing import List

import numpy as np

from .config import settings
from .doc_store import ParentDocStore
from .embeddings import encode_for_rescore


def aggregate_child_first(hits: List[dict], top_n: int | None = None) -> List[dict]:
    """Strategy 1: max-pool child scores into unique parents."""
    n = top_n or settings.top_n_parents
    best: dict[str, dict] = {}
    for h in hits:
        pid = h["parent_id"]
        score = h.get("rescore", h.get("hybrid_score", 0.0))
        cur = best.get(pid)
        if cur is None or score > cur["parent_score"]:
            best[pid] = {
                "parent_id": pid,
                "parent_score": score,
                "best_child_id": h["child_id"],
                "best_child_text": h["text"],
            }
    ranked = sorted(best.values(), key=lambda x: x["parent_score"], reverse=True)
    return ranked[:n]


def aggregate_parent_first(
    query: str,
    hits: List[dict],
    doc_store: ParentDocStore,
    top_n: int | None = None,
) -> List[dict]:
    """Strategy 2: dedupe parents, then rescore using parent-level content."""
    n = top_n or settings.top_n_parents
    parent_ids: List[str] = []
    seen = set()
    for h in hits:
        if h["parent_id"] not in seen:
            seen.add(h["parent_id"])
            parent_ids.append(h["parent_id"])

    parents = doc_store.get_many(parent_ids)
    if not parents:
        return []

    vecs = encode_for_rescore([query] + [p.text for p in parents])
    q, pmat = vecs[0], vecs[1:]
    sims = pmat @ q

    ranked = [
        {"parent_id": p.parent_id, "parent_score": float(s)}
        for p, s in zip(parents, sims.tolist())
    ]
    ranked.sort(key=lambda x: x["parent_score"], reverse=True)
    return ranked[:n]
