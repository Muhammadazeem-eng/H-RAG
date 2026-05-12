"""Embedding-based rescoring (paper §3.2).

The paper does NOT use the reranker as a cross-encoder. Instead it rescores
candidates by cosine similarity between the query embedding and the child
embedding produced by BAAI/bge-reranker-v2-m3, 'enabling efficient
large-scale retrieval while maintaining ranking quality'.
"""

from __future__ import annotations

from typing import List

import numpy as np

from .embeddings import encode_for_rescore


def rescore_children(query: str, hits: List[dict]) -> List[dict]:
    """Replace the hybrid score with embedding cosine sim from the reranker."""
    if not hits:
        return hits
    texts = [query] + [h["text"] for h in hits]
    vecs = encode_for_rescore(texts)
    q = vecs[0]
    children = vecs[1:]
    sims = children @ q  # already normalized -> cosine
    for h, s in zip(hits, sims.tolist()):
        h["rescore"] = float(s)
    hits.sort(key=lambda x: x["rescore"], reverse=True)
    return hits
