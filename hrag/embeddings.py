"""Embedding models (paper Table 1).

- Indexing/retrieval: BAAI/bge-large-en-v1.5
- Rescoring:          BAAI/bge-reranker-v2-m3 used via embedding cosine
                      (paper §3.2: 'this rescoring step relies solely on
                       embedding similarity', not cross-encoder logits).

Both are loaded lazily as singletons so importing the module is cheap.
"""

from __future__ import annotations

from functools import lru_cache
from typing import List

import numpy as np

from .config import settings


@lru_cache(maxsize=2)
def _load(name: str):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(name)


def encode_query(text: str) -> np.ndarray:
    model = _load(settings.embed_model)
    vec = model.encode([text], normalize_embeddings=True)[0]
    return np.asarray(vec, dtype=np.float32)


def encode_passages(texts: List[str], batch_size: int = 32) -> np.ndarray:
    model = _load(settings.embed_model)
    vecs = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.asarray(vecs, dtype=np.float32)


def encode_for_rescore(texts: List[str]) -> np.ndarray:
    """Embeddings produced by the reranker model, used for cosine rescoring."""
    model = _load(settings.reranker_model)
    vecs = model.encode(
        texts,
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.asarray(vecs, dtype=np.float32)
