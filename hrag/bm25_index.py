"""BM25 sparse index over child chunks (paper §3.2: 'keyword-based lexical
matching' component of hybrid search).

We persist the BM25 corpus alongside the vector store so the index can be
rebuilt at startup without re-ingesting documents.
"""

from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import List

from rank_bm25 import BM25Okapi

from .config import settings


_TOKEN = re.compile(r"[A-Za-z0-9]+")


def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN.findall(text)]


class BM25Index:
    def __init__(self, path: Path | str | None = None):
        self.path = Path(path or settings.bm25_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._bm25: BM25Okapi | None = None
        self._child_ids: List[str] = []
        self._parent_ids: List[str] = []
        self._texts: List[str] = []
        if self.path.exists():
            self._load()

    def build(self, children: List[dict]) -> None:
        """Build (or rebuild) the BM25 index from a list of child dicts."""
        self._child_ids = [c["child_id"] for c in children]
        self._parent_ids = [c["parent_id"] for c in children]
        self._texts = [c["text"] for c in children]
        tokenized = [_tokenize(t) for t in self._texts]
        self._bm25 = BM25Okapi(tokenized) if tokenized else None
        self._save()

    def search(self, query: str, top_k: int) -> List[dict]:
        if self._bm25 is None or not self._child_ids:
            return []
        scores = self._bm25.get_scores(_tokenize(query))
        if not len(scores):
            return []
        # argpartition then sort the head -> O(n + k log k).
        import numpy as np
        k = min(top_k, len(scores))
        top_idx = np.argpartition(-scores, k - 1)[:k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        return [
            {
                "child_id": self._child_ids[i],
                "parent_id": self._parent_ids[i],
                "text": self._texts[i],
                "sparse_score": float(scores[i]),
            }
            for i in top_idx
        ]

    def _save(self) -> None:
        with open(self.path, "wb") as f:
            pickle.dump(
                {
                    "child_ids": self._child_ids,
                    "parent_ids": self._parent_ids,
                    "texts": self._texts,
                    "bm25": self._bm25,
                },
                f,
            )

    def _load(self) -> None:
        with open(self.path, "rb") as f:
            data = pickle.load(f)
        self._child_ids = data["child_ids"]
        self._parent_ids = data["parent_ids"]
        self._texts = data["texts"]
        self._bm25 = data["bm25"]
