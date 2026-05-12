"""Child-chunk vector store (paper Figure 1: 'Vector Database (Child Chunks)').

The paper uses Weaviate's hybrid store; we use a local ChromaDB persistent
collection. Sparse matching is handled separately by bm25_index.py to keep
the hybrid score formula explicit and educational.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

import numpy as np

from .chunking import ChildChunk
from .config import settings
from .embeddings import encode_passages, encode_query


class ChildVectorStore:
    def __init__(self, path: Path | str | None = None, collection: str = "hrag_children"):
        import chromadb
        from chromadb.config import Settings as ChromaSettings
        self.path = Path(path or settings.chroma_path)
        self.path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(self.path),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._col = self._client.get_or_create_collection(
            name=collection,
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, chunks: Sequence[ChildChunk]) -> None:
        if not chunks:
            return
        texts = [c.text for c in chunks]
        vectors = encode_passages(texts)
        self._col.add(
            ids=[c.child_id for c in chunks],
            embeddings=vectors.tolist(),
            documents=texts,
            metadatas=[
                {"parent_id": c.parent_id, "chunk_index": c.chunk_index}
                for c in chunks
            ],
        )

    def dense_search(self, query: str, top_k: int) -> List[dict]:
        """Cosine search over child embeddings."""
        qv = encode_query(query)
        res = self._col.query(
            query_embeddings=[qv.tolist()],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        hits: List[dict] = []
        for cid, doc, meta, dist in zip(
            res["ids"][0],
            res["documents"][0],
            res["metadatas"][0],
            res["distances"][0],
        ):
            # Chroma cosine distance = 1 - cosine_similarity -> recover sim.
            score = float(1.0 - dist)
            hits.append(
                {
                    "child_id": cid,
                    "parent_id": meta.get("parent_id"),
                    "text": doc,
                    "dense_score": score,
                }
            )
        return hits

    def all_children(self) -> List[dict]:
        """Return every child (used to rebuild BM25 index after ingestion)."""
        res = self._col.get(include=["documents", "metadatas"])
        out = []
        for cid, doc, meta in zip(res["ids"], res["documents"], res["metadatas"]):
            out.append(
                {
                    "child_id": cid,
                    "parent_id": meta.get("parent_id"),
                    "text": doc,
                }
            )
        return out

    def count(self) -> int:
        return self._col.count()
