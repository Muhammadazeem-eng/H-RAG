"""Document ingestion pipeline (paper §3.1, Figure 1 left half).

For every input document:
  1. Build hierarchical parent/child representation (chunking.py).
  2. Persist parents to the doc store (doc_store.py).
  3. Embed and persist children in the vector DB (vector_store.py).
  4. Rebuild the BM25 sparse index (bm25_index.py).

After ingestion both retrieval streams are ready for hybrid search.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

from tqdm import tqdm

from .bm25_index import BM25Index
from .chunking import chunk_document
from .doc_store import ParentDocStore
from .vector_store import ChildVectorStore


def ingest_texts(
    texts: Iterable[str],
    sources: Iterable[str] | None = None,
    *,
    vector_store: ChildVectorStore | None = None,
    doc_store: ParentDocStore | None = None,
    bm25: BM25Index | None = None,
    show_progress: bool = True,
) -> dict:
    """Ingest raw text documents and rebuild the BM25 index."""
    vector_store = vector_store or ChildVectorStore()
    doc_store = doc_store or ParentDocStore()
    bm25 = bm25 or BM25Index()

    sources = list(sources) if sources else []
    texts = list(texts)
    if sources and len(sources) != len(texts):
        raise ValueError("sources must align with texts when provided")

    total_children = 0
    iterable = tqdm(texts, desc="ingest", disable=not show_progress)
    for i, text in enumerate(iterable):
        src = sources[i] if sources else ""
        parent, children = chunk_document(text, source=src)
        if not children:
            continue
        doc_store.upsert_many([parent])
        vector_store.add(children)
        total_children += len(children)

    # Rebuild BM25 over the full child collection so it stays consistent
    # with the vector store after each ingestion run.
    bm25.build(vector_store.all_children())

    return {
        "documents": len(texts),
        "parents": doc_store.count(),
        "children_added": total_children,
        "children_total": vector_store.count(),
    }


def ingest_directory(
    directory: str | Path,
    *,
    glob: str = "*.txt",
    **kwargs,
) -> dict:
    directory = Path(directory)
    paths: List[Path] = sorted(directory.glob(glob))
    texts = [p.read_text(encoding="utf-8") for p in paths]
    sources = [str(p) for p in paths]
    return ingest_texts(texts, sources, **kwargs)
