"""Hierarchical parent-child chunking (paper §3.1, §3.5).

Each document becomes:
  - 1 PARENT  : the full document text (preserved for generation context).
  - N CHILDREN: overlapping sentence windows (chunk_sentences=3, stride=2).

Children are what we index and retrieve over. Parents are what we hand to
the LLM after parent-level aggregation.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from typing import List

from .config import settings


@dataclass
class ParentDoc:
    parent_id: str
    text: str
    source: str = ""
    metadata: dict = field(default_factory=dict)


@dataclass
class ChildChunk:
    child_id: str
    parent_id: str
    text: str
    chunk_index: int


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")


def split_sentences(text: str) -> List[str]:
    """Rule-based sentence splitter (paper §3.1).

    NLTK's punkt is used if available; otherwise we fall back to a regex
    that splits on terminal punctuation followed by whitespace + uppercase.
    """
    text = text.strip()
    if not text:
        return []
    try:
        import nltk
        try:
            nltk.data.find("tokenizers/punkt_tab")
        except LookupError:
            try:
                nltk.download("punkt_tab", quiet=True)
            except Exception:
                pass
        return [s.strip() for s in nltk.sent_tokenize(text) if s.strip()]
    except Exception:
        return [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]


def make_child_chunks(
    parent_id: str,
    text: str,
    chunk_sentences: int | None = None,
    stride: int | None = None,
) -> List[ChildChunk]:
    """Sliding sentence window: 3-sentence chunks, stride 2 (paper §3.5)."""
    n = chunk_sentences or settings.chunk_sentences
    s = stride or settings.chunk_stride
    sents = split_sentences(text)
    if not sents:
        return []

    chunks: List[ChildChunk] = []
    i, idx = 0, 0
    while i < len(sents):
        window = sents[i : i + n]
        if not window:
            break
        chunks.append(
            ChildChunk(
                child_id=f"{parent_id}::c{idx}",
                parent_id=parent_id,
                text=" ".join(window),
                chunk_index=idx,
            )
        )
        idx += 1
        if i + n >= len(sents):
            break
        i += s
    return chunks


def chunk_document(
    text: str,
    source: str = "",
    metadata: dict | None = None,
) -> tuple[ParentDoc, List[ChildChunk]]:
    """Build a (parent, children) pair from a raw document."""
    parent_id = str(uuid.uuid4())
    parent = ParentDoc(
        parent_id=parent_id,
        text=text,
        source=source,
        metadata=metadata or {},
    )
    children = make_child_chunks(parent_id, text)
    return parent, children
