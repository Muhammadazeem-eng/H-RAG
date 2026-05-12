"""Full H-RAG query-time pipeline (paper §3.2 - §3.4, Figure 1 right half).

Wires together all components into a single `answer()` call:

  question + history
        |
        v
  query_rewriter  ->  rewritten query
        |
        v
  HybridRetriever (dense + sparse fusion)         [paper §3.2]
        |
        v
  embedding-based rescoring                        [paper §3.2]
        |
        v
  parent aggregation (max child score)             [paper §3.3]
        |
        v
  generator (instruction-tuned LLM)                [paper §3.4]
        |
        v
  grounded answer
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

from .aggregator import aggregate_child_first, aggregate_parent_first
from .bm25_index import BM25Index
from .config import settings
from .doc_store import ParentDocStore
from .generator import generate_answer
from .query_rewriter import rewrite_query
from .rescorer import rescore_children
from .retriever import HybridRetriever
from .vector_store import ChildVectorStore


@dataclass
class HRAGResult:
    answer: str
    rewritten_query: str
    parent_ids: List[str]
    parents: List[dict] = field(default_factory=list)
    debug: dict = field(default_factory=dict)


class HRAGPipeline:
    def __init__(
        self,
        *,
        vector_store: ChildVectorStore | None = None,
        doc_store: ParentDocStore | None = None,
        bm25: BM25Index | None = None,
        retriever: HybridRetriever | None = None,
    ):
        self.vector_store = vector_store or ChildVectorStore()
        self.doc_store = doc_store or ParentDocStore()
        self.bm25 = bm25 or BM25Index()
        self.retriever = retriever or HybridRetriever(self.vector_store, self.bm25)

    def retrieve(
        self,
        question: str,
        history: List[Tuple[str, str]] | None = None,
    ) -> tuple[str, List[dict], List[dict]]:
        """Run rewrite + hybrid + rescore + aggregate. Returns (q', children, parents)."""
        history = history or []
        rewritten = rewrite_query(question, history)

        hits = self.retriever.search(rewritten, top_k=settings.top_k_children)
        hits = rescore_children(rewritten, hits)

        if settings.rank_parents:
            parents_ranked = aggregate_child_first(hits, top_n=settings.top_n_parents)
        else:
            parents_ranked = aggregate_parent_first(
                rewritten, hits, self.doc_store, top_n=settings.top_n_parents
            )

        return rewritten, hits, parents_ranked

    def answer(
        self,
        question: str,
        history: List[Tuple[str, str]] | None = None,
    ) -> HRAGResult:
        history = history or []
        rewritten, hits, parents_ranked = self.retrieve(question, history)

        parent_ids = [p["parent_id"] for p in parents_ranked]
        parent_docs = self.doc_store.get_many(parent_ids)
        passages = [p.text for p in parent_docs]

        answer = generate_answer(rewritten, passages, history)

        return HRAGResult(
            answer=answer,
            rewritten_query=rewritten,
            parent_ids=parent_ids,
            parents=[
                {
                    "parent_id": p.parent_id,
                    "source": p.source,
                    "score": next(
                        (r["parent_score"] for r in parents_ranked if r["parent_id"] == p.parent_id),
                        0.0,
                    ),
                }
                for p in parent_docs
            ],
            debug={
                "n_children_retrieved": len(hits),
                "alpha": settings.hybrid_alpha,
                "top_k_children": settings.top_k_children,
                "top_n_parents": settings.top_n_parents,
                "rank_parents": settings.rank_parents,
            },
        )
