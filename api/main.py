"""FastAPI orchestrator (paper Figure 1 top: 'API Orchestrator').

Endpoints:
  POST /ingest        - ingest a list of documents
  POST /ingest_dir    - ingest a local directory of .txt files
  POST /retrieve      - run retrieval only (Task A diagnostic)
  POST /chat          - full multi-turn RAG (Task C)
  GET  /health        - basic liveness + index counts

Sessions are tracked server-side; clients pass a `session_id` so the
orchestrator can pull chat history before query rewriting.
"""

from __future__ import annotations

import uuid
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from hrag.bm25_index import BM25Index
from hrag.config import settings
from hrag.doc_store import ParentDocStore
from hrag.ingest import ingest_directory, ingest_texts
from hrag.pipeline import HRAGPipeline
from hrag.session_store import SessionStore
from hrag.vector_store import ChildVectorStore


app = FastAPI(title="H-RAG", version="0.1.0")

# Shared singletons -- the FastAPI process owns one of each.
_vector_store = ChildVectorStore()
_doc_store = ParentDocStore()
_bm25 = BM25Index()
_sessions = SessionStore()
_pipeline = HRAGPipeline(
    vector_store=_vector_store, doc_store=_doc_store, bm25=_bm25
)


class IngestTextsRequest(BaseModel):
    documents: List[str]
    sources: Optional[List[str]] = None


class IngestDirRequest(BaseModel):
    directory: str
    glob: str = "*.txt"


class RetrieveRequest(BaseModel):
    question: str
    session_id: Optional[str] = None
    top_k_children: Optional[int] = None
    top_n_parents: Optional[int] = None


class ChatRequest(BaseModel):
    question: str
    session_id: Optional[str] = Field(
        default=None,
        description="If omitted, a new session_id is generated and returned.",
    )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "parents": _doc_store.count(),
        "children": _vector_store.count(),
        "config": {
            "embed_model": settings.embed_model,
            "reranker_model": settings.reranker_model,
            "alpha": settings.hybrid_alpha,
            "top_k_children": settings.top_k_children,
            "top_n_parents": settings.top_n_parents,
            "rank_parents": settings.rank_parents,
        },
    }


@app.post("/ingest")
def ingest(req: IngestTextsRequest):
    stats = ingest_texts(
        req.documents,
        req.sources,
        vector_store=_vector_store,
        doc_store=_doc_store,
        bm25=_bm25,
        show_progress=False,
    )
    return stats


@app.post("/ingest_dir")
def ingest_dir(req: IngestDirRequest):
    stats = ingest_directory(
        req.directory,
        glob=req.glob,
        vector_store=_vector_store,
        doc_store=_doc_store,
        bm25=_bm25,
        show_progress=False,
    )
    return stats


@app.post("/retrieve")
def retrieve(req: RetrieveRequest):
    history = _sessions.recent_turns(req.session_id) if req.session_id else []
    rewritten, hits, parents = _pipeline.retrieve(req.question, history)
    return {
        "rewritten_query": rewritten,
        "n_children": len(hits),
        "parents": parents,
    }


@app.post("/chat")
def chat(req: ChatRequest):
    if not _vector_store.count():
        raise HTTPException(
            status_code=400, detail="No documents indexed yet. Call /ingest first."
        )

    session_id = req.session_id or str(uuid.uuid4())
    history = _sessions.recent_turns(session_id)

    result = _pipeline.answer(req.question, history)
    _sessions.append_turn(session_id, req.question, result.answer)

    return {
        "session_id": session_id,
        "answer": result.answer,
        "rewritten_query": result.rewritten_query,
        "parents": result.parents,
        "debug": result.debug,
    }


def run() -> None:
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=False,
    )


if __name__ == "__main__":
    run()
