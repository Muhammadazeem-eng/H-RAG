# H-RAG: Hierarchical Parent–Child Retrieval for Multi-Turn RAG

An educational, modular re-implementation of **H-RAG** (Elchafei et al., SemEval-2026 Task 8, MTRAGEval) — a hierarchical parent–child retrieval pipeline for multi-turn conversational RAG.

> Reference: *H-RAG at SemEval-2026 Task 8: Hierarchical Parent–Child Retrieval for Multi-Turn RAG Conversations* — [arXiv:2605.00631](https://arxiv.org/abs/2605.00631)

The original paper uses **Weaviate + GPT-5**. This repo swaps in **ChromaDB + OpenAI Chat Completions** so anyone can clone and run it locally without an external vector DB. Every other component (chunking, hybrid retrieval, embedding-based rescoring, parent aggregation, prompts) is implemented faithfully.

---

## 1. The paper in one diagram

```
   user query ──► API Orchestrator ──► (pull session history)
                       │
                       ▼
              context-dependent?  ── yes ──► LLM query rewriter
                       │
                       ▼  rewritten query
   ┌──────────────────────────────────────────────────────────┐
   │  Hybrid dense + sparse retrieval over CHILD chunks       │
   │     score = α·dense + (1-α)·sparse        (α = 0.7)      │
   ├──────────────────────────────────────────────────────────┤
   │  Embedding-based rescore (BAAI/bge-reranker-v2-m3)       │
   ├──────────────────────────────────────────────────────────┤
   │  Parent aggregation: parent_score = max(child_scores)    │
   └──────────────────────────────────────────────────────────┘
                       │  top-n parent documents
                       ▼
                 LLM answer generation
```

---

## 2. File map (each file = one paper component)

| File | Paper section | What it does |
|------|---------------|--------------|
| [hrag/config.py](hrag/config.py) | Table 1 | All hyperparameters (α, k, n, model names, chunking). |
| [hrag/chunking.py](hrag/chunking.py) | §3.1, §3.5 | Sentence splitter + sliding 3-sentence / stride-2 child chunks. |
| [hrag/embeddings.py](hrag/embeddings.py) | Table 1 | Wraps BAAI/bge-large-en-v1.5 (indexing) + bge-reranker-v2-m3 (rescore). |
| [hrag/doc_store.py](hrag/doc_store.py) | Fig. 1 "Docs Store" | SQLite key-value store for parent documents. |
| [hrag/vector_store.py](hrag/vector_store.py) | Fig. 1 "Vector DB" | ChromaDB persistent collection for child chunks. |
| [hrag/bm25_index.py](hrag/bm25_index.py) | §3.2 sparse | BM25Okapi over the same child corpus. |
| [hrag/retriever.py](hrag/retriever.py) | §3.2 | Hybrid `score = α·dense + (1-α)·sparse` with per-query min-max normalization. |
| [hrag/rescorer.py](hrag/rescorer.py) | §3.2 | Cosine rescore using bge-reranker-v2-m3 **embeddings** (not cross-encoder). |
| [hrag/aggregator.py](hrag/aggregator.py) | §3.3 | Max-pool child scores into parents; supports both ranking strategies. |
| [hrag/query_rewriter.py](hrag/query_rewriter.py) | §3.2 | LLM rewriter; returns the question unchanged when it's already standalone. |
| [hrag/generator.py](hrag/generator.py) | §3.4 | Instruction-tuned generation conditioned on parent passages. |
| [hrag/prompts.py](hrag/prompts.py) | Appendix A | Verbatim Rewrite + Generation prompt templates. |
| [hrag/llm_client.py](hrag/llm_client.py) | — | Shared OpenAI client used by rewriter + generator. |
| [hrag/session_store.py](hrag/session_store.py) | Fig. 1 "Chat History DB" | SQLite session history (3-turn window). |
| [hrag/ingest.py](hrag/ingest.py) | §3.1 | Ingestion orchestration (chunk → embed → store → rebuild BM25). |
| [hrag/pipeline.py](hrag/pipeline.py) | §3.2–§3.4 | End-to-end: rewrite → retrieve → rescore → aggregate → generate. |
| [api/main.py](api/main.py) | Fig. 1 "API Orchestrator" | FastAPI: `/ingest`, `/retrieve`, `/chat`, `/health`. |
| [demo.py](demo.py) | — | 3-turn conversation demo on the sample corpus. |

---

## 3. Setup

### 3.1 Create a virtual environment

```powershell
# Windows / PowerShell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

```bash
# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3.2 Install dependencies

```bash
pip install -r requirements.txt
```

### 3.3 Configure environment

```bash
cp .env.example .env
# edit .env and set OPENAI_API_KEY
```

Models from HuggingFace (`BAAI/bge-large-en-v1.5`, `BAAI/bge-reranker-v2-m3`) download on first use (~2–3 GB total).

---

## 4. Run the demo

```bash
python demo.py
```

Output for each turn includes the rewritten query, the top-N retrieved parents with scores, and the generated answer.

---

## 5. Run the API

```bash
python -m api.main
# or: uvicorn api.main:app --reload
```

| Endpoint | Body | Purpose |
|----------|------|---------|
| `POST /ingest` | `{"documents": ["..."], "sources": ["..."]}` | Ingest raw text documents. |
| `POST /ingest_dir` | `{"directory": "./data/sample"}` | Ingest a folder of `.txt` files. |
| `POST /retrieve` | `{"question": "...", "session_id": "..."}` | Retrieval only (Task A). |
| `POST /chat` | `{"question": "...", "session_id": "..."}` | Full RAG turn (Task C). |
| `GET /health` | — | Index counts + config. |

Quick test once the server is up:

```bash
curl -X POST http://localhost:8000/ingest_dir \
     -H "Content-Type: application/json" \
     -d '{"directory": "./data/sample"}'

curl -X POST http://localhost:8000/chat \
     -H "Content-Type: application/json" \
     -d '{"question": "What problem does H-RAG solve?"}'
```

---

## 6. Tuning knobs (paper Table 2 ablation)

The ablation in Table 2 of the paper shows ranking strategy dominates and α is relatively insensitive. The defaults below match the paper's submission row:

| Setting | Default | Notes |
|---------|---------|-------|
| `HRAG_HYBRID_ALPHA` | `0.7` | Best at 0.7; differences <0.003 nDCG@5 across {0.5, 0.7, 0.9}. |
| `HRAG_TOP_K_CHILDREN` | `50` | Larger pools (k=50) add ranking noise; k=30 peaks. |
| `HRAG_TOP_N_PARENTS` | `5` | Parents handed to the generator. |
| `HRAG_RANK_PARENTS` | `true` | +0.0197 nDCG@5 vs. parent-first rescoring. |
| `HRAG_CHUNK_SENTENCES` / `STRIDE` | `3 / 2` | Sentence-window chunking. |

---

## 7. Differences vs. the original submission

| Aspect | Paper | This repo |
|--------|-------|-----------|
| Vector store | Weaviate hybrid | ChromaDB + BM25Okapi |
| LLM | GPT-5 | OpenAI Chat Completions (`gpt-4o` default) |
| Hybrid score | Weaviate-internal | Explicit α·dense + (1-α)·sparse (min-max normalized) |
| Reranker usage | Embedding cosine (§3.2) | Same — uses `sentence-transformers` on `BAAI/bge-reranker-v2-m3` |
| Everything else | — | Faithful to §3 and Appendix A |

---

## 8. Project layout

```
H-RAG/
├── hrag/                  # library modules (one paper component per file)
├── api/main.py            # FastAPI orchestrator
├── data/sample/           # 4-doc demo corpus
├── demo.py                # 3-turn conversation demo
├── requirements.txt
├── .env.example
└── README.md
```
