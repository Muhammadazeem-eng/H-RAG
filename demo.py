"""End-to-end demo for H-RAG.

Run after installing requirements and setting OPENAI_API_KEY:

    python demo.py

This script:
  1. Ingests data/sample/*.txt into the parent doc store + child vector store.
  2. Runs a 3-turn conversation through the full H-RAG pipeline.
  3. Prints each rewritten query, retrieved parents, and the generated answer.

The conversation includes a context-dependent follow-up so you can see the
query rewriter in action.
"""

from __future__ import annotations

from pathlib import Path

from hrag.ingest import ingest_directory
from hrag.pipeline import HRAGPipeline


SAMPLE_DIR = Path(__file__).parent / "data" / "sample"


def banner(text: str) -> None:
    bar = "=" * len(text)
    print(f"\n{bar}\n{text}\n{bar}")


def main() -> None:
    banner("1. Ingesting sample corpus")
    stats = ingest_directory(SAMPLE_DIR)
    print(stats)

    banner("2. Building pipeline")
    pipeline = HRAGPipeline()

    history: list[tuple[str, str]] = []
    turns = [
        "What problem is H-RAG designed to solve?",
        "How does it chunk documents?",                       # follow-up: 'it' -> H-RAG
        "What were its final scores on Task A and Task C?",   # follow-up: 'its' -> H-RAG
    ]

    for turn_idx, question in enumerate(turns, start=1):
        banner(f"Turn {turn_idx}: {question}")
        result = pipeline.answer(question, history=history)
        print(f"[rewritten] {result.rewritten_query}")
        print("[parents]")
        for p in result.parents:
            src = p.get("source") or p["parent_id"][:8]
            print(f"  - {src}  score={p['score']:.4f}")
        print(f"[answer]\n{result.answer}")
        history.append((question, result.answer))


if __name__ == "__main__":
    main()
