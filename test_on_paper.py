"""Test the H-RAG pipeline on the H-RAG paper PDF itself.

Steps:
  1. Wipe any previous index (so re-running gives clean results).
  2. Extract text from 2605.00631v1.pdf.
  3. Split into logical sections (by '\n\n' paragraphs of reasonable size)
     so we have several "documents" rather than a single mega-doc.
  4. Ingest -> rebuilds the child vector store + BM25.
  5. Run a 4-turn conversation that exercises retrieval, follow-ups
     (context-dependent queries), and ablation-table recall.
"""

from __future__ import annotations

import io
import re
import shutil
import sys
from pathlib import Path

# Windows consoles default to cp1252 which can't encode the Greek alpha
# (and other glyphs) found in the paper. Force UTF-8 stdout/stderr.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from hrag.config import settings
from hrag.ingest import ingest_texts
from hrag.pipeline import HRAGPipeline


PDF_PATH = Path(__file__).parent / "2605.00631v1.pdf"
MIN_SECTION_CHARS = 600
MAX_SECTION_CHARS = 4000


def extract_pdf_text(pdf_path: Path) -> str:
    """Extract text using PyPDF2 (a transitive dep of chromadb). Falls back
    to pypdf if PyPDF2 is unavailable."""
    try:
        from pypdf import PdfReader
    except ImportError:
        from PyPDF2 import PdfReader  # type: ignore
    reader = PdfReader(str(pdf_path))
    pages = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            pages.append("")
    return "\n\n".join(pages)


def split_into_sections(text: str) -> list[str]:
    """Greedy paragraph packer: merge paragraphs until each chunk is in the
    [MIN, MAX] char range. This gives the parent-doc store several coherent
    documents rather than one giant one."""
    text = re.sub(r"\r\n", "\n", text)
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    sections: list[str] = []
    buf = ""
    for para in paragraphs:
        if not buf:
            buf = para
            continue
        if len(buf) >= MIN_SECTION_CHARS and len(buf) + len(para) > MAX_SECTION_CHARS:
            sections.append(buf)
            buf = para
        else:
            buf = f"{buf}\n\n{para}"
    if buf:
        sections.append(buf)
    return sections


def wipe_store() -> None:
    """Delete the previous .hrag_store so this run is reproducible."""
    store = Path("./.hrag_store")
    if store.exists():
        shutil.rmtree(store, ignore_errors=True)


def banner(text: str) -> None:
    bar = "=" * min(len(text), 80)
    print(f"\n{bar}\n{text}\n{bar}", flush=True)


def main() -> int:
    if not PDF_PATH.exists():
        print(f"ERROR: PDF not found at {PDF_PATH}", file=sys.stderr)
        return 1

    from hrag.doc_store import ParentDocStore
    existing = ParentDocStore().count()
    if existing == 0:
        banner("0. Resetting H-RAG store (.hrag_store/)")
        wipe_store()

        banner("1. Extracting PDF text")
        text = extract_pdf_text(PDF_PATH)
        print(f"  total chars: {len(text):,}")

        banner("2. Splitting paper into logical sections")
        sections = split_into_sections(text)
        print(f"  sections: {len(sections)}")
        for i, s in enumerate(sections):
            head = re.sub(r"\s+", " ", s[:80])
            print(f"  [{i:02d}] ({len(s):>5} chars)  {head}...")

        banner("3. Ingesting into H-RAG")
        sources = [f"{PDF_PATH.name}#section{i:02d}" for i in range(len(sections))]
        stats = ingest_texts(sections, sources)
        print(f"  {stats}")
    else:
        banner(f"0-3. Store already populated ({existing} parents) - skipping ingestion")

    banner("4. Building pipeline")
    pipeline = HRAGPipeline()
    print(
        "  config: "
        f"alpha={settings.hybrid_alpha} "
        f"top_k={settings.top_k_children} "
        f"top_n={settings.top_n_parents} "
        f"rank_parents={settings.rank_parents}"
    )

    history: list[tuple[str, str]] = []
    turns = [
        "What problem does H-RAG aim to solve?",
        "How are documents chunked during ingestion?",                            # ref: 'documents' (H-RAG's)
        "Which embedding and reranker models does it use?",                       # ref: 'it' -> H-RAG
        "What were the final scores on Task A and Task C?",                       # ref: implicit (H-RAG's submission)
        "What does the ablation table say about ranking strategy vs alpha?",      # tests ablation recall
    ]

    for i, q in enumerate(turns, 1):
        banner(f"Turn {i}: {q}")
        result = pipeline.answer(q, history=history)
        print(f"\n[rewritten query]\n  {result.rewritten_query}")
        print("\n[retrieved parents]")
        for p in result.parents:
            print(f"  - {p.get('source', p['parent_id'])}  score={p['score']:.4f}")
        print(f"\n[answer]\n{result.answer}")
        history.append((q, result.answer))

    banner("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
