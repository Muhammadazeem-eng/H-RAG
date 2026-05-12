"""Configuration loaded from environment / .env.

All hyperparameters come from Table 1 and Section 3.5 of the paper.
"""

from __future__ import annotations

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="HRAG_", extra="ignore")

    # LLM (Table 1: GPT-5; we expose as gen + rewrite models)
    openai_api_key: str = ""
    gen_model: str = "gpt-4o"
    rewrite_model: str = "gpt-4o-mini"
    gen_temperature: float = 0.7
    rewrite_temperature: float = 0.2
    max_completion_tokens: int = 4096

    # Models (Table 1)
    embed_model: str = "BAAI/bge-large-en-v1.5"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"

    # Retrieval (Table 1)
    hybrid_alpha: float = 0.7     # 70% dense / 30% sparse
    top_k_children: int = 50      # initial candidate set k
    top_n_parents: int = 5        # parents handed to generator
    rank_parents: bool = True     # child-first ranking + parent aggregation

    # Chunking (§3.5: 3-sentence chunks, stride 2)
    chunk_sentences: int = 3
    chunk_stride: int = 2

    # Storage
    chroma_path: Path = Path("./.hrag_store/chroma")
    docstore_path: Path = Path("./.hrag_store/parents.sqlite")
    session_path: Path = Path("./.hrag_store/sessions.sqlite")
    bm25_path: Path = Path("./.hrag_store/bm25.pkl")

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    @property
    def openai_key(self) -> str:
        # Tolerate either OPENAI_API_KEY or HRAG_OPENAI_API_KEY.
        import os
        return self.openai_api_key or os.environ.get("OPENAI_API_KEY", "")


settings = Settings()
