"""LLM-based query rewriter (paper §3.2).

Reformulates context-dependent multi-turn questions into standalone queries.
For already-standalone turns the prompt instructs the model to return the
question unchanged - the paper notes this rule was 'essential for short
factoid turns where LLM-generated rewrites tended to over-elaborate'.
"""

from __future__ import annotations

from typing import List, Tuple

from .config import settings
from .llm_client import chat
from .prompts import QUERY_REWRITE_SYSTEM, format_query_rewrite_user


def rewrite_query(question: str, history: List[Tuple[str, str]]) -> str:
    if not history:
        # Nothing to disambiguate against -> skip the LLM call entirely.
        return question
    user_msg = format_query_rewrite_user(history[-3:], question)
    try:
        rewritten = chat(
            model=settings.rewrite_model,
            system=QUERY_REWRITE_SYSTEM,
            user=user_msg,
            temperature=settings.rewrite_temperature,
            max_tokens=256,
        )
    except Exception:
        # Retrieval should still work if the rewriter is unavailable.
        return question
    return rewritten or question
