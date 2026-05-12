"""Answer generator (paper §3.4).

Conditioned on the reconstructed parent-level context and the rewritten
query. The system prompt forbids referencing 'the context' or numeric
citations, matching the Appendix A template.
"""

from __future__ import annotations

from typing import List, Tuple

from .config import settings
from .llm_client import chat
from .prompts import GENERATION_SYSTEM, format_generation_user


def generate_answer(
    question: str,
    parent_passages: List[str],
    history: List[Tuple[str, str]],
) -> str:
    user_msg = format_generation_user(history[-3:], parent_passages, question)
    return chat(
        model=settings.gen_model,
        system=GENERATION_SYSTEM,
        user=user_msg,
        temperature=settings.gen_temperature,
        max_tokens=settings.max_completion_tokens,
    )
