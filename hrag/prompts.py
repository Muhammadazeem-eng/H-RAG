"""Prompt templates verbatim from Appendix A of the paper.

Two prompts:
  1. Query Rewrite Prompt - rewrites context-dependent turns into standalone
     questions, returns the original verbatim when already standalone.
  2. Generation Prompt    - grounded answer that NEVER cites 'the context'
     or source numbers like [1].
"""

from __future__ import annotations

from typing import List, Tuple

QUERY_REWRITE_SYSTEM = (
    "Given the conversation history, rewrite the new user question into a "
    "standalone and specific query suitable for retrieval.\n\n"
    "Important Rules:\n"
    "- If the question is already clear and standalone, return it EXACTLY as is\n"
    "- If the question contains pronouns or references to earlier dialogue, "
    "rewrite it using the necessary context\n"
    "- Do NOT invent information or change the original meaning\n"
    "- Return only the rewritten question with NO explanation"
)


def format_query_rewrite_user(history: List[Tuple[str, str]], user_question: str) -> str:
    """Render the user message for the rewriter.

    history: list of (Q, A) tuples, most recent 3 turns recommended by the paper.
    """
    history_text = (
        "\n".join(f"Q: {q}\nA: {a}" for q, a in history) if history else "(none)"
    )
    return (
        f"Conversation History:\n{history_text}\n\n"
        f"New Question: {user_question}\n\n"
        "Expected Output: Rewritten standalone question (or unchanged original "
        "question if already standalone)."
    )


GENERATION_SYSTEM = (
    "You are a helpful AI assistant engaged in a conversation with a user. "
    "Answer the user's question naturally and directly, as if you already "
    "know the information.\n\n"
    "Important Rules:\n"
    "- Do NOT mention 'the context', 'the provided information', 'according "
    "to the documents', or similar phrases\n"
    "- Do NOT reference source numbers such as [1], [2], etc.\n"
    "- Respond conversationally as if the knowledge is your own\n"
    "- If insufficient information is available, state this naturally "
    "without mentioning missing context\n"
    "- Maintain continuity with the conversation history"
)


def format_generation_user(
    history: List[Tuple[str, str]],
    parent_passages: List[str],
    user_question: str,
) -> str:
    history_block = (
        "\n".join(f"Q: {q}\nA: {a}" for q, a in history) if history else "(none)"
    )
    knowledge_block = "\n\n".join(
        f"[{i+1}] {p}" for i, p in enumerate(parent_passages)
    )
    return (
        f"Conversation History:\n{history_block}\n\n"
        f"Retrieved Background Knowledge:\n{knowledge_block}\n\n"
        f"Current User Question: {user_question}\n\n"
        "Respond naturally and conversationally:"
    )
