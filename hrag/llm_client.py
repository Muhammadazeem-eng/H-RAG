"""Thin OpenAI client wrapper (paper Table 1: GPT-5 with chat completions).

Centralized so query_rewriter.py and generator.py share one client and one
place to swap models or providers.
"""

from __future__ import annotations

from functools import lru_cache

from .config import settings


@lru_cache(maxsize=1)
def get_client():
    # Pull OPENAI_API_KEY out of .env into os.environ on first use.
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    from openai import OpenAI
    key = settings.openai_key
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add it to your environment or .env file."
        )
    return OpenAI(api_key=key)


def chat(
    *,
    model: str,
    system: str,
    user: str,
    temperature: float,
    max_tokens: int | None = None,
) -> str:
    client = get_client()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return (resp.choices[0].message.content or "").strip()
