"""LLM instances for the aviation multi-agent graph.

Supervisor routing uses OpenAI; spoke/synthesis work uses Gemini Flash.
Both read API keys from the environment (loaded by app.py via dotenv).
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


def _usable_key(name: str) -> bool:
    """True only if the env var looks like a real key (not a .env.example placeholder)."""
    value = os.getenv(name, "")
    return bool(value) and "your_" not in value and "api_key_here" not in value


def _build_supervisor_llm() -> Any | None:
    """OpenAI chat model for the supervisor / router node."""
    if not _usable_key("OPENAI_API_KEY"):
        logger.warning("OPENAI_API_KEY missing; supervisor_llm unavailable.")
        return None
    try:
        from langchain_openai import ChatOpenAI

        # Override with SUPERVISOR_MODEL if set (default: gpt-4o-mini)
        model_name = os.getenv("SUPERVISOR_MODEL", "gpt-4o-mini")
        return ChatOpenAI(model=model_name, temperature=0)
    except Exception:
        logger.exception("Failed to initialize supervisor_llm (OpenAI).")
        return None


def _build_spoke_llm() -> Any | None:
    """Gemini Flash model for spoke planning / synthesis work."""
    if not _usable_key("GOOGLE_API_KEY"):
        logger.warning("GOOGLE_API_KEY missing; spoke_llm unavailable.")
        return None
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI

        # Override with SPOKE_MODEL if set (default: gemini-3.6-flash)
        model_name = os.getenv("SPOKE_MODEL", "gemini-3.6-flash")
        return ChatGoogleGenerativeAI(model=model_name, temperature=0)
    except Exception:
        logger.exception("Failed to initialize spoke_llm (Gemini).")
        return None


# Exported instances used by nodes.py (None when the matching API key is absent)
supervisor_llm = _build_supervisor_llm()
spoke_llm = _build_spoke_llm()
