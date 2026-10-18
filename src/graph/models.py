"""LLM instances for the aviation multi-agent graph.

Supervisor routing uses OpenAI; spoke/synthesis work uses Gemini Flash.
Both read API keys from the environment (loaded by app.py via dotenv).

Models are built lazily and rebuilt when SUPERVISOR_MODEL / SPOKE_MODEL change
so a Streamlit rerun after editing `.env` picks up the new id without a
manual process kill (module-level singletons previously froze the first model).
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

_supervisor_llm: Any | None = None
_supervisor_model_name: str | None = None
_spoke_llm: Any | None = None
_spoke_model_name: str | None = None


def _usable_key(name: str) -> bool:
    """True only if the env var looks like a real key (not a .env.example placeholder)."""
    value = os.getenv(name, "")
    return bool(value) and "your_" not in value and "api_key_here" not in value


def get_supervisor_llm() -> Any | None:
    """OpenAI chat model for the supervisor / router node (env-fresh)."""
    global _supervisor_llm, _supervisor_model_name

    if not _usable_key("OPENAI_API_KEY"):
        logger.warning("OPENAI_API_KEY missing; supervisor_llm unavailable.")
        _supervisor_llm = None
        _supervisor_model_name = None
        return None

    model_name = (os.getenv("SUPERVISOR_MODEL") or "gpt-4o-mini").strip()
    if _supervisor_llm is not None and _supervisor_model_name == model_name:
        return _supervisor_llm

    try:
        from langchain_openai import ChatOpenAI

        _supervisor_llm = ChatOpenAI(model=model_name, temperature=0)
        _supervisor_model_name = model_name
        logger.info("supervisor_llm ready: %s", model_name)
        return _supervisor_llm
    except Exception:
        logger.exception("Failed to initialize supervisor_llm (OpenAI).")
        _supervisor_llm = None
        _supervisor_model_name = None
        return None


def get_spoke_llm() -> Any | None:
    """Gemini model for spoke planning / synthesis (env-fresh)."""
    global _spoke_llm, _spoke_model_name

    if not _usable_key("GOOGLE_API_KEY"):
        logger.warning("GOOGLE_API_KEY missing; spoke_llm unavailable.")
        _spoke_llm = None
        _spoke_model_name = None
        return None

    # Default away from gemini-3.6-flash — free-tier daily caps are per-model.
    model_name = (os.getenv("SPOKE_MODEL") or "gemini-2.5-flash").strip()
    if _spoke_llm is not None and _spoke_model_name == model_name:
        return _spoke_llm

    try:
        from langchain_google_genai import ChatGoogleGenerativeAI

        _spoke_llm = ChatGoogleGenerativeAI(model=model_name, temperature=0)
        _spoke_model_name = model_name
        logger.info("spoke_llm ready: %s", model_name)
        return _spoke_llm
    except Exception:
        logger.exception("Failed to initialize spoke_llm (Gemini).")
        _spoke_llm = None
        _spoke_model_name = None
        return None


# Back-compat for `from src.graph.models import spoke_llm` — still resolves once
# at import. Prefer get_spoke_llm() / get_supervisor_llm() in node code.
supervisor_llm = get_supervisor_llm()
spoke_llm = get_spoke_llm()
