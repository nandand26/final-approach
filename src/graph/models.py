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
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from src.graph.state import SPOKE_CAPS, VALID_FANOUT_SPOKES

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


class PlannedStep(BaseModel):
    """One spoke CALL the router scheduled, plus the exact text that call must run."""

    worker: Literal["opensky", "flightaware", "tavily"] = Field(
        description="Fan-out spoke id (graph node name).",
    )
    task: str = Field(
        min_length=1,
        description=(
            "Exact instruction for that spoke. flightaware: flight ident. "
            "opensky: icao24 hex or airport ICAO question. "
            "tavily: web, NOTAM, or discovery query. Must be non-empty."
        ),
    )

    @field_validator("task")
    @classmethod
    def _strip_task(cls, value: str) -> str:
        """Reject a blank task. A spoke with no instruction is not a valid step."""
        task = value.strip()
        if not task:
            raise ValueError("task must be non-empty")
        return task


class SupervisorPlan(BaseModel):
    """Supervisor output for one user turn: intent, then spokes + pack order.

    `intent` picks the graph path (known telemetry, discovery+HITL, research, chat).
    `spokes_to_call` is the Send() set for telemetry_known, or `tavily` alone for
    discovery / research. Empty means synthesizer only.
    """

    intent: Literal[
        "telemetry_known",
        "telemetry_discovery",
        "general_knowledge",
        "direct_chat",
    ] = Field(
        description=(
            "telemetry_known: a flight ident is known (BA117) or the user just "
            "picked one. You choose which spokes run; they run in parallel. "
            "telemetry_discovery: live status by airline/route with NO ident. "
            "Tavily only, then the human picks an ident. "
            "general_knowledge: research with no live tracking. Tavily only. "
            "direct_chat: greeting or reformat. No spokes."
        ),
    )
    flight_number: str | None = Field(
        default=None,
        description=(
            "Marketed flight ident (BA117, SQ32, ID8085) or null. "
            "Required for telemetry_known when the user named a flight. "
            "Null for telemetry_discovery, general_knowledge, direct_chat, "
            "and for hex-only OpenSky."
        ),
    )
    search_query: str | None = Field(
        default=None,
        description=(
            "Tavily query for telemetry_discovery or general_knowledge. "
            "Null when no web search is needed."
        ),
    )
    spokes_to_call: list[Literal["opensky", "flightaware", "tavily"]] = Field(
        default_factory=list,
        description=(
            "Domain spokes to execute IN PARALLEL. Allowed: "
            f"{list(VALID_FANOUT_SPOKES)}. List each spoke once; repeat calls "
            "to a spoke go in `plan`. Caps on distinct tasks per spoke: "
            f"{ {k: SPOKE_CAPS[k] for k in VALID_FANOUT_SPOKES} }. "
            "telemetry_known: any subset of opensky, flightaware, tavily that "
            "this question needs. Those spokes run in parallel. "
            "telemetry_discovery MUST be exactly ['tavily']. "
            "general_knowledge MUST be exactly ['tavily']. "
            "direct_chat MUST be []."
        ),
    )
    context_priority_order: list[Literal["opensky", "flightaware", "tavily"]] = Field(
        default_factory=list,
        description=(
            "Order in which the synthesizer should pack spoke notes. "
            "Allowed values only: opensky, flightaware, tavily. "
            "Any other name is dropped. Duplicates are dropped. "
            "IF safety-critical spokes are selected — flightaware (NOAA METARs "
            "are fetched inside FlightAware only when a delay/hold/divert is "
            "reported) and/or tavily NOTAMs — they MUST be placed at the "
            "absolute beginning or end of this list to mitigate LLM attention "
            "degradation (lost-in-the-middle)."
        ),
    )

    @field_validator("context_priority_order", mode="before")
    @classmethod
    def _keep_known_spokes(cls, value: Any) -> list[str]:
        """Drop unknown pack-order ids before this plan reaches the synthesizer."""
        if value is None:
            return []
        items = [value] if isinstance(value, str) else list(value)
        seen: set[str] = set()
        kept: list[str] = []
        for item in items:
            name = str(item).strip()
            if name not in VALID_FANOUT_SPOKES or name in seen:
                continue
            seen.add(name)
            kept.append(name)
        return kept

    extract_memory: bool = Field(
        default=False,
        description=(
            "Memory Gate. True ONLY if the latest user message explicitly dictates "
            "how the assistant should behave, format answers, or remember personal "
            "context for future interactions. False for standard aviation lookups."
        ),
    )
    reasoning: str = Field(
        default="",
        description=(
            "Brief note on reference resolution, why each spoke was scheduled, "
            "why context_priority_order is arranged as it is, and extract_memory."
        ),
    )
    plan: list[PlannedStep] = Field(
        default_factory=list,
        description=(
            "Required execution contract. One {worker, task} row per spoke CALL. "
            "Every id in spokes_to_call needs at least one row, and no row may "
            "name a spoke that was not selected. task must be non-empty. "
            "The same worker may appear more than once ONLY with genuinely "
            "different tasks, up to the per-spoke cap. NEVER repeat an identical "
            "task for the same worker. "
            "Validation fails if a selected spoke has no row or a row names a "
            "spoke that was not selected. "
            "Empty only when spokes_to_call is empty (direct_chat)."
        ),
    )

    @field_validator("plan", mode="before")
    @classmethod
    def _drop_unknown_plan_steps(cls, value: Any) -> list:
        """Drop workers outside opensky | flightaware | tavily. Keep the rest.

        Runs before each row is parsed as a PlannedStep. A bad worker such as
        "noaa" or "synthesizer" is removed here so it does not reject the
        whole SupervisorPlan. spokes_to_call is not available yet.
        """
        if value is None:
            return []
        # Structured output is a list; a single dict/object is still accepted.
        items = value if isinstance(value, list) else [value]
        kept: list = []
        for item in items:
            if isinstance(item, dict):
                worker = str(item.get("worker") or "").strip()
                # Unknown id: skip this row, keep any later valid rows.
                if worker not in VALID_FANOUT_SPOKES:
                    continue
                kept.append({**item, "worker": worker})
                continue
            worker = str(getattr(item, "worker", "") or "").strip()
            if worker not in VALID_FANOUT_SPOKES:
                continue
            kept.append(item)
        return kept

    @model_validator(mode="after")
    def _align_plan_to_spokes(self) -> SupervisorPlan:
        """Require the plan's workers to equal the spokes in spokes_to_call.

        Runs after both fields are parsed. A selected spoke with no row, or a
        row for a spoke that was not selected, raises. The same worker may have
        several rows (different tasks); exact duplicates and the per-spoke cap
        are handled in nodes.py. Blank tasks never reach here: PlannedStep
        already rejects them.
        """
        selected = list(dict.fromkeys(self.spokes_to_call))
        workers = [step.worker for step in self.plan]
        if set(workers) != set(selected):
            missing = [name for name in selected if name not in workers]
            extra = [name for name in workers if name not in set(selected)]
            raise ValueError(
                "plan must match spokes_to_call exactly. "
                f"missing={missing or 'none'} extra={extra or 'none'}."
            )
        return self
