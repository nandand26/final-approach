"""LangGraph node implementations for supervisor and spoke workers.

This file is the "brains" of each graph step:
- Helpers: small utilities (read user text, pick an LLM, keyword routing, etc.)
- Nodes: supervisor, Spoke 1 (OpenSky physical airspace), Spoke 2 (live ops), synthesizer

LangGraph calls a node with the current AgentState, and the node returns
a partial update (usually new messages + next_worker / final_report).
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any, Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from src.graph.state import SPOKE_CAPS, AgentState
from src.graph.models import spoke_llm, supervisor_llm
from src.mcp_servers.opensky_mcp import OPENSKY_TOOLS, invoke_opensky_tool
from src.tools.live_ops import flightaware_live_status, tavily_aviation_web_search

logger = logging.getLogger(__name__)

# Allowed spoke names that may appear in the plan queue / next_worker.
SPOKE_WORKERS = ("Spoke_1_OpenSky", "Spoke_2_LiveOps", "Spoke_3_WebSearch")
# Allowed values for state["next_worker"] (spokes + synthesizer handoff).
VALID_WORKERS = {
    "Spoke_1_OpenSky",
    "Spoke_2_LiveOps",
    "Spoke_3_WebSearch",
    "FINISH",
}

# Acronyms that look like airline idents (letter+digit) but are not flight numbers.
_FLIGHT_IDENT_STOPWORDS = frozenset(
    {"MEL", "CAT", "UTC", "ILS", "VFR", "IFR", "FDR", "EHM", "FL"}
)
_FOLLOWUP_IDENT_RE = re.compile(
    r"\b("
    r"this flight|that flight|same plane|same flight|"
    r"mentioned before|mentioned earlier|"
    r"earlier flight|previous flight|prior flight|"
    r"flight i mentioned|flight you mentioned|"
    r"the (?:one|flight) i mentioned|the (?:one|flight) you mentioned|"
    r"\bit\b"
    r")\b",
    flags=re.IGNORECASE,
)

# Lazy Mem0 client — long-term, cross-session prefs keyed by Google OAuth user_id.
# Initialized once per process; failures never take down the LangGraph turn.
_memory_client: Any | None = None
_memory_init_attempted = False


def _get_memory() -> Any | None:
    """Return a local mem0.Memory() instance, or None if unavailable.

    Mem0 OSS defaults to `gpt-5-mini` with temperature=0.1, but that model
    rejects non-default temperature (OpenAI 400). Pin a chat model that
    accepts 0.1 — prefer MEM0_LLM_MODEL, else SUPERVISOR_MODEL, else gpt-4o-mini.
    """
    global _memory_client, _memory_init_attempted
    if _memory_client is not None:
        return _memory_client
    if _memory_init_attempted:  # already tried once this process
        return None
    _memory_init_attempted = True
    try:
        from mem0 import Memory

        mem0_model = (
            (os.getenv("MEM0_LLM_MODEL") or "").strip()
            or (os.getenv("SUPERVISOR_MODEL") or "").strip()
            or "gpt-4o-mini"
        )
        _memory_client = Memory.from_config(
            {
                "llm": {
                    "provider": "openai",
                    "config": {
                        "model": mem0_model,
                        "temperature": 0.1,
                    },
                },
            }
        )
        return _memory_client
    except Exception:  # noqa: BLE001
        logger.exception("Mem0 Memory() init failed; continuing without long-term memory.")
        _memory_client = None
        return None


def _memory_fact_texts(raw: Any, *, limit: int = 3) -> list[str]:
    """Normalize Mem0 search payloads into plain fact strings."""
    rows: list[Any]
    if isinstance(raw, dict):
        rows = list(raw.get("results") or raw.get("memories") or [])
    elif isinstance(raw, list):
        rows = raw
    else:
        rows = []

    facts: list[str] = []
    for row in rows:
        if isinstance(row, dict):
            text = str(row.get("memory") or row.get("text") or row.get("data") or "").strip()
        else:
            text = str(row).strip()
        if text:
            facts.append(text)
        if len(facts) >= limit:
            break
    return facts


def _search_user_memories(query: str, user_id: str, *, limit: int = 3) -> list[str]:
    """Fetch cross-session Mem0 facts for this Google identity.

    Prefer the call shape from the project spec
    (`search(query=..., user_id=..., limit=...)`). Fall back to newer
    `filters={"user_id": ...}` if the installed SDK rejects top-level user_id.
    """
    if not user_id or not (query or "").strip():
        return []
    memory = _get_memory()
    if memory is None:
        return []
    try:
        try:
            raw = memory.search(query=query, user_id=user_id, limit=limit)
        except (TypeError, ValueError):
            # Newer Mem0 SDKs reject top-level user_id; use filters= instead.
            raw = memory.search(
                query,
                filters={"user_id": user_id},
                limit=limit,
            )
        return _memory_fact_texts(raw, limit=limit)
    except Exception:  # noqa: BLE001
        logger.exception("Mem0 search failed for user_id=%s; skipping memory inject.", user_id)
        return []


def _add_user_memory(interaction: list[dict[str, str]], user_id: str) -> None:
    """Persist a user/assistant turn into Mem0. Never raises into the graph."""
    if not user_id or not interaction:
        return
    memory = _get_memory()
    if memory is None:
        return
    try:
        memory.add(interaction, user_id=user_id)
    except Exception:  # noqa: BLE001
        logger.exception(
            "Mem0 add failed for user_id=%s; graph response still returned.",
            user_id,
        )


# ---------------------------------------------------------------------------
# Small helpers used by the nodes below
# ---------------------------------------------------------------------------

def _latest_user_text(state: AgentState) -> str:
    """Pull the most recent user question out of the message list."""
    # Walk messages from newest -> oldest so we get the latest user ask
    for message in reversed(state.get("messages", [])):
        # Case 1: normal LangChain HumanMessage object
        if isinstance(message, HumanMessage):
            content = message.content
            # Content is usually a string; convert if somehow it is not
            return content if isinstance(content, str) else str(content)

        # Case 2: plain dict-style message, e.g. {"role": "user", "content": "..."}
        # getattr(..., None) safely reads .type if it exists
        role = getattr(message, "type", None) or (
            message.get("role") if isinstance(message, dict) else None
        )
        if role in {"human", "user"}:
            content = (
                message.content
                if hasattr(message, "content")
                else message.get("content", "")
            )
            return content if isinstance(content, str) else str(content)

    # No user message found
    return ""


def _message_text(message: Any) -> str:
    """Normalize any message shape into a plain string of its content."""
    if isinstance(message, dict):
        return str(message.get("content", "")) # if message is a dict, return the content
    # LangChain message objects store text on .content
    return str(getattr(message, "content", "")) # if message is a LangChain message object, return the content  


def _llm_text(response: Any) -> str:
    """Pull plain text from an LLM response (string or Gemini content-block list).

    Newer Gemini models often return content like:
      [{'type': 'text', 'text': '...', 'extras': {...}}]
    instead of a plain string. OpenAI usually returns a string.
    """
    content = getattr(response, "content", response) # if response is a dict, return the content else return the response
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip() # if content is a string, return the content stripped of whitespace

    # List of text blocks (Gemini Flash / multimodal-style payloads)
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                # Prefer the human-readable text field; ignore signatures/extras
                text = block.get("text")
                if text:
                    parts.append(str(text))
            else:
                text = getattr(block, "text", None)
                if text:
                    parts.append(str(text))
        return "\n".join(part for part in parts if part).strip() # join the parts with a newline and strip the whitespace

    return str(content).strip()


def _spoke_results_present(turn_notes: list[str]) -> tuple[bool, bool]:
    """Check whether Spoke 1 / Spoke 2 already ran at least once THIS turn.

    Looks only at turn_notes (scratch), not durable conversation messages.
    Returns:
        (spoke_1_done, spoke_2_done)
    """
    spoke_1 = False
    spoke_2 = False
    for note in turn_notes:
        text = str(note)
        if "[Spoke_1_OpenSky]" in text:
            spoke_1 = True
        if "[Spoke_2_LiveOps]" in text:
            spoke_2 = True
    return spoke_1, spoke_2


def _empty_spoke_counts() -> dict[str, int]:
    return {name: 0 for name in SPOKE_WORKERS}


def _spoke_execution_counts(turn_notes: list[str]) -> dict[str, int]:
    """Count how many times each spoke already ran in THIS user-message turn."""
    counts = _empty_spoke_counts()
    for note in turn_notes:
        text = str(note)
        if "[Spoke_1_OpenSky]" in text:
            counts["Spoke_1_OpenSky"] += 1
        if "[Spoke_2_LiveOps]" in text:
            counts["Spoke_2_LiveOps"] += 1
        if "[Spoke_3_WebSearch]" in text:
            counts["Spoke_3_WebSearch"] += 1
    return counts


def _step_worker(step: Any) -> str:
    """Extract worker name from a plan step (dict or legacy string)."""
    if isinstance(step, dict):
        return str(step.get("worker") or "").strip()
    if isinstance(step, BaseModel):
        return str(getattr(step, "worker", "") or "").strip()
    return str(step or "").strip()


def _step_task(step: Any) -> str:
    """Extract task string from a plan step (dict / PlannedStep / empty)."""
    if isinstance(step, dict):
        return str(step.get("task") or "").strip()
    if isinstance(step, BaseModel):
        return str(getattr(step, "task", "") or "").strip()
    return ""


def _normalize_plan_steps(plan: Any) -> list[dict[str, str]]:
    """Coerce LLM / heuristic plan items into [{"worker", "task"}, ...]."""
    normalized: list[dict[str, str]] = []
    for step in list(plan or []):
        worker = _step_worker(step)
        if worker not in SPOKE_WORKERS:
            continue
        task = _step_task(step)
        normalized.append({"worker": worker, "task": task})
    return normalized


def _enforce_spoke_caps(
    plan: list[Any],
    already_ran: dict[str, int] | None = None,
) -> list[dict[str, str]]:
    """Defensive queue sanitizer: valid spokes only + per-worker SPOKE_CAPS.

    WHY this exists on top of the LLM schema prompt:
    - LLMs can still emit duplicates or invalid names.
    - Mid-turn re-entry must also respect how many times a spoke already executed.
    - Caps are counted against (already_ran + scheduled-in-this-plan).
    """
    already_ran = already_ran or _empty_spoke_counts()
    scheduled = _empty_spoke_counts()
    cleaned: list[dict[str, str]] = []

    for step in _normalize_plan_steps(plan):
        worker = step["worker"]
        cap = SPOKE_CAPS.get(worker, 0)
        total_if_added = already_ran.get(worker, 0) + scheduled.get(worker, 0) + 1
        if total_if_added > cap:
            continue
        cleaned.append(step)
        scheduled[worker] = scheduled.get(worker, 0) + 1

    return cleaned


def _heuristic_plan(user_text: str, history_blob: str) -> list[dict[str, str]]:
    """Offline Plan-and-Execute fallback when supervisor_llm is unavailable.

    Domain split:
      Spoke 1 → physical ADS-B / OpenSky (icao24, altitude, airport movements)
      Spoke 2 → FlightAware commercial status + NOAA METARs
      Spoke 3 → Tavily web research (schedules, airline/route without flight #)
    """
    text = f"{user_text}\n{history_blob}".lower()
    latest = (user_text or "").lower()
    task = (user_text or "").strip()

    # Physical airspace / ADS-B style asks
    wants_opensky = any(
        signal in text
        for signal in (
            "icao24",
            "transponder",
            "ads-b",
            "adsb",
            "altitude",
            "squawk",
            "on_ground",
            "ground speed",
            "vertical rate",
            "trajectory",
            "track waypoints",
            "opensky",
            "bbox",
            "airspace",
            "arrivals at",
            "departures from",
            "airport arrivals",
            "airport departures",
        )
    ) or bool(re.search(r"\b[0-9a-f]{6}\b", text))  # bare icao24 hex

    has_flight_ident = bool(re.search(r"\b[a-z]{2,3}\d{1,5}\b", latest))

    # Airline / route without a concrete flight number → web research first
    wants_web = any(
        signal in text
        for signal in (
            "tavily",
            "web search",
            "research",
            "news",
            "recall",
            "advisory",
            "schedule",
            "timetable",
            "which flight",
            "what flight",
            "find flight",
            "flight number",
            "transit",
            "connection",
            "from ",
            " to ",
        )
    ) and not has_flight_ident

    # Explicit airline+route asks without an ident (e.g. Batik Air India → KL)
    airline_route_without_ident = (
        not has_flight_ident
        and any(
            signal in latest
            for signal in (
                "air ",
                " airlines",
                " airways",
                "batik",
                "flight from",
                "flights from",
                "to kl",
                "to kuala",
            )
        )
    )

    # Commercial / ops / weather (FlightAware path)
    wants_live = any(
        signal in text
        for signal in (
            "live",
            "weather",
            "now",
            "status",
            "ops",
            "operational",
            "flightaware",
            "gate",
            "eta",
            "enroute",
            "metar",
            "aircraft type",
            "codeshare",
            "a380",
            "b787",
            "delay",
        )
    ) or has_flight_ident or airline_route_without_ident

    conversational = any(
        signal in text
        for signal in (
            "summarize",
            "reformat",
            "rewrite",
            "thanks",
            "thank you",
            "hello",
            "hi ",
            "what did you say",
            "repeat that",
        )
    )
    if conversational and not wants_opensky and not wants_live and not wants_web:
        return []

    plan: list[dict[str, str]] = []
    if wants_opensky:
        plan.append({"worker": "Spoke_1_OpenSky", "task": task})

    # Live commercial ask without a flight number → discover idents via web, then FA
    if wants_live and (wants_web or airline_route_without_ident) and not has_flight_ident:
        plan.append({"worker": "Spoke_3_WebSearch", "task": task})
        plan.append({"worker": "Spoke_2_LiveOps", "task": task})
    else:
        if wants_web and not wants_live:
            plan.append({"worker": "Spoke_3_WebSearch", "task": task})
        if wants_live:
            plan.append({"worker": "Spoke_2_LiveOps", "task": task})

    # Ambiguous aviation ask → prefer commercial live ops
    if not plan and user_text.strip():
        plan = [{"worker": "Spoke_2_LiveOps", "task": task}]

    return _enforce_spoke_caps(plan)


class PlannedStep(BaseModel):
    """One sub-task in the supervisor's ordered execution queue."""

    worker: Literal["Spoke_1_OpenSky", "Spoke_2_LiveOps", "Spoke_3_WebSearch"] = Field(
        description="Which spoke should execute this sub-task.",
    )
    task: str = Field(
        description=(
            "Targeted sub-query for WebSearch, flight ident/route goal for LiveOps, "
            "or icao24/airport task for OpenSky."
        ),
    )


class SupervisorPlan(BaseModel):
    """Structured plan the supervisor LLM must return at turn start.

    `plan` is an ordered queue of {worker, task} steps. Empty list means
    conversational bypass → go straight to FINISH / synthesizer (no tool calls).

    `extract_memory` is the Memory Gate: True only for explicit durable prefs.
    """

    plan: list[PlannedStep] = Field(
        default_factory=list,
        description=(
            "Ordered execution queue of PlannedStep objects. Allowed workers: "
            "'Spoke_1_OpenSky', 'Spoke_2_LiveOps', 'Spoke_3_WebSearch'. "
            "Use [] when the answer needs no tools (greetings, reformat, "
            "questions answerable from existing chat context alone). "
            f"Per-turn caps: {SPOKE_CAPS}."
        ),
    )
    extract_memory: bool = Field(
        default=False,
        description=(
            "Memory Gate. True ONLY if the latest user message explicitly dictates "
            "how the assistant should behave, format answers, or remember personal "
            "context for future interactions. False for standard aviation lookups, "
            "flight status, ADS-B, weather, or conversational filler."
        ),
    )
    reasoning: str = Field(
        default="",
        description=(
            "Brief note on how chat history resolved ambiguous references "
            "(e.g. 'that flight' → AI8303), why each spoke was scheduled, and "
            "why extract_memory is true/false."
        ),
    )


class WebSearchRefine(BaseModel):
    """Structured refinement of a Tavily query using prior spoke facts."""

    refined_query: str = Field(
        description=(
            "Short Tavily search query that incorporates concrete facts "
            "(flight numbers, aircraft types, tails) from prior spoke outputs "
            "without repeating prior queries verbatim."
        ),
    )


def _heuristic_extract_memory(user_text: str) -> bool:
    """Offline Memory Gate when supervisor_llm is unavailable.

    Conservative: True only when the text clearly states a durable preference /
    standing instruction. Default False so standard ops queries never trigger Mem0 add.
    """
    text = (user_text or "").lower()
    if not text.strip():
        return False
    preference_markers = (
        "always ",
        "never ",
        "from now on",
        "remember that",
        "remember i",
        "remember I'm",
        "remember i'm",
        "prefer ",
        "i prefer",
        "i am a ",
        "i'm a ",
        "keep your answers",
        "format answers",
        "in bullet",
        "in meters",
        "in feet",
        "metric",
        "dispatcher",
    )
    return any(marker in text for marker in preference_markers)


def _llm_generate_plan(
    user_text: str,
    history_blob: str,
    memory_facts: list[str] | None = None,
) -> tuple[list[dict[str, str]], bool]:
    """Ask the supervisor LLM for an ordered {worker,task} plan + Memory Gate flag.

    CONTEXT AWARENESS:
    - `history_blob` is the recent `messages` transcript (user + prior finals).
    - The prompt explicitly requires resolving pronouns / implicit refs
      ("that flight", "yesterday's delay") against that history BEFORE planning.
    - `memory_facts` are cross-session Mem0 preferences (units, formatting, etc.).
    - `extract_memory` is True only for explicit durable preference statements.
    """
    model = supervisor_llm
    if model is None:
        return (
            _heuristic_plan(user_text, history_blob),
            _heuristic_extract_memory(user_text),
        )

    memory_block = ""
    if memory_facts:
        bullets = "\n".join(f"- {fact}" for fact in memory_facts)
        memory_block = (
            "\n\nLONG-TERM USER MEMORY (cross-session preferences from Mem0):\n"
            f"{bullets}\n"
            "Honor these preferences when planning (e.g. metric units, dispatch "
            "formatting, preferred spokes/detail level). Pass intent through "
            "turn_notes so spokes/synthesizer can adhere as well.\n"
            "IMPORTANT: Existing Mem0 facts do NOT by themselves mean "
            "extract_memory=true for this turn — only the latest user message does.\n"
        )

    system_prompt = (
        "You are the aviation fleet intelligence supervisor operating under a Plan-and-Execute architecture.\n"
        "At the START of each user turn you must (1) build an ordered `plan` queue of "
        "{worker, task} steps and (2) set the Memory Gate flag `extract_memory`.\n\n"
        "CONTEXT & AMBIGUITY RESOLUTION:\n"
        "- Read the full chat history alongside the latest user query.\n"
        "- Resolve implicit references using prior context before picking spokes.\n"
        f"{memory_block}"
        "SPOKE CAPABILITIES (strict domain split):\n"
        "- Spoke_1_OpenSky = physical ADS-B radar (icao24 hex, airport arrivals/"
        "departures, tracks). DO NOT use for commercial flight numbers, schedules, "
        "equipment types, METAR, or web research.\n"
        "- Spoke_2_LiveOps = FlightAware commercial status & NOAA METARs "
        "(flight numbers, tail registrations, codeshares). NO Tavily / web search.\n"
        "- Spoke_3_WebSearch = Tavily web search (general aviation research, route "
        "schedules, finding flight numbers for an airline/route when no flight ident "
        "is given, airport transit).\n\n"
        "PLANNING & ORDERING RULES:\n"
        "- Known flight number / delay / weather / aircraft-type → Spoke_2_LiveOps "
        "with that ident/goal in `task`.\n"
        "- Live flight by airline/route WITHOUT a flight number "
        "(e.g. \"Batik Air flight from India to KL\") → schedule Spoke_3_WebSearch "
        "FIRST (discover flight numbers) THEN Spoke_2_LiveOps (live AeroAPI status). "
        "Put a focused discovery query in the WebSearch `task` and a status-check "
        "goal in the LiveOps `task`.\n"
        "- icao24 / altitude / trajectory / ADS-B airport movements → Spoke_1_OpenSky.\n"
        "- Pure research / schedules / transit with no live FA need → Spoke_3_WebSearch alone.\n"
        "- Hybrid queries may include multiple spokes, ordered by what the user needs first.\n"
        f"- Per-spoke caps this turn: {SPOKE_CAPS}. Do not exceed them.\n"
        "- NEVER place 'FINISH' or synthesizer nodes inside the `plan` array.\n"
        "- Every plan item MUST include a non-empty `task` string tailored to that spoke.\n\n"
        "DIRECT BYPASS RULE:\n"
        "- If conversational / reformat / answerable from chat alone, return an empty plan: `[]`.\n\n"
        "MEMORY GATE (`extract_memory`) — treat Mem0 as an immutable configuration file:\n"
        "- Set extract_memory=true ONLY if the latest user message explicitly dictates how "
        "you should behave, format answers, or remember their personal context for future "
        "interactions. Capture only explicit, durable rules.\n"
        "- Set extract_memory=false for standard aviation questions, flight lookups, "
        "ADS-B/radar, weather/delay asks, or conversational filler.\n"
        "- FALSE examples: \"What is the status of BA117?\", "
        "\"Pull live ADS-B for icao24 4cc2a3.\", "
        "\"Is the flight delayed due to weather?\"\n"
        "- TRUE examples: \"Always report altitudes in meters, never feet.\", "
        "\"I am a commercial dispatcher, keep your answers in bullet points.\"\n"
        "- When in doubt, set extract_memory=false."
    )
    try:
        structured = model.with_structured_output(SupervisorPlan)
        result = structured.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(
                    content=(
                        "Recent conversation history (use this to resolve references):\n"
                        f"{history_blob or '(none)'}\n\n"
                        f"Latest user query:\n{user_text}\n\n"
                        "Return a SupervisorPlan with an ordered `plan` of "
                        "{worker, task} steps and extract_memory true/false per "
                        "the Memory Gate rules."
                    )
                ),
            ]
        )
        if isinstance(result, dict):
            result = SupervisorPlan.model_validate(result)
        raw_plan = list(getattr(result, "plan", []) or [])
        extract_memory = bool(getattr(result, "extract_memory", False))
        return _enforce_spoke_caps(raw_plan), extract_memory
    except Exception:
        logger.exception("Supervisor plan generation failed; using heuristic plan.")
        return (
            _heuristic_plan(user_text, history_blob),
            _heuristic_extract_memory(user_text),
        )


# ---------------------------------------------------------------------------
# Graph nodes (these are the functions wired in workflow.py)
# ---------------------------------------------------------------------------

def supervisor_node(state: AgentState) -> dict:
    """Plan-and-Execute supervisor: build a turn plan once, then pop/execute it.

    Lifecycle for ONE user message turn:
    1) Turn start (`turn_notes` empty after app reset) → generate `plan` from
       latest query + full recent `messages` (resolves 'that flight', etc.).
    2) Each supervisor visit → pop the next step from `plan` into
       `next_worker` + `current_task`.
    3) Empty `plan` → `FINISH` (synthesizer). An empty plan at generation time
       is an intentional conversational/direct bypass (saves tool latency/cost).
    4) Hard caps: per-worker limits from SPOKE_CAPS (2 / 2 / 5).
    """
    user_text = _latest_user_text(state)
    user_id = str(state.get("user_id") or "").strip()
    turn_notes = list(state.get("turn_notes") or [])
    plan = _normalize_plan_steps(state.get("plan") or [])

    # Build a compact history blob from durable messages for reference resolution.
    # WHY messages (not turn_notes): prior final answers contain flight numbers /
    # delays the user may refer to implicitly in the new prompt.
    history_parts = [_message_text(m)[:400] for m in state.get("messages", [])[-8:]]
    history_blob = "\n---\n".join(part for part in history_parts if part.strip())

    # ----- Phase A: TURN START — generate the plan once -----
    # Signal: turn_notes was reset to [] by app.py at the start of this prompt.
    if not turn_notes:
        # Mem0 retrieval at plan-build time so supervisor_llm honors cross-session prefs.
        memory_facts = _search_user_memories(user_text, user_id, limit=3)
        memory_note = ""
        if memory_facts:
            memory_note = " Mem0 prefs: " + "; ".join(memory_facts[:3]) + "."

        plan, extract_memory = _llm_generate_plan(
            user_text, history_blob, memory_facts=memory_facts
        )
        # Defensive re-check with zero prior executions this turn.
        plan = _enforce_spoke_caps(plan, already_ran=_empty_spoke_counts())
        gate_note = f" extract_memory={extract_memory}."

        if not plan:
            # Empty plan → immediate synthesizer handoff.
            # WHY: conversational / reformat / context-only asks do not need tools;
            # skipping spokes cuts latency and API overhead.
            return {
                "plan": [],
                "next_worker": "FINISH",
                "current_task": "",
                "extract_memory": extract_memory,
                "turn_notes": [
                    "[supervisor] Empty plan (conversational/direct bypass) → FINISH. "
                    "No spokes scheduled; synthesizer will use chat context only."
                    + memory_note
                    + gate_note
                ],
            }

        # Pop the first planned step; remaining queue stays in state for later hops.
        head = plan[0]
        next_worker = head["worker"]
        current_task = head.get("task") or user_text
        remaining = plan[1:]
        workers_for_note = [next_worker] + [s["worker"] for s in remaining]
        return {
            "plan": remaining,
            "next_worker": next_worker,
            "current_task": current_task,
            "extract_memory": extract_memory,
            "turn_notes": [
                f"[supervisor] Plan generated from query + chat history: "
                f"{workers_for_note}. "
                f"Executing {next_worker} "
                f"(task={current_task!r}; caps={SPOKE_CAPS})."
                + memory_note
                + gate_note
            ],
        }

    # ----- Phase B: IN-FLIGHT — continue draining the existing plan -----
    already_ran = _spoke_execution_counts(turn_notes)
    # Re-sanitize remaining queue against how many times each spoke already ran.
    plan = _enforce_spoke_caps(plan, already_ran=already_ran)

    if not plan:
        # All planned steps done (or nothing left after cap filtering) → synthesize.
        return {
            "plan": [],
            "next_worker": "FINISH",
            "current_task": "",
            "turn_notes": [
                "[supervisor] Plan queue empty after spoke execution(s) → FINISH."
            ],
        }

    head = plan[0]
    next_worker = head["worker"]
    current_task = head.get("task") or user_text
    remaining = plan[1:]
    cap = SPOKE_CAPS.get(next_worker, 0)

    # Final defensive gate: never dispatch a spoke that already hit the per-turn cap.
    if already_ran.get(next_worker, 0) >= cap:
        return {
            "plan": [],
            "next_worker": "FINISH",
            "current_task": "",
            "turn_notes": [
                f"[supervisor] Spoke cap reached for {next_worker} "
                f"({cap}/turn). Forcing FINISH."
            ],
        }

    return {
        "plan": remaining,
        "next_worker": next_worker,
        "current_task": current_task,
        "turn_notes": [
            f"[supervisor] Continuing plan → {next_worker} "
            f"(task={current_task!r}). Remaining workers: "
            f"{[s['worker'] for s in remaining]}."
        ],
    }


# ---------------------------------------------------------------------------
# Spoke 1 — OpenSky Physical Airspace Engine (no SQL / no sqlglot)
# ---------------------------------------------------------------------------

OpenSkyToolName = Literal[
    "get_live_aircraft_telemetry",
    "get_airport_arrivals",
    "get_airport_departures",
    "get_aircraft_flight_history",
    "get_aircraft_trajectory_track",
]


class OpenSkyToolPlan(BaseModel):
    """Which OpenSky MCP tool to call + arguments (structured; not free-form)."""

    tool: OpenSkyToolName = Field(
        description=(
            "OpenSky tool to invoke. Use telemetry/track/history for icao24; "
            "use airport arrivals/departures for ICAO airport movement logs."
        )
    )
    icao24: str | None = Field(
        default=None,
        description="24-bit transponder hex (e.g. 3c675a). Required for aircraft tools.",
    )
    airport: str | None = Field(
        default=None,
        description="4-letter ICAO airport (e.g. VIDP, WSSS, EDDF). Required for airport tools.",
    )
    begin: str | int | None = Field(
        default=None,
        description="Window start: Unix epoch or ISO-8601. Optional; defaults applied in MCP.",
    )
    end: str | int | None = Field(
        default=None,
        description="Window end: Unix epoch or ISO-8601. Optional; defaults applied in MCP.",
    )
    bbox: str | None = Field(
        default=None,
        description="Optional bbox 'min_lat,max_lat,min_lon,max_lon' for telemetry.",
    )
    t: str | int | None = Field(
        default=0,
        description="Track time (epoch) for get_aircraft_trajectory_track; 0 = live/current.",
    )


def _heuristic_opensky_plan(user_text: str) -> OpenSkyToolPlan:
    """Regex/keyword fallback when Gemini structured planning is unavailable."""
    text = user_text.strip()
    lower = text.lower()
    now = int(time.time())

    # icao24: six hex chars, not a common English word
    hex_match = re.search(r"\b([0-9A-Fa-f]{6})\b", text)
    icao24 = hex_match.group(1).lower() if hex_match else None

    # Airport ICAO (4 letters) near arrival/departure words
    apt_match = re.search(
        r"\b(?:at|airport|arrivals?|departures?|into|out of)\s+([A-Za-z]{4})\b",
        text,
        flags=re.IGNORECASE,
    )
    airport = apt_match.group(1).upper() if apt_match else None
    if airport is None:
        bare_apt = re.search(r"\b([A-Z]{4})\b", text.upper())
        if bare_apt and bare_apt.group(1) not in {"THAT", "THIS", "FROM", "WITH", "HAVE"}:
            # Only treat as airport if arrivals/departures mentioned
            if "arrival" in lower or "departure" in lower:
                airport = bare_apt.group(1)

    if "track" in lower or "trajectory" in lower or "waypoint" in lower:
        return OpenSkyToolPlan(
            tool="get_aircraft_trajectory_track",
            icao24=icao24 or "3c675a",
            t=0,
        )
    if "arrival" in lower and airport:
        return OpenSkyToolPlan(
            tool="get_airport_arrivals",
            airport=airport,
            begin=now - 2 * 3600,
            end=now,
        )
    if "departure" in lower and airport:
        return OpenSkyToolPlan(
            tool="get_airport_departures",
            airport=airport,
            begin=now - 2 * 3600,
            end=now,
        )
    if "history" in lower or "legs" in lower or "past flights" in lower:
        return OpenSkyToolPlan(
            tool="get_aircraft_flight_history",
            icao24=icao24 or "3c675a",
            begin=now - 24 * 3600,
            end=now,
        )

    # Default physical ask → live telemetry
    return OpenSkyToolPlan(
        tool="get_live_aircraft_telemetry",
        icao24=icao24 or "3c675a",
        bbox=None,
    )


def _plan_opensky_tool(user_text: str) -> tuple[OpenSkyToolPlan, str]:
    """Ask Gemini which OpenSky tool to call; fall back to heuristics."""
    model = spoke_llm
    if model is not None:
        try:
            structured = model.with_structured_output(OpenSkyToolPlan)
            plan = structured.invoke(
                [
                    SystemMessage(
                        content=(
                            "You select ONE OpenSky Network tool for physical ADS-B airspace data.\n"
                            "Tools:\n"
                            "- get_live_aircraft_telemetry(icao24, bbox?): live vectors\n"
                            "- get_airport_arrivals(airport ICAO, begin, end)\n"
                            "- get_airport_departures(airport ICAO, begin, end)\n"
                            "- get_aircraft_flight_history(icao24, begin, end)\n"
                            "- get_aircraft_trajectory_track(icao24, t?)\n"
                            "DO NOT select tools for commercial flight numbers (AI8303), schedules, "
                            "A380 equipment, or METAR — those belong to Spoke 2.\n"
                            "icao24 must be 6 hex chars. Airports must be 4-letter ICAO."
                        )
                    ),
                    HumanMessage(content=user_text),
                ]
            )
            if isinstance(plan, dict):
                plan = OpenSkyToolPlan.model_validate(plan)
            if plan.tool not in OPENSKY_TOOLS:
                raise ValueError(f"Invalid tool {plan.tool}")
            return plan, "llm_structured"
        except Exception:
            logger.exception("OpenSky tool planning failed; using heuristic plan.")

    return _heuristic_opensky_plan(user_text), "heuristic_fallback"


def spoke_1_opensky_node(state: AgentState) -> dict:
    """Specialist 1: Physical Airspace Engine via OpenSky MCP tools.

    Plans one OpenSky tool call (structured LLM or heuristic), invokes the
    matching MCP tool function, and writes the JSON result into turn_notes.
    """
    user_text = _latest_user_text(state) or "live aircraft telemetry"
    history = _conversation_texts(state, limit=4)
    planning_text = user_text
    if history:
        planning_text = (
            f"Latest user query: {user_text}\n\n"
            f"Recent conversation context:\n" + "\n---\n".join(history)
        )

    # 1) Decide which OpenSky tool + args
    plan, source = _plan_opensky_tool(planning_text)

    # 2) Invoke the MCP tool implementation (in-process; same code as stdio server)
    kwargs: dict[str, Any] = {
        "icao24": plan.icao24,
        "airport": plan.airport,
        "begin": plan.begin,
        "end": plan.end,
        "bbox": plan.bbox,
        "t": plan.t,
    }
    result = invoke_opensky_tool(plan.tool, **kwargs)

    content = (
        f"[Spoke_1_OpenSky] source={source}\n"
        f"tool={plan.tool}\n"
        f"args={ {k: v for k, v in kwargs.items() if v is not None} }\n\n"
        f"OpenSky result:\n{result}"
    )
    return {"turn_notes": [content]}


# Backward-compatible alias if anything still imports the old name
spoke_1_sqlite_node = spoke_1_opensky_node


# ---------------------------------------------------------------------------
# Conversation-aware flight/tail resolution (Spoke 2 follow-up memory)
# ---------------------------------------------------------------------------

def _conversation_texts(state: AgentState, limit: int = 8) -> list[str]:
    """Newest-last plain-text snippets from durable chat messages.

    REVIEW: helper so spokes/synthesizer can see rolling chat history
    (app.py already passes messages; we just weren't using them for tools).
    """
    texts: list[str] = []
    for message in state.get("messages", [])[-limit:]:
        text = _message_text(message).strip()
        if text:
            texts.append(text)
    return texts


def _normalize_ident_text(user_text: str) -> str:
    """Uppercase + glue 'AI 8303' / 'EK 568' into AI8303 / EK568."""
    return re.sub(r"\b([A-Z]{2,3})\s+(\d{1,5})\b", r"\1\2", user_text.upper())


def _find_airline_flight_in_text(user_text: str) -> str | None:
    """Prefer marketed airline flight numbers (EK568, AI8303, SQ509)."""
    text = _normalize_ident_text(user_text)
    # Also accept slash pairs from web copy: EK568/569 → first leg EK568
    slash = re.search(r"\b([A-Z]{2,3})(\d{1,5})\s*/\s*\d{1,5}\b", text)
    if slash:
        candidate = f"{slash.group(1)}{slash.group(2)}"
        if candidate not in _FLIGHT_IDENT_STOPWORDS:
            return candidate
    flight = re.search(r"\b([A-Z]{2,3}\d{1,5})\b", text)
    if flight:
        candidate = flight.group(1)
        if candidate not in _FLIGHT_IDENT_STOPWORDS:
            return candidate
    return None


def _find_ident_in_text(user_text: str) -> str | None:
    """Find airline flight ident, US tail, registration, or ATC callsign.

    Commercial flight numbers are preferred over US N-tails: web notes often
    contain engine/token noise like 'N1' that must not beat 'EK568'.
    US N-numbers require ≥2 digits (N12 / N12345) — bare 'N1' is rejected.
    """
    text = _normalize_ident_text(user_text)

    # 1) Airline flight first (this product's primary LiveOps key)
    airline = _find_airline_flight_in_text(user_text)
    if airline:
        return airline

    # 2) US N-number — ≥2 digits so engine parameter 'N1' cannot win
    tail = re.search(r"\bN\d{2,5}[A-Z]{0,2}\b", text)
    if tail:
        return tail.group(0)

    # 3) Registration with hyphen: D-AIZZ, 9V-SKQ, A6-EEP, G-EUUY
    reg = re.search(r"\b([A-Z0-9]{1,2}-[A-Z0-9]{2,5})\b", text)
    if reg:
        return reg.group(1)

    # 4) ATC callsign e.g. DLH6K
    callsign = re.search(r"\b([A-Z]{3}\d[A-Z0-9]{0,3})\b", text)
    if callsign:
        candidate = callsign.group(1)
        if candidate not in _FLIGHT_IDENT_STOPWORDS:
            return candidate

    return None


def _resolve_flight_ident(state: AgentState) -> str | None:
    """Resolve flight/tail with priority that avoids cross-turn regex bleed.

    Order:
      1) `current_task` + latest user message
      2) `[Spoke_3_WebSearch]` notes — airline flights only (Tavily discovery)
      3) Prior messages only when the latest prompt has explicit follow-up phrases
    """
    current_task = str(state.get("current_task") or "").strip()
    latest = _latest_user_text(state) or ""

    for text in (current_task, latest):
        if not text:
            continue
        found = _find_ident_in_text(text)
        if found:
            return found

    # Web research notes: only accept airline idents (EK568), never N-tails/regs
    # from noisy article bodies (engine N1, special liveries A6-EEP, etc.).
    for note in state.get("turn_notes") or []:
        text = str(note)
        if text.startswith("[Spoke_3_WebSearch]"):
            found = _find_airline_flight_in_text(text)
            if found:
                return found

    if latest and _FOLLOWUP_IDENT_RE.search(latest):
        prior = list(state.get("messages") or [])[:-1]
        for message in reversed(prior):
            found = _find_ident_in_text(_message_text(message))
            if found:
                return found
    return None


def _extract_flight_ident(user_text: str) -> str:
    """Backward-compatible single-string extractor (tests / callers)."""
    return _find_ident_in_text(user_text) or "N12345"


def spoke_2_liveops_node(state: AgentState) -> dict:
    """Specialist 2: FlightAware commercial status + conditional NOAA METARs.

    Tavily web search lives on Spoke_3_WebSearch — not here.
    Ident resolution prefers current_task / latest message / Spoke 3 notes.
    """
    user_text = _latest_user_text(state) or "current flight operations"
    task = str(state.get("current_task") or "").strip() or user_text
    ident = _resolve_flight_ident(state)

    # Only hit FlightAware when we actually know which flight/tail
    if ident:
        fa_result = flightaware_live_status.invoke({"ident": ident})
    else:
        fa_result = (
            '{"source":"skipped","note":"No flight number or tail found in '
            'current_task, latest message, Spoke_3 notes, or (for follow-ups) '
            'prior chat."}'
        )

    content = (
        f"[Spoke_2_LiveOps]\n"
        f"Resolved ident from conversation: {ident or 'NONE'}\n"
        f"Current task: {task}\n"
        f"Latest user text: {user_text}\n"
        f"FlightAware ({ident or 'skipped'}):\n{fa_result}"
    )
    # Return to supervisor so Plan-and-Execute can pop the next plan step (or FINISH).
    return {"turn_notes": [content]}


def spoke_3_websearch_node(state: AgentState) -> dict:
    """Specialist 3: Tavily aviation web research (decoupled from LiveOps).

    Fast path (0 LLM): first spoke this turn → search `current_task` as-is.
    Chained path (1 spoke_llm): refine query using prior spoke tool outputs.
    """
    # Prefer supervisor's hop-local `current_task`; fall back to the latest user ask.
    user_text = _latest_user_text(state) or ""
    task_query = str(state.get("current_task") or "").strip() or user_text
    if not task_query:
        task_query = "aviation operations"

    # Did any spoke already write tool output THIS turn?
    # startswith(tuple) matches notes beginning with any of these prefixes.
    # Example: LiveOps ran first → we may refine Tavily using the FA ident it found.
    prior_spoke_outputs = [
        n
        for n in (state.get("turn_notes") or [])
        if str(n).startswith(("[Spoke_1", "[Spoke_2", "[Spoke_3"))
    ]

    # Default = fast path: use the supervisor task string unchanged (0 LLM calls).
    final_query = task_query
    refine_source = "task_direct"  # logged into turn_notes for metro / debugging

    if prior_spoke_outputs:
        # Chained path: something already ran (OpenSky / LiveOps / earlier WebSearch).
        # One Gemini call can fold concrete facts into a sharper Tavily query.
        model = spoke_llm
        if model is not None:
            try:
                # Cap size: last 4 spoke notes, each truncated, so the refine prompt stays small.
                prior_blob = "\n\n".join(str(n)[:1200] for n in prior_spoke_outputs[-4:])
                # Force shape {"refined_query": "..."} via WebSearchRefine (same idea as PlannedStep).
                structured = model.with_structured_output(WebSearchRefine)
                refined = structured.invoke(
                    [
                        SystemMessage(
                            content=(
                                "You refine a Tavily web-search query for aviation research.\n"
                                "Use concrete facts from prior spoke outputs "
                                "(flight numbers, aircraft types, tail regs, airports) "
                                "to sharpen the query. Do not repeat prior queries verbatim. "
                                "Keep the refined query short and searchable."
                            )
                        ),
                        HumanMessage(
                            content=(
                                f"Original task query:\n{task_query}\n\n"
                                f"Prior spoke outputs:\n{prior_blob}\n\n"
                                "Return a refined_query string."
                            )
                        ),
                    ]
                )
                # Some model wrappers return a dict instead of a Pydantic instance.
                if isinstance(refined, dict):
                    refined = WebSearchRefine.model_validate(refined)
                candidate = str(getattr(refined, "refined_query", "") or "").strip()
                if candidate:
                    final_query = candidate
                    refine_source = "llm_chained"
            except Exception:  # noqa: BLE001
                # Soft-fail: web search still runs with the original task_query.
                logger.exception(
                    "Spoke_3 query refine failed; using task_query unchanged."
                )
                refine_source = "task_fallback"

    # Always hit Tavily (mock path inside the tool if TAVILY_API_KEY is missing).
    tv_result = tavily_aviation_web_search.invoke({"query": final_query})

    # Prefix matters: Spoke 2's _resolve_flight_ident scans "[Spoke_3_WebSearch]"
    # notes to pick up flight numbers discovered by web research.
    content = (
        f"[Spoke_3_WebSearch] Query: {final_query}\n"
        f"Refine: {refine_source}\n"
        f"{tv_result}"
    )
    # Scratch only — supervisor pops the next plan step (or FINISH) on the next hop.
    return {"turn_notes": [content]}


def synthesizer_node(state: AgentState) -> dict:
    """Writer node: turn this turn's scratch notes into one final chat answer.

    REVIEW: now receives recent conversation + resolved flight/tail so
    follow-up answers stay about AI8303 (etc.), not a lost identifier.
    Also stronger rules around FlightAware headline / codeshares / mocks.
    Always appends a durable OpenSky vector block when Spoke 1 returned states,
    so follow-up turns can re-read callsign / icao24 from messages.
    """
    chunks: list[str] = [str(note) for note in (state.get("turn_notes") or [])]
    history = _conversation_texts(state, limit=6)  # REVIEW: chat context
    resolved = _resolve_flight_ident(state)  # REVIEW: bind pronouns to a flight
    opensky_block = _format_opensky_durable_block(chunks)

    model = spoke_llm
    if model is not None and chunks:
        try:
            response = model.invoke(
                [
                    SystemMessage(
                        content=(
                            "Synthesize a concise aviation intelligence report "
                            "from supervisor and spoke outputs.\n"
                            "Rules:\n"
                            # REVIEW: follow-up + codeshare + empty-SQLite rules
                            "- Use recent conversation context for follow-up questions "
                            "(pronouns like 'this flight', '2 months from now').\n"
                            f"- Active flight/tail in this conversation: {resolved or 'unknown'}.\n"
                            "- If FlightAware JSON has flights or a headline, treat that "
                            "as ground truth for live status, route, aircraft_type, and "
                            "registration. Never claim 'no live data' when flights exist.\n"
                            "- If queried_is_codeshare is true, say the query number is a "
                            "codeshare and report the operating flight + aircraft type.\n"
                            "- OpenSky ADS-B gaps do not mean a commercial flight "
                            "does not exist — prefer FlightAware for marketed flight numbers.\n"
                            "- When OpenSky live states exist, you MUST include icao24, "
                            "callsign, altitude, ground speed, vertical rate, track, "
                            "lat/lon, on_ground, and squawk in the answer (do not drop them).\n"
                            "- Do NOT invent marketed flight numbers (LHxxxx) or aircraft "
                            "types from callsign alone — only state what tools returned.\n"
                            "- Ignore mock payloads when source is flightaware_error_fallback "
                            "unless you clearly label them as mock.\n"
                            "- For future-schedule equipment questions, prefer Tavily / "
                            "schedule research; say when only typical fleet is known and "
                            "exact tail is assigned close to departure.\n"
                            "- Lead with status, aircraft, route, and times when relevant.\n"
                            "- NEVER paste raw turn_notes, JSON blobs, or '[Spoke_…]' / "
                            "'[supervisor]' lines. Write a clean passenger-facing answer.\n"
                            "- If FlightAware returned empty/error for a bad ident (e.g. N1) "
                            "but Tavily named real flights (e.g. EK568/EK569), trust Tavily "
                            "for the flight numbers and say live status was unavailable."
                        )
                    ),
                    HumanMessage(
                        content=(
                            # REVIEW: pass chat history into synthesis, not notes-only
                            "Recent conversation:\n"
                            + ("\n---\n".join(history) if history else "(none)")
                            + "\n\nSpoke / supervisor notes:\n"
                            + "\n\n".join(chunks[-8:])
                        )
                    ),
                ]
            )
            final_report = _llm_text(response)
            # If the model echoes scratch instead of synthesizing, use fallback.
            if (
                not final_report.strip()
                or final_report.lstrip().startswith(("[Spoke_", "[supervisor]"))
            ):
                final_report = _fallback_report(chunks)
        except Exception:  # noqa: BLE001
            logger.exception("Synthesizer LLM failed; using fallback report.")
            final_report = _fallback_report(chunks)
    else:
        final_report = _fallback_report(chunks)

    # Deterministic stamp so OpenSky ids survive into durable messages even if
    # the LLM paraphrases/omits fields (critical for follow-up Spoke 2 / regex).
    if opensky_block and opensky_block not in final_report:
        final_report = f"{final_report.rstrip()}\n\n{opensky_block}"

    # Memory Gate: Mem0 add ONLY when supervisor set extract_memory=True.
    # Standard aviation lookups skip ingestion to avoid wasted extract/embed calls.
    user_id = str(state.get("user_id") or "").strip()
    user_text = _latest_user_text(state)
    extract_memory = bool(state.get("extract_memory"))
    if extract_memory and user_id and user_text and final_report:
        interaction = [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": final_report},
        ]
        _add_user_memory(interaction, user_id)

    return {
        "final_report": final_report,
        "messages": [AIMessage(content=final_report)],
        "next_worker": "FINISH",
    }


def _extract_json_blobs_from_text(text: str) -> list[Any]:
    """Pull JSON objects/arrays embedded in spoke turn_notes."""
    blobs: list[Any] = []
    decoder = json.JSONDecoder()
    i = 0
    while i < len(text):
        if text[i] not in "{[":
            i += 1
            continue
        try:
            value, end = decoder.raw_decode(text, i)
            blobs.append(value)
            i = end
        except json.JSONDecodeError:
            i += 1
    return blobs


def _format_opensky_durable_block(chunks: list[str]) -> str | None:
    """Build a fixed OpenSky summary for messages (follow-up / regex friendly)."""
    joined = "\n".join(chunks)
    if "[Spoke_1_OpenSky]" not in joined:
        return None

    state: dict[str, Any] | None = None
    icao24 = None
    for blob in _extract_json_blobs_from_text(joined):
        if not isinstance(blob, dict) or blob.get("ok") is False:
            continue
        data = blob.get("data")
        if isinstance(data, dict) and isinstance(data.get("states"), list) and data["states"]:
            first = data["states"][0]
            if isinstance(first, dict):
                state = first
                icao24 = first.get("icao24") or blob.get("icao24")
                break
        if isinstance(data, dict) and data.get("icao24"):
            icao24 = data.get("icao24")

    if state is None and not icao24:
        # Still stamp tool error context so the gap is visible in chat history
        if "OpenSky result:" in joined and '"ok": false' in joined.replace(" ", ""):
            return None
        return None

    def _m_to_ft(meters: Any) -> str:
        if meters is None:
            return "—"
        try:
            return f"{int(round(float(meters) * 3.28084)):,} ft"
        except (TypeError, ValueError):
            return str(meters)

    def _ms_to_kt(ms: Any) -> str:
        if ms is None:
            return "—"
        try:
            return f"{int(round(float(ms) * 1.94384))} kt"
        except (TypeError, ValueError):
            return str(ms)

    def _ms_to_fpm(ms: Any) -> str:
        if ms is None:
            return "—"
        try:
            return f"{int(round(float(ms) * 196.85))} fpm"
        except (TypeError, ValueError):
            return str(ms)

    if state is None:
        return (
            "### OpenSky vector (durable)\n"
            f"- icao24: `{icao24}`\n"
            "- (no live state row in this turn)"
        )

    callsign = (str(state.get("callsign") or "").strip() or "—")
    lines = [
        "### OpenSky vector (durable)",
        f"- icao24: `{state.get('icao24') or icao24 or '—'}`",
        f"- callsign: `{callsign}`",
        f"- altitude: {_m_to_ft(state.get('baro_altitude'))}",
        f"- ground_speed: {_ms_to_kt(state.get('velocity'))}",
        f"- vertical_rate: {_ms_to_fpm(state.get('vertical_rate'))}",
        f"- true_track_deg: {state.get('true_track') if state.get('true_track') is not None else '—'}",
        f"- lat: {state.get('latitude') if state.get('latitude') is not None else '—'}",
        f"- lon: {state.get('longitude') if state.get('longitude') is not None else '—'}",
        f"- on_ground: {state.get('on_ground') if state.get('on_ground') is not None else '—'}",
        f"- squawk: {state.get('squawk') or '—'}",
    ]
    return "\n".join(lines)


def _fallback_report(chunks: list[str]) -> str:
    """Build a short readable report when the synthesizer LLM is unavailable."""
    if not chunks:
        return (
            "Aviation Intelligence Report\n"
            "No worker outputs were available to synthesize."
        )

    # Prefer airline idents discovered in web notes over dumping raw JSON.
    web_idents: list[str] = []
    for note in chunks:
        if str(note).startswith("[Spoke_3_WebSearch]"):
            # Collect a few distinct flight numbers from the Tavily payload text.
            text = _normalize_ident_text(str(note))
            for match in re.finditer(r"\b([A-Z]{2,3}\d{1,5})\b", text):
                cand = match.group(1)
                if cand in _FLIGHT_IDENT_STOPWORDS or cand in web_idents:
                    continue
                web_idents.append(cand)
                if len(web_idents) >= 6:
                    break

    lines = [
        "Aviation Fleet & Operational Intelligence Report",
        "==============================================",
    ]
    if web_idents:
        lines.append(
            "Flight numbers found in web research: " + ", ".join(web_idents)
        )
        lines.append(
            "(Live FlightAware status was unavailable or used a bad ident; "
            "re-ask with a specific flight number like EK568 for live status.)"
        )
    else:
        # Last resort: truncated plain text without supervisor routing noise
        useful = [
            str(c)
            for c in chunks[-4:]
            if str(c).startswith(("[Spoke_1", "[Spoke_2", "[Spoke_3"))
        ]
        body = "\n\n".join(useful) if useful else "\n\n".join(str(c) for c in chunks[-3:])
        if len(body) > 1800:
            body = body[:1800].rstrip() + "…"
        lines.append(body)
    lines.append("==============================================")
    lines.append("End of report.")
    report = "\n".join(lines)
    block = _format_opensky_durable_block(chunks)
    if block:
        report = f"{report}\n\n{block}"
    return report
