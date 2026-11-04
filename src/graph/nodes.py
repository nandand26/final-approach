"""LangGraph node implementations for supervisor and spoke workers.

This file is the "brains" of each graph step:
- Helpers: small utilities (read user text, pick an LLM, keyword routing, etc.)
- Nodes: supervisor (fan-out plan), opensky, flightaware, tavily,
  synthesizer (priority-packed fan-in). NOAA METARs enrich FlightAware
  only when AeroAPI reports delay/hold/divert.

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
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field, field_validator

from src.graph.state import (
    MESSAGE_WINDOW,
    SPOKE_CAPS,
    VALID_FANOUT_SPOKES,
    WORKER_TO_FANOUT,
    AgentState,
    task_key, # Task key is a unique identifier for a task. It is used to identify the task and to avoid duplicate tasks.
)
from src.graph.models import (
    SupervisorPlan,
    get_spoke_llm,
    get_supervisor_llm,
)
from src.mcp_servers.opensky_mcp import OPENSKY_TOOLS, invoke_opensky_tool
from src.tools.live_ops import (
    flightaware_live_status,
    tavily_aviation_web_search,
)

logger = logging.getLogger(__name__)

# Spoke ids are VALID_FANOUT_SPOKES: opensky, flightaware, tavily.
# WORKER_TO_FANOUT is the only alias table. It rewrites Spoke_1_OpenSky,
# noaa, and the other old names into those three ids before any check.

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


def _note_spoke(note: Any) -> str:
    """Canonical fan-out id on a turn_note dict (or legacy string).

    Dict notes already store the id in ``spoke``. Old string notes only have a
    bracket tag such as ``[Spoke_2_LiveOps]``. Both go through WORKER_TO_FANOUT.
    """
    if isinstance(note, dict):
        raw = str(note.get("spoke") or "").strip()
        return WORKER_TO_FANOUT.get(raw, raw.lower())
    for tag in re.findall(r"\[([^\]]+)\]", str(note)): # 
        mapped = WORKER_TO_FANOUT.get(tag.strip())
        if mapped:
            return mapped
    return ""


def _note_content(note: Any) -> str:
    """Plain text payload of a turn_note dict or legacy string."""
    if isinstance(note, dict):
        return str(note.get("content") or "")
    return str(note)


def _spoke_note(spoke: str, content: str) -> dict[str, str]:
    """Pack a spoke/supervisor note for operator.add fan-in."""
    return {"spoke": spoke, "content": content}


def _empty_spoke_counts() -> dict[str, int]:
    return {name: 0 for name in VALID_FANOUT_SPOKES}


def _spoke_execution_counts(turn_notes: list) -> dict[str, int]:
    """Count how many times each spoke already ran in THIS user-message turn."""
    counts = _empty_spoke_counts()
    for note in turn_notes:
        spoke = _note_spoke(note)
        if spoke in counts:
            counts[spoke] += 1
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
        raw = _step_worker(step)
        worker = WORKER_TO_FANOUT.get(raw, raw.strip().lower())
        if worker not in VALID_FANOUT_SPOKES:
            continue
        task = _step_task(step)
        normalized.append({"worker": worker, "task": task})
    return normalized


def _enforce_spoke_caps(
    plan: list[Any],
    already_ran: dict[str, int] | None = None,
) -> list[dict[str, str]]:
    """Defensive plan sanitizer: valid spokes, no duplicate calls, per-worker caps.

    A row is one spoke call. Identity is (worker, task_key(task)):
    - An exact duplicate of an earlier row is dropped. The FIRST row is kept
      exactly as written. task_key is only a comparison key; it is never stored.
    - SPOKE_CAPS counts DISTINCT tasks per worker (plus already_ran). Different
      tasks for the same worker are all kept, up to the cap, in plan order.
    - LLMs can still emit duplicates, invalid names, or too many calls.
    """
    already_ran = already_ran or _empty_spoke_counts()
    scheduled = _empty_spoke_counts()
    seen: set[tuple[str, str]] = set()
    cleaned: list[dict[str, str]] = []

    for step in _normalize_plan_steps(plan):
        worker = step["worker"]
        call = (worker, task_key(step["task"]))
        if call in seen:
            continue
        cap = SPOKE_CAPS.get(worker, 0)
        total_if_added = already_ran.get(worker, 0) + scheduled.get(worker, 0) + 1
        if total_if_added > cap:
            continue
        seen.add(call)
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
    # Same three paths as the supervisor prompt. No ident means Tavily only.
    if has_flight_ident and wants_live:
        plan.append({"worker": "flightaware", "task": task})
        plan.append({"worker": "opensky", "task": task})
        if wants_web or any(
            token in text for token in ("notam", "news", "advisory", "recall", "press")
        ):
            plan.append({"worker": "tavily", "task": task})
    elif wants_opensky:
        plan.append({"worker": "opensky", "task": task})
    elif not has_flight_ident and (wants_live or airline_route_without_ident or wants_web):
        plan.append({"worker": "tavily", "task": task})

    return _enforce_spoke_caps(plan)


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


def _unique_fanout(names: list[str]) -> list[str]:
    """Keep first-seen valid fan-out ids only.

    Review: this is the allow-list used before Send. Aliases such as
    Spoke_2_LiveOps or noaa are rewritten by WORKER_TO_FANOUT, then anything
    that is still not opensky|flightaware|tavily is dropped. Duplicates go too.
    """
    seen: set[str] = set()
    out: list[str] = []
    for raw in names:
        worker = WORKER_TO_FANOUT.get(str(raw or "").strip(), str(raw or "").strip().lower())
        if worker not in VALID_FANOUT_SPOKES or worker in seen:
            continue
        seen.add(worker)
        out.append(worker)
    return out


def _default_context_priority_order(spokes: list[str]) -> list[str]:
    """Pin FlightAware / Tavily at the start when selected (safety-critical edges).

    Review: pack order for the synthesizer, not the order spokes execute.
    FlightAware may carry NOAA METARs on delay; Tavily carries NOTAMs.
    Those two are placed first so they are not buried in the middle of the prompt.
    OpenSky, when selected, follows them.
    """
    selected = _unique_fanout(spokes)
    safety = [s for s in ("flightaware", "tavily") if s in selected]
    rest = [s for s in selected if s not in safety]
    return safety + rest


def _sanitize_context_priority_order(order: Any) -> list[str]:
    """Keep only real spoke ids from the model's pack-order list.

    Review: strict match only, no alias rewrite. "noaa" or "synthesizer" is
    dropped here. A duplicate id is kept once, in the original order.
    """
    seen: set[str] = set()
    out: list[str] = []
    for item in list(order or []):
        name = str(item).strip()
        if name not in VALID_FANOUT_SPOKES or name in seen:
            continue
        seen.add(name)
        out.append(name)
    return out


def _plan_from_spokes(
    spokes: list[str],
    user_text: str,
    plan_steps: list[dict[str, str]] | None = None,
    *,
    flight_number: str | None = None,
    search_query: str | None = None,
) -> list[dict[str, str]]:
    """One {worker, task} row for every spoke CALL the router will Send.

    Review: the supervisor's own rows win, in plan order. The same worker may
    appear more than once with different tasks. Exact duplicates and the cap
    are already applied by _enforce_spoke_caps before this runs. A selected
    spoke that has no row at all gets one fallback row: FlightAware gets the
    ident, Tavily gets the search query, OpenSky gets the user question. The
    returned list is what state["plan"] stores.
    """
    wanted = _unique_fanout(spokes)
    rows: list[dict[str, str]] = []
    for step in plan_steps or []:
        worker = WORKER_TO_FANOUT.get(step.get("worker", ""), step.get("worker", ""))
        task = str(step.get("task") or "").strip()
        # Skip a row for a spoke that is not being sent, or a blank task.
        # The fallback below fills a spoke that ends up with no row.
        if worker not in wanted or not task:
            continue
        rows.append({"worker": worker, "task": task})

    ident = (flight_number or "").strip()
    query = (search_query or "").strip()
    fallback = (user_text or "").strip()

    def _default_task(spoke: str) -> str:
        if spoke == "flightaware" and ident:
            return ident
        if spoke == "tavily" and query:
            return query
        return fallback

    have = {row["worker"] for row in rows}
    for spoke in wanted:
        if spoke not in have:
            rows.append({"worker": spoke, "task": _default_task(spoke)})
    return rows


def _task_for_spoke(state: AgentState, spoke: str) -> str:
    """Task string for the spoke call that is running now.

    Review: lookup order is the Send payload's own "task" (the router puts the
    plan row's task there), then that spoke's first plan row, then current_task,
    then the latest user message. OpenSky, FlightAware, and Tavily all call
    this. Fan-in does not, and it has no payload task.
    """
    sent = str(state.get("task") or "").strip()
    if sent:
        return sent
    for step in _normalize_plan_steps(state.get("plan") or []):
        if step.get("worker") == spoke and step.get("task"):
            return str(step["task"])
    return (
        str(state.get("current_task") or "").strip()
        or _latest_user_text(state)
        or ""
    )


def _reorder_turn_notes(notes: list, priority_order: list[str]) -> list:
    """Sort turn_notes to match sanitized context_priority_order.

    Review: this is fan-in packing, after every spoke has written its note.
    1) Bucket notes by spoke id. Supervisor notes are leftovers.
    2) Emit buckets in the model's pack order.
    3) A spoke that returned data but was missing from that order is appended,
       so a payload is not dropped.
    4) Supervisor and other non-spoke notes go last.
    """
    order = _sanitize_context_priority_order(priority_order)
    buckets: dict[str, list] = {name: [] for name in VALID_FANOUT_SPOKES}
    leftovers: list = []
    for note in list(notes or []):
        spoke = _note_spoke(note)
        if spoke in buckets:
            buckets[spoke].append(note)
        else:
            leftovers.append(note)

    ordered: list = []
    used: set[str] = set()
    for spoke in order:
        ordered.extend(buckets.get(spoke) or [])
        used.add(spoke)
    # Spoke returned data but was left out of context_priority_order.
    for spoke in VALID_FANOUT_SPOKES:
        if spoke not in used and buckets.get(spoke):
            ordered.extend(buckets[spoke])
    ordered.extend(leftovers)
    return ordered


def _format_notes_markdown(notes: list) -> str:
    """Turn the sorted notes into one markdown string for the synthesizer LLM.

    Review: each section is "## <spoke id>" plus the note text. This string
    is the only tool output the writer model sees.
    """
    sections: list[str] = []
    for note in notes:
        spoke = _note_spoke(note) or "note"
        content = _note_content(note).strip()
        if not content:
            continue
        sections.append(f"## {spoke}\n\n{content}")
    return "\n\n".join(sections)


_SUPERVISOR_INTENTS = (
    "telemetry_known",
    "telemetry_discovery",
    "general_knowledge",
    "direct_chat",
)


def _offline_supervisor(
    user_text: str,
    history_blob: str,
) -> tuple[list[dict[str, str]], list[str], list[str], bool, str, str | None, str | None]:
    """Keyword plan used only when the supervisor model is unavailable.

    Does not rewrite a model decision. Labels intent from the spokes this
    fallback itself chose, using the same three paths as the prompt.
    """
    plan = _heuristic_plan(user_text, history_blob)
    spokes = _unique_fanout([step["worker"] for step in plan])
    ident = _find_ident_in_text(user_text)
    blob = f"{user_text}\n{history_blob}".lower()
    live_route = any(
        token in blob
        for token in ("status", "track", "live", "delay", "eta", "flight from", "flights from")
    )
    if not spokes:
        intent, flight_number, search_query = "direct_chat", None, None
    elif "flightaware" in spokes or "opensky" in spokes:
        intent = "telemetry_known"
        flight_number = ident
        search_query = user_text if "tavily" in spokes else None
    elif live_route:
        intent = "telemetry_discovery"
        flight_number, search_query = None, user_text
    else:
        intent = "general_knowledge"
        flight_number, search_query = None, user_text
    plan = _plan_from_spokes(
        spokes,
        user_text,
        plan,
        flight_number=flight_number,
        search_query=search_query,
    )
    return (
        plan,
        spokes,
        _default_context_priority_order(spokes),
        _heuristic_extract_memory(user_text),
        intent,
        flight_number,
        search_query,
    )


def _llm_generate_plan(
    user_text: str,
    history_blob: str,
    memory_facts: list[str] | None = None,
) -> tuple[list[dict[str, str]], list[str], list[str], bool, str, str | None, str | None]:
    """Ask the supervisor LLM for intent, spokes, pack order, and Memory Gate.

    Returns (plan, spokes, priority, extract_memory, intent, flight_number, search_query).
    """
    model = get_supervisor_llm()
    if model is None:
        return _offline_supervisor(user_text, history_blob)

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
        "You are the aviation fleet intelligence supervisor. You alone choose the "
        "path. Classify EVERY turn into exactly one intent, then choose spokes and "
        "the task each spoke runs. Nothing downstream will change your intent or "
        "your spoke list.\n\n"
        "HARD RULES:\n"
        "1. Known ident. The latest message contains a flight ident (BA117, SQ32, "
        "ID8085), or the user just replied with one. intent is telemetry_known. "
        "You choose which of opensky, flightaware, and tavily are needed. "
        "Every spoke you put in spokes_to_call runs in parallel. "
        "flight_number is that ident.\n"
        "2. Unknown ident. Live status by airline or route with NO ident "
        "(Batik Air from KL to BLR). intent is telemetry_discovery. "
        "spokes_to_call is exactly [tavily]. flight_number is null. "
        "Never flightaware. Never opensky. Tavily finds candidates, then a human "
        "picks the ident. Do not pick one yourself.\n"
        "3. General research. Regulations, specs, theory, or news with no live "
        "tracking. intent is general_knowledge. spokes_to_call is exactly [tavily]. "
        "Never flightaware. Never opensky. flight_number is null.\n"
        "4. Greeting or reformat, or anything the chat already answers. "
        "intent is direct_chat. spokes_to_call is [] and plan is [].\n\n"
        "EXAMPLES:\n"
        "- 'Status of BA117' → telemetry_known, flight_number BA117, "
        "the spokes you need (for example flightaware, or flightaware + opensky).\n"
        "- 'Status of BA117 and Heathrow NOTAMs' → telemetry_known, "
        "flightaware and tavily in parallel.\n"
        "- 'Status of Batik Air from KL to BLR' → telemetry_discovery, "
        "[tavily] only, flight_number null.\n"
        "- User then replies 'ID8085' → telemetry_known for ID8085, "
        "whatever spokes that ident needs, in parallel.\n"
        "- 'What is a METAR?' → general_knowledge, [tavily].\n"
        "- 'Hi' → direct_chat, no spokes.\n\n"
        "CONTEXT:\n"
        "- Read the full chat history before classifying. Resolve 'that flight' "
        "and a bare ident from the earlier turn.\n"
        f"{memory_block}"
        "SPOKE CAPABILITIES:\n"
        "- opensky = ADS-B by icao24 hex or airport ICAO. Soft-fails without a hex.\n"
        "- flightaware = AeroAPI status. NOAA METARs run INSIDE this spoke only "
        "on delay/hold/divert. Requires a flight ident.\n"
        "- tavily = web research, NOTAMs, and flight-number discovery.\n\n"
        "PLAN (`plan`) — required, one row per spoke CALL:\n"
        "- worker is the spoke id. task is the exact text that call must run.\n"
        "- flightaware task = the flight ident only (BA117).\n"
        "- opensky task = the icao24 hex or airport ICAO question.\n"
        "- tavily task = the web, NOTAM, or discovery query.\n"
        "- List each spoke once in spokes_to_call. Every spoke in spokes_to_call "
        "needs at least one plan row, and plan rows may use only those spokes. "
        "Do not omit plan when spokes_to_call is non-empty.\n"
        "- You may plan more than one call to the same spoke ONLY when the tasks "
        "are genuinely different (for example flightaware BA117 and flightaware "
        "BA118), up to the cap. NEVER repeat an identical task for the same spoke.\n"
        "- telemetry_discovery and general_knowledge stay Tavily only. Several "
        "tavily rows are allowed when the tasks differ, up to the cap. Never add "
        "opensky or flightaware on those intents.\n\n"
        "Also return flight_number (ident or null) and search_query "
        "(Tavily string, or null when Tavily is not selected).\n"
        f"Cap on DISTINCT tasks per spoke ({ {k: SPOKE_CAPS[k] for k in VALID_FANOUT_SPOKES} }).\n"
        "NEVER emit noaa, FINISH, or synthesizer as a spoke.\n\n"
        "CONTEXT PRIORITY ORDER (pack order, not execution order):\n"
        "- If flightaware and/or tavily are selected, put them at the absolute "
        "start OR end of context_priority_order (lost-in-the-middle). "
        "Only ids: opensky, flightaware, tavily.\n\n"
        "MEMORY GATE (`extract_memory`):\n"
        "- true ONLY when the latest message explicitly sets how you should "
        "behave, format answers, or remember personal context.\n"
        "- false for flight lookups, ADS-B, weather, news, and chat.\n"
        "- Existing Mem0 facts do not make this turn true. When in doubt, false."
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
                        "Return a SupervisorPlan with intent, flight_number, search_query, "
                        "spokes_to_call, plan (one {worker, task} per spoke call), "
                        "context_priority_order, and extract_memory."
                    )
                ),
            ]
        )
        if isinstance(result, dict):
            result = SupervisorPlan.model_validate(result)
        raw_plan = _enforce_spoke_caps(list(getattr(result, "plan", []) or []))
        spokes = _unique_fanout(list(getattr(result, "spokes_to_call", []) or []))
        if not spokes and raw_plan:
            spokes = _unique_fanout([step["worker"] for step in raw_plan])
        intent = str(getattr(result, "intent", "") or "").strip()
        if intent not in _SUPERVISOR_INTENTS:
            intent = "direct_chat"
        # Discovery and research are Tavily only. A greeting runs nothing.
        # A known ident keeps whatever spokes the supervisor selected.
        if intent in {"telemetry_discovery", "general_knowledge"}:
            spokes = ["tavily"]
        elif intent == "direct_chat":
            spokes = []
        flight_number = (getattr(result, "flight_number", None) or "").strip() or None
        search_query = (getattr(result, "search_query", None) or "").strip() or None
        plan = _plan_from_spokes(
            spokes,
            user_text,
            raw_plan,
            flight_number=flight_number,
            search_query=search_query,
        )
        priority = _sanitize_context_priority_order(
            getattr(result, "context_priority_order", None)
        )
        if not priority:
            priority = _default_context_priority_order(spokes)
        selected = set(spokes)
        if ("flightaware" in selected or "tavily" in selected) and priority:
            mid = priority[1:-1] if len(priority) > 2 else []
            if "flightaware" in mid or "tavily" in mid:
                priority = _default_context_priority_order(spokes)
        extract_memory = bool(getattr(result, "extract_memory", False))
        return plan, spokes, priority, extract_memory, intent, flight_number, search_query
    except Exception:
        logger.exception("Supervisor plan generation failed; using heuristic plan.")
        return _offline_supervisor(user_text, history_blob)


# ---------------------------------------------------------------------------
# Graph nodes (these are the functions wired in workflow.py)
# ---------------------------------------------------------------------------

def supervisor_node(state: AgentState) -> dict:
    """Classify the turn, then expose spokes for Send / discovery / bypass.

    Routing itself lives in workflow._route_after_supervisor (reads `intent`).
    Discovery does not call FlightAware until the human picks an ident and
    this node runs again.
    """
    user_text = _latest_user_text(state)
    user_id = str(state.get("user_id") or "").strip()

    history_parts = [
        _message_text(m) for m in state.get("messages", [])[-MESSAGE_WINDOW:]
    ]
    history_blob = "\n---\n".join(part for part in history_parts if part.strip())

    memory_facts = _search_user_memories(user_text, user_id, limit=3)
    memory_note = ""
    if memory_facts:
        memory_note = " Mem0 prefs: " + "; ".join(memory_facts[:3]) + "."

    plan, spokes, priority, extract_memory, intent, flight_number, search_query = (
        _llm_generate_plan(user_text, history_blob, memory_facts=memory_facts)
    )
    plan = _enforce_spoke_caps(plan, already_ran=_empty_spoke_counts())
    spokes = _unique_fanout(spokes or [s["worker"] for s in plan])
    plan = _plan_from_spokes(
        spokes,
        user_text,
        plan,
        flight_number=flight_number,
        search_query=search_query,
    )
    priority = _sanitize_context_priority_order(priority) or _default_context_priority_order(
        spokes
    )
    gate_note = f" extract_memory={extract_memory}."
    current_task = (plan[0].get("task") if plan else "") or (search_query or user_text)

    if intent == "direct_chat" or not spokes:
        note = (
            "[supervisor] Empty plan (conversational/direct bypass) → FINISH. "
            f"intent={intent or 'direct_chat'}."
            + memory_note
            + gate_note
        )
        return {
            "intent": "direct_chat" if not spokes else intent,
            "flight_number": flight_number,
            "search_query": search_query,
            "plan": [],
            "spokes_to_call": [],
            "context_priority_order": [],
            "next_worker": "FINISH",
            "current_task": "",
            "extract_memory": extract_memory,
            "turn_notes": [_spoke_note("supervisor", note)],
        }

    task_bits = ", ".join(f"{step['worker']}={step['task']!r}" for step in plan)
    note = (
        f"[supervisor] Plan generated from query + chat history: {spokes}. "
        f"intent={intent}. flight_number={flight_number or 'NONE'}. "
        f"Tasks: {task_bits}. Priority {priority}."
        + memory_note
        + gate_note
    )
    return {
        "intent": intent,
        "flight_number": flight_number,
        "search_query": search_query,
        "plan": plan,
        "spokes_to_call": spokes,
        "context_priority_order": priority,
        "next_worker": spokes[0],
        "current_task": current_task,
        "extract_memory": extract_memory,
        "turn_notes": [_spoke_note("supervisor", note)],
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


# Wall-clock cap for Gemini OpenSky tool planning. Gemini 429 retries can
# otherwise stall Spoke 1 for minutes (SDK has no reliable default timeout).
_OPENSKY_PLAN_LLM_TIMEOUT_SEC = 20.0


def _extract_opensky_icao24(user_text: str) -> str | None:
    """Six-char ADS-B hex, not a marketed airline flight like AI101."""
    text = user_text.strip()
    # Prefer explicit hex that is not purely letters+digits flight-shaped
    for match in re.finditer(r"\b([0-9A-Fa-f]{6})\b", text):
        token = match.group(1)
        # Airline idents like "AI0101" are rare; real hex usually has a-f or mixed.
        if re.fullmatch(r"[A-Za-z]{2,3}\d{1,4}", token):
            continue
        return token.lower()
    return None


def _heuristic_opensky_plan(user_text: str) -> OpenSkyToolPlan:
    """Regex/keyword fallback when Gemini structured planning is unavailable."""
    text = user_text.strip()
    lower = text.lower()
    now = int(time.time())

    icao24 = _extract_opensky_icao24(text)

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
            icao24=icao24,
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
            icao24=icao24,
            begin=now - 24 * 3600,
            end=now,
        )

    # Default physical ask → live telemetry (icao24 may be None → soft error)
    return OpenSkyToolPlan(
        tool="get_live_aircraft_telemetry",
        icao24=icao24,
        bbox=None,
    )


def _plan_opensky_tool(user_text: str) -> tuple[OpenSkyToolPlan, str]:
    """Ask Gemini which OpenSky tool to call; fall back to heuristics.

    Skips the LLM when there is no icao24/airport signal (commercial flight
    numbers alone cannot key OpenSky) so Spoke 1 does not stall on Gemini.
    """
    lower = user_text.lower()
    has_hex = _extract_opensky_icao24(user_text) is not None
    has_airport = bool(
        re.search(
            r"\b(?:at|airport|arrivals?|departures?|into|out of)\s+[A-Za-z]{4}\b",
            user_text,
            flags=re.IGNORECASE,
        )
        or (
            ("arrival" in lower or "departure" in lower)
            and re.search(r"\b[A-Z]{4}\b", user_text.upper())
        )
    )
    if not has_hex and not has_airport:
        return _heuristic_opensky_plan(user_text), "heuristic_no_icao24"

    model = get_spoke_llm()
    if model is not None:
        from concurrent.futures import ThreadPoolExecutor
        from concurrent.futures import TimeoutError as FuturesTimeout

        try:
            structured = model.with_structured_output(OpenSkyToolPlan)

            def _invoke() -> Any:
                return structured.invoke(
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

            with ThreadPoolExecutor(max_workers=1) as pool:
                plan = pool.submit(_invoke).result(timeout=_OPENSKY_PLAN_LLM_TIMEOUT_SEC)
            if isinstance(plan, dict):
                plan = OpenSkyToolPlan.model_validate(plan)
            if plan.tool not in OPENSKY_TOOLS:
                raise ValueError(f"Invalid tool {plan.tool}")
            return plan, "llm_structured"
        except FuturesTimeout:
            logger.warning(
                "OpenSky tool planning exceeded %.0fs; using heuristic plan.",
                _OPENSKY_PLAN_LLM_TIMEOUT_SEC,
            )
        except Exception:
            logger.exception("OpenSky tool planning failed; using heuristic plan.")

    return _heuristic_opensky_plan(user_text), "heuristic_fallback"


def spoke_1_opensky_node(state: AgentState) -> dict:
    """Specialist 1: Physical Airspace Engine via OpenSky MCP tools.

    Plans one OpenSky tool call (structured LLM or heuristic), invokes the
    matching MCP tool function, and writes the JSON result into turn_notes.
    """
    try:
        user_text = _task_for_spoke(state, "opensky") or "live aircraft telemetry"
        history = _conversation_texts(state, limit=MESSAGE_WINDOW)
        planning_text = user_text
        if history:
            planning_text = (
                f"Latest user query: {user_text}\n\n"
                f"Recent conversation context:\n" + "\n---\n".join(history)
            )

        plan, source = _plan_opensky_tool(planning_text)
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
        return {"turn_notes": [_spoke_note("opensky", content)]}
    except Exception as exc:  # noqa: BLE001
        logger.exception("OpenSky spoke failed; returning error note.")
        return {
            "turn_notes": [
                _spoke_note("opensky", f"[opensky] error: {exc}")
            ]
        }


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


class CandidateFlights(BaseModel):
    """Marketed flight idents pulled from Tavily notes. Extraction only."""

    idents: list[str] = Field(
        default_factory=list,
        description="Up to 6 marketed flight identifiers, such as ID8085 or BA117.",
    )

    @field_validator("idents")
    @classmethod
    def _cap_idents(cls, value: list[str]) -> list[str]:
        """Keep the first 6 distinct idents. A longer list must not fail validation."""
        found: list[str] = []
        for ident in value or []:
            text = str(ident).strip()
            if text and text not in found:
                found.append(text)
            if len(found) >= 6:
                break
        return found


def _extract_candidate_flights(notes: list) -> list[str]:
    """Ask Gemini for idents in the raw Tavily notes. It does not write the question."""
    raw_notes = [
        _note_content(note).strip()
        for note in notes or []
        if _note_spoke(note) == "tavily" and _note_content(note).strip()
    ]
    if not raw_notes:
        return []
    model = get_spoke_llm()
    if model is None:
        return []
    try:
        structured = model.with_structured_output(CandidateFlights)
        result = structured.invoke(
            [
                SystemMessage(
                    content=(
                        "Extract marketed flight identifiers from the research notes. "
                        "Return at most 6 idents, for example ID8085 or BA117. "
                        "Return only the ident strings in `idents`. "
                        "Do not write a question, sentence, or explanation."
                    )
                ),
                HumanMessage(content="\n\n".join(raw_notes)),
            ]
        )
        if isinstance(result, dict):
            result = CandidateFlights.model_validate(result)
        return list(getattr(result, "idents", []) or [])
    except Exception:
        logger.exception("Candidate flight extraction failed.")
        return []


def clarification_node(state: AgentState) -> Command:
    """Path 2 pause: show Tavily candidates, then loop back to the supervisor.

    Gemini only extracts idents. This node fills a fixed question string.
    `interrupt()` checkpoints here. Streamlit resumes with Command(resume=...).
    The human reply is the only new message — discovery JSON stays out of `messages`.
    No static edge leaves this node; `Command(goto="supervisor")` is the only route.
    """
    idents = _extract_candidate_flights(list(state.get("turn_notes") or []))
    if idents:
        question = (
            "I found these candidate flights: "
            f"{', '.join(idents)}. Reply with the exact identifier to track."
        )
    else:
        question = (
            "I couldn't pick a flight number from web research. "
            "Reply with the Flight Number or Tail Number to track (for example BA117)."
        )
    # Pause. interrupt() checkpoints this thread and does not continue to the
    # supervisor. Example payload after Tavily finds two Batik flights:
    #   {"question": "I found these candidate flights: ID8085, ID8086. "
    #    "Reply with the exact identifier to track.",
    #    "options": ["ID8085", "ID8086"]}
    #
    # Streamlit reads it on the next rerun: app_graph.get_state() exposes
    # snapshot.interrupts, and _interrupt_payload pulls out this dict.
    # _format_interrupt_markdown shows `question` in an assistant bubble.
    # The idents are already inside that sentence, so `options` is not drawn
    # again as buttons or a second list. The chat box placeholder becomes
    # "Reply with a flight ident...".
    #
    # Resume is not a new turn. The user types ID8085 and Streamlit streams
    # Command(resume="ID8085"). That string is what interrupt() returns here
    # as user_selection.
    #
    # Hand-off: store HumanMessage("ID8085"), clear turn_notes, and
    # Command(goto="supervisor"). The next rerun shows ID8085 as the user
    # bubble. The supervisor classifies it again, usually telemetry_known,
    # and can Send FlightAware and OpenSky.
    user_selection = interrupt({"question": question, "options": idents})
    selection = str(user_selection or "").strip() or "none"
    return Command(
        goto="supervisor",
        update={
            "messages": [HumanMessage(content=selection)],
            "turn_notes": [],
        },
    )


def _resolve_flight_ident(state: AgentState) -> str | None:
    """Resolve flight/tail with priority that avoids cross-turn regex bleed.

    Order:
      0) This Send call's own task, when it contains an ident. Two FlightAware
         calls for different flights must not both fall back to `flight_number`.
      1) Supervisor `flight_number` (typed by the user or chosen at clarification)
      2) `current_task` + latest user message
      3) Tavily notes from this turn (usually empty under parallel Send)
      4) Prior messages only when the latest prompt has explicit follow-up phrases
    """
    sent = str(state.get("task") or "").strip()
    if sent:
        found_in_task = _find_ident_in_text(sent)
        if found_in_task:
            return found_in_task

    explicit = str(state.get("flight_number") or "").strip()
    if explicit:
        return _find_ident_in_text(explicit) or explicit.upper()

    flightaware_task = _task_for_spoke(state, "flightaware")
    current_task = str(state.get("current_task") or "").strip()
    latest = _latest_user_text(state) or ""

    for text in (flightaware_task, current_task, latest):
        if not text:
            continue
        found = _find_ident_in_text(text)
        if found:
            return found

    # Web research notes: only accept airline idents (EK568), never N-tails/regs
    # from noisy article bodies (engine N1, special liveries A6-EEP, etc.).
    for note in state.get("turn_notes") or []:
        spoke = _note_spoke(note)
        text = _note_content(note)
        if spoke == "tavily" or text.startswith("[Spoke_3_WebSearch]"):
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

    NOAA is not a separate spoke. `flightaware_live_status` calls
    `_enrich_formatted_with_noaa` only when AeroAPI reports delay/hold/divert.
    """
    try:
        user_text = _latest_user_text(state) or "current flight operations"
        task = _task_for_spoke(state, "flightaware") or user_text
        ident = _resolve_flight_ident(state)

        if ident:
            fa_result = flightaware_live_status.invoke({"ident": ident})
        else:
            fa_result = (
                '{"source":"skipped","note":"No flight number or tail found in '
                "current_task, latest message, Tavily notes, or (for follow-ups) "
                'prior chat."}'
            )

        content = (
            f"[Spoke_2_LiveOps]\n"
            f"Resolved ident from conversation: {ident or 'NONE'}\n"
            f"Current task: {task}\n"
            f"Latest user text: {user_text}\n"
            f"FlightAware ({ident or 'skipped'}):\n{fa_result}"
        )
        return {"turn_notes": [_spoke_note("flightaware", content)]}
    except Exception as exc:  # noqa: BLE001
        logger.exception("FlightAware spoke failed; returning error note.")
        return {
            "turn_notes": [
                _spoke_note("flightaware", f"[flightaware] error: {exc}")
            ]
        }


def spoke_3_websearch_node(state: AgentState) -> dict:
    """Specialist 3: Tavily aviation web research / NOTAMs.

    Parallel fan-out typically takes the fast path (supervisor task as-is).
    If sibling notes are already in this Send copy, optionally refine once.
    """
    try:
        user_text = _latest_user_text(state) or ""
        task_query = _task_for_spoke(state, "tavily") or user_text or "aviation operations"

        prior_spoke_outputs = [
            _note_content(n)
            for n in (state.get("turn_notes") or [])
            if _note_spoke(n) in VALID_FANOUT_SPOKES
        ]

        final_query = task_query
        refine_source = "task_direct"

        if prior_spoke_outputs:
            model = get_spoke_llm()
            if model is not None:
                try:
                    prior_blob = "\n\n".join(str(n)[:1200] for n in prior_spoke_outputs[-4:])
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
                    if isinstance(refined, dict):
                        refined = WebSearchRefine.model_validate(refined)
                    candidate = str(getattr(refined, "refined_query", "") or "").strip()
                    if candidate:
                        final_query = candidate
                        refine_source = "llm_chained"
                except Exception:  # noqa: BLE001
                    logger.exception(
                        "Tavily query refine failed; using task_query unchanged."
                    )
                    refine_source = "task_fallback"

        tv_result = tavily_aviation_web_search.invoke({"query": final_query})
        content = (
            f"[Spoke_3_WebSearch] Query: {final_query}\n"
            f"Refine: {refine_source}\n"
            f"{tv_result}"
        )
        return {"turn_notes": [_spoke_note("tavily", content)]}
    except Exception as exc:  # noqa: BLE001
        logger.exception("Tavily spoke failed; returning error note.")
        return {
            "turn_notes": [_spoke_note("tavily", f"[tavily] error: {exc}")]
        }


def synthesizer_node(state: AgentState) -> dict:
    """Writer node: reorder fan-in notes, then synthesize one final chat answer.

    Reads `turn_notes` + sanitized `context_priority_order`, packs a markdown
    string in that sequence, and hands it to the synthesis LLM.
    """
    raw_notes = list(state.get("turn_notes") or [])
    priority = _sanitize_context_priority_order(
        state.get("context_priority_order")
    )
    ordered_notes = _reorder_turn_notes(raw_notes, priority)
    packed_markdown = _format_notes_markdown(ordered_notes) # turns the sorted notes into a markdown string to feed to the LLM
    chunks: list[str] = [ # this is the list of notes that will be fed to the LLM
        _note_content(note) for note in ordered_notes if _note_content(note).strip()
    ]
    history = _conversation_texts(state, limit=MESSAGE_WINDOW)
    resolved = _resolve_flight_ident(state)
    opensky_block = _format_opensky_durable_block(chunks)

    model = get_spoke_llm()
    if model is not None and packed_markdown:
        try:
            response = model.invoke(
                [
                    SystemMessage(
                        content=(
                            "Synthesize a concise aviation intelligence report "
                            "from supervisor and spoke outputs.\n"
                            "Notes are packed in context_priority_order (safety-critical "
                            "FlightAware, which may include NOAA METARs on delay, and "
                            "Tavily NOTAMs sit at an edge — treat those sections as "
                            "high-priority).\n"
                            "Rules:\n"
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
                            "'[supervisor]' / markdown headers. Write a clean passenger-facing answer.\n"
                            "- If FlightAware returned empty/error for a bad ident (e.g. N1) "
                            "but Tavily named real flights (e.g. EK568/EK569), trust Tavily "
                            "for the flight numbers and say live status was unavailable."
                        )
                    ),
                    HumanMessage(
                        content=(
                            "Recent conversation:\n"
                            + ("\n---\n".join(history) if history else "(none)")
                            + "\n\nSpoke / supervisor notes (priority-packed):\n"
                            + packed_markdown
                        )
                    ),
                ]
            )
            final_report = _llm_text(response)
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
    if (
        "[Spoke_1_OpenSky]" not in joined
        and "OpenSky result:" not in joined
        and "[opensky]" not in joined.lower()
    ):
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
        blob = str(note)
        if blob.startswith("[Spoke_3_WebSearch]") or blob.startswith("[tavily]"):
            # Collect a few distinct flight numbers from the Tavily payload text.
            text = _normalize_ident_text(blob)
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
            if str(c).startswith(("[Spoke_1", "[Spoke_2", "[Spoke_3", "[opensky", "[flightaware", "[tavily"))
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
