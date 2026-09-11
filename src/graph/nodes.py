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
import re
import time
from typing import Any, Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from src.graph.state import MAX_SPOKE_EXECUTIONS_PER_TURN, AgentState
from src.graph.models import spoke_llm, supervisor_llm
from src.mcp_servers.opensky_mcp import OPENSKY_TOOLS, invoke_opensky_tool
from src.tools.live_ops import flightaware_live_status, tavily_aviation_web_search

logger = logging.getLogger(__name__)

# Allowed spoke names that may appear in the plan queue / next_worker.
SPOKE_WORKERS = ("Spoke_1_OpenSky", "Spoke_2_LiveOps")
# Allowed values for state["next_worker"] (spokes + synthesizer handoff).
VALID_WORKERS = {"Spoke_1_OpenSky", "Spoke_2_LiveOps", "FINISH"}


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


def _spoke_execution_counts(turn_notes: list[str]) -> dict[str, int]:
    """Count how many times each spoke already ran in THIS user-message turn.

    Used to enforce MAX_SPOKE_EXECUTIONS_PER_TURN (hard cap = 2).
    """
    counts = {"Spoke_1_OpenSky": 0, "Spoke_2_LiveOps": 0}
    for note in turn_notes:
        text = str(note)
        if "[Spoke_1_OpenSky]" in text:
            counts["Spoke_1_OpenSky"] += 1
        if "[Spoke_2_LiveOps]" in text:
            counts["Spoke_2_LiveOps"] += 1
    return counts


def _enforce_spoke_caps(plan: list[str], already_ran: dict[str, int] | None = None) -> list[str]:
    """Defensive queue sanitizer: keep only valid spokes and enforce the 2-run cap.

    WHY this exists on top of the LLM schema prompt:
    - LLMs can still emit duplicates or invalid names.
    - Mid-turn re-entry must also respect how many times a spoke already executed.
    - Caps are counted against (already_ran + scheduled-in-this-plan).
    """
    already_ran = already_ran or {"Spoke_1_OpenSky": 0, "Spoke_2_LiveOps": 0}
    scheduled: dict[str, int] = {"Spoke_1_OpenSky": 0, "Spoke_2_LiveOps": 0}
    cleaned: list[str] = []

    for step in plan:
        if step not in SPOKE_WORKERS:
            # Drop unknown tokens (e.g. FINISH accidentally placed inside plan).
            continue
        total_if_added = already_ran.get(step, 0) + scheduled.get(step, 0) + 1 # 
        # Strict per-spoke hard cap for a single user message turn.
        if total_if_added > MAX_SPOKE_EXECUTIONS_PER_TURN:
            continue
        cleaned.append(step)
        scheduled[step] = scheduled.get(step, 0) + 1

    return cleaned


def _heuristic_plan(user_text: str, history_blob: str) -> list[str]:
    """Offline Plan-and-Execute fallback when supervisor_llm is unavailable.

    Domain split:
      Spoke 1 → physical ADS-B / OpenSky (icao24, altitude, airport movements)
      Spoke 2 → commercial / web (flight numbers, METAR, Tavily)
    """
    text = f"{user_text}\n{history_blob}".lower()

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

    # Commercial / ops / weather / web
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
            "news",
            "recall",
            "advisory",
            "tavily",
            "metar",
            "aircraft type",
            "codeshare",
            "a380",
            "b787",
            "schedule",
            "delay",
        )
    ) or bool(re.search(r"\b[a-z]{2,3}\d{1,5}\b", text))  # flight numbers like ai8303

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
    if conversational and not wants_opensky and not wants_live:
        return []

    plan: list[str] = []
    if wants_opensky:
        plan.append("Spoke_1_OpenSky")
    if wants_live:
        plan.append("Spoke_2_LiveOps")

    # Ambiguous aviation ask → prefer commercial live ops (flight numbers are common)
    if not plan and user_text.strip():
        plan = ["Spoke_2_LiveOps"]

    return _enforce_spoke_caps(plan)


class SupervisorPlan(BaseModel):
    """Structured plan the supervisor LLM must return at turn start.

    `plan` is an ordered queue of spokes. Empty list means conversational
    bypass → go straight to FINISH / synthesizer (no tool calls).
    """

    plan: list[str] = Field(
        default_factory=list,
        description=(
            "Ordered execution queue. Allowed values only: "
            "'Spoke_1_OpenSky' and/or 'Spoke_2_LiveOps'. "
            "Use [] when the answer needs no tools (greetings, reformat, "
            "questions answerable from existing chat context alone). "
            f"Each spoke may appear at most {MAX_SPOKE_EXECUTIONS_PER_TURN} times."
        ),
    )
    reasoning: str = Field(
        default="",
        description=(
            "Brief note on how chat history resolved ambiguous references "
            "(e.g. 'that flight' → AI8303) and why each spoke was scheduled."
        ),
    )


def _llm_generate_plan(user_text: str, history_blob: str) -> list[str]:
    """Ask the supervisor LLM for an ordered plan, then enforce the spoke cap.

    CONTEXT AWARENESS:
    - `history_blob` is the recent `messages` transcript (user + prior finals).
    - The prompt explicitly requires resolving pronouns / implicit refs
      ("that flight", "yesterday's delay") against that history BEFORE planning.
    """
    model = supervisor_llm
    if model is None:
        return _heuristic_plan(user_text, history_blob)

    system_prompt = (
        "You are the aviation fleet intelligence supervisor operating under a Plan-and-Execute architecture.\n"
        "Your sole job at the START of each user turn is to build an ordered `plan` queue of spoke worker nodes.\n\n"
        "CONTEXT & AMBIGUITY RESOLUTION:\n"
        "- Read the full chat history alongside the latest user query.\n"
        "- Resolve implicit references using prior context before picking spokes.\n\n"
        "SPOKE CAPABILITIES (strict domain split):\n"
        "- Spoke_1_OpenSky (Physical Airspace Engine / OpenSky ADS-B):\n"
        "  icao24 transponder hex IDs, live GPS/altitude/speed/heading/squawk/on_ground,\n"
        "  trajectory waypoints, airport ADS-B arrival/departure movement logs.\n"
        "  DO NOT use for commercial flight numbers (AI8303, SQ403), schedules, A380/B787 equipment, METAR, or news.\n"
        "- Spoke_2_LiveOps (Commercial Intelligence & Web):\n"
        "  commercial flight numbers, FlightAware status, codeshares, airline delays,\n"
        "  equipment/type tags, NOAA METARs, Tavily web research.\n\n"
        "PLANNING & ORDERING RULES:\n"
        "- Flight number / delay / weather / aircraft-type questions → Spoke_2_LiveOps.\n"
        "- icao24 / altitude / trajectory / ADS-B airport movements → Spoke_1_OpenSky.\n"
        "- Hybrid queries may include both, ordered by what the user needs first.\n"
        f"- Each spoke may appear in the `plan` array AT MOST {MAX_SPOKE_EXECUTIONS_PER_TURN} times per user turn.\n"
        "- NEVER place 'FINISH' or synthesizer nodes inside the `plan` array.\n\n"
        "DIRECT BYPASS RULE:\n"
        "- If conversational / reformat / answerable from chat alone, return an empty plan: `[]`."
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
                        "Return a SupervisorPlan with an ordered `plan` list."
                    )
                ),
            ]
        )
        if isinstance(result, dict):
            result = SupervisorPlan.model_validate(result)
        raw_plan = list(getattr(result, "plan", []) or [])
        # Schema/prompt AND defensive sanitizer both enforce the 2-run cap.
        return _enforce_spoke_caps(raw_plan)
    except Exception:
        logger.exception("Supervisor plan generation failed; using heuristic plan.")
        return _heuristic_plan(user_text, history_blob)


# ---------------------------------------------------------------------------
# Graph nodes (these are the functions wired in workflow.py)
# ---------------------------------------------------------------------------

def supervisor_node(state: AgentState) -> dict:
    """Plan-and-Execute supervisor: build a turn plan once, then pop/execute it.

    Lifecycle for ONE user message turn:
    1) Turn start (`turn_notes` empty after app reset) → generate `plan` from
       latest query + full recent `messages` (resolves 'that flight', etc.).
    2) Each supervisor visit → pop the next spoke from `plan` into `next_worker`.
    3) Empty `plan` → `FINISH` (synthesizer). An empty plan at generation time
       is an intentional conversational/direct bypass (saves tool latency/cost).
    4) Hard cap: no spoke runs more than MAX_SPOKE_EXECUTIONS_PER_TURN (2).
    """
    user_text = _latest_user_text(state)
    turn_notes = list(state.get("turn_notes") or [])
    plan = list(state.get("plan") or [])

    # Build a compact history blob from durable messages for reference resolution.
    # WHY messages (not turn_notes): prior final answers contain flight numbers /
    # delays the user may refer to implicitly in the new prompt.
    history_parts = [_message_text(m)[:400] for m in state.get("messages", [])[-8:]]
    history_blob = "\n---\n".join(part for part in history_parts if part.strip())

    # ----- Phase A: TURN START — generate the plan once -----
    # Signal: turn_notes was reset to [] by app.py at the start of this prompt.
    if not turn_notes:
        plan = _llm_generate_plan(user_text, history_blob)
        # Defensive re-check with zero prior executions this turn.
        plan = _enforce_spoke_caps(plan, already_ran={"Spoke_1_OpenSky": 0, "Spoke_2_LiveOps": 0})

        if not plan:
            # Empty plan → immediate synthesizer handoff.
            # WHY: conversational / reformat / context-only asks do not need tools;
            # skipping spokes cuts latency and API overhead.
            return {
                "plan": [],
                "next_worker": "FINISH",
                "turn_notes": [
                    "[supervisor] Empty plan (conversational/direct bypass) → FINISH. "
                    "No spokes scheduled; synthesizer will use chat context only."
                ],
            }

        # Pop the first planned spoke; remaining queue stays in state for later hops.
        next_worker = plan[0]
        remaining = plan[1:]
        return {
            "plan": remaining,
            "next_worker": next_worker,
            "turn_notes": [
                f"[supervisor] Plan generated from query + chat history: "
                f"{[next_worker] + remaining}. "
                f"Executing {next_worker} "
                f"(cap={MAX_SPOKE_EXECUTIONS_PER_TURN}/spoke/turn)."
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
            "turn_notes": [
                "[supervisor] Plan queue empty after spoke execution(s) → FINISH."
            ],
        }

    next_worker = plan[0]
    remaining = plan[1:]

    # Final defensive gate: never dispatch a spoke that already hit the per-turn cap.
    if already_ran.get(next_worker, 0) >= MAX_SPOKE_EXECUTIONS_PER_TURN:
        return {
            "plan": [],
            "next_worker": "FINISH",
            "turn_notes": [
                f"[supervisor] Spoke cap reached for {next_worker} "
                f"({MAX_SPOKE_EXECUTIONS_PER_TURN}/turn). Forcing FINISH."
            ],
        }

    return {
        "plan": remaining,
        "next_worker": next_worker,
        "turn_notes": [
            f"[supervisor] Continuing plan → {next_worker}. Remaining queue: {remaining}."
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


def _find_ident_in_text(user_text: str) -> str | None:
    """Find a US tail, airline flight ident, registration, or ATC callsign.

    REVIEW: old regex `\bN[0-9A-Z]{2,6}\b` matched the English word NOW as a
    fake tail, which broke follow-ups containing 'from now'.
    New US-tail pattern requires digits: N12345 / N123AB.
    Also picks European/Asian regs (D-AIZZ, 9V-SKQ) and callsigns (DLH6K)
    so follow-ups after an OpenSky turn can feed FlightAware.
    """
    # Normalize "AI 8303" -> "AI8303"
    text = re.sub(r"\b([A-Z]{2,3})\s+(\d{1,5})\b", r"\1\2", user_text.upper())

    # REVIEW: US N-numbers always include digits — do NOT match words like NOW
    tail = re.search(r"\bN\d{1,5}[A-Z]{0,2}\b", text)
    if tail:
        return tail.group(0)

    # Airline flight: UAL123 / AA100 / AI8303 / SQ403 (up to 5 digits)
    flight = re.search(r"\b([A-Z]{2,3}\d{1,5})\b", text)
    if flight:
        return flight.group(1)

    # Registration with hyphen: D-AIZZ, 9V-SKQ, G-EUUY
    reg = re.search(r"\b([A-Z0-9]{1,2}-[A-Z0-9]{2,5})\b", text)
    if reg:
        return reg.group(1)

    # ATC callsign e.g. DLH6K (airline trigraph + alphanumeric suffix)
    callsign = re.search(r"\b([A-Z]{3}\d[A-Z0-9]{0,3})\b", text)
    if callsign:
        return callsign.group(1)

    return None


def _resolve_flight_ident(state: AgentState) -> str | None:
    """Resolve flight/tail from the latest user text, else earlier chat history.

    REVIEW: this is the main follow-up-memory fix. Example:
      Turn 1: 'status of AI8303...'
      Turn 2: 'if I fly this 2 months from now, which aircraft...'
    Without history walk, Turn 2 lost AI8303 (and wrongly used NOW).
    """
    # Prefer an explicit ident in the newest user message
    latest = _latest_user_text(state)
    from_latest = _find_ident_in_text(latest) if latest else None
    if from_latest:
        return from_latest

    # Otherwise walk conversation newest -> oldest (user + prior final answers)
    for message in reversed(state.get("messages", [])):
        found = _find_ident_in_text(_message_text(message))
        if found:
            return found
    return None


def _extract_flight_ident(user_text: str) -> str:
    """Backward-compatible single-string extractor (tests / callers)."""
    return _find_ident_in_text(user_text) or "N12345"


def spoke_2_liveops_node(state: AgentState) -> dict:
    """Specialist 2: FlightAware telemetry + Tavily aviation web research.

    REVIEW: used to call `_extract_flight_ident(latest_user_text_only)`.
    Now resolves ident from full chat, skips FA if none, and feeds history
    into the Tavily query for schedule/equipment follow-ups.
    """
    user_text = _latest_user_text(state) or "current flight operations"
    # REVIEW: conversation-aware ident (not latest message only)
    ident = _resolve_flight_ident(state)
    history = _conversation_texts(state, limit=6)
    history_blob = "\n".join(f"- {chunk[:400]}" for chunk in history[-6:])

    # REVIEW: only hit FlightAware when we actually know which flight/tail
    if ident:
        fa_result = flightaware_live_status.invoke({"ident": ident})
    else:
        fa_result = (
            '{"source":"skipped","note":"No flight number or tail found in '
            'this message or recent chat history."}'
        )

    # REVIEW: enrich Tavily with resolved flight + recent chat for follow-ups
    tv_query = user_text
    if ident:
        tv_query = (
            f"{user_text}\n\n"
            f"Resolved flight/tail from conversation: {ident}\n"
            f"Focus web research on scheduled aircraft type / equipment for {ident}, "
            f"including typical fleet for this route when looking months ahead."
        )
    if history_blob:
        tv_query = f"{tv_query}\n\nRecent conversation:\n{history_blob}"

    tv_result = tavily_aviation_web_search.invoke({"query": tv_query})

    content = (
        f"[Spoke_2_LiveOps]\n"
        f"Resolved ident from conversation: {ident or 'NONE'}\n"
        f"Latest user text: {user_text}\n"
        f"FlightAware ({ident or 'skipped'}):\n{fa_result}\n\n"
        f"Tavily research:\n{tv_result}"
    )
    # Return to supervisor so Plan-and-Execute can pop the next plan step (or FINISH).
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
                            "- Lead with status, aircraft, route, and times when relevant."
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
        except Exception:  # noqa: BLE001
            final_report = _fallback_report(chunks)
    else:
        final_report = _fallback_report(chunks)

    # Deterministic stamp so OpenSky ids survive into durable messages even if
    # the LLM paraphrases/omits fields (critical for follow-up Spoke 2 / regex).
    if opensky_block and opensky_block not in final_report:
        final_report = f"{final_report.rstrip()}\n\n{opensky_block}"

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
    """Build a simple final report without calling an LLM."""
    if not chunks:
        return (
            "Aviation Intelligence Report\n"
            "No worker outputs were available to synthesize."
        )
    # Keep only the last few chunks so the report stays readable
    body = "\n\n".join(chunks[-6:])
    report = (
        "Aviation Fleet & Operational Intelligence Report\n"
        "==============================================\n"
        f"{body}\n"
        "==============================================\n"
        "End of report."
    )
    block = _format_opensky_durable_block(chunks)
    if block:
        report = f"{report}\n\n{block}"
    return report
