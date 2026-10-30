"""Shared LangGraph agent state for the aviation orchestration hub."""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

# add_messages tells LangGraph to APPEND new messages instead of replacing them
from langgraph.graph.message import add_messages

# Mem0 (`mem0ai`) is an optional long-term memory backend. Do NOT import it here:
# `import mem0` has side effects (creates ~/.mem0) and must stay lazy in nodes.py.
# AgentState.user_id is the identity stamp Mem0 scopes against.

# MESSAGE_WINDOW: hard cap on durable chat history kept in AgentState.messages.
# 20 messages is 10 question-answer pairs. Each user line and each assistant
# reply counts as one. Older messages are dropped.
MESSAGE_WINDOW = 20

# Fan-out spoke ids used by Send() and context_priority_order.
VALID_FANOUT_SPOKES = ("opensky", "flightaware", "tavily")

# Per-spoke hard caps for a single user-message turn (one hop each in fan-out).
SPOKE_CAPS = {
    "opensky": 2,
    "flightaware": 2,
    "tavily": 2,
}

WORKER_TO_FANOUT = {
    "opensky": "opensky",
    "flightaware": "flightaware",
    "tavily": "tavily",
    # NOAA METARs live inside FlightAware (delay/hold/divert only).
    "noaa": "flightaware",
    "Spoke_1_OpenSky": "opensky",
    "Spoke_2_LiveOps": "flightaware",
    "Spoke_3_WebSearch": "tavily",
    "spoke_1_opensky": "opensky",
    "spoke_2_liveops": "flightaware",
    "spoke_3_websearch": "tavily",
}


def add_messages_window(left: list, right: list) -> list:
    """Append chat messages, then keep only the newest N (fixed-size queue)."""
    # 1) Merge old messages (left) with new ones (right) via LangGraph helper
    merged = add_messages(left, right)
    merged_list: list[Any] = list(merged)
    # 2) Trim to last MESSAGE_WINDOW so context never grows unbounded
    if len(merged_list) > MESSAGE_WINDOW:
        return merged_list[-MESSAGE_WINDOW:]
    return merged_list


def merge_turn_notes(left: list[dict], right: list[dict]) -> list[dict]:
    """Reducer for per-turn scratch notes: operator.add + explicit turn reset.

    Parallel Send() hops must APPEND (map-reduce fan-in). Checkpointer turns
    must still wipe scratch so the next prompt cannot reuse last turn's notes.

    Reset behavior (unchanged):
    - app.py starts each new user prompt with turn_notes=[]
    - right == []  → CLEAR scratch for this turn (do not append)
    - right == [notes...] → operator.add onto current turn scratch
    """
    if right == []:
        return []
    return operator.add(list(left or []), list(right or []))


class AgentState(TypedDict):
    """Graph state passed between supervisor and spoke workers."""

    # --- durable across turns (per thread_id) ---
    # Rolling chat memory (user + final answers only), capped at MESSAGE_WINDOW.
    messages: Annotated[list, add_messages_window]

    # Permanent Google OAuth identity (sub / email fallback from auth_sessions.db).
    # Used as Mem0 `user_id` so preferences survive across threads and sessions.
    # Not a chat transcript field — identity stamp refreshed by app.py each turn.
    user_id: str  # this is for memzero

    # --- turn-local scratch (must reset every new user message) ---
    # Supervisor/spoke internals for THIS ask only.
    # Each item is {"spoke": "<fan-out id>", "content": "<note text>"}.
    # Typed as list[dict] with operator.add fan-in for parallel Send() writes.
    # merge_turn_notes IS operator.add plus the existing `right == []` clear
    # (app.py passes turn_notes=[] each prompt). Do not drop that reset.
    turn_notes: Annotated[list[dict], merge_turn_notes]

    # One {worker, task} per spoke in spokes_to_call. The router still fans out
    # from spokes_to_call; each spoke reads its own task from this list.
    plan: list[dict[str, str]]

    # Route class for this supervisor visit:
    # telemetry_known | telemetry_discovery | general_knowledge | direct_chat
    intent: str

    # Marketed flight ident when Path 1 can call FlightAware (e.g. BA117).
    # None until the user typed one or picked one at the clarification interrupt.
    flight_number: str | None

    # Tavily query for discovery or general research. Not a chat transcript field.
    search_query: str | None

    # Fan-out targets (graph node names): opensky | flightaware | tavily
    spokes_to_call: list[str]

    # Synthesizer pack order (may differ from execution order).
    # Safety-critical FlightAware (may include NOAA METARs on delay) and
    # Tavily NOTAMs belong at the start or end.
    context_priority_order: list[str]

    # Current hop after a pop: fan-out id | FINISH (metro / Send payload)
    next_worker: str

    # Hop-local sub-query the active spoke should execute (from plan head's "task").
    # Cleared to "" when FINISH. Replace semantics (no reducer).
    current_task: str

    # Memory Gate: supervisor sets True only when this turn contains an explicit,
    # durable preference/rule worth writing to Mem0. Synthesizer adds only if True.
    # Reset rule: app.py passes extract_memory=False at every new prompt.
    extract_memory: bool

    # Synthesizer's final answer string for this turn
    final_report: str
