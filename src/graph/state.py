"""Shared LangGraph agent state for the aviation orchestration hub."""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

# add_messages tells LangGraph to APPEND new messages instead of replacing them
from langgraph.graph.message import add_messages

# MESSAGE_WINDOW: hard cap on durable chat history kept in AgentState.messages.
# Only the newest N user+assistant messages survive; older ones are dropped.
MESSAGE_WINDOW = 10

# Hard cap: each spoke may run at most this many times in one user-message turn
MAX_SPOKE_EXECUTIONS_PER_TURN = 2


def add_messages_window(left: list, right: list) -> list:
    """Append chat messages, then keep only the newest N (fixed-size queue)."""
    # 1) Merge old messages (left) with new ones (right) via LangGraph helper
    merged = add_messages(left, right)
    merged_list: list[Any] = list(merged)
    # 2) Trim to last MESSAGE_WINDOW so context never grows unbounded
    if len(merged_list) > MESSAGE_WINDOW:
        return merged_list[-MESSAGE_WINDOW:]
    return merged_list


def merge_turn_notes(left: list[str], right: list[str]) -> list[str]:
    """Reducer for per-turn scratch notes with explicit turn resets.

    WHY this is needed with a checkpointer:
    - `messages` should persist across turns in one thread.
    - `turn_notes` should *not* persist forever, otherwise every new prompt
      reuses old scratch and can route/summarize incorrectly.

    Reset behavior (IMPORTANT):
    - app.py starts each new user prompt with turn_notes=[]
    - right == []  → CLEAR scratch for this turn (do not append)
    - right == [notes...] → APPEND onto current turn scratch
    """
    # Explicit reset signal from app.py / turn start
    if right == []:
        return []
    # Normal path: keep prior notes this turn + add new ones from a node
    return list(left or []) + list(right or [])


class AgentState(TypedDict):
    """Graph state passed between supervisor and spoke workers."""

    # --- durable across turns (per thread_id) ---
    # Rolling chat memory (user + final answers only), capped at MESSAGE_WINDOW.
    messages: Annotated[list, add_messages_window]

    # --- turn-local scratch (must reset every new user message) ---
    # Supervisor/spoke internals for THIS ask only.
    # Reset rule: app passes turn_notes=[] → merge_turn_notes clears it.
    turn_notes: Annotated[list[str], merge_turn_notes]

    # Plan-and-Execute queue: ordered spokes still left to run this turn.
    # No Annotated reducer → each write REPLACES the whole list (not append).
    # Reset rule: app passes plan=[] at turn start so last turn's queue cannot leak.
    # Supervisor pops the head into next_worker; remaining stays here.
    plan: list[str]

    # Current hop after a pop: Spoke_1_OpenSky | Spoke_2_LiveOps | FINISH
    next_worker: str

    # Synthesizer's final answer string for this turn
    final_report: str
