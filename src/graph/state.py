"""Shared LangGraph agent state for the aviation orchestration hub."""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

# add_messages tells LangGraph to APPEND new messages instead of replacing them
from langgraph.graph.message import add_messages

# Mem0 (`mem0ai`) is an optional long-term memory backend. Do NOT import it here:
# `import mem0` has side effects (creates ~/.mem0) and must stay lazy in nodes.py.
# AgentState.user_id is the identity stamp Mem0 scopes against.

# MESSAGE_WINDOW: hard cap on durable chat history kept in AgentState.messages.
# Only the newest N user+assistant messages survive; older ones are dropped.
MESSAGE_WINDOW = 10

# Per-spoke hard caps for a single user-message turn (Plan-and-Execute queue).
SPOKE_CAPS = {
    "Spoke_1_OpenSky": 2,
    "Spoke_2_LiveOps": 2,
    "Spoke_3_WebSearch": 5,
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

    # Permanent Google OAuth identity (sub / email fallback from auth_sessions.db).
    # Used as Mem0 `user_id` so preferences survive across threads and sessions.
    # Not a chat transcript field — identity stamp refreshed by app.py each turn.
    user_id: str  # this is for memzero

    # --- turn-local scratch (must reset every new user message) ---
    # Supervisor/spoke internals for THIS ask only.
    # Reset rule: app passes turn_notes=[] → merge_turn_notes clears it.
    turn_notes: Annotated[list[str], merge_turn_notes]

    # Plan-and-Execute queue: ordered sub-tasks still left to run this turn.
    # Each item is {"worker": Spoke_*, "task": "<sub-query>"}.
    # No Annotated reducer → each write REPLACES the whole list (not append).
    # Reset rule: app passes plan=[] at turn start so last turn's queue cannot leak.
    # Supervisor pops the head into next_worker + current_task; remaining stays here.
    plan: list[dict[str, str]]

    # Current hop after a pop: Spoke_1_OpenSky | Spoke_2_LiveOps | Spoke_3_WebSearch | FINISH
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
