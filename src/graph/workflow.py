"""LangGraph StateGraph supervisor workflow with Plan-and-Execute delegation.

The supervisor builds a turn-local `plan` queue once per user message, then
pops spokes sequentially. Spokes always return to the supervisor so remaining
plan steps can run. Empty plan / FINISH routes to the synthesizer.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Literal

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph

from src.graph.nodes import (
    spoke_1_opensky_node,
    spoke_2_liveops_node,
    spoke_3_websearch_node,
    supervisor_node,
    synthesizer_node,
)
from src.graph.state import AgentState

# Project root: src/graph/workflow.py -> src -> aviation_agent/
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_DB_PATH = _PROJECT_ROOT / "checkpoints.db"


def _route_supervisor(
    state: AgentState,
) -> Literal[
    "spoke_1_opensky",
    "spoke_2_liveops",
    "spoke_3_websearch",
    "synthesizer",
]:
    """Map supervisor's next_worker choice to an actual graph node name."""
    nxt = state.get("next_worker", "FINISH")
    if nxt == "Spoke_1_OpenSky":
        return "spoke_1_opensky"
    if nxt == "Spoke_2_LiveOps":
        return "spoke_2_liveops"
    if nxt == "Spoke_3_WebSearch":
        return "spoke_3_websearch"
    return "synthesizer"


def _build_sqlite_checkpointer() -> SqliteSaver:
    """Durable checkpointer so chat history survives Streamlit restarts.

    check_same_thread=False is required: Streamlit/LangGraph may touch the
    connection from more than one thread. SqliteSaver serializes access with
    an internal lock.
    """
    conn = sqlite3.connect(
        str(CHECKPOINT_DB_PATH),
        check_same_thread=False,
    )
    saver = SqliteSaver(conn)
    saver.setup()
    return saver


workflow = StateGraph(AgentState)

workflow.add_node("supervisor", supervisor_node)
workflow.add_node("spoke_1_opensky", spoke_1_opensky_node)
workflow.add_node("spoke_2_liveops", spoke_2_liveops_node)
workflow.add_node("spoke_3_websearch", spoke_3_websearch_node)
workflow.add_node("synthesizer", synthesizer_node)

workflow.set_entry_point("supervisor")

workflow.add_conditional_edges(
    "supervisor",
    _route_supervisor,
    {
        "spoke_1_opensky": "spoke_1_opensky",
        "spoke_2_liveops": "spoke_2_liveops",
        "spoke_3_websearch": "spoke_3_websearch",
        "synthesizer": "synthesizer",
    },
)

workflow.add_edge("spoke_1_opensky", "supervisor")
workflow.add_edge("spoke_2_liveops", "supervisor")
workflow.add_edge("spoke_3_websearch", "supervisor")
workflow.add_edge("synthesizer", END)

# On-disk SQLite checkpointer: each thread_id keeps isolated chat memory
# that survives process / Streamlit server restarts.
app_graph = workflow.compile(checkpointer=_build_sqlite_checkpointer())
