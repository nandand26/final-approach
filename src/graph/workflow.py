"""LangGraph router.

Path 1 (telemetry_known): one Send() per plan row, in parallel. The same spoke
may run more than once with different tasks. Each Send carries its own task.
Path 2 (telemetry_discovery): one Send() per Tavily plan row, then
clarification_node interrupt → Command(goto="supervisor"). No static edge
off clarification_node.
Path 3a (general_knowledge): one Send() per Tavily plan row, then synthesizer.
Path 3b (direct_chat): synthesizer. No tools.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph
from langgraph.types import Send

from src.graph.nodes import (
    clarification_node,
    spoke_1_opensky_node,
    spoke_2_liveops_node,
    spoke_3_websearch_node,
    supervisor_node,
    synthesizer_node,
)
from src.graph.state import VALID_FANOUT_SPOKES, AgentState

# Project root: src/graph/workflow.py -> src -> aviation_agent/
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_DB_PATH = _PROJECT_ROOT / "checkpoints.db"


def _route_after_supervisor(state: AgentState):
    """Intent → a Send list, or the synthesizer.

    Discovery and general research already stored a Tavily-only plan, so the
    sends stay Tavily-only. After those calls finish, _route_after_tavily
    sends discovery to clarification and research to the synthesizer. A
    greeting goes straight to the synthesizer.
    """
    intent = str(state.get("intent") or "")
    spokes = [
        spoke
        for spoke in list(state.get("spokes_to_call") or [])
        if spoke in VALID_FANOUT_SPOKES
    ]
    if intent == "direct_chat" or not spokes:
        return "synthesizer"
    # The supervisor already applied the cap, dropped duplicate tasks, and
    # kept only spokes_to_call. One Send per remaining plan row. The row's
    # task rides in the payload so two calls to the same spoke stay distinct.
    sends: list[Send] = []
    for step in list(state.get("plan") or []):
        if not isinstance(step, dict):
            continue
        worker = str(step.get("worker") or "").strip()
        task = str(step.get("task") or "").strip()
        if not worker or not task:
            continue
        sends.append(Send(worker, {**state, "task": task}))
    return sends


def _route_after_tavily(state: AgentState) -> str:
    """Discovery pauses for a human ident. Every other Tavily hop synthesizes."""
    if str(state.get("intent") or "") == "telemetry_discovery":
        return "clarification_node"
    return "synthesizer"


def _build_sqlite_checkpointer() -> SqliteSaver:
    """Durable checkpointer so chat history survives Streamlit restarts.

    check_same_thread=False is required: Streamlit/LangGraph may touch the
    connection from more than one thread. SqliteSaver serializes access with
    an internal lock. Required for interrupt()/resume.
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
workflow.add_node("opensky", spoke_1_opensky_node)
workflow.add_node("flightaware", spoke_2_liveops_node)
workflow.add_node("tavily", spoke_3_websearch_node)
# destinations= declares the Command(goto=) target only. No static edge,
# so this node does not also fall through to END.
workflow.add_node(
    "clarification_node",
    clarification_node,
    destinations=("supervisor",),
)
workflow.add_node("synthesizer", synthesizer_node)

workflow.set_entry_point("supervisor")

workflow.add_conditional_edges(
    "supervisor",
    _route_after_supervisor,
    [
        "opensky",
        "flightaware",
        "tavily",
        "synthesizer",
    ],
)

# Discovery vs research. clarification_node has NO static edge — it returns
# Command(goto="supervisor") only.
workflow.add_conditional_edges(
    "tavily",
    _route_after_tavily,
    ["clarification_node", "synthesizer"],
)

workflow.add_edge("opensky", "synthesizer")
workflow.add_edge("flightaware", "synthesizer")
workflow.add_edge("synthesizer", END)

# On-disk SQLite checkpointer: each thread_id keeps isolated chat memory
# that survives process / Streamlit server restarts.
app_graph = workflow.compile(checkpointer=_build_sqlite_checkpointer())
