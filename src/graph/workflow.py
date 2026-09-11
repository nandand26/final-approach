"""LangGraph StateGraph supervisor workflow with Plan-and-Execute delegation.

The supervisor builds a turn-local `plan` queue once per user message, then
pops spokes sequentially. Spokes always return to the supervisor so remaining
plan steps can run. Empty plan / FINISH routes to the synthesizer.
"""

from __future__ import annotations

from typing import Literal

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from src.graph.nodes import (
    spoke_1_opensky_node,
    spoke_2_liveops_node,
    supervisor_node,
    synthesizer_node,
)
from src.graph.state import AgentState


def _route_supervisor(
    state: AgentState,
) -> Literal["spoke_1_opensky", "spoke_2_liveops", "synthesizer"]:
    """Map supervisor's next_worker choice to an actual graph node name."""
    nxt = state.get("next_worker", "FINISH")
    if nxt == "Spoke_1_OpenSky":
        return "spoke_1_opensky"
    if nxt == "Spoke_2_LiveOps":
        return "spoke_2_liveops"
    return "synthesizer"


workflow = StateGraph(AgentState)

workflow.add_node("supervisor", supervisor_node)
workflow.add_node("spoke_1_opensky", spoke_1_opensky_node)
workflow.add_node("spoke_2_liveops", spoke_2_liveops_node)
workflow.add_node("synthesizer", synthesizer_node)

workflow.set_entry_point("supervisor")

workflow.add_conditional_edges(
    "supervisor",
    _route_supervisor,
    {
        "spoke_1_opensky": "spoke_1_opensky",
        "spoke_2_liveops": "spoke_2_liveops",
        "synthesizer": "synthesizer",
    },
)

workflow.add_edge("spoke_1_opensky", "supervisor")
workflow.add_edge("spoke_2_liveops", "supervisor")
workflow.add_edge("synthesizer", END)

# In-process checkpointer: each thread_id keeps isolated chat memory
app_graph = workflow.compile(checkpointer=MemorySaver())
