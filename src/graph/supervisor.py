"""Supervisor routing lives in `src/graph/nodes.py` (`supervisor_node`).

Three paths (`SupervisorPlan.intent`):
  - telemetry_known → parallel Send(flightaware, opensky[, tavily])
  - telemetry_discovery → tavily, then clarification_node interrupt,
    Command(goto="supervisor") after the user picks an ident
  - general_knowledge → tavily or synthesizer (no flightaware / opensky)
  - direct_chat → synthesizer

NOAA METARs stay inside FlightAware (delay/hold/divert only).
Pydantic schema: `SupervisorPlan` in `src/graph/models.py`.
"""

from src.graph.models import SupervisorPlan
from src.graph.nodes import clarification_node, supervisor_node

__all__ = ["supervisor_node", "clarification_node", "SupervisorPlan"]
