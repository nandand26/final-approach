"""Supervisor routing lives in `src/graph/nodes.py` (`supervisor_node`).

Domain split (Plan-and-Execute):
  - Spoke_1_OpenSky → physical ADS-B / OpenSky MCP tools
  - Spoke_2_LiveOps → commercial FlightAware + NOAA + Tavily
  - FINISH → synthesizer

This module exists as a pointer for docs/reviewers; import the node from nodes.py.
"""

from src.graph.nodes import supervisor_node

__all__ = ["supervisor_node"]
