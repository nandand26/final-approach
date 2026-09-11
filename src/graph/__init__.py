"""Graph package: LangGraph state, nodes, and compiled workflow."""

# Re-export so callers can do: from src.graph import app_graph
from src.graph.workflow import app_graph

__all__ = ["app_graph"]
