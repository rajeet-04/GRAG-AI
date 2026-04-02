"""GRAG AI agent orchestration module.

Implements the 5-agent orchestration pattern:
1. Ingestion Agent — text → entities + relations
2. Query Agent — natural language → Cypher
3. KR/KB Search — parallel Neo4j + ChromaDB retrieval
4. Context Builder — merge KR + KB context with token budget
5. Explanation Agent — xAI reasoning output with Mermaid

Exports core LangGraph types and graph factory for API layer use.
"""

from app.agents.context_builder import context_builder_node, merge_with_budget
from app.agents.explanation_agent import explanation_agent_node
from app.agents.graph import create_agent_graph
from app.agents.ingestion_agent import create_ingestion_state, ingestion_agent_node
from app.agents.state import GraphState, create_initial_state, get_trace

__all__ = [
    # State types and helpers
    "GraphState",
    "create_initial_state",
    "get_trace",
    # Graph factory
    "create_agent_graph",
    # Agent nodes
    "ingestion_agent_node",
    "create_ingestion_state",
    "explanation_agent_node",
    "context_builder_node",
    "merge_with_budget",
]
