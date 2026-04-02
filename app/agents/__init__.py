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
from app.agents.graph_builder import graph_builder_agent_node
from app.agents.ingestion_agent import create_ingestion_state, ingestion_agent_node
from app.agents.query_agent import query_agent_node
from app.agents.state import GraphState, create_initial_state, get_trace
from app.agents.tools import (
    AGENT_TOOLS,
    chroma_episodic_search,
    chroma_semantic_search,
    neo4j_search,
)

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
    "query_agent_node",
    "context_builder_node",
    "merge_with_budget",
    "explanation_agent_node",
    "graph_builder_agent_node",
    # Tools
    "AGENT_TOOLS",
    "neo4j_search",
    "chroma_episodic_search",
    "chroma_semantic_search",
]
