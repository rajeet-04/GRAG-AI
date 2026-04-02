"""LangGraph agent graph with checkpointing for the GRAG AI system.

Agent Graph Topology:
  1. Ingestion Agent (optional — for document ingestion)
  2. Graph Builder Agent (optional — writes ingested data to Neo4j)
  3. Query Agent → extracts intent + generates Cypher
  4. Parallel KR (Neo4j) + KB (ChromaDB) search
  5. Context Builder → merges with token budget enforcement
  6. Explanation Agent → generates xAI output with Mermaid

Flow (document ingestion):
  START → Ingestion → Graph Builder → Query Agent → [KR Search | KB Search] → Context Builder → Explanation → END

Flow (query mode):
  START → Ingestion (skip) → Query Agent → [KR Search | KB Search] → Context Builder → Explanation → END
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langgraph.graph import END, START, StateGraph

from app.agents.state import GraphState

# ---------------------------------------------------------------------------
# Real agent node imports
# ---------------------------------------------------------------------------

from app.agents.ingestion_agent import ingestion_agent_node  # noqa: F401
from app.agents.graph_builder import graph_builder_agent_node  # noqa: F401

# ---------------------------------------------------------------------------
# Stub node functions — will be replaced by real agent implementations
# ---------------------------------------------------------------------------


def query_agent_node(state: GraphState) -> dict[str, Any]:
    """Query Agent: extracts search intent and generates Cypher query.

    Stub — will be implemented with local qwen3.5:9b for fast intent extraction.
    """
    return {"agent_trace": ["query_agent:stub"]}


def kr_search_node(state: GraphState) -> dict[str, Any]:
    """KR Search: Neo4j graph traversal for knowledge retrieval.

    Stub — will use Neo4jClient with multi-hop traversal (2-4 hops max).
    """
    return {"agent_trace": ["kr_search:stub"]}


def kb_search_node(state: GraphState) -> dict[str, Any]:
    """KB Search: ChromaDB semantic memory retrieval.

    Stub — will use existing KB memory services (episodic + semantic).
    """
    return {"agent_trace": ["kb_search:stub"]}


def context_builder_node(state: GraphState) -> dict[str, Any]:
    """Context Builder: merges KR + KB results with token budget enforcement.

    Stub — will implement strict priority truncation at 8192 token budget.
    """
    return {"agent_trace": ["context_builder:stub"]}


def explanation_agent_node(state: GraphState) -> dict[str, Any]:
    """Explanation Agent: generates xAI output with Mermaid reasoning path.

    Stub — will produce Markdown answer + reasoning steps + Mermaid diagram.
    """
    return {"agent_trace": ["explanation_agent:stub"]}


def error_handler_node(state: GraphState) -> dict[str, Any]:
    """Error handler: routes errors back to failing agent for self-correction.

    Stub — implements cyclic recovery loops per CONTEXT.md decision #5.
    """
    retry = state.get("retry_count", 0) + 1
    return {"agent_trace": ["error_handler:stub"], "retry_count": retry}


# ---------------------------------------------------------------------------
# Conditional routing
# ---------------------------------------------------------------------------


def _route_after_ingestion(state: GraphState) -> str:
    """Route after Ingestion Agent.

    If entities were extracted, proceed to Graph Builder to persist them.
    Otherwise (query mode or empty ingestion), skip directly to Query Agent.
    """
    entities = state.get("entities", [])
    if entities:
        return "graph_builder"
    return "query_agent"


def _route_after_query(state: GraphState) -> list[str]:
    """Route after Query Agent: parallel KR + KB search, or skip to context.

    If the query agent produced a cypher_query, we need retrieval.
    Otherwise (e.g., simple greeting), skip directly to context builder.
    """
    cypher = state.get("cypher_query", "")
    if cypher:
        # Parallel fan-out: both KR and KB search run concurrently
        return ["kr_search", "kb_search"]
    # No graph query needed — go straight to context building
    return ["context_builder"]


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def _build_graph() -> StateGraph:
    """Build the StateGraph with all agent nodes and edges.

    Topology (document ingestion mode):
      START → ingestion → graph_builder → query_agent → [kr_search, kb_search] → context_builder → explanation → END

    Topology (query mode — ingestion returns no entities):
      START → ingestion → query_agent → [kr_search, kb_search] → context_builder → explanation → END

    Error handling nodes are wired for future use.
    """
    graph = StateGraph(GraphState)

    # ── Add nodes ──────────────────────────────────────────────
    graph.add_node("ingestion", ingestion_agent_node)
    graph.add_node("graph_builder", graph_builder_agent_node)
    graph.add_node("query_agent", query_agent_node)
    graph.add_node("kr_search", kr_search_node)
    graph.add_node("kb_search", kb_search_node)
    graph.add_node("context_builder", context_builder_node)
    graph.add_node("explanation", explanation_agent_node)
    graph.add_node("error_handler", error_handler_node)

    # ── Entry edge ─────────────────────────────────────────────
    graph.add_edge(START, "ingestion")

    # ── Ingestion → conditional routing ────────────────────────
    # If ingestion produced entities → Graph Builder
    # If no ingestion text (query mode) → skip to Query Agent
    graph.add_conditional_edges(
        "ingestion",
        _route_after_ingestion,
        ["graph_builder", "query_agent"],
    )

    # ── Graph Builder → Query Agent ────────────────────────────
    graph.add_edge("graph_builder", "query_agent")

    # ── Query Agent → conditional fan-out ──────────────────────
    graph.add_conditional_edges(
        "query_agent",
        _route_after_query,
        ["kr_search", "kb_search", "context_builder"],
    )

    # ── Parallel retrieval → Context Builder ───────────────────
    # Both KR and KB converge at context_builder
    graph.add_edge("kr_search", "context_builder")
    graph.add_edge("kb_search", "context_builder")

    # ── Context Builder → Explanation ──────────────────────────
    graph.add_edge("context_builder", "explanation")

    # ── Explanation → END ──────────────────────────────────────
    graph.add_edge("explanation", END)

    return graph


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_CHECKPOINT_DIR = Path("./data/checkpoints")
_CHECKPOINT_DB = _CHECKPOINT_DIR / "agent_graph.db"


def create_agent_graph() -> Any:
    """Create and compile the GRAG AI agent graph with SQLite checkpointing.

    Returns:
        A compiled LangGraph application ready for invocation.

    The checkpointer persists state at every node transition, enabling:
    - Crash recovery: resume from last successful node
    - Human-in-the-loop: pause and inspect state mid-execution
    - Debugging: replay any execution from checkpoint
    """
    import sqlite3

    from langgraph.checkpoint.sqlite import SqliteSaver

    # Ensure checkpoint directory exists
    _CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(_CHECKPOINT_DB), check_same_thread=False)
    checkpointer = SqliteSaver(conn)

    graph = _build_graph()
    return graph.compile(checkpointer=checkpointer)
