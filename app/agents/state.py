"""GraphState TypedDict for the 5-agent LangGraph workflow.

Defines the shared state that flows through the agent graph:
  Ingestion → Query Agent → [KR Search | KB Search] → Context Builder → Explanation

All agent nodes read from and write to this state dict.
"""

import operator
from typing import Annotated, TypedDict


class GraphState(TypedDict):
    """Shared state for the GRAG AI agent graph.

    Keys are organized by which agent populates them:
    - User input: provided at graph entry
    - Query Agent: intent extraction + Cypher generation
    - KR/KB retrieval: parallel search results
    - Context Builder: merged + token-counted context
    - Explanation: final answer with xAI reasoning
    - Error handling: retry logic and error accumulation
    - Metadata: execution tracing
    """

    # ── User input ──────────────────────────────────────────────
    user_query: str
    session_id: str
    ingestion_text: str  # Optional: populated for document ingestion mode

    # ── Ingestion Agent output ──────────────────────────────────
    entities: Annotated[list[dict], operator.add]  # Extracted entities
    relations: Annotated[list[dict], operator.add]  # Extracted relations
    graph_write_result: dict  # Result from Graph Builder (Neo4j write)

    # ── Query Agent output ──────────────────────────────────────
    search_intent: str
    temporal_filters: dict
    cypher_query: str

    # ── KR retrieval results (Neo4j) ───────────────────────────
    kr_entities: list[dict]
    kr_relations: list[dict]
    kr_paths: list[dict]

    # ── KB retrieval results (ChromaDB) ────────────────────────
    episodic_memories: list[dict]
    semantic_preferences: list[dict]
    web_search_results: list[dict]

    # ── Ranked results (late fusion) ───────────────────────────
    ranked_entities: list[dict]  # Deduplicated, scored entities
    ranked_relations: list[dict]  # Relations with path info

    # ── Merged context ─────────────────────────────────────────
    merged_context: str
    token_count: int
    context_truncated: bool  # True if any truncation occurred
    truncation_warning: str | None  # Warning message when 80% threshold exceeded

    # ── Explanation output ─────────────────────────────────────
    answer: str
    reasoning_steps: list[str]
    mermaid_path: str
    confidence_scores: dict
    citation_map: dict[str, int]  # Maps "[1]" -> reasoning step index

    # ── Error handling ─────────────────────────────────────────
    errors: Annotated[list[dict], operator.add]
    retry_count: int

    # ── Metadata ───────────────────────────────────────────────
    agent_trace: Annotated[list[str], operator.add]


def create_initial_state(user_query: str, session_id: str) -> GraphState:
    """Create a fresh GraphState with defaults for all keys.

    Args:
        user_query: The user's natural language question.
        session_id: Unique session identifier for checkpointing.

    Returns:
        A fully initialized GraphState ready for graph entry.
    """
    return GraphState(
        # User input
        user_query=user_query,
        session_id=session_id,
        ingestion_text="",
        # Ingestion Agent output
        entities=[],
        relations=[],
        graph_write_result={},
        # Query Agent output
        search_intent="",
        temporal_filters={},
        cypher_query="",
        # KR retrieval results
        kr_entities=[],
        kr_relations=[],
        kr_paths=[],
        # KB retrieval results
        episodic_memories=[],
        semantic_preferences=[],
        web_search_results=[],
        # Ranked results
        ranked_entities=[],
        ranked_relations=[],
        # Merged context
        merged_context="",
        token_count=0,
        context_truncated=False,
        truncation_warning=None,
        # Explanation output
        answer="",
        reasoning_steps=[],
        mermaid_path="",
        confidence_scores={},
        citation_map={},
        # Error handling
        errors=[],
        retry_count=0,
        # Metadata
        agent_trace=[],
    )


def get_trace(state: GraphState) -> str:
    """Return the agent execution trace as a formatted string.

    Args:
        state: The current GraphState.

    Returns:
        Newline-separated agent trace entries.
    """
    return "\n".join(state.get("agent_trace", []))
