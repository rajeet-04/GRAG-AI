"""GraphState TypedDict for the 5-agent LangGraph workflow.

Defines the shared state that flows through the agent graph:
  Ingestion → Query Agent → [KR Search | KB Search] → Context Builder → Explanation

All agent nodes read from and write to this state dict.
"""

from typing import TypedDict


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

    # ── Merged context ─────────────────────────────────────────
    merged_context: str
    token_count: int

    # ── Explanation output ─────────────────────────────────────
    answer: str
    reasoning_steps: list[str]
    mermaid_path: str
    confidence_scores: dict

    # ── Error handling ─────────────────────────────────────────
    errors: list[dict]
    retry_count: int

    # ── Metadata ───────────────────────────────────────────────
    agent_trace: list[str]


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
        # Merged context
        merged_context="",
        token_count=0,
        # Explanation output
        answer="",
        reasoning_steps=[],
        mermaid_path="",
        confidence_scores={},
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
