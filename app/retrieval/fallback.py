"""Vector fallback system for weak graph queries.

Triggers:
1. Cypher returns 0 results
2. Average graph confidence < 0.70
3. Query Agent flags intent as ambiguous

Executes:
- ChromaDB episodic search (top-20)
- ChromaDB semantic search (top-10)
- Merges and deduplicates results
"""

from __future__ import annotations

import asyncio
from typing_extensions import TypedDict

import structlog

from app.database.chroma_client import get_chromadb_client
from app.schemas.retrieval import RetrievalConfig

logger = structlog.get_logger()


class FallbackTrigger(TypedDict):
    """Why fallback was triggered."""

    reason: str  # "zero_results" | "low_confidence" | "ambiguous_intent"
    details: str


class FallbackResult(TypedDict):
    """Result from vector fallback execution."""

    results: list[dict]
    trigger: FallbackTrigger
    combined_scores: list[float]


def check_fallback_trigger(
    graph_results: list[dict],
    avg_confidence: float | None = None,
    intent_is_ambiguous: bool = False,
    config: RetrievalConfig | None = None,
) -> FallbackTrigger | None:
    """Check if graph results warrant vector fallback.

    Args:
        graph_results: Results from graph traversal (empty = trigger)
        avg_confidence: Average confidence of graph results (None = no scores)
        intent_is_ambiguous: True if Query Agent flagged ambiguous intent
        config: Retrieval configuration (uses defaults if None)

    Returns:
        FallbackTrigger if fallback needed, None if graph results are sufficient

    Decision order (first match wins):
    1. Zero results -> fallback always
    2. No confidence scores -> assume weak, fallback
    3. Low confidence (< 0.70) -> fallback
    4. Ambiguous intent -> fallback
    """
    config = config or RetrievalConfig()

    # Signal 1: Zero results
    if len(graph_results) == 0:
        return FallbackTrigger(
            reason="zero_results",
            details="Cypher query returned no results",
        )

    # Signal 2: No confidence scores
    if avg_confidence is None:
        return FallbackTrigger(
            reason="low_confidence",
            details="No confidence scores available from graph",
        )

    # Signal 3: Low confidence
    if avg_confidence < config.confidence_threshold:
        return FallbackTrigger(
            reason="low_confidence",
            details=f"Average confidence {avg_confidence:.2f} < threshold {config.confidence_threshold}",
        )

    # Signal 4: Ambiguous intent
    if intent_is_ambiguous:
        return FallbackTrigger(
            reason="ambiguous_intent",
            details="Query Agent flagged intent as ambiguous",
        )

    return None


async def execute_vector_fallback(
    query: str,
    top_k: int | None = None,
    config: RetrievalConfig | None = None,
) -> FallbackResult:
    """Execute vector fallback search via ChromaDB.

    Searches both episodic and semantic memory, merges results,
    and returns deduplicated entities with similarity scores.

    Args:
        query: Original user query for semantic search
        top_k: Number of results per collection (default from config)
        config: Retrieval configuration

    Returns:
        FallbackResult with combined results and scores
    """
    config = config or RetrievalConfig()
    top_k = top_k or config.fallback_top_k

    logger.info("fallback.executing", query=query[:50], top_k=top_k)

    chroma_client = get_chromadb_client()

    # Execute both searches in parallel
    episodic_task = _search_episodic(chroma_client, query, top_k)
    semantic_task = _search_semantic(chroma_client, query, max(1, top_k // 2))

    episodic_results, semantic_results = await asyncio.gather(
        episodic_task, semantic_task
    )

    # Merge and deduplicate by entity/content
    combined = merge_fallback_results(episodic_results, semantic_results)

    # Apply similarity threshold filter
    threshold = config.similarity_threshold
    filtered = [r for r in combined if r["similarity"] >= threshold]

    logger.info(
        "fallback.complete",
        episodic=len(episodic_results),
        semantic=len(semantic_results),
        combined=len(combined),
        filtered=len(filtered),
    )

    return FallbackResult(
        results=filtered,
        trigger=FallbackTrigger(
            reason="fallback_executed",
            details=f"Found {len(filtered)} results above threshold {threshold}",
        ),
        combined_scores=[r["similarity"] for r in filtered],
    )


async def _search_episodic(client, query: str, limit: int) -> list[dict]:
    """Search episodic memory."""
    try:
        collection = client.get_episodic_collection()
        embedding_service = client._embedding_service

        embedding = await embedding_service.embed_text(query)

        results = collection.query(
            query_embeddings=[embedding],
            n_results=limit,
            include=["documents", "metadatas", "distances"],
        )

        items = []
        if results["ids"] and results["ids"][0]:
            for i, ep_id in enumerate(results["ids"][0]):
                distance = results["distances"][0][i] if "distances" in results else 1.0
                similarity = 1.0 - distance

                items.append(
                    {
                        "id": ep_id,
                        "type": "episodic",
                        "content": results["documents"][0][i],
                        "similarity": similarity,
                        "metadata": results["metadatas"][0][i],
                    }
                )

        return items
    except Exception as e:
        logger.error("fallback.episodic_error", error=str(e))
        return []


async def _search_semantic(client, query: str, limit: int) -> list[dict]:
    """Search semantic memory."""
    try:
        collection = client.get_semantic_collection()
        embedding_service = client._embedding_service

        embedding = await embedding_service.embed_text(query)

        results = collection.query(
            query_embeddings=[embedding],
            n_results=limit,
            include=["documents", "metadatas", "distances"],
        )

        items = []
        if results["ids"] and results["ids"][0]:
            for i, pref_id in enumerate(results["ids"][0]):
                distance = results["distances"][0][i] if "distances" in results else 1.0
                similarity = 1.0 - distance

                items.append(
                    {
                        "id": pref_id,
                        "type": "semantic",
                        "content": results["documents"][0][i],
                        "similarity": similarity,
                        "metadata": results["metadatas"][0][i],
                    }
                )

        return items
    except Exception as e:
        logger.error("fallback.semantic_error", error=str(e))
        return []


def merge_fallback_results(
    episodic: list[dict],
    semantic: list[dict],
) -> list[dict]:
    """Merge episodic and semantic results, deduplicating by content.

    Prioritizes episodic results when same content appears in both.
    """
    seen = set()
    merged = []

    # Add episodic first (higher priority)
    for item in episodic:
        content_key = item["content"][:100]
        if content_key not in seen:
            seen.add(content_key)
            merged.append(item)

    # Add semantic (lower priority)
    for item in semantic:
        content_key = item["content"][:100]
        if content_key not in seen:
            seen.add(content_key)
            merged.append(item)

    # Sort by similarity descending
    merged.sort(key=lambda x: x["similarity"], reverse=True)

    return merged
