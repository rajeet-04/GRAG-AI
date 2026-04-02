"""Result ranking with confidence-weighted late fusion.

Merges graph (KR) and vector (KB) results:
- Graph gets 0.6 weight (higher confidence, domain knowledge)
- Vector gets 0.4 weight (semantic matching)
- Deduplication by Entity ID
- Unified scoring on 0-1 scale
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import structlog

from app.schemas.retrieval import RetrievalResult

logger = structlog.get_logger()


# Weight constants (from CONTEXT.md Decision 4)
GRAPH_WEIGHT = 0.6
VECTOR_WEIGHT = 0.4

# Vector-only penalty factor (no domain knowledge validation)
VECTOR_ONLY_PENALTY = 0.7


@dataclass
class RankingConfig:
    """Configuration for result ranking.

    Attributes:
        graph_weight: Weight for graph confidence in late fusion (0.0-1.0).
        vector_weight: Weight for vector similarity in late fusion (0.0-1.0).
        min_unified_score: Minimum unified score threshold (inclusive).
        max_results: Maximum number of results to return.
        vector_only_penalty: Penalty multiplier for vector-only results.
    """

    graph_weight: float = GRAPH_WEIGHT
    vector_weight: float = VECTOR_WEIGHT
    min_unified_score: float = 0.0
    max_results: int = 50
    vector_only_penalty: float = VECTOR_ONLY_PENALTY


class ScoredResult(dict):
    """Result with unified confidence score.

    Keys:
        entity_id: Unique entity identifier for deduplication.
        entity_name: Human-readable entity name.
        entity_type: Entity type (Person, Project, etc.).
        unified_score: Final score on 0.0-1.0 scale after late fusion.
        source: Result source — "graph", "vector", or "merged".
        confidence_graph: Raw graph confidence (None if vector-only).
        similarity_vector: Vector similarity score (None if graph-only).
        path: Graph traversal path segments (None if vector-only).
        reasoning_path: Associated relations for xAI (populated by enrich step).
    """

    pass


def normalize_distance_to_similarity(distance: float) -> float:
    """Convert distance to similarity score.

    ChromaDB returns distances (lower = better).
    Convert to similarity (higher = better) for unified scoring.

    Args:
        distance: ChromaDB distance value (typically 0-1).

    Returns:
        Similarity score (0-1), clamped to valid range.
    """
    similarity = 1.0 - distance
    return max(0.0, min(1.0, similarity))


def merge_graph_and_vector(
    graph_results: List[Dict[str, Any]],
    vector_results: List[Dict[str, Any]],
    config: Optional[RankingConfig] = None,
) -> List[ScoredResult]:
    """Merge graph and vector results with confidence-weighted late fusion.

    Algorithm:
        1. Create unified map keyed by entity_id
        2. For each entity present in both sources:
           unified_score = graph_weight * graph_conf + vector_weight * vector_sim
        3. For entities in only one source:
           - Graph only: score = graph_conf * 1.0 (full weight, domain validated)
           - Vector only: score = vector_sim * 0.7 (penalized, no domain knowledge)
        4. Deduplicate by entity_id, sort by unified_score descending

    Args:
        graph_results: List of graph entities with confidence.
        vector_results: List of vector results with similarity or distance.
        config: Ranking configuration (uses defaults if None).

    Returns:
        List of ScoredResult sorted by unified_score descending.
    """
    config = config or RankingConfig()

    unified: Dict[str, ScoredResult] = {}

    # ── Process graph results (higher priority) ───────────────────────
    for r in graph_results:
        entity_id = r.get("entity_id") or r.get("id")
        if not entity_id:
            continue

        confidence = r.get("confidence") or r.get("unified_score") or 0.0
        confidence = max(0.0, min(1.0, float(confidence)))

        unified[entity_id] = ScoredResult(
            entity_id=entity_id,
            entity_name=r.get("entity_name") or r.get("name", ""),
            entity_type=r.get("entity_type") or r.get("type", ""),
            unified_score=confidence,
            source="graph",
            confidence_graph=confidence,
            similarity_vector=None,
            path=r.get("path") or r.get("kr_paths"),
            reasoning_path=[],
        )

    # ── Process vector results, merge or add ──────────────────────────
    for r in vector_results:
        entity_id = r.get("id") or r.get("entity_id")
        if not entity_id:
            continue

        # Convert distance to similarity if distance field present
        distance = r.get("distance")
        if distance is not None:
            similarity = normalize_distance_to_similarity(float(distance))
        else:
            similarity = float(r.get("similarity", 0.0))
        similarity = max(0.0, min(1.0, similarity))

        if entity_id in unified:
            # Entity in both sources — late fusion merge
            existing = unified[entity_id]
            graph_conf = existing.get("confidence_graph") or 0.0

            merged_score = (
                config.graph_weight * graph_conf + config.vector_weight * similarity
            )

            unified[entity_id] = ScoredResult(
                entity_id=entity_id,
                entity_name=existing.get("entity_name") or r.get("name", ""),
                entity_type=existing.get("entity_type") or r.get("type", ""),
                unified_score=merged_score,
                source="merged",
                confidence_graph=graph_conf,
                similarity_vector=similarity,
                path=existing.get("path"),
                reasoning_path=existing.get("reasoning_path", []),
            )
        else:
            # Entity in vector only — penalized (no domain validation)
            penalized_score = similarity * config.vector_only_penalty

            unified[entity_id] = ScoredResult(
                entity_id=entity_id,
                entity_name=r.get("name") or r.get("entity_name", ""),
                entity_type=r.get("type") or r.get("entity_type", ""),
                unified_score=penalized_score,
                source="vector",
                confidence_graph=None,
                similarity_vector=similarity,
                path=None,
                reasoning_path=[],
            )

    # ── Sort by unified score descending ──────────────────────────────
    ranked = sorted(
        unified.values(),
        key=lambda x: x["unified_score"],
        reverse=True,
    )

    # ── Filter by minimum score and limit results ────────────────────
    filtered = [r for r in ranked if r["unified_score"] >= config.min_unified_score][
        : config.max_results
    ]

    logger.info(
        "ranker.merged",
        graph_count=len(graph_results),
        vector_count=len(vector_results),
        unique_entities=len(unified),
        final_count=len(filtered),
    )

    return filtered


def rank_retrieval_results(
    kr_entities: List[Dict[str, Any]],
    kr_relations: List[Dict[str, Any]],
    episodic_memories: List[Dict[str, Any]],
    semantic_preferences: List[Dict[str, Any]],
    config: Optional[RankingConfig] = None,
) -> tuple[List[ScoredResult], List[Dict[str, Any]]]:
    """Rank and deduplicate all retrieval results for Context Builder.

    Takes results from both KR (graph) and KB (vector) layers,
    merges them with late fusion, and returns clean arrays.

    Args:
        kr_entities: Graph entities with confidence.
        kr_relations: Graph relations (not ranked, passed through).
        episodic_memories: Episodic memories with distances.
        semantic_preferences: Semantic preferences with distances.
        config: Ranking configuration.

    Returns:
        Tuple of (ranked_entities, ranked_relations):
        - ranked_entities: ScoredResult list sorted by unified_score.
        - ranked_relations: Relations with reasoning_path info for xAI.
    """
    config = config or RankingConfig()

    # Combine vector results from episodic and semantic
    vector_results = episodic_memories + semantic_preferences

    # Merge graph and vector with late fusion
    ranked = merge_graph_and_vector(
        graph_results=kr_entities,
        vector_results=vector_results,
        config=config,
    )

    # Enrich with relation info where available
    ranked_relations = _enrich_with_relations(ranked, kr_relations)

    return ranked, ranked_relations


def _enrich_with_relations(
    ranked_entities: List[ScoredResult],
    relations: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Add relation paths to ranked entities for xAI.

    For each ranked entity, find associated relations that form
    the reasoning path. Relations are matched by source_id or target_id.
    """
    # Build entity_id -> relations map
    entity_rels: Dict[str, List[Dict]] = {}
    for rel in relations:
        source = rel.get("source_id") or rel.get("source", {}).get("id", "")
        target = rel.get("target_id") or rel.get("target", {}).get("id", "")
        if source:
            entity_rels.setdefault(source, []).append(rel)
        if target:
            entity_rels.setdefault(target, []).append(rel)

    # Add relations to ranked results
    enriched = []
    for entity in ranked_entities:
        entity_dict = dict(entity)
        entity_dict["reasoning_path"] = entity_rels.get(entity["entity_id"], [])
        enriched.append(entity_dict)

    return enriched


def create_retrieval_result(entity: ScoredResult) -> RetrievalResult:
    """Convert ScoredResult to RetrievalResult for Context Builder.

    Args:
        entity: ScoredResult with unified score and metadata.

    Returns:
        RetrievalResult TypedDict for downstream processing.
    """
    return RetrievalResult(
        entity_id=entity["entity_id"],
        entity_name=entity["entity_name"],
        entity_type=entity["entity_type"],
        unified_score=entity["unified_score"],
        source=entity["source"],
        path=entity.get("path"),
    )
