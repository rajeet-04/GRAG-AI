"""Retrieval schemas for Phase 7: Multi-hop BFS traversal."""

from dataclasses import dataclass
from typing import List, Optional

from typing_extensions import TypedDict


@dataclass
class RetrievalConfig:
    """Configuration for retrieval engine.

    All values configurable via app/config.py settings.
    """

    similarity_threshold: float = 0.75
    """Minimum similarity score for vector fallback results (0.0-1.0)."""

    fallback_top_k: int = 20
    """Number of results to return from vector fallback search."""

    confidence_threshold: float = 0.70
    """Minimum average confidence to avoid fallback trigger (0.0-1.0)."""

    max_traversal_depth: int = 4
    """Maximum BFS traversal depth (2-4 hops per RETR-01)."""

    max_results_per_hop: int = 50
    """Maximum results to return per hop (prevents BFS explosion)."""

    query_timeout_seconds: float = 3.0
    """Hard timeout for Cypher query execution."""


class PathSegment(TypedDict):
    """Single hop in a traversal path."""

    source_id: str
    source_name: str
    relation_type: str
    relation_id: str
    target_id: str
    target_name: str
    valid_from: str
    valid_to: Optional[str]
    confidence: float


class TraversalResult(TypedDict):
    """Result from multi-hop BFS traversal."""

    entity_id: str
    entity_name: str
    entity_type: str
    confidence: float
    depth: int
    path: List["PathSegment"]


class RetrievalResult(TypedDict):
    """Unified retrieval result after ranking."""

    entity_id: str
    entity_name: str
    entity_type: str
    unified_score: float
    source: str  # "graph" | "vector" | "merged"
    path: Optional[List[PathSegment]]
