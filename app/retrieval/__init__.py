"""Retrieval module for GRAG AI."""

try:
    from app.retrieval.fallback import (
        FallbackResult,
        FallbackTrigger,
        check_fallback_trigger,
        execute_vector_fallback,
    )
except ImportError:
    pass  # chromadb may not be installed in test environments

try:
    from app.retrieval.ranker import (
        RankingConfig,
        ScoredResult,
        merge_graph_and_vector,
        normalize_distance_to_similarity,
        rank_retrieval_results,
    )
except ImportError:
    pass

__all__ = [
    "FallbackTrigger",
    "FallbackResult",
    "check_fallback_trigger",
    "execute_vector_fallback",
    "RankingConfig",
    "ScoredResult",
    "merge_graph_and_vector",
    "normalize_distance_to_similarity",
    "rank_retrieval_results",
]
