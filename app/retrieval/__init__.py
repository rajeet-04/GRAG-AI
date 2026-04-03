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

try:
    from app.retrieval.web_search import (
        execute_web_search,
        format_web_search_results,
        is_latest_data_query,
        should_trigger_web_search,
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
    "execute_web_search",
    "format_web_search_results",
    "is_latest_data_query",
    "should_trigger_web_search",
]
