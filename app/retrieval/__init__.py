"""Retrieval module for GRAG AI."""

from app.retrieval.fallback import (
    FallbackResult,
    FallbackTrigger,
    check_fallback_trigger,
    execute_vector_fallback,
)

__all__ = [
    "FallbackTrigger",
    "FallbackResult",
    "check_fallback_trigger",
    "execute_vector_fallback",
]
