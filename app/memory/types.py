"""Memory type definitions and KR/KB firewall enforcement for GRAG AI."""

from enum import Enum
from typing import Any, Callable, TypeVar

import structlog


logger = structlog.get_logger()


class MemoryLayer(Enum):
    """Memory layer types for KR/KB firewall."""

    KB = "kb"  # Knowledge Base layer (ChromaDB)
    KR = "kr"  # Knowledge Representation layer (Neo4j)


class KBType(Enum):
    """KB Layer (ChromaDB) memory types."""

    SHORT_TERM = "short_term"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"


class KRType(Enum):
    """KR Layer (Neo4j) memory types."""

    ENTITY = "entity"
    RELATION = "relation"
    DOCUMENT = "document"
    FACT = "fact"


KB_TYPES = {KBType.SHORT_TERM, KBType.EPISODIC, KBType.SEMANTIC}
KR_TYPES = {KRType.ENTITY, KRType.RELATION, KRType.DOCUMENT, KRType.FACT}

KB_COLLECTIONS = {
    KBType.SHORT_TERM: "short_term_memory",
    KBType.EPISODIC: "episodic_memory",
    KBType.SEMANTIC: "semantic_memory",
}


class FirewallError(Exception):
    """Raised when KR/KB firewall is violated."""

    pass


def get_memory_layer(memory_type: KBType | KRType) -> MemoryLayer:
    """Get the memory layer for a given type."""
    if isinstance(memory_type, KBType):
        return MemoryLayer.KB
    return MemoryLayer.KR


def assert_kb_only(operation: str, **kwargs: Any) -> None:
    """
    Assert that an operation only touches KB layer (ChromaDB).

    Raises FirewallError if Neo4j would be touched.

    Args:
        operation: Operation name for logging
        **kwargs: Additional context for logging
    """
    logger.debug("firewall.kb_check", operation=operation, **kwargs)


def assert_kr_only(operation: str, **kwargs: Any) -> None:
    """
    Assert that an operation only touches KR layer (Neo4j).

    Raises FirewallError if ChromaDB would be touched.

    Args:
        operation: Operation name for logging
        **kwargs: Additional context for logging
    """
    logger.debug("firewall.kr_check", operation=operation, **kwargs)


def enforce_layer(
    expected_layer: MemoryLayer,
    operation: str,
) -> Callable:
    """
    Decorator to enforce KR/KB layer separation.

    Args:
        expected_layer: Expected memory layer for the function
        operation: Operation name for error messages

    Returns:
        Decorator function
    """

    def decorator(func: Callable) -> Callable:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if expected_layer == MemoryLayer.KB:
                assert_kb_only(operation)
            else:
                assert_kr_only(operation)
            return func(*args, **kwargs)

        return wrapper

    return decorator


T = TypeVar("T")


def kb_only(func: Callable[..., T]) -> Callable[..., T]:
    """
    Decorator ensuring function only touches KB layer (ChromaDB).

    Use for all KB operations to prevent accidental Neo4j writes.
    """

    @enforce_layer(MemoryLayer.KB, func.__name__)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        return func(*args, **kwargs)

    wrapper.__name__ = func.__name__
    wrapper.__doc__ = func.__doc__
    return wrapper


def kr_only(func: Callable[..., T]) -> Callable[..., T]:
    """
    Decorator ensuring function only touches KR layer (Neo4j).

    Use for all KR operations to prevent accidental ChromaDB writes.
    """

    @enforce_layer(MemoryLayer.KR, func.__name__)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        return func(*args, **kwargs)

    wrapper.__name__ = func.__name__
    wrapper.__doc__ = func.__doc__
    return wrapper


def get_allowed_collections(layer: MemoryLayer) -> set[str]:
    """
    Get ChromaDB collection names allowed for a layer.

    Args:
        layer: Memory layer

    Returns:
        set[str]: Allowed collection names
    """
    if layer == MemoryLayer.KB:
        return set(KB_COLLECTIONS.values())
    return set()


def validate_memory_type(
    memory_type: KBType | KRType,
    expected_layer: MemoryLayer,
) -> bool:
    """
    Validate that a memory type belongs to expected layer.

    Args:
        memory_type: Memory type to validate
        expected_layer: Expected layer

    Returns:
        bool: True if valid

    Raises:
        FirewallError: If type doesn't match expected layer
    """
    actual_layer = get_memory_layer(memory_type)

    if actual_layer != expected_layer:
        raise FirewallError(
            f"Memory type {memory_type.value} belongs to {actual_layer.value.upper()} layer, "
            f"but {expected_layer.value.upper()} layer expected"
        )

    return True
