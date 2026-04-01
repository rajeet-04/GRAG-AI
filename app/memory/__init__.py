"""Memory module for GRAG AI KB (Knowledge Base) layer.

KB LAYER STRUCTURE:
    - Short-term: Session messages (short_term_memory collection)
    - Episodic: Interaction summaries (episodic_memory collection)
    - Semantic: User preferences (semantic_memory collection)

KR/KB FIREWALL:
    - KB Layer: ChromaDB only (this module)
    - KR Layer: Neo4j only (app.graphiti, app.schemas.graph_schema)

NEVER import neo4j modules here. NEVER write to Neo4j from this module.
"""

from app.memory.short_term import ShortTermMemory, get_short_term_memory
from app.memory.episodic import EpisodicMemory, get_episodic_memory
from app.memory.semantic import SemanticMemory, get_semantic_memory
from app.memory.types import (
    MemoryLayer,
    KBType,
    KRType,
    FirewallError,
    kb_only,
    kr_only,
    KB_TYPES,
    KR_TYPES,
)

__all__ = [
    "ShortTermMemory",
    "get_short_term_memory",
    "EpisodicMemory",
    "get_episodic_memory",
    "SemanticMemory",
    "get_semantic_memory",
    "MemoryLayer",
    "KBType",
    "KRType",
    "FirewallError",
    "kb_only",
    "kr_only",
    "KB_TYPES",
    "KR_TYPES",
]
