"""Text normalization utilities for entity processing."""

import unicodedata
import re
from typing import Optional


def normalize_entity_name(name: Optional[str]) -> str:
    """
    Normalize entity name for comparison.

    Args:
        name: Entity name to normalize

    Returns:
        Normalized name string
    """
    if not name:
        return ""

    # Convert to lowercase
    normalized = name.lower()

    # Remove extra whitespace
    normalized = re.sub(r"\s+", " ", normalized).strip()

    # Normalize unicode characters
    normalized = unicodedata.normalize("NFKD", normalized)

    # Remove diacritics
    normalized = "".join(c for c in normalized if not unicodedata.combining(c))

    return normalized


def normalize_entity_description(description: Optional[str]) -> str:
    """
    Normalize entity description for embedding.

    Args:
        description: Entity description to normalize

    Returns:
        Normalized description string
    """
    if not description:
        return ""

    # Convert to lowercase
    normalized = description.lower()

    # Remove extra whitespace
    normalized = re.sub(r"\s+", " ", normalized).strip()

    return normalized
