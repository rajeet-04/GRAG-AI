"""Graphiti configuration for temporal versioning."""

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings


class GraphitiConfig(BaseSettings):
    """Configuration for Graphiti episodic memory system."""

    graphiti_node_labels: dict[str, str] = Field(
        default_factory=lambda: {
            "entity": "graphiti_Entity",
            "entity_fact": "graphiti_EntityFact",
            "episode": "graphiti_Episode",
            "relation": "graphiti_Relation",
        },
        description="Node labels used by Graphiti (separate from manual KR schema)",
    )

    graphiti_episode_history_window: int = Field(
        default=30,
        description="Number of days to search for related episodes",
    )

    graphiti_max_entities: int = Field(
        default=50,
        description="Maximum entities to extract per episode",
    )

    graphiti_entity_similarity_threshold: float = Field(
        default=0.85,
        description="Minimum similarity for entity matching",
    )

    model_config = {
        "env_prefix": "GRAPHITI_",
        "extra": "ignore",
    }


def get_graphiti_config() -> GraphitiConfig:
    """Get Graphiti configuration instance."""
    return GraphitiConfig()
