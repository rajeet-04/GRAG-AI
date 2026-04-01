"""KB Memory schemas for GRAG AI.

Defines Pydantic models for semantic and episodic memory stores.
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class SemanticMemory(BaseModel):
    """
    Semantic memory model for user profiles and preferences.

    Represents long-term, stable traits about the user that don't
    have a specific timestamp.

    Examples:
    - "The user prefers Python over Java"
    - "User is a B.Tech CSE student"
    - "User is building a romanization service"
    - "User wants practical implementation explanations"
    """

    user_id: str = Field(..., description="User identifier")
    trait_type: str = Field(
        ...,
        description="Type of trait (e.g., 'language_preference', 'skill_level', 'project_interest')",
    )
    content: str = Field(..., description="The actual trait value")
    updated_at: datetime = Field(
        default_factory=datetime.utcnow, description="Last update time"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Optional extra data"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "user_id": "user_123",
                    "trait_type": "language_preference",
                    "content": "Python",
                    "updated_at": "2026-04-01T12:00:00Z",
                    "metadata": {"priority": "high"},
                }
            ]
        }
    }


class EpisodicMemory(BaseModel):
    """
    Episodic memory model for session snapshots.

    Represents time-bound, session-specific interactions that capture
    what happened at a specific moment in time.

    Examples:
    - "In Tuesday's session, the user was debugging an AI agricultural supervisor"
    - "Yesterday, the user asked for a summary of Graphiti's temporal logic"
    """

    user_id: str = Field(..., description="User identifier")
    session_id: str = Field(..., description="Session identifier")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="When this episode occurred"
    )
    summary: str = Field(..., description="LLM-generated summary of the interaction")
    content: str = Field(..., description="Raw interaction text or details")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional: entities mentioned, tools used, etc.",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "user_id": "user_123",
                    "session_id": "session_456",
                    "timestamp": "2026-04-01T14:30:00Z",
                    "summary": "User worked on optimizing AgroEngine supervisor logic",
                    "content": "User asked about graph traversal patterns for Neo4j",
                    "metadata": {
                        "entities": ["AgroEngine", "Neo4j"],
                        "tools_used": ["cypher", "python"],
                    },
                }
            ]
        }
    }


class SemanticMemoryQuery(BaseModel):
    """Query model for semantic memory retrieval."""

    query_text: str = Field(..., description="Text to search for")
    user_id: str | None = Field(None, description="Filter by user (optional)")
    trait_types: list[str] | None = Field(None, description="Filter by trait types")
    n_results: int = Field(default=5, description="Number of results to return")


class EpisodicMemoryQuery(BaseModel):
    """Query model for episodic memory retrieval."""

    query_text: str | None = Field(None, description="Text to search for")
    user_id: str | None = Field(None, description="Filter by user")
    session_id: str | None = Field(None, description="Filter by session")
    time_range: tuple[datetime, datetime] | None = Field(
        None, description="Filter by time range (start, end)"
    )
    n_results: int = Field(default=5, description="Number of results to return")
