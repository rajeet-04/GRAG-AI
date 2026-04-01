"""Semantic memory (user preferences) for GRAG AI KB layer."""

from datetime import datetime
from typing import Any
import uuid

import structlog

from app.database.chroma_client import ChromaDBClient


logger = structlog.get_logger()


class SemanticMemory:
    """
    Semantic memory store for user preferences.

    Stores user preferences as vector embeddings for semantic search.
    Automatically tracks preference confidence and source.

    KB LAYER: All operations write to ChromaDB only.
    """

    PREFERENCE_TYPES = [
        "coding_style",
        "response_format",
        "topics",
        "communication_style",
        "language",
        "domain_expertise",
        "other",
    ]

    def __init__(self, chroma_client: ChromaDBClient) -> None:
        """Initialize semantic memory with ChromaDB client."""
        self._chroma = chroma_client
        self._embedding_service = chroma_client._embedding_service

    def _get_collection(self):
        """Get the semantic memory collection."""
        return self._chroma.get_semantic_collection()

    def _generate_id(self, user_id: str, preference_type: str) -> str:
        """Generate unique preference ID."""
        return f"pref_{user_id[:8]}_{preference_type}_{uuid.uuid4().hex[:8]}"

    async def store_preference(
        self,
        user_id: str,
        preference_type: str,
        content: str,
        source: str = "conversation",
        confidence: float = 0.8,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Store a user preference.

        Args:
            user_id: User identifier
            preference_type: Type of preference
            content: Preference description
            source: How preference was learned
            confidence: Confidence score (0-1)
            metadata: Additional metadata

        Returns:
            str: Preference ID
        """
        collection = self._get_collection()
        pref_id = self._generate_id(user_id, preference_type)
        ts = datetime.utcnow()

        embedding = await self._embedding_service.embed_text(content)

        collection.upsert(
            ids=[pref_id],
            documents=[content],
            embeddings=[embedding],
            metadatas=[
                {
                    "user_id": user_id,
                    "preference_type": preference_type,
                    "source": source,
                    "confidence": confidence,
                    "created_at": ts.isoformat(),
                    "updated_at": ts.isoformat(),
                    "type": "preference",
                    **(metadata or {}),
                }
            ],
        )

        logger.info(
            "semantic_memory.preference_stored",
            pref_id=pref_id,
            user_id=user_id,
            preference_type=preference_type,
        )
        return pref_id

    async def get_user_preferences(
        self,
        user_id: str,
        preference_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Get all preferences for a user.

        Args:
            user_id: User to query
            preference_type: Optional type filter

        Returns:
            list[dict]: User preferences
        """
        collection = self._get_collection()

        where_filter: dict[str, Any] = {"user_id": user_id}
        if preference_type:
            where_filter["preference_type"] = preference_type

        results = collection.get(
            where=where_filter,
            include=["documents", "metadatas"],
        )

        preferences = []
        if results["ids"]:
            for i, pref_id in enumerate(results["ids"]):
                metadata = results["metadatas"][i]
                preferences.append(
                    {
                        "id": pref_id,
                        "content": results["documents"][i],
                        "preference_type": metadata.get("preference_type", ""),
                        "source": metadata.get("source", ""),
                        "confidence": metadata.get("confidence", 0.0),
                        "created_at": metadata.get("created_at", ""),
                        "updated_at": metadata.get("updated_at", ""),
                    }
                )

        return preferences

    async def find_similar_preferences(
        self,
        query: str,
        user_id: str,
        n: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Semantic search for preferences.

        Args:
            query: Search query
            user_id: User to search within
            n: Number of results

        Returns:
            list[dict]: Matching preferences
        """
        collection = self._get_collection()

        embedding = await self._embedding_service.embed_text(query)

        results = collection.query(
            query_embeddings=[embedding],
            n_results=n,
            where={"user_id": user_id},
        )

        preferences = []
        if results["ids"] and results["ids"][0]:
            for i, pref_id in enumerate(results["ids"][0]):
                metadata = results["metadatas"][0][i]
                preferences.append(
                    {
                        "id": pref_id,
                        "content": results["documents"][0][i],
                        "preference_type": metadata.get("preference_type", ""),
                        "source": metadata.get("source", ""),
                        "confidence": metadata.get("confidence", 0.0),
                        "distance": results["distances"][0][i]
                        if "distances" in results
                        else None,
                    }
                )

        logger.debug(
            "semantic_memory.preferences_found",
            query=query,
            user_id=user_id,
            results=len(preferences),
        )
        return preferences

    async def update_preference(
        self,
        preference_id: str,
        content: str,
        confidence: float | None = None,
    ) -> bool:
        """
        Update an existing preference.

        Args:
            preference_id: Preference to update
            content: New content
            confidence: Optional new confidence

        Returns:
            bool: True if updated
        """
        collection = self._get_collection()

        results = collection.get(
            where={"id": preference_id},
            include=["metadatas"],
        )

        if not results["ids"]:
            logger.warning("semantic_memory.pref_not_found", pref_id=preference_id)
            return False

        metadata = results["metadatas"][0]
        embedding = await self._embedding_service.embed_text(content)

        updated_metadata = {
            **metadata,
            "updated_at": datetime.utcnow().isoformat(),
        }
        if confidence is not None:
            updated_metadata["confidence"] = confidence

        collection.upsert(
            ids=[preference_id],
            documents=[content],
            embeddings=[embedding],
            metadatas=[updated_metadata],
        )

        logger.info("semantic_memory.preference_updated", pref_id=preference_id)
        return True

    async def delete_preference(self, preference_id: str) -> bool:
        """
        Delete a preference.

        Args:
            preference_id: Preference to delete

        Returns:
            bool: True if deleted
        """
        collection = self._get_collection()
        try:
            collection.delete(ids=[preference_id])
            logger.info("semantic_memory.preference_deleted", pref_id=preference_id)
            return True
        except Exception as e:
            logger.error(
                "semantic_memory.delete_error", pref_id=preference_id, error=str(e)
            )
            return False

    def get_preference_types(self) -> list[str]:
        """Get all valid preference types."""
        return self.PREFERENCE_TYPES.copy()


_semantic_memory: SemanticMemory | None = None


def get_semantic_memory() -> SemanticMemory:
    """Get singleton semantic memory instance."""
    global _semantic_memory
    if _semantic_memory is None:
        from app.database.chroma_client import get_chromadb_client

        _semantic_memory = SemanticMemory(get_chromadb_client())
    return _semantic_memory
