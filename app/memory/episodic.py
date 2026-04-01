"""Episodic memory (interaction summaries) for GRAG AI KB layer."""

from datetime import datetime
from typing import Any
import uuid

import structlog

from app.database.chroma_client import ChromaDBClient


logger = structlog.get_logger()


class EpisodicMemory:
    """
    Episodic memory store for interaction summaries.

    Stores LLM-generated summaries of interactions with temporal metadata.
    Episodes capture key facts, entities, topics, and sentiment.

    KB LAYER: All operations write to ChromaDB only.
    """

    def __init__(self, chroma_client: ChromaDBClient) -> None:
        """Initialize episodic memory with ChromaDB client."""
        self._chroma = chroma_client
        self._embedding_service = chroma_client._embedding_service

    def _get_collection(self):
        """Get the episodic memory collection."""
        return self._chroma.get_episodic_collection()

    def _generate_id(self, session_id: str, timestamp: datetime) -> str:
        """Generate unique episode ID."""
        ts_str = timestamp.isoformat().replace(":", "-").replace(".", "-")
        return f"ep_{session_id[:8]}_{ts_str}_{uuid.uuid4().hex[:8]}"

    async def create_episode(
        self,
        session_id: str,
        user_id: str,
        summary: str,
        content: str,
        key_entities: list[str] | None = None,
        sentiment: str = "neutral",
        topics: list[str] | None = None,
        duration_seconds: int = 0,
        message_count: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Create and store an episodic memory.

        Args:
            session_id: Session identifier
            user_id: User identifier
            summary: LLM-generated summary
            content: Raw interaction content
            key_entities: Entities discussed
            sentiment: Sentiment (positive, neutral, negative)
            topics: Topic tags
            duration_seconds: Session duration
            message_count: Number of messages
            metadata: Additional metadata

        Returns:
            str: Episode ID
        """
        collection = self._get_collection()
        ts = datetime.utcnow()
        ep_id = self._generate_id(session_id, ts)

        combined_text = f"Summary: {summary}\n\nContent: {content}"
        embedding = await self._embedding_service.embed_text(combined_text)

        collection.add(
            ids=[ep_id],
            documents=[combined_text],
            embeddings=[embedding],
            metadatas=[
                {
                    "user_id": user_id,
                    "session_id": session_id,
                    "timestamp": ts.isoformat(),
                    "summary": summary,
                    "key_entities": ",".join(key_entities) if key_entities else "",
                    "sentiment": sentiment,
                    "topics": ",".join(topics) if topics else "",
                    "duration_seconds": duration_seconds,
                    "message_count": message_count,
                    "type": "episode",
                    **(metadata or {}),
                }
            ],
        )

        logger.info(
            "episodic_memory.episode_created",
            ep_id=ep_id,
            session_id=session_id,
            sentiment=sentiment,
        )
        return ep_id

    async def get_episodes_for_session(self, session_id: str) -> list[dict[str, Any]]:
        """
        Get all episodes for a session.

        Args:
            session_id: Session to query

        Returns:
            list[dict]: Episodes for the session
        """
        collection = self._get_collection()

        results = collection.get(
            where={"session_id": session_id},
            include=["documents", "metadatas"],
        )

        episodes = []
        if results["ids"]:
            for i, ep_id in enumerate(results["ids"]):
                metadata = results["metadatas"][i]
                episodes.append(
                    {
                        "id": ep_id,
                        "summary": metadata.get("summary", ""),
                        "content": results["documents"][i],
                        "sentiment": metadata.get("sentiment", "neutral"),
                        "topics": metadata.get("topics", "").split(",")
                        if metadata.get("topics")
                        else [],
                        "key_entities": metadata.get("key_entities", "").split(",")
                        if metadata.get("key_entities")
                        else [],
                        "timestamp": metadata.get("timestamp", ""),
                    }
                )

        episodes.sort(key=lambda e: e.get("timestamp", ""), reverse=True)
        return episodes

    async def get_episodes_by_topic(
        self, topic: str, user_id: str | None = None, n: int = 10
    ) -> list[dict[str, Any]]:
        """
        Find episodes by topic.

        Args:
            topic: Topic to search
            user_id: Optional user filter
            n: Number of results

        Returns:
            list[dict]: Matching episodes
        """
        collection = self._get_collection()

        where_filter: dict[str, Any] = {}
        if user_id:
            where_filter["user_id"] = user_id

        embedding = await self._embedding_service.embed_text(f"Topic: {topic}")

        results = collection.query(
            query_embeddings=[embedding],
            n_results=n,
            where=where_filter if where_filter else None,
        )

        episodes = []
        if results["ids"] and results["ids"][0]:
            for i, ep_id in enumerate(results["ids"][0]):
                metadata = results["metadatas"][0][i]
                episodes.append(
                    {
                        "id": ep_id,
                        "session_id": metadata.get("session_id", ""),
                        "summary": metadata.get("summary", ""),
                        "sentiment": metadata.get("sentiment", "neutral"),
                        "topics": metadata.get("topics", "").split(",")
                        if metadata.get("topics")
                        else [],
                        "timestamp": metadata.get("timestamp", ""),
                        "distance": results["distances"][0][i]
                        if "distances" in results
                        else None,
                    }
                )

        return episodes

    async def get_recent_episodes(
        self, user_id: str | None = None, n: int = 10
    ) -> list[dict[str, Any]]:
        """
        Get most recent episodes.

        Args:
            user_id: Optional user filter
            n: Number of episodes

        Returns:
            list[dict]: Recent episodes
        """
        collection = self._get_collection()

        where_filter = {"user_id": user_id} if user_id else None

        results = collection.query(
            query_texts=[""],
            n_results=n,
            where=where_filter,
        )

        episodes = []
        if results["ids"] and results["ids"][0]:
            for i, ep_id in enumerate(results["ids"][0]):
                metadata = results["metadatas"][0][i]
                episodes.append(
                    {
                        "id": ep_id,
                        "session_id": metadata.get("session_id", ""),
                        "summary": metadata.get("summary", ""),
                        "sentiment": metadata.get("sentiment", "neutral"),
                        "topics": metadata.get("topics", "").split(",")
                        if metadata.get("topics")
                        else [],
                        "timestamp": metadata.get("timestamp", ""),
                    }
                )

        episodes.sort(key=lambda e: e.get("timestamp", ""), reverse=True)
        return episodes

    async def delete_episode(self, episode_id: str) -> bool:
        """
        Delete an episode.

        Args:
            episode_id: Episode to delete

        Returns:
            bool: True if deleted
        """
        collection = self._get_collection()
        try:
            collection.delete(ids=[episode_id])
            logger.info("episodic_memory.episode_deleted", ep_id=episode_id)
            return True
        except Exception as e:
            logger.error("episodic_memory.delete_error", ep_id=episode_id, error=str(e))
            return False


_episodic_memory: EpisodicMemory | None = None


def get_episodic_memory() -> EpisodicMemory:
    """Get singleton episodic memory instance."""
    global _episodic_memory
    if _episodic_memory is None:
        from app.database.chroma_client import get_chromadb_client

        _episodic_memory = EpisodicMemory(get_chromadb_client())
    return _episodic_memory
