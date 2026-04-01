"""Short-term memory (session storage) for GRAG AI KB layer."""

from datetime import datetime, timedelta
from typing import Any
import uuid

import structlog

from app.database.chroma_client import ChromaDBClient


logger = structlog.get_logger()


class ShortTermMemory:
    """
    Short-term memory store for session messages.

    Stores recent conversation messages with session tracking.
    Messages are automatically timestamped and sessionized.

    KB LAYER: All operations write to ChromaDB only.
    """

    COLLECTION_NAME = "short_term_memory"
    DEFAULT_SESSION_TTL = timedelta(hours=24)

    def __init__(self, chroma_client: ChromaDBClient) -> None:
        """Initialize short-term memory with ChromaDB client."""
        self._chroma = chroma_client
        self._initialized = False

    async def init(self) -> bool:
        """Initialize the short-term memory collection."""
        try:
            self._chroma._client.get_or_create_collection(
                name=self.COLLECTION_NAME,
                metadata={"description": "Short-term memory for session messages"},
            )
            self._initialized = True
            logger.info("short_term_memory.initialized")
            return True
        except Exception as e:
            logger.error("short_term_memory.init_error", error=str(e))
            return False

    def _get_collection(self):
        """Get the short-term memory collection."""
        return self._chroma._client.get_or_create_collection(
            name=self.COLLECTION_NAME,
        )

    def _generate_id(self, session_id: str, timestamp: datetime) -> str:
        """Generate unique message ID."""
        ts_str = timestamp.isoformat().replace(":", "-").replace(".", "-")
        return f"msg_{session_id[:8]}_{ts_str}_{uuid.uuid4().hex[:8]}"

    async def store_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None,
        timestamp: datetime | None = None,
    ) -> str:
        """
        Store a message in short-term memory.

        Args:
            session_id: Session identifier
            role: Message role ("user" or "assistant")
            content: Message content
            metadata: Optional additional metadata
            timestamp: Message timestamp (defaults to now)

        Returns:
            str: Message ID
        """
        collection = self._get_collection()
        ts = timestamp or datetime.utcnow()
        msg_id = self._generate_id(session_id, ts)

        collection.add(
            ids=[msg_id],
            documents=[content],
            metadatas=[
                {
                    "session_id": session_id,
                    "role": role,
                    "timestamp": ts.isoformat(),
                    "created_at": ts.isoformat(),
                    "type": "session_message",
                    **(metadata or {}),
                }
            ],
        )

        logger.debug(
            "short_term_memory.message_stored",
            msg_id=msg_id,
            session_id=session_id,
            role=role,
        )
        return msg_id

    async def get_session_messages(
        self,
        session_id: str,
        limit: int = 50,
        include_metadata: bool = False,
    ) -> list[dict[str, Any]]:
        """
        Get all messages for a session.

        Args:
            session_id: Session to retrieve
            limit: Maximum messages to return
            include_metadata: Include metadata in results

        Returns:
            list[dict]: Messages in chronological order
        """
        collection = self._get_collection()

        results = collection.get(
            where={"session_id": session_id},
            limit=limit,
            include=["documents", "metadatas"] if include_metadata else ["documents"],
        )

        messages = []
        if results["ids"]:
            for i, msg_id in enumerate(results["ids"]):
                msg = {"id": msg_id, "content": results["documents"][i]}
                if include_metadata and "metadatas" in results:
                    msg["metadata"] = results["metadatas"][i]
                messages.append(msg)

        messages.sort(
            key=lambda m: m.get("metadata", {}).get("timestamp", ""),
            reverse=False,
        )

        logger.debug(
            "short_term_memory.session_retrieved",
            session_id=session_id,
            count=len(messages),
        )
        return messages

    async def get_session_context(
        self,
        session_id: str,
        n: int = 10,
    ) -> list[dict[str, str]]:
        """
        Get last N messages for context injection.

        Args:
            session_id: Session to query
            n: Number of recent messages

        Returns:
            list[dict]: Messages with role and content only
        """
        collection = self._get_collection()

        results = collection.get(
            where={"session_id": session_id},
            limit=n,
            include=["documents", "metadatas"],
        )

        messages = []
        if results["ids"]:
            for i, msg_id in enumerate(results["ids"]):
                if "metadatas" in results:
                    messages.append(
                        {
                            "role": results["metadatas"][i].get("role", "user"),
                            "content": results["documents"][i],
                        }
                    )

        messages.sort(
            key=lambda m: m.get("timestamp", ""),
            reverse=True,
        )

        return messages[:n]

    async def clear_session(self, session_id: str) -> int:
        """
        Delete all messages for a session.

        Args:
            session_id: Session to clear

        Returns:
            int: Number of messages deleted
        """
        collection = self._get_collection()

        results = collection.get(
            where={"session_id": session_id},
            include=["ids"],
        )

        count = 0
        if results["ids"]:
            collection.delete(ids=results["ids"])
            count = len(results["ids"])

        logger.info(
            "short_term_memory.session_cleared",
            session_id=session_id,
            messages_deleted=count,
        )
        return count

    async def get_session_stats(self, session_id: str) -> dict[str, Any]:
        """
        Get statistics for a session.

        Args:
            session_id: Session to analyze

        Returns:
            dict: Session statistics
        """
        collection = self._get_collection()

        results = collection.get(
            where={"session_id": session_id},
            include=["metadatas"],
        )

        if not results["ids"]:
            return {"message_count": 0, "user_messages": 0, "assistant_messages": 0}

        user_count = 0
        assistant_count = 0
        timestamps = []

        for metadata in results.get("metadatas", []):
            role = metadata.get("role", "")
            if role == "user":
                user_count += 1
            elif role == "assistant":
                assistant_count += 1
            if "timestamp" in metadata:
                timestamps.append(metadata["timestamp"])

        timestamps.sort()

        return {
            "message_count": len(results["ids"]),
            "user_messages": user_count,
            "assistant_messages": assistant_count,
            "session_start": timestamps[0] if timestamps else None,
            "session_end": timestamps[-1] if timestamps else None,
        }


_short_term_memory: ShortTermMemory | None = None


async def get_short_term_memory() -> ShortTermMemory:
    """Get singleton short-term memory instance."""
    global _short_term_memory
    if _short_term_memory is None:
        from app.database.chroma_client import get_chromadb_client

        _short_term_memory = ShortTermMemory(get_chromadb_client())
        await _short_term_memory.init()
    return _short_term_memory
