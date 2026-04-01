"""ChromaDB client for GRAG AI KB (Knowledge Base) memory.

Provides dual-collection architecture:
- semantic_memory: User profiles, preferences, long-term traits (upsert semantics)
- episodic_memory: Session snapshots, time-bound interactions (append + timestamp query)
"""

from datetime import datetime
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings
import structlog

from app.config import get_settings
from app.llm.embedding import get_embedding_service


logger = structlog.get_logger()


class ChromaDBClient:
    """
    ChromaDB client for KB memory storage.

    Manages two collections:
    - semantic_memory: Upsert-based storage for user traits
    - episodic_memory: Append-based storage for session history
    """

    COLLECTION_SEMANTIC = "semantic_memory"
    COLLECTION_EPISODIC = "episodic_memory"

    def __init__(self, persist_path: str | None = None) -> None:
        """Initialize ChromaDB client with persistence."""
        self.settings = get_settings()
        self.persist_path = persist_path or str(self.settings.chromadb_path)

        Path(self.persist_path).mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(
            path=self.persist_path,
            settings=ChromaSettings(
                anonymized_telemetry=False,
                allow_reset=True,
            ),
        )

        self._embedding_service = get_embedding_service()
        self._initialized = False

        logger.info("chromadb.initialized", path=self.persist_path)

    def _generate_id(self, prefix: str, *parts: str) -> str:
        """Generate a unique ID for a document."""
        import hashlib

        key = "_".join(str(p) for p in parts)
        hash_key = hashlib.md5(key.encode()).hexdigest()[:8]
        return f"{prefix}_{hash_key}"

    async def init_collections(self) -> dict[str, bool]:
        """
        Initialize both memory collections.

        Returns:
            dict: Status of each collection initialization
        """
        results = {}

        try:
            self._client.get_or_create_collection(
                name=self.COLLECTION_SEMANTIC,
                metadata={
                    "description": "Semantic memory for user profiles and preferences"
                },
            )
            results[self.COLLECTION_SEMANTIC] = True
            logger.info("chromadb.collection.created", name=self.COLLECTION_SEMANTIC)
        except Exception as e:
            logger.error(
                "chromadb.collection.error", name=self.COLLECTION_SEMANTIC, error=str(e)
            )
            results[self.COLLECTION_SEMANTIC] = False

        try:
            self._client.get_or_create_collection(
                name=self.COLLECTION_EPISODIC,
                metadata={"description": "Episodic memory for session snapshots"},
            )
            results[self.COLLECTION_EPISODIC] = True
            logger.info("chromadb.collection.created", name=self.COLLECTION_EPISODIC)
        except Exception as e:
            logger.error(
                "chromadb.collection.error", name=self.COLLECTION_EPISODIC, error=str(e)
            )
            results[self.COLLECTION_EPISODIC] = False

        self._initialized = True
        return results

    def get_semantic_collection(self):
        """Get the semantic memory collection."""
        return self._client.get_or_create_collection(
            name=self.COLLECTION_SEMANTIC,
        )

    def get_episodic_collection(self):
        """Get the episodic memory collection."""
        return self._client.get_or_create_collection(
            name=self.COLLECTION_EPISODIC,
        )

    async def upsert_semantic(
        self,
        user_id: str,
        trait_type: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Upsert a semantic memory entry.

        Updates existing entry if user_id + trait_type exists,
        otherwise creates new entry.

        Args:
            user_id: User identifier
            trait_type: Type of trait
            content: Trait value
            metadata: Optional extra data

        Returns:
            str: Document ID
        """
        collection = self.get_semantic_collection()
        doc_id = self._generate_id("sem", user_id, trait_type)

        embedding = await self._embedding_service.embed_text(content)

        collection.upsert(
            ids=[doc_id],
            documents=[content],
            embeddings=[embedding],
            metadatas=[
                {
                    "user_id": user_id,
                    "trait_type": trait_type,
                    "updated_at": datetime.utcnow().isoformat(),
                    **(metadata or {}),
                }
            ],
        )

        logger.info(
            "chromadb.semantic.upserted",
            doc_id=doc_id,
            user_id=user_id,
            trait_type=trait_type,
        )
        return doc_id

    async def query_semantic(
        self,
        query_text: str,
        user_id: str | None = None,
        n_results: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Query semantic memory.

        Args:
            query_text: Search query
            user_id: Optional user filter
            n_results: Number of results

        Returns:
            list: Matching semantic memories
        """
        collection = self.get_semantic_collection()
        embedding = await self._embedding_service.embed_text(query_text)

        where_filter = {"user_id": user_id} if user_id else None

        results = collection.query(
            query_embeddings=[embedding],
            n_results=n_results,
            where=where_filter,
        )

        memories = []
        if results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                memories.append(
                    {
                        "id": doc_id,
                        "content": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "distance": results["distances"][0][i]
                        if "distances" in results
                        else None,
                    }
                )

        logger.info(
            "chromadb.semantic.queried", query=query_text, results=len(memories)
        )
        return memories

    async def add_episode(
        self,
        session_id: str,
        user_id: str,
        summary: str,
        content: str,
        metadata: dict[str, Any] | None = None,
        timestamp: datetime | None = None,
    ) -> str:
        """
        Add an episodic memory entry.

        Appends a new episode (never updates existing).

        Args:
            session_id: Session identifier
            user_id: User identifier
            summary: LLM-generated summary
            content: Raw interaction content
            metadata: Optional extra data
            timestamp: Episode timestamp (defaults to now)

        Returns:
            str: Document ID
        """
        collection = self.get_episodic_collection()
        ts = timestamp or datetime.utcnow()
        doc_id = self._generate_id("ep", user_id, session_id, ts.isoformat())

        combined_text = f"Summary: {summary}\n\nContent: {content}"
        embedding = await self._embedding_service.embed_text(combined_text)

        collection.add(
            ids=[doc_id],
            documents=[combined_text],
            embeddings=[embedding],
            metadatas=[
                {
                    "user_id": user_id,
                    "session_id": session_id,
                    "timestamp": ts.isoformat(),
                    "summary": summary,
                    **(metadata or {}),
                }
            ],
        )

        logger.info(
            "chromadb.episode.added",
            doc_id=doc_id,
            user_id=user_id,
            session_id=session_id,
        )
        return doc_id

    async def query_episodes(
        self,
        user_id: str | None = None,
        session_id: str | None = None,
        time_range: tuple[datetime, datetime] | None = None,
        n_results: int = 5,
        query_text: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Query episodic memory.

        Args:
            user_id: Filter by user
            session_id: Filter by session
            time_range: Filter by time range (start, end)
            n_results: Number of results
            query_text: Optional semantic search

        Returns:
            list: Matching episodes
        """
        collection = self.get_episodic_collection()

        where_filter: dict[str, Any] = {}
        if user_id:
            where_filter["user_id"] = user_id
        if session_id:
            where_filter["session_id"] = session_id
        if time_range:
            where_filter["timestamp"] = {
                "$gte": time_range[0].isoformat(),
                "$lte": time_range[1].isoformat(),
            }

        if query_text:
            embedding = await self._embedding_service.embed_text(query_text)
            results = collection.query(
                query_embeddings=[embedding],
                n_results=n_results,
                where=where_filter if where_filter else None,
            )
        else:
            results = collection.query(
                query_texts=[""],
                n_results=n_results,
                where=where_filter if where_filter else None,
            )

        episodes = []
        if results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                episodes.append(
                    {
                        "id": doc_id,
                        "content": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "distance": results["distances"][0][i]
                        if "distances" in results
                        else None,
                    }
                )

        logger.info(
            "chromadb.episodes.queried", query=query_text, results=len(episodes)
        )
        return episodes

    def reset(self) -> None:
        """Reset all collections (dangerous!)."""
        self._client.reset()
        self._initialized = False
        logger.warning("chromadb.reset")


_chromadb_client: ChromaDBClient | None = None


def get_chromadb_client() -> ChromaDBClient:
    """
    Get singleton ChromaDB client instance.

    Returns:
        ChromaDBClient: Shared client instance
    """
    global _chromadb_client
    if _chromadb_client is None:
        _chromadb_client = ChromaDBClient()
    return _chromadb_client
