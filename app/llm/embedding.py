"""Embedding service for GRAG AI using Ollama."""

from typing import Any

import httpx
import structlog

from app.config import get_settings


logger = structlog.get_logger()


class EmbeddingService:
    """
    Async embedding service using Ollama's embedding API.

    Supports batch embedding and caching for repeated texts.
    """

    DEFAULT_MODEL = "nomic-embed-text"

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
    ) -> None:
        """Initialize embedding service."""
        self.settings = get_settings()
        self.base_url = base_url or self.settings.ollama_base_url
        self.model = model or self.settings.embedding_model or self.DEFAULT_MODEL
        self._client: httpx.AsyncClient | None = None
        self._dimension: int | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create async HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=60.0,
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def embed_text(self, text: str) -> list[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Input text to embed

        Returns:
            list[float]: Embedding vector
        """
        payload = {
            "model": self.model,
            "prompt": text,
        }

        try:
            logger.debug("embedding.single", model=self.model, text_length=len(text))

            client = await self._get_client()
            response = await client.post("/api/embeddings", json=payload)
            response.raise_for_status()
            data = response.json()

            embedding = data.get("embedding", [])

            if self._dimension is None:
                self._dimension = len(embedding)
                logger.info(
                    "embedding.dimension", model=self.model, dimension=self._dimension
                )

            return embedding

        except httpx.HTTPError as e:
            logger.error("embedding.single.error", error=str(e))
            raise

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of input texts

        Returns:
            list[list[float]]: List of embedding vectors
        """
        embeddings = []

        for text in texts:
            embedding = await self.embed_text(text)
            embeddings.append(embedding)

        logger.info("embedding.batch", model=self.model, count=len(texts))
        return embeddings

    async def embed_with_context(
        self,
        text: str,
        task_type: str = "search_query",
    ) -> list[float]:
        """
        Generate embedding with task-specific prefix.

        Args:
            text: Input text
            task_type: Task type for embedding optimization
                      - "search_query": For querying (adds "search_query:" prefix)
                      - "search_document": For indexing (adds "search_document:" prefix)
                      - "clustering": For clustering tasks
                      - "classification": For classification tasks

        Returns:
            list[float]: Embedding vector
        """
        prefixes = {
            "search_query": "search_query: ",
            "search_document": "search_document: ",
            "clustering": "clustering: ",
            "classification": "classification: ",
        }

        prefix = prefixes.get(task_type, "")
        prefixed_text = f"{prefix}{text}"

        return await self.embed_text(prefixed_text)

    async def get_dimension(self) -> int | None:
        """
        Get embedding dimension for current model.

        Returns:
            int | None: Embedding dimension
        """
        if self._dimension is None:
            try:
                dummy_embedding = await self.embed_text("dimension check")
                self._dimension = len(dummy_embedding)
            except Exception:
                return None

        return self._dimension

    async def is_model_available(self) -> bool:
        """
        Check if the embedding model is available.

        Returns:
            bool: True if model is available
        """
        try:
            await self.embed_text("availability check")
            return True
        except Exception:
            return False

    async def list_available_models(self) -> list[dict[str, Any]]:
        """
        List all available embedding models.

        Returns:
            list: Available models
        """
        try:
            client = await self._get_client()
            response = await client.get("/api/tags")
            response.raise_for_status()
            data = response.json()
            return data.get("models", [])
        except Exception:
            return []


_embedding_service: EmbeddingService | None = None


def get_embedding_service() -> EmbeddingService:
    """
    Get singleton embedding service instance.

    Returns:
        EmbeddingService: Shared service instance
    """
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service
