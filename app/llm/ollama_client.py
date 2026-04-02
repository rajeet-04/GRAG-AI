"""Ollama LLM client for GRAG AI."""

from __future__ import annotations

from typing import Any

import httpx
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import get_settings


logger = structlog.get_logger()


class OllamaClient:
    """
    Async client for Ollama API with OpenAI-compatible interface.

    Provides chat completions with retry logic and error handling.
    """

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
    ) -> None:
        """Initialize Ollama client with settings."""
        self.settings = get_settings()
        self.base_url = base_url or self.settings.ollama_base_url
        self.model = model or self.settings.ollama_model
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create async HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=120.0,
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    async def _request(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        """
        Make HTTP request to Ollama with retry logic.

        Args:
            endpoint: API endpoint
            payload: Request payload

        Returns:
            dict: API response
        """
        client = await self._get_client()
        response = await client.post(endpoint, json=payload)
        response.raise_for_status()
        return response.json()

    async def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        stream: bool = False,
    ) -> dict[str, Any]:
        """
        Generate chat completion.

        Args:
            messages: List of message dicts with "role" and "content"
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens to generate
            stream: Whether to stream response

        Returns:
            dict: Response with "content", "usage", and "model"
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
        }

        try:
            logger.info(
                "ollama.chat.request",
                model=self.model,
                message_count=len(messages),
            )

            response = await self._request("/api/chat", payload)

            result = {
                "content": response.get("message", {}).get("content", ""),
                "usage": response.get("usage", {}),
                "model": response.get("model", self.model),
                "done": response.get("done", True),
            }

            logger.info(
                "ollama.chat.response",
                model=self.model,
                content_length=len(result["content"]),
            )

            return result

        except httpx.HTTPError as e:
            logger.error("ollama.chat.error", error=str(e))
            raise

    async def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        system: str | None = None,
    ) -> dict[str, Any]:
        """
        Generate text completion (non-chat interface).

        Args:
            prompt: Input prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens
            system: Optional system prompt

        Returns:
            dict: Response with "content"
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if system:
            payload["system"] = system

        try:
            logger.info("ollama.generate.request", model=self.model)

            response = await self._request("/api/generate", payload)

            return {
                "content": response.get("response", ""),
                "usage": response.get("usage", {}),
                "model": response.get("model", self.model),
            }

        except httpx.HTTPError as e:
            logger.error("ollama.generate.error", error=str(e))
            raise

    async def list_models(self) -> list[dict[str, Any]]:
        """
        List available models.

        Returns:
            list: Available models
        """
        try:
            client = await self._get_client()
            response = await client.get("/api/tags")
            response.raise_for_status()
            data = response.json()
            return data.get("models", [])
        except httpx.HTTPError as e:
            logger.error("ollama.list_models.error", error=str(e))
            return []

    async def is_available(self) -> bool:
        """
        Check if Ollama service is available.

        Returns:
            bool: True if service responds
        """
        try:
            models = await self.list_models()
            return len(models) >= 0
        except Exception:
            return False


_ollama_client: OllamaClient | None = None


def get_ollama_client() -> OllamaClient:
    """
    Get singleton Ollama client instance.

    Returns:
        OllamaClient: Shared client instance
    """
    global _ollama_client
    if _ollama_client is None:
        _ollama_client = OllamaClient()
    return _ollama_client
