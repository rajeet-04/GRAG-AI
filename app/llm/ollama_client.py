"""Ollama LLM client for GRAG AI.

Implements the Ollama native API per AGENT/OLLAMA.md:

Local (Docker):
  POST http://ollama:11434/api/chat
  Body: { model, messages, options: { temperature, num_predict }, think, stream }
  Response: { message: { content, thinking }, done, model }

Cloud (ollama.com direct API):
  POST https://ollama.com/api/chat
  Header: Authorization: Bearer <key>
  Body: { model, messages, options: { temperature, num_predict }, think, stream }
  Response: same as local

Note: qwen3.5 is a thinking model — set think=False for JSON extraction tasks.
"""

from __future__ import annotations

from typing import Any

import httpx
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import get_settings


logger = structlog.get_logger()


class OllamaClient:
    """
    Async client for Ollama API (local Docker or ollama.com cloud).

    Uses the native Ollama /api/chat endpoint for both local and cloud,
    with Authorization Bearer for cloud access.
    """

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        use_cloud: bool = False,
    ) -> None:
        """Initialize Ollama client.

        Args:
            base_url: Override base URL (default from settings)
            model: Override model name
            api_key: Override API key
            use_cloud: If True, use cloud config (ollama.com API with key)
        """
        self.settings = get_settings()

        if use_cloud:
            # Cloud: https://ollama.com  (native /api/chat, not OpenAI-compat)
            self.base_url = base_url or "https://ollama.com"
            self.model = model or self.settings.ollama_cloud_model
            self.api_key = api_key or self.settings.ollama_cloud_api_key
        else:
            # Local Docker: http://ollama:11434
            self.base_url = base_url or self.settings.ollama_base_url
            self.model = model or self.settings.ollama_model
            self.api_key = api_key or getattr(self.settings, "ollama_api_key", None)

        self.use_cloud = use_cloud
        self._client: httpx.AsyncClient | None = None

    def _get_headers(self) -> dict[str, str]:
        """Build request headers — adds Bearer token if API key is set."""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or lazily create the async HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=180.0,
                headers=self._get_headers(),
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
        """POST to Ollama endpoint with retry logic.

        Both local and cloud use the same native Ollama API format.
        Retries up to 3 times with exponential backoff (2s → 10s).
        """
        client = await self._get_client()
        try:
            response = await client.post(endpoint, json=payload)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.warning(
                "ollama.request.retry",
                endpoint=endpoint,
                model=payload.get("model"),
                error=str(e),
            )
            raise

    async def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        stream: bool = False,
        think: bool = False,
    ) -> dict[str, Any]:
        """Generate chat completion using native Ollama /api/chat.

        Per OLLAMA.md:
        - Local:  POST /api/chat  (no auth needed)
        - Cloud:  POST /api/chat with Authorization: Bearer <key>
        - qwen3.5 is a thinking model; set think=False for JSON extraction
          to avoid the thinking trace polluting structured output.

        Args:
            messages:    List of {"role": ..., "content": ...} dicts
            temperature: Sampling temperature (0-1)
            max_tokens:  Max tokens to generate (num_predict in Ollama)
            stream:      Whether to stream (always False for our pipeline)
            think:       Enable reasoning trace (False = faster, JSON-safe)

        Returns:
            dict with keys: content (str), thinking (str), model (str), done (bool)
        """
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
            "think": think,
            "stream": stream,
        }

        try:
            logger.info(
                "ollama.chat.request",
                model=self.model,
                use_cloud=self.use_cloud,
                base_url=self.base_url,
                message_count=len(messages),
                think=think,
            )

            response = await self._request("/api/chat", payload)

            message = response.get("message", {})
            content = message.get("content", "")
            thinking = message.get("thinking", "")

            logger.info(
                "ollama.chat.response",
                model=self.model,
                content_length=len(content),
                has_thinking=bool(thinking),
                done=response.get("done", True),
            )

            return {
                "content": content,
                "thinking": thinking,
                "usage": response.get("prompt_eval_count", 0),
                "model": response.get("model", self.model),
                "done": response.get("done", True),
            }

        except Exception as e:
            logger.error("ollama.chat.error", model=self.model, error=str(e))
            raise

    async def chat_stream(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        think: bool = False,
    ):
        """Stream chat completion from Ollama — yields text chunks as they arrive.

        Uses Ollama's native streaming API (stream=true in /api/chat).
        Each line of the NDJSON response is decoded and the content delta
        is yielded immediately so callers can pipe tokens to the client.

        Per OLLAMA.md: thinking-capable models emit a `thinking` field
        alongside content. When think=False, only `content` deltas are yielded.

        Args:
            messages:    Conversation messages
            temperature: Sampling temperature
            max_tokens:  Max tokens (num_predict)
            think:       Emit thinking traces (False = content-only)

        Yields:
            str: Text delta chunks from the model
        """
        import json as _json

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
            "think": think,
            "stream": True,
        }

        client = await self._get_client()

        logger.info(
            "ollama.chat_stream.start",
            model=self.model,
            use_cloud=self.use_cloud,
            message_count=len(messages),
        )

        try:
            async with client.stream("POST", "/api/chat", json=payload) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        data = _json.loads(line)
                    except _json.JSONDecodeError:
                        continue

                    message = data.get("message", {})
                    # Skip thinking trace tokens when think=False
                    if message.get("thinking"):
                        continue
                    delta = message.get("content", "")
                    if delta:
                        yield delta

                    if data.get("done"):
                        break

        except Exception as e:
            logger.error("ollama.chat_stream.error", model=self.model, error=str(e))
            raise

    async def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        system: str | None = None,
        think: bool = False,
    ) -> dict[str, Any]:
        """Generate text completion (non-chat interface).

        Uses /api/generate for single-turn prompt completion.
        """
        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
            "think": think,
            "stream": False,
        }

        if system:
            payload["system"] = system

        try:
            logger.info("ollama.generate.request", model=self.model)
            response = await self._request("/api/generate", payload)
            return {
                "content": response.get("response", ""),
                "thinking": response.get("thinking", ""),
                "model": response.get("model", self.model),
            }
        except Exception as e:
            logger.error("ollama.generate.error", model=self.model, error=str(e))
            raise

    async def list_models(self) -> list[dict[str, Any]]:
        """List models available on this Ollama instance."""
        try:
            client = await self._get_client()
            response = await client.get("/api/tags")
            response.raise_for_status()
            return response.json().get("models", [])
        except Exception as e:
            logger.error("ollama.list_models.error", error=str(e))
            return []

    async def is_available(self) -> bool:
        """Check if Ollama service is reachable."""
        try:
            models = await self.list_models()
            return len(models) >= 0
        except Exception:
            return False


_ollama_client: OllamaClient | None = None


def get_ollama_client() -> OllamaClient:
    """Get singleton local OllamaClient instance."""
    global _ollama_client
    if _ollama_client is None:
        _ollama_client = OllamaClient()
    return _ollama_client
