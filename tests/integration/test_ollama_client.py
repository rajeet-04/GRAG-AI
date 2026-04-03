"""Integration tests for Ollama client."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from app.llm.ollama_client import OllamaClient, get_ollama_client


class TestOllamaClient:
    """Tests for OllamaClient functionality."""

    @pytest.fixture
    def client(self):
        """Create OllamaClient instance for testing."""
        return OllamaClient(use_cloud=False)

    def test_client_initialization_local(self):
        """Test client initializes with local settings."""
        client = OllamaClient(use_cloud=False)
        assert client.base_url == "http://localhost:11434"
        assert client.use_cloud is False

    def test_client_initialization_cloud(self):
        """Test client initializes with cloud settings."""
        client = OllamaClient(use_cloud=True)
        assert "api.ollama.com" in client.base_url
        assert client.use_cloud is True

    @pytest.mark.asyncio
    async def test_is_available_returns_bool(self, client):
        """Test is_available returns a boolean."""
        with patch.object(client, "list_models", new_callable=AsyncMock) as mock_list:
            mock_list.return_value = [{"name": "qwen2.5:7b"}]
            result = await client.is_available()
            assert isinstance(result, bool)

    @pytest.mark.asyncio
    async def test_list_models_returns_list(self, client):
        """Test list_models returns a list of model dicts."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "models": [
                {"name": "qwen2.5:7b", "size": 3826793472},
                {"name": "nomic-embed-text", "size": 274000000},
            ]
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(
            client, "_get_client", new_callable=AsyncMock
        ) as mock_get_client:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            result = await client.list_models()
            assert isinstance(result, list)
            assert len(result) == 2
            assert "name" in result[0]

    @pytest.mark.asyncio
    async def test_chat_returns_dict_with_keys(self, client):
        """Test chat returns dict with 'content', 'usage', 'model' keys."""
        mock_response = {
            "message": {"content": "Test response"},
            "usage": {"prompt_tokens": 10, "completion_tokens": 20},
            "model": "qwen2.5:7b",
            "done": True,
        }

        with patch.object(client, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response

            result = await client.chat(
                messages=[{"role": "user", "content": "Hello"}],
                temperature=0.7,
                max_tokens=100,
            )

            assert isinstance(result, dict)
            assert "content" in result
            assert "usage" in result
            assert "model" in result
            assert result["content"] == "Test response"

    @pytest.mark.asyncio
    async def test_generate_returns_dict_with_content(self, client):
        """Test generate returns dict with 'content' key."""
        mock_response = {
            "response": "Generated text",
            "usage": {"prompt_tokens": 5, "completion_tokens": 15},
            "model": "qwen2.5:7b",
        }

        with patch.object(client, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response

            result = await client.generate(
                prompt="Write a story",
                temperature=0.8,
                max_tokens=200,
            )

            assert isinstance(result, dict)
            assert "content" in result
            assert result["content"] == "Generated text"

    @pytest.mark.asyncio
    async def test_web_search_returns_results(self, client):
        """Test web_search returns raw search payload."""
        client.api_key = "test-key"

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "results": [
                {
                    "title": "Result 1",
                    "url": "https://example.com",
                    "content": "Snippet",
                }
            ]
        }

        mock_http_client = AsyncMock()
        mock_http_client.__aenter__.return_value = mock_http_client
        mock_http_client.__aexit__.return_value = None
        mock_http_client.post = AsyncMock(return_value=mock_response)

        with patch("app.llm.ollama_client.httpx.AsyncClient", return_value=mock_http_client):
            result = await client.web_search("latest ai", max_results=3)

        assert "results" in result
        assert len(result["results"]) == 1

    @pytest.mark.asyncio
    async def test_web_fetch_returns_content(self, client):
        """Test web_fetch returns fetched page payload."""
        client.api_key = "test-key"

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "title": "Example",
            "content": "Body",
            "links": ["https://example.com"],
        }

        mock_http_client = AsyncMock()
        mock_http_client.__aenter__.return_value = mock_http_client
        mock_http_client.__aexit__.return_value = None
        mock_http_client.post = AsyncMock(return_value=mock_response)

        with patch("app.llm.ollama_client.httpx.AsyncClient", return_value=mock_http_client):
            result = await client.web_fetch("https://example.com")

        assert result["title"] == "Example"
        assert "content" in result


class TestOllamaClientSingleton:
    """Tests for singleton pattern."""

    def test_get_ollama_client_returns_client(self):
        """Test get_ollama_client returns an OllamaClient."""
        client = get_ollama_client()
        assert isinstance(client, OllamaClient)

    def test_get_ollama_client_returns_same_instance(self):
        """Test singleton returns same instance."""
        client1 = get_ollama_client()
        client2 = get_ollama_client()
        assert client1 is client2
