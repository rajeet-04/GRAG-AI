"""Tests for Ollama web-search fallback helpers."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.retrieval.web_search import (
    execute_web_search,
    format_web_search_results,
    is_latest_data_query,
    should_trigger_web_search,
)


class TestWebSearchSignals:
    """Signal and trigger behavior tests."""

    def test_detects_latest_query(self):
        assert is_latest_data_query("What is the latest OpenAI release?") is True
        assert is_latest_data_query("Tell me about Tesla") is False

    def test_triggers_when_no_local_context(self):
        assert (
            should_trigger_web_search(
                query="Explain this topic",
                kr_entities=[],
                episodic_memories=[],
                semantic_preferences=[],
            )
            is True
        )

    def test_triggers_when_latest_even_with_context(self):
        assert (
            should_trigger_web_search(
                query="latest ai news",
                kr_entities=[{"name": "AI"}],
                episodic_memories=[],
                semantic_preferences=[],
            )
            is True
        )


class TestWebSearchExecution:
    """Execution path tests with mocked network."""

    @pytest.mark.asyncio
    async def test_execute_web_search_returns_normalized_results(self):
        fake_settings = SimpleNamespace(
            query_enable_web_search_fallback=True,
            ollama_enable_web_search=True,
            ollama_cloud_api_key="test-key",
            ollama_web_search_max_results=3,
        )

        mock_client = MagicMock()
        mock_client.web_search = AsyncMock(
            return_value={
                "results": [
                    {
                        "title": "Result 1",
                        "url": "https://example.com/1",
                        "content": "Snippet 1",
                    }
                ]
            }
        )

        with patch("app.retrieval.web_search.get_settings", return_value=fake_settings):
            with patch("app.retrieval.web_search.OllamaClient", return_value=mock_client):
                results = await execute_web_search("latest updates")

        assert len(results) == 1
        assert results[0]["title"] == "Result 1"
        assert results[0]["url"] == "https://example.com/1"

    @pytest.mark.asyncio
    async def test_execute_web_search_skips_without_api_key(self):
        fake_settings = SimpleNamespace(
            query_enable_web_search_fallback=True,
            ollama_enable_web_search=True,
            ollama_cloud_api_key="",
            ollama_web_search_max_results=5,
        )

        with patch("app.retrieval.web_search.get_settings", return_value=fake_settings):
            results = await execute_web_search("latest updates")

        assert results == []


class TestWebSearchFormatting:
    """Formatting behavior tests."""

    def test_formats_web_results_for_context(self):
        context = format_web_search_results(
            [
                {
                    "title": "Ollama Update",
                    "url": "https://ollama.com/blog",
                    "content": "New scheduling engine is available.",
                }
            ]
        )

        assert "Web Search Results" in context
        assert "Ollama Update" in context
        assert "Source: https://ollama.com/blog" in context
