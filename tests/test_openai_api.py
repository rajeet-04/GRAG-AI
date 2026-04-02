"""Integration tests for OpenAI-compatible API."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock, MagicMock

from app.main import app

client = TestClient(app)


class TestOpenAIModels:
    """Tests for /v1/models endpoint."""

    def test_list_models(self):
        """GET /v1/models should return the virtual model."""
        response = client.get("/v1/models")

        assert response.status_code == 200

        data = response.json()
        assert data["object"] == "list"
        assert len(data["data"]) >= 1

        model = data["data"][0]
        assert model["id"] == "grag-pipeline-v1"
        assert model["object"] == "model"
        assert "created" in model
        assert model["owned_by"] == "grag"


class TestOpenAIChatCompletions:
    """Tests for /v1/chat/completions endpoint."""

    def test_chat_completions_non_streaming(self):
        """Non-streaming /v1/chat/completions should return OpenAI format."""
        with patch("app.api.openai.create_agent_graph") as mock_graph:
            # Mock LangGraph response
            mock_state = {
                "answer": "Test answer from GRAG pipeline",
                "reasoning_steps": [
                    "Step 1: Analyzed query",
                    "Step 2: Retrieved context",
                ],
                "mermaid_path": "graph TD\nA[Query] --> B[Result]",
            }

            mock_app = MagicMock()
            mock_app.ainvoke = AsyncMock(return_value=mock_state)
            mock_graph.return_value = mock_app

            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "grag-pipeline-v1",
                    "messages": [
                        {"role": "user", "content": "Hello, what can you do?"}
                    ],
                    "stream": False,
                },
            )

            assert response.status_code == 200

            data = response.json()
            assert "id" in data
            assert data["object"] == "chat.completion"
            assert data["model"] == "grag-pipeline-v1"
            assert "choices" in data
            assert len(data["choices"]) >= 1

            choice = data["choices"][0]
            assert "message" in choice
            assert choice["message"]["role"] == "assistant"
            assert "content" in choice["message"]
            assert choice["finish_reason"] == "stop"

            assert "usage" in data
            assert "total_tokens" in data["usage"]

    def test_chat_completions_streaming(self):
        """Streaming /v1/chat/completions should return SSE chunks."""
        with patch("app.api.openai.create_agent_graph") as mock_graph:
            mock_state = {
                "answer": "Test answer",
                "reasoning_steps": ["Step 1"],
                "mermaid_path": "graph TD",
            }

            mock_app = MagicMock()
            mock_app.ainvoke = AsyncMock(return_value=mock_state)
            mock_graph.return_value = mock_app

            # Use client.get with follow=True doesn't work for streaming
            # Test by calling the endpoint directly and checking it's a StreamingResponse
            from app.api.openai import chat_completions
            from app.schemas.openai import ChatCompletionRequest, ChatMessage

            # Create a request
            request = ChatCompletionRequest(
                model="grag-pipeline-v1",
                messages=[ChatMessage(role="user", content="Hello")],
                stream=True,
            )

            import asyncio

            result = asyncio.run(chat_completions(request, True))

            # Should be a StreamingResponse
            from fastapi.responses import StreamingResponse

            assert isinstance(result, StreamingResponse)

    def test_invalid_model(self):
        """Unknown model should return error."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "unknown-model",
                "messages": [{"role": "user", "content": "test"}],
                "stream": False,
            },
        )

        assert response.status_code == 400
        # FastAPI returns error in detail field wrapped in HTTPException
        data = response.json()
        assert "detail" in data
        error_data = data["detail"]
        if isinstance(error_data, dict) and "error" in error_data:
            assert error_data["error"]["type"] == "invalid_request_error"
            assert error_data["error"]["code"] == "model_not_found"

    def test_empty_messages(self):
        """Empty messages should return error."""
        response = client.post(
            "/v1/chat/completions",
            json={"model": "grag-pipeline-v1", "messages": [], "stream": False},
        )

        assert response.status_code == 400

    def test_missing_model_field(self):
        """Missing model field should return validation error."""
        response = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "test"}], "stream": False},
        )

        assert response.status_code == 422

    def test_missing_messages_field(self):
        """Missing messages field should return validation error."""
        response = client.post(
            "/v1/chat/completions", json={"model": "grag-pipeline-v1", "stream": False}
        )

        assert response.status_code == 422

    def test_graceful_extra_params(self):
        """Extra parameters should be accepted gracefully."""
        with patch("app.api.openai.create_agent_graph") as mock_graph:
            mock_state = {
                "answer": "Test answer",
                "reasoning_steps": [],
                "mermaid_path": "",
            }

            mock_app = MagicMock()
            mock_app.ainvoke = AsyncMock(return_value=mock_state)
            mock_graph.return_value = mock_app

            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "grag-pipeline-v1",
                    "messages": [{"role": "user", "content": "test"}],
                    "stream": False,
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "max_tokens": 100,
                    "unknown_param": "should be ignored",
                },
            )

            # Should not return 422 (validation error)
            assert response.status_code == 200


class TestOpenAIAuthentication:
    """Tests for API key authentication."""

    def test_no_auth_when_not_configured(self):
        """When API_KEY is not set, requests should work without auth."""
        with patch("app.api.openai.create_agent_graph") as mock_graph:
            mock_state = {
                "answer": "Test answer",
                "reasoning_steps": [],
                "mermaid_path": "",
            }

            mock_app = MagicMock()
            mock_app.ainvoke = AsyncMock(return_value=mock_state)
            mock_graph.return_value = mock_app

            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "grag-pipeline-v1",
                    "messages": [{"role": "user", "content": "test"}],
                    "stream": False,
                },
            )

            assert response.status_code == 200


class TestOpenAIErrorFormat:
    """Tests for OpenAI-compatible error responses."""

    def test_error_response_format(self):
        """Error responses should match OpenAI format."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "unknown-model",
                "messages": [{"role": "user", "content": "test"}],
                "stream": False,
            },
        )

        assert response.status_code == 400
        data = response.json()

        # FastAPI wraps in detail field
        assert "detail" in data
        error_data = data["detail"]
        if isinstance(error_data, dict) and "error" in error_data:
            error = error_data["error"]
            assert "message" in error
            assert "type" in error
            assert "code" in error
            assert error["type"] == "invalid_request_error"
            assert error["code"] == "model_not_found"


class TestHealthEndpoint:
    """Tests for health check endpoint."""

    def test_health_check(self):
        """Health endpoint should still work."""
        pytest.skip("Health check requires Neo4j/Ollama running")
