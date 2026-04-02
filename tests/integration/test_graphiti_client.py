"""Integration tests for Graphiti client."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone

from app.graphiti.client import GraphitiClient, get_graphiti_client


class TestGraphitiClient:
    """Tests for GraphitiClient functionality."""

    @pytest.fixture
    def mock_driver(self):
        """Create mock Neo4j driver."""
        driver = MagicMock()
        return driver

    @pytest.fixture
    def graphiti_client(self, mock_driver):
        """Create GraphitiClient instance for testing."""
        return GraphitiClient(driver=mock_driver)

    def test_client_initialization(self, graphiti_client, mock_driver):
        """Test client initializes correctly."""
        assert graphiti_client._driver is mock_driver
        assert graphiti_client._graphiti is None
        assert graphiti_client.is_available is False

    @pytest.mark.asyncio
    async def test_initialize_without_graphiti_core(self, mock_driver):
        """Test initialize handles missing graphiti gracefully."""
        client = GraphitiClient(driver=mock_driver)
        await client.initialize()
        assert client.is_available is False

    def test_is_available_property(self, graphiti_client):
        """Test is_available returns bool."""
        assert isinstance(graphiti_client.is_available, bool)

    def test_node_labels_property(self, graphiti_client):
        """Test node_labels returns dict."""
        labels = graphiti_client.node_labels
        assert isinstance(labels, dict)
        assert "entity" in labels
        assert labels["entity"] == "graphiti_Entity"

    @pytest.mark.asyncio
    async def test_add_episode_returns_none_when_unavailable(self, graphiti_client):
        """Test add_episode returns None when Graphiti not available."""
        result = await graphiti_client.add_episode(
            name="test_episode",
            episode_body="Test content",
            source_name="test_source",
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_search_returns_empty_list_when_unavailable(self, graphiti_client):
        """Test search returns empty list when Graphiti not available."""
        result = await graphiti_client.search(query="test query")
        assert result == []

    @pytest.mark.asyncio
    async def test_verify_label_separation_returns_dict(
        self, graphiti_client, mock_driver
    ):
        """Test verify_label_separation returns dict with expected keys."""
        mock_session = AsyncMock()
        mock_result = AsyncMock()
        mock_result.data.return_value = [
            {"label": "graphiti_Entity", "count": 10},
            {"label": "Entity", "count": 5},
            {"label": "Document", "count": 3},
        ]
        mock_session.run.return_value = mock_result
        mock_driver.session.return_value.__aenter__ = AsyncMock(
            return_value=mock_session
        )
        mock_driver.session.return_value.__aexit__ = AsyncMock(return_value=None)

        result = await graphiti_client.verify_label_separation()

        assert isinstance(result, dict)
        assert "no_collision" in result
        assert "graphiti_labels_exist" in result
        assert "kr_labels_exist" in result


class TestGraphitiClientSingleton:
    """Tests for singleton pattern."""

    @pytest.fixture
    def mock_driver(self):
        return MagicMock()

    @pytest.mark.asyncio
    async def test_get_graphiti_client_returns_client(self, mock_driver):
        """Test get_graphiti_client returns GraphitiClient."""
        with patch("app.graphiti.client.GraphitiClient") as mock_client_class:
            mock_instance = GraphitiClient(driver=mock_driver)
            mock_client_class.return_value = mock_instance

            result = await get_graphiti_client(mock_driver)

            assert isinstance(result, GraphitiClient)
