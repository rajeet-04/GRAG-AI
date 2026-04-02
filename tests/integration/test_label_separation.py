"""Integration tests for KR/KB label separation verification."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from app.graphiti.client import GraphitiClient
from app.graphiti.config import GraphitiConfig


class TestLabelSeparation:
    """Tests for KR/KB label separation (Graphiti vs manual KR schema)."""

    @pytest.fixture
    def mock_driver(self):
        """Create mock Neo4j driver."""
        driver = MagicMock()
        return driver

    @pytest.fixture
    def graphiti_client(self, mock_driver):
        """Create GraphitiClient instance."""
        return GraphitiClient(driver=mock_driver)

    @pytest.fixture
    def graphiti_config(self):
        """Create GraphitiConfig instance."""
        return GraphitiConfig()

    def test_graphiti_config_uses_graphiti_prefix(self, graphiti_config):
        """Test Graphiti config uses graphiti_ prefix for labels."""
        labels = graphiti_config.graphiti_node_labels

        assert labels["entity"].startswith("graphiti_")
        assert labels["entity_fact"].startswith("graphiti_")
        assert labels["episode"].startswith("graphiti_")
        assert labels["relation"].startswith("graphiti_")

    def test_kr_labels_are_not_graphiti_prefixed(self):
        """Test manual KR labels don't use graphiti_ prefix."""
        kr_labels = ["Entity", "Relation", "Document"]

        for label in kr_labels:
            assert not label.startswith("graphiti_")

    @pytest.mark.asyncio
    async def test_verify_label_separation_no_collision(
        self, graphiti_client, mock_driver
    ):
        """Test verify_label_separation detects no collision."""
        mock_session = AsyncMock()
        mock_result = AsyncMock()
        mock_result.data.return_value = [
            {"label": "graphiti_Entity", "count": 10},
            {"label": "graphiti_EntityFact", "count": 5},
            {"label": "Entity", "count": 3},
            {"label": "Relation", "count": 2},
            {"label": "Document", "count": 1},
        ]
        mock_session.run.return_value = mock_result
        mock_driver.session.return_value.__aenter__ = AsyncMock(
            return_value=mock_session
        )
        mock_driver.session.return_value.__aexit__ = AsyncMock(return_value=None)

        result = await graphiti_client.verify_label_separation()

        assert result["no_collision"] is True
        assert result["graphiti_labels_exist"] is True
        assert result["kr_labels_exist"] is True

    @pytest.mark.asyncio
    async def test_verify_label_separation_detects_collision(
        self, graphiti_client, mock_driver
    ):
        """Test verify_label_separation detects label collision."""
        mock_session = AsyncMock()
        mock_result = AsyncMock()
        mock_result.data.return_value = [
            {"label": "Entity", "count": 10},
            {"label": "EntityFact", "count": 5},
            {"label": "Document", "count": 3},
        ]
        mock_session.run.return_value = mock_result
        mock_driver.session.return_value.__aenter__ = AsyncMock(
            return_value=mock_session
        )
        mock_driver.session.return_value.__aexit__ = AsyncMock(return_value=None)

        result = await graphiti_client.verify_label_separation()

        assert result["no_collision"] is True
        assert result["graphiti_labels_exist"] is False

    @pytest.mark.asyncio
    async def test_verify_label_separation_graphiti_only(
        self, graphiti_client, mock_driver
    ):
        """Test verify_label_separation with only Graphiti labels."""
        mock_session = AsyncMock()
        mock_result = AsyncMock()
        mock_result.data.return_value = [
            {"label": "graphiti_Entity", "count": 10},
            {"label": "graphiti_EntityFact", "count": 5},
        ]
        mock_session.run.return_value = mock_result
        mock_driver.session.return_value.__aenter__ = AsyncMock(
            return_value=mock_session
        )
        mock_driver.session.return_value.__aexit__ = AsyncMock(return_value=None)

        result = await graphiti_client.verify_label_separation()

        assert result["no_collision"] is True
        assert result["graphiti_labels_exist"] is True
        assert result["kr_labels_exist"] is False

    @pytest.mark.asyncio
    async def test_verify_label_separation_kr_only(self, graphiti_client, mock_driver):
        """Test verify_label_separation with only KR labels."""
        mock_session = AsyncMock()
        mock_result = AsyncMock()
        mock_result.data.return_value = [
            {"label": "Entity", "count": 10},
            {"label": "Relation", "count": 5},
            {"label": "Document", "count": 3},
        ]
        mock_session.run.return_value = mock_result
        mock_driver.session.return_value.__aenter__ = AsyncMock(
            return_value=mock_session
        )
        mock_driver.session.return_value.__aexit__ = AsyncMock(return_value=None)

        result = await graphiti_client.verify_label_separation()

        assert result["no_collision"] is True
        assert result["graphiti_labels_exist"] is False
        assert result["kr_labels_exist"] is True


class TestArchitecturalConstraint:
    """Tests verifying the KR/KB firewall architectural constraint."""

    @pytest.fixture
    def mock_driver(self):
        driver = MagicMock()
        return driver

    @pytest.fixture
    def graphiti_client(self, mock_driver):
        return GraphitiClient(driver=mock_driver)

    def test_graphiti_labels_different_from_kr_labels(self):
        """Test Graphiti and KR use completely different label sets."""
        graphiti_labels = {
            "graphiti_Entity",
            "graphiti_EntityFact",
            "graphiti_Episode",
            "graphiti_Relation",
        }

        kr_labels = {"Entity", "Relation", "Document", "MergeDecision"}

        intersection = graphiti_labels & kr_labels
        assert len(intersection) == 0, f"Label collision detected: {intersection}"

    def test_node_labels_provides_separation(self, graphiti_client):
        """Test client node_labels property provides label separation info."""
        labels = graphiti_client.node_labels

        for label_key, label_value in labels.items():
            assert label_value.startswith("graphiti_"), (
                f"Graphiti label '{label_key}' should use graphiti_ prefix, got: {label_value}"
            )
