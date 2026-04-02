"""Integration tests for the retrieval pipeline.

Tests the full flow from KR search through fallback, ranking, and
context building.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch


class TestRetrievalPipeline:
    """Integration tests for full retrieval pipeline."""

    def test_graph_state_has_ranked_fields(self):
        """GraphState should have ranked_entities and ranked_relations."""
        from app.agents.state import create_initial_state

        state = create_initial_state(user_query="test", session_id="test-session")
        assert "ranked_entities" in state
        assert "ranked_relations" in state
        assert state["ranked_entities"] == []
        assert state["ranked_relations"] == []

    @pytest.mark.asyncio
    async def test_context_builder_ranks_results(self):
        """Context Builder should call ranking before merge."""
        from app.agents.context_builder import context_builder_node

        state = {
            "user_query": "test",
            "session_id": "test-session",
            "kr_entities": [
                {"entity_id": "e1", "entity_name": "Entity 1", "confidence": 0.9},
            ],
            "kr_relations": [],
            "episodic_memories": [],
            "semantic_preferences": [],
            "agent_trace": [],
        }

        result = await context_builder_node(state)

        assert "ranked_entities" in result
        assert len(result["ranked_entities"]) == 1
        assert result["ranked_entities"][0]["entity_id"] == "e1"
        assert result["ranked_entities"][0]["unified_score"] == 0.9

    @pytest.mark.asyncio
    async def test_context_builder_merges_graph_and_vector(self):
        """Context Builder merges graph and vector with late fusion."""
        from app.agents.context_builder import context_builder_node

        state = {
            "user_query": "test",
            "session_id": "test-session",
            "kr_entities": [
                {"entity_id": "e1", "entity_name": "Entity 1", "confidence": 0.9},
            ],
            "kr_relations": [],
            "episodic_memories": [
                {"id": "ep1", "content": "Memory 1", "distance": 0.2},
            ],
            "semantic_preferences": [],
            "agent_trace": [],
        }

        result = await context_builder_node(state)

        # e1 (graph) + ep1 (vector-only) = 2 entities
        assert len(result["ranked_entities"]) == 2
        assert result["merged_context"] != ""
        assert result["token_count"] > 0

    @pytest.mark.asyncio
    async def test_context_builder_empty_results(self):
        """Empty results produce empty context."""
        from app.agents.context_builder import context_builder_node

        state = {
            "user_query": "test",
            "session_id": "test-session",
            "kr_entities": [],
            "kr_relations": [],
            "episodic_memories": [],
            "semantic_preferences": [],
            "agent_trace": [],
        }

        result = await context_builder_node(state)

        assert result["merged_context"] == ""
        assert result["token_count"] == 0
        assert "No results to merge" in result["agent_trace"][-1]

    @pytest.mark.asyncio
    async def test_context_builder_deduplicates(self):
        """Same entity from graph and vector is merged, not duplicated."""
        from app.agents.context_builder import context_builder_node

        state = {
            "user_query": "test",
            "session_id": "test-session",
            "kr_entities": [
                {"entity_id": "e1", "entity_name": "Entity 1", "confidence": 0.9},
            ],
            "kr_relations": [],
            "episodic_memories": [
                {"id": "e1", "content": "Entity 1 memory", "distance": 0.2},
            ],
            "semantic_preferences": [],
            "agent_trace": [],
        }

        result = await context_builder_node(state)

        assert len(result["ranked_entities"]) == 1
        assert result["ranked_entities"][0]["source"] == "merged"
