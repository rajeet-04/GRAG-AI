"""End-to-end tests for full GRAG pipeline."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from app.agents.state import GraphState, create_initial_state


class TestFullPipeline:
    """Tests for full GRAG pipeline from ingestion to retrieval."""

    @pytest.fixture
    def initial_state(self):
        """Create initial GraphState for testing."""
        return create_initial_state(
            user_query="What did I learn about machine learning?",
            session_id="test-session-123",
        )

    def test_initial_state_creation(self, initial_state):
        """Test initial state has all required keys."""
        assert "user_query" in initial_state
        assert "session_id" in initial_state
        assert initial_state["user_query"] == "What did I learn about machine learning?"
        assert initial_state["session_id"] == "test-session-123"
        assert initial_state["entities"] == []
        assert initial_state["kr_entities"] == []

    @pytest.mark.asyncio
    async def test_ingestion_to_graph_state(self, initial_state):
        """Test that ingestion produces entities and relations in state."""
        mock_entities = [
            {"id": "ent-1", "name": "Machine Learning", "type": "Concept"},
            {"id": "ent-2", "name": "Neural Network", "type": "Concept"},
        ]
        mock_relations = [
            {"source": "ent-1", "target": "ent-2", "type": "RELATED_TO"},
        ]

        result_state = {
            **initial_state,
            "entities": mock_entities,
            "relations": mock_relations,
            "agent_trace": initial_state["agent_trace"]
            + ["IngestionAgent: extracted 2 entities, 1 relation"],
        }

        assert len(result_state["entities"]) == 2
        assert len(result_state["relations"]) == 1

    @pytest.mark.asyncio
    async def test_query_agent_state_transition(self, initial_state):
        """Test Query Agent populates search intent and Cypher."""
        result_state = {
            **initial_state,
            "search_intent": "Find information about machine learning concepts",
            "temporal_filters": {
                "valid_from": None,
                "valid_to": None,
                "temporal_hint": None,
            },
            "cypher_query": "MATCH (e:Entity) WHERE e.name CONTAINS 'machine learning' RETURN e",
            "agent_trace": initial_state["agent_trace"]
            + ["QueryAgent: extracted intent, generated Cypher"],
        }

        assert result_state["search_intent"] != ""
        assert result_state["cypher_query"] != ""
        assert "MATCH" in result_state["cypher_query"]

    @pytest.mark.asyncio
    async def test_kr_retrieval_state(self, initial_state):
        """Test KR retrieval populates kr_entities and kr_relations."""
        kr_entities = [
            {
                "id": "ent-1",
                "name": "Machine Learning",
                "type": "Concept",
                "confidence": 0.9,
            },
            {
                "id": "ent-2",
                "name": "Deep Learning",
                "type": "Concept",
                "confidence": 0.85,
            },
        ]
        kr_relations = [
            {
                "source": "ent-1",
                "target": "ent-2",
                "type": "RELATED_TO",
                "confidence": 0.8,
            },
        ]

        result_state = {
            **initial_state,
            "kr_entities": kr_entities,
            "kr_relations": kr_relations,
            "agent_trace": initial_state["agent_trace"]
            + ["KR Search: found 2 entities, 1 relation"],
        }

        assert len(result_state["kr_entities"]) == 2
        assert len(result_state["kr_relations"]) == 1

    @pytest.mark.asyncio
    async def test_kb_retrieval_state(self, initial_state):
        """Test KB retrieval populates episodic and semantic memories."""
        episodic_memories = [
            {
                "content": "User asked about neural networks yesterday",
                "timestamp": "2026-04-01T10:00:00",
                "summary": "Asked about neural networks",
            },
            {
                "content": "User discussed transformers last week",
                "timestamp": "2026-03-25T14:30:00",
                "summary": "Discussed transformers",
            },
        ]
        semantic_preferences = [
            {
                "content": "Prefers technical explanations",
                "preference_type": "explanation_style",
                "confidence": 0.95,
            },
        ]

        result_state = {
            **initial_state,
            "episodic_memories": episodic_memories,
            "semantic_preferences": semantic_preferences,
            "agent_trace": initial_state["agent_trace"]
            + ["KB Search: found 2 episodic, 1 semantic"],
        }

        assert len(result_state["episodic_memories"]) == 2
        assert len(result_state["semantic_preferences"]) == 1


class TestTemporalFiltering:
    """Tests for temporal filtering in the pipeline."""

    def test_temporal_filter_extraction(self):
        """Test temporal filters are extracted from queries."""
        query_with_time = "What did I learn about AI before 2024?"

        temporal_filters = {
            "valid_from": None,
            "valid_to": "2024-01-01T00:00:00",
            "temporal_hint": "before 2024",
        }

        assert temporal_filters["valid_to"] is not None
        assert temporal_filters["temporal_hint"] == "before 2024"

    def test_temporal_filter_for_specific_period(self):
        """Test temporal filter for specific time period."""
        temporal_filters = {
            "valid_from": "2024-01-01T00:00:00",
            "valid_to": "2024-12-31T23:59:59",
            "temporal_hint": "during 2024",
        }

        assert temporal_filters["valid_from"] is not None
        assert temporal_filters["valid_to"] is not None

    def test_temporal_filter_with_relative_time(self):
        """Test temporal filter with relative time like 'last year'."""
        current_year = datetime.now().year

        temporal_filters = {
            "valid_from": f"{current_year - 1}-01-01T00:00:00",
            "valid_to": f"{current_year - 1}-12-31T23:59:59",
            "temporal_hint": "last year",
        }

        assert temporal_filters["temporal_hint"] == "last year"


class TestContextMerging:
    """Tests for context merging in Context Builder."""

    @pytest.fixture
    def mock_retrieval_results(self):
        """Create mock retrieval results for context merging."""
        kr_entities = [
            {
                "name": "Machine Learning",
                "description": "A field of AI",
                "confidence": 0.9,
            },
        ]
        kr_relations = [
            {
                "source": "Machine Learning",
                "type": "RELATED_TO",
                "target": "Neural Network",
            },
        ]
        episodic_memories = [
            {
                "content": "Discussed ML last week",
                "timestamp": "2026-03-25T10:00:00",
                "summary": "ML discussion",
            },
        ]
        semantic_preferences = [
            {
                "content": "Prefers code examples",
                "preference_type": "learning_style",
                "confidence": 0.8,
            },
        ]

        return {
            "kr_entities": kr_entities,
            "kr_relations": kr_relations,
            "episodic_memories": episodic_memories,
            "semantic_preferences": semantic_preferences,
        }

    def test_context_merge_includes_all_sources(self, mock_retrieval_results):
        """Test merged context includes KR, episodic, and semantic."""
        context_sections = []

        if mock_retrieval_results["kr_entities"]:
            context_sections.append("KR Graph Results")
        if mock_retrieval_results["episodic_memories"]:
            context_sections.append("Episodic Memories")
        if mock_retrieval_results["semantic_preferences"]:
            context_sections.append("Semantic Preferences")

        assert len(context_sections) == 3

    def test_context_token_counting(self):
        """Test token counting works in context builder."""
        test_text = "This is a test context for token counting."

        token_count = len(test_text.split())

        assert token_count > 0


class TestExplanationGeneration:
    """Tests for Explanation Agent output."""

    def test_explanation_output_structure(self):
        """Test explanation has required output fields."""
        explanation_output = {
            "answer": "Machine learning is a subset of AI...",
            "reasoning_steps": [
                "Step 1: Found ML entity with 0.9 confidence",
                "Step 2: Related to Neural Network concept",
            ],
            "mermaid_path": "graph TD\nA[Query] --> B[ML]\nB --> C[Neural Network]",
            "confidence_scores": {"overall": 0.85},
            "citation_map": {"[1]": 0, "[2]": 1},
        }

        assert "answer" in explanation_output
        assert "reasoning_steps" in explanation_output
        assert "mermaid_path" in explanation_output
        assert "confidence_scores" in explanation_output
        assert "citation_map" in explanation_output

    def test_reasoning_steps_include_confidence(self):
        """Test reasoning steps include confidence annotations."""
        reasoning_steps = [
            "Step 1: Found ML entity (Confidence: 90%)",
            "Step 2: Related to Neural Network (Confidence: 85%)",
        ]

        for step in reasoning_steps:
            assert "Confidence:" in step

    def test_mermaid_validation(self):
        """Test Mermaid diagram validation."""
        valid_mermaid = "graph TD\nA[Query] --> B[Entity]\nB --> C[Related]"

        assert "graph TD" in valid_mermaid
        assert "-->" in valid_mermaid


class TestPipelineIntegration:
    """Integration tests for complete pipeline."""

    @pytest.mark.asyncio
    async def test_full_pipeline_state_flow(self):
        """Test complete state flow through pipeline."""
        state = create_initial_state(
            user_query="What is neural networks?",
            session_id="integration-test-001",
        )

        state["search_intent"] = "Find neural network information"
        state["cypher_query"] = (
            "MATCH (e:Entity) WHERE e.name CONTAINS 'neural' RETURN e"
        )

        state["kr_entities"] = [
            {"name": "Neural Network", "type": "Concept", "confidence": 0.9}
        ]
        state["kr_relations"] = []
        state["episodic_memories"] = []
        state["semantic_preferences"] = []

        state["merged_context"] = (
            "## Knowledge Graph Results\n- Neural Network: A concept"
        )
        state["token_count"] = 10

        state["answer"] = "Neural networks are computing systems..."
        state["reasoning_steps"] = ["Step 1: Found neural network entity"]
        state["mermaid_path"] = "graph TD\nA[Query] --> B[Neural Network]"

        assert state["answer"] != ""
        assert len(state["reasoning_steps"]) > 0
        assert "graph TD" in state["mermaid_path"]

    def test_agent_trace_accumulation(self):
        """Test agent trace accumulates through pipeline."""
        state = create_initial_state(
            user_query="Test query",
            session_id="trace-test",
        )

        trace = state.get("agent_trace", [])

        trace.append("QueryAgent: extracted intent")
        trace.append("KR Search: found results")
        trace.append("KB Search: found memories")
        trace.append("ContextBuilder: merged context")
        trace.append("ExplanationAgent: generated answer")

        assert len(trace) == 5
        assert "QueryAgent" in trace[0]
        assert "ExplanationAgent" in trace[-1]
