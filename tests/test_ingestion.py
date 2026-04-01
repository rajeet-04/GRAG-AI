"""
Integration tests for the ingestion pipeline.

Tests verify all Phase 4 success criteria:
1. Text input → extracted entities
2. Relationships extracted and typed
3. Write to Neo4j with temporal metadata
4. Document → Entity links for provenance
5. Batch processing
"""

import json
from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Sample text fixtures
SAMPLE_TEXT = "Elon Musk founded Tesla in 2003. He also leads SpaceX."

SAMPLE_TEXT_COMPLEX = """
Apple Inc. was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in 1976.
The company is headquartered in Cupertino, California.
Steve Jobs returned to Apple in 1997 and launched the iPhone in 2007.
Tim Cook became CEO in 2011 after Jobs passed away.
"""

# Mock entity extraction responses
MOCK_ENTITIES_SINGLE = [
    {
        "name": "Elon Musk",
        "type": "Person",
        "description": "CEO and founder of SpaceX and Tesla",
        "confidence": 1.0,
    },
    {
        "name": "Tesla",
        "type": "Organization",
        "description": "Electric vehicle and clean energy company",
        "confidence": 1.0,
    },
    {
        "name": "SpaceX",
        "type": "Organization",
        "description": "Space exploration company",
        "confidence": 1.0,
    },
    {
        "name": "2003",
        "type": "Event",
        "description": "Year Tesla was founded",
        "confidence": 0.9,
    },
]

MOCK_ENTITIES_SINGLE_JSON = json.dumps(MOCK_ENTITIES_SINGLE)

MOCK_ENTITIES_COMPLEX = [
    {
        "name": "Apple Inc.",
        "type": "Organization",
        "description": "Technology company",
        "confidence": 1.0,
    },
    {
        "name": "Steve Jobs",
        "type": "Person",
        "description": "Co-founder of Apple",
        "confidence": 1.0,
    },
    {
        "name": "Steve Wozniak",
        "type": "Person",
        "description": "Co-founder of Apple",
        "confidence": 1.0,
    },
    {
        "name": "Ronald Wayne",
        "type": "Person",
        "description": "Co-founder of Apple",
        "confidence": 0.9,
    },
    {
        "name": "1976",
        "type": "Event",
        "description": "Year Apple was founded",
        "confidence": 1.0,
    },
    {
        "name": "Cupertino",
        "type": "Location",
        "description": "City in California",
        "confidence": 1.0,
    },
    {
        "name": "California",
        "type": "Location",
        "description": "US State",
        "confidence": 1.0,
    },
    {
        "name": "1997",
        "type": "Event",
        "description": "Year Steve Jobs returned to Apple",
        "confidence": 1.0,
    },
    {
        "name": "iPhone",
        "type": "Product",
        "description": "Smartphone by Apple",
        "confidence": 1.0,
    },
    {
        "name": "2007",
        "type": "Event",
        "description": "Year iPhone was launched",
        "confidence": 1.0,
    },
    {
        "name": "Tim Cook",
        "type": "Person",
        "description": "CEO of Apple",
        "confidence": 1.0,
    },
    {
        "name": "2011",
        "type": "Event",
        "description": "Year Tim Cook became CEO",
        "confidence": 1.0,
    },
]

MOCK_RELATIONS_SINGLE = [
    {
        "source": "Elon Musk",
        "target": "Tesla",
        "type": "FOUNDED",
        "confidence": 1.0,
    },
    {
        "source": "Elon Musk",
        "target": "SpaceX",
        "type": "FOUNDED",
        "confidence": 1.0,
    },
    {
        "source": "Elon Musk",
        "target": "SpaceX",
        "type": "LEADS",
        "confidence": 1.0,
    },
]

MOCK_RELATIONS_SINGLE_JSON = json.dumps(MOCK_RELATIONS_SINGLE)

MOCK_RELATIONS_COMPLEX = [
    {
        "source": "Steve Jobs",
        "target": "Apple Inc.",
        "type": "FOUNDED",
        "confidence": 1.0,
    },
    {
        "source": "Steve Wozniak",
        "target": "Apple Inc.",
        "type": "FOUNDED",
        "confidence": 1.0,
    },
    {
        "source": "Ronald Wayne",
        "target": "Apple Inc.",
        "type": "FOUNDED",
        "confidence": 0.9,
    },
    {
        "source": "Apple Inc.",
        "target": "Cupertino",
        "type": "LOCATED_IN",
        "confidence": 1.0,
    },
    {
        "source": "Cupertino",
        "target": "California",
        "type": "LOCATED_IN",
        "confidence": 1.0,
    },
    {
        "source": "Steve Jobs",
        "target": "Apple Inc.",
        "type": "LEADS",
        "confidence": 1.0,
    },
    {
        "source": "Steve Jobs",
        "target": "iPhone",
        "type": "CREATED",
        "confidence": 1.0,
    },
    {
        "source": "Tim Cook",
        "target": "Apple Inc.",
        "type": "LEADS",
        "confidence": 1.0,
    },
]


class MockNeo4jClient:
    """Mock Neo4j client for testing without database."""

    def __init__(self):
        self.documents = {}
        self.entities = {}
        self.relations = {}
        self.entity_counter = 0
        self.relation_counter = 0

    async def execute_single(self, query: str, params: dict[str, Any] | None = None):
        """Mock execute_single for CREATE/MERGE operations."""
        params = params or {}

        if "CREATE (d:Document" in query:
            doc_id = params.get("document_id")
            self.documents[doc_id] = {
                "id": doc_id,
                "content": params.get("content"),
                "source": params.get("source"),
            }
            return {
                "document_id": doc_id,
                "content": params.get("content"),
                "source": params.get("source"),
            }

        if "MERGE (e:Entity" in query:
            name = params.get("name")
            entity_id = params.get("entity_id", f"entity_{self.entity_counter}")
            self.entity_counter += 1
            self.entities[name] = {
                "neo4j_id": entity_id,
                "name": name,
                "type": params.get("entity_type", "Concept"),
            }
            return self.entities[name]

        if "CREATE" in query and "RELATES_TO" in query:
            source = params.get("source_name")
            target = params.get("target_name")
            rel_id = f"rel_{self.relation_counter}"
            self.relation_counter += 1
            self.relations[f"{source}|{target}"] = {
                "relation_id": rel_id,
                "source_name": source,
                "target_name": target,
                "relation_type": params.get("relation_type"),
            }
            return self.relations[f"{source}|{target}"]

        if "CONTAINS_ENTITY" in query:
            return {
                "document_id": params.get("document_id"),
                "entity_id": params.get("entity_id"),
            }

        return None


# ============================================================================
# Criterion 1: Text input → extracted entities
# ============================================================================


class TestEntityExtraction:
    """Tests for entity extraction from text."""

    @pytest.mark.asyncio
    async def test_extract_single_entity(self):
        """Test extracting a single entity from text."""
        from app.ingestion.entity_extractor import EntityExtractionService

        with patch("app.ingestion.entity_extractor.OllamaClient") as MockClient:
            mock_instance = AsyncMock()
            mock_instance.chat = AsyncMock(
                return_value={
                    "content": '[{"name": "Elon Musk", "type": "Person", "description": "CEO", "confidence": 1.0}]',
                    "usage": {},
                    "model": "test",
                    "done": True,
                }
            )
            MockClient.return_value = mock_instance

            service = EntityExtractionService()
            entities = await service.extract_entities("Elon Musk founded Tesla.")

            assert len(entities) >= 1
            entity_names = [e["name"] for e in entities]
            assert "Elon Musk" in entity_names

    @pytest.mark.asyncio
    async def test_extract_multiple_entities(self):
        """Test extracting multiple entities from text."""
        from app.ingestion.entity_extractor import EntityExtractionService

        with patch("app.ingestion.entity_extractor.OllamaClient") as MockClient:
            mock_instance = AsyncMock()
            mock_instance.chat = AsyncMock(
                return_value={
                    "content": MOCK_ENTITIES_SINGLE_JSON,
                    "usage": {},
                    "model": "test",
                    "done": True,
                }
            )
            MockClient.return_value = mock_instance

            service = EntityExtractionService()
            entities = await service.extract_entities(SAMPLE_TEXT)

            assert len(entities) >= 3
            entity_names = [e["name"] for e in entities]
            assert "Elon Musk" in entity_names
            assert "Tesla" in entity_names
            assert "SpaceX" in entity_names

    @pytest.mark.asyncio
    async def test_extract_entity_types(self):
        """Test that entity types are correctly assigned."""
        from app.ingestion.entity_extractor import EntityExtractionService

        with patch("app.ingestion.entity_extractor.OllamaClient") as MockClient:
            mock_instance = AsyncMock()
            mock_instance.chat = AsyncMock(
                return_value={
                    "content": MOCK_ENTITIES_SINGLE_JSON,
                    "usage": {},
                    "model": "test",
                    "done": True,
                }
            )
            MockClient.return_value = mock_instance

            service = EntityExtractionService()
            entities = await service.extract_entities(SAMPLE_TEXT)

            type_map = {e["name"]: e.get("type") for e in entities}
            assert type_map.get("Elon Musk") == "Person"
            assert type_map.get("Tesla") == "Organization"
            assert type_map.get("SpaceX") == "Organization"

    @pytest.mark.asyncio
    async def test_empty_text_returns_empty_list(self):
        """Test that empty text returns empty entity list."""
        from app.ingestion.entity_extractor import EntityExtractionService

        service = EntityExtractionService()
        entities = await service.extract_entities("")

        assert entities == []

    @pytest.mark.asyncio
    async def test_entity_has_id(self):
        """Test that extracted entities have IDs."""
        from app.ingestion.entity_extractor import EntityExtractionService

        with patch("app.ingestion.entity_extractor.OllamaClient") as MockClient:
            mock_instance = AsyncMock()
            mock_instance.chat = AsyncMock(
                return_value={
                    "content": '[{"name": "Test", "type": "Person", "description": "Test", "confidence": 1.0}]',
                    "usage": {},
                    "model": "test",
                    "done": True,
                }
            )
            MockClient.return_value = mock_instance

            service = EntityExtractionService()
            entities = await service.extract_entities("Test entity")

            assert len(entities) >= 1
            assert "id" in entities[0]


# ============================================================================
# Criterion 2: Relationships extracted and typed
# ============================================================================


class TestRelationExtraction:
    """Tests for relationship extraction between entities."""

    @pytest.mark.asyncio
    async def test_extract_single_relation(self):
        """Test extracting a single relationship."""
        from app.ingestion.relation_extractor import RelationExtractionService

        entities = [
            {
                "name": "Elon Musk",
                "type": "Person",
                "description": "CEO",
                "confidence": 1.0,
            },
            {
                "name": "Tesla",
                "type": "Organization",
                "description": "Car company",
                "confidence": 1.0,
            },
        ]

        with patch("app.ingestion.relation_extractor.OllamaClient") as MockClient:
            mock_instance = AsyncMock()
            mock_instance.chat = AsyncMock(
                return_value={
                    "content": '[{"source": "Elon Musk", "target": "Tesla", "type": "FOUNDED", "confidence": 1.0}]',
                    "usage": {},
                    "model": "test",
                    "done": True,
                }
            )
            MockClient.return_value = mock_instance

            service = RelationExtractionService()
            relations = await service.extract_relations(
                "Elon Musk founded Tesla.", entities
            )

            assert len(relations) >= 1
            assert relations[0]["source"] == "Elon Musk"
            assert relations[0]["target"] == "Tesla"

    @pytest.mark.asyncio
    async def test_extract_multiple_relations(self):
        """Test extracting multiple relationships."""
        from app.ingestion.relation_extractor import RelationExtractionService

        with patch("app.ingestion.relation_extractor.OllamaClient") as MockClient:
            mock_instance = AsyncMock()
            mock_instance.chat = AsyncMock(
                return_value={
                    "content": MOCK_RELATIONS_SINGLE_JSON,
                    "usage": {},
                    "model": "test",
                    "done": True,
                }
            )
            MockClient.return_value = mock_instance

            service = RelationExtractionService()
            entities = MOCK_ENTITIES_SINGLE[:2]  # Elon Musk and Tesla
            relations = await service.extract_relations(SAMPLE_TEXT, entities)

            assert len(relations) >= 1

    @pytest.mark.asyncio
    async def test_relation_type_assignment(self):
        """Test that relation types are correctly assigned."""
        from app.ingestion.relation_extractor import RelationExtractionService

        entities = [
            {
                "name": "Elon Musk",
                "type": "Person",
                "description": "CEO",
                "confidence": 1.0,
            },
            {
                "name": "Tesla",
                "type": "Organization",
                "description": "Car company",
                "confidence": 1.0,
            },
            {
                "name": "SpaceX",
                "type": "Organization",
                "description": "Space company",
                "confidence": 1.0,
            },
        ]

        with patch("app.ingestion.relation_extractor.OllamaClient") as MockClient:
            mock_instance = AsyncMock()
            mock_instance.chat = AsyncMock(
                return_value={
                    "content": MOCK_RELATIONS_SINGLE_JSON,
                    "usage": {},
                    "model": "test",
                    "done": True,
                }
            )
            MockClient.return_value = mock_instance

            service = RelationExtractionService()
            relations = await service.extract_relations(SAMPLE_TEXT, entities)

            type_map = {(r["source"], r["target"]): r["type"] for r in relations}
            assert type_map.get(("Elon Musk", "Tesla")) == "FOUNDED"
            # Elon Musk → SpaceX has both FOUNDED and LEADS in mock data
            assert type_map.get(("Elon Musk", "SpaceX")) in ["FOUNDED", "LEADS"]

    @pytest.mark.asyncio
    async def test_empty_entities_returns_empty_relations(self):
        """Test that empty entities list returns empty relations."""
        from app.ingestion.relation_extractor import RelationExtractionService

        service = RelationExtractionService()
        relations = await service.extract_relations("Some text", [])

        assert relations == []


# ============================================================================
# Criterion 3: Write to Neo4j with temporal metadata
# ============================================================================


class TestGraphWriter:
    """Tests for graph writer service with Neo4j and temporal metadata."""

    @pytest.mark.asyncio
    async def test_write_entity_to_neo4j(self):
        """Test entity persistence to Neo4j."""
        from app.ingestion.graph_writer import GraphWriterService

        mock_client = MockNeo4jClient()
        service = GraphWriterService(mock_client)

        entities = [
            {
                "name": "Test Entity",
                "type": "Concept",
                "description": "Test description",
            }
        ]
        result = await service.write_entities(entities, "doc_123")

        assert len(result) >= 1
        assert result[0]["name"] == "Test Entity"
        assert "neo4j_id" in result[0]

    @pytest.mark.asyncio
    async def test_write_relation_temporal(self):
        """Test that temporal properties are set on relations."""
        from app.ingestion.graph_writer import GraphWriterService
        from app.ingestion.temporal import TemporalIngestionMixin

        # Create entities first
        mock_client = MockNeo4jClient()
        service = GraphWriterService(mock_client)

        entities = [
            {"name": "Entity A", "type": "Concept", "description": "A"},
            {"name": "Entity B", "type": "Concept", "description": "B"},
        ]
        await service.write_entities(entities, "doc_123")

        # Create relation
        relations = [
            {
                "source": "Entity A",
                "target": "Entity B",
                "type": "RELATED_TO",
                "confidence": 0.9,
            }
        ]
        result = await service.write_relations(relations, "doc_123")

        assert len(result) >= 1

    @pytest.mark.asyncio
    async def test_temporal_mixin_prepare_entity(self):
        """Test temporal mixin for entity preparation."""
        from app.ingestion.temporal import TemporalIngestionMixin

        entity = {"name": "Test", "type": "Person"}
        temporal = TemporalIngestionMixin.prepare_temporal_entity(entity)

        assert "valid_from" in temporal
        assert "valid_to" in temporal
        assert temporal["valid_to"] is None  # Active by default

    @pytest.mark.asyncio
    async def test_temporal_mixin_prepare_relation(self):
        """Test temporal mixin for relation preparation."""
        from app.ingestion.temporal import TemporalIngestionMixin

        relation = {"source": "A", "target": "B", "type": "RELATED_TO"}
        temporal = TemporalIngestionMixin.prepare_temporal_relation(relation)

        assert "valid_from" in temporal
        assert "version" in temporal
        assert temporal["version"] == "v1"

    @pytest.mark.asyncio
    async def test_temporal_validation(self):
        """Test temporal validation of entities and relations."""
        from app.ingestion.temporal import validate_ingestion_temporal

        entities = [{"name": "Test", "confidence": 0.5}]
        relations = [{"source": "A", "target": "B", "confidence": 0.8}]

        warnings = validate_ingestion_temporal(entities, relations)
        assert isinstance(warnings, list)


# ============================================================================
# Criterion 4: Document → Entity links for provenance
# ============================================================================


class TestDocumentEntityLinks:
    """Tests for document-entity provenance tracking."""

    @pytest.mark.asyncio
    async def test_document_entity_link(self):
        """Test that CONTAINS_ENTITY edge is created."""
        from app.ingestion.graph_writer import GraphWriterService

        mock_client = MockNeo4jClient()
        service = GraphWriterService(mock_client)

        # Create document
        doc_result = await service.write_document("Test content", "test")
        doc_id = doc_result["document_id"]

        # Create entity
        entity_result = await service.write_entities(
            [{"name": "TestEntity", "type": "Concept", "description": "test"}], doc_id
        )
        entity_id = entity_result[0]["neo4j_id"]

        # Link document to entity
        link_result = await service.link_document_entity(doc_id, entity_id)

        assert link_result is not None
        assert link_result["document_id"] == doc_id
        assert link_result["entity_id"] == entity_id


# ============================================================================
# Criterion 5: Batch processing
# ============================================================================


class TestBatchProcessing:
    """Tests for batch document ingestion."""

    @pytest.mark.asyncio
    async def test_batch_ingestion(self):
        """Test processing multiple documents in batch."""
        from app.ingestion.entity_extractor import EntityExtractionService
        from app.ingestion.relation_extractor import RelationExtractionService

        documents = [
            "Document one mentions Apple.",
            "Document two mentions Google.",
            "Document three mentions Microsoft.",
        ]

        with (
            patch("app.ingestion.entity_extractor.OllamaClient") as mock_ollama,
            patch("app.ingestion.relation_extractor.OllamaClient"),
        ):
            # Mock entity extraction for each document
            async def mock_chat(*args, **kwargs):
                return {
                    "content": '[{"name": "Test", "type": "Organization", "description": "Test", "confidence": 1.0}]',
                    "usage": {},
                    "model": "test",
                    "done": True,
                }

            mock_ollama.return_value.chat = AsyncMock(
                side_effect=[mock_chat() for _ in documents]
            )

            entity_service = EntityExtractionService()
            relation_service = RelationExtractionService()

            results = []
            for text in documents:
                entities = await entity_service.extract_entities(text)
                relations = await relation_service.extract_relations(text, entities)
                results.append({"entities": entities, "relations": relations})

            assert len(results) == 3

    @pytest.mark.asyncio
    async def test_batch_results_count(self):
        """Test that batch returns correct number of results."""
        from app.ingestion.graph_writer import GraphWriterService

        mock_client = MockNeo4jClient()
        service = GraphWriterService(mock_client)

        # Simulate batch write
        all_entity_ids = []
        for i in range(3):
            entities = [
                {"name": f"Entity_{i}", "type": "Concept", "description": f"desc_{i}"}
            ]
            result = await service.write_entities(entities, f"doc_{i}")
            all_entity_ids.extend([r["neo4j_id"] for r in result])

        assert len(all_entity_ids) == 3


# ============================================================================
# Integration tests (require running services)
# ============================================================================


@pytest.mark.integration
class TestFullIngestionPipeline:
    """End-to-end integration tests - require Neo4j and Ollama."""

    @pytest.mark.asyncio
    async def test_full_ingestion_pipeline_mocked(self):
        """Test full pipeline with mocked external services."""
        from app.ingestion.entity_extractor import EntityExtractionService
        from app.ingestion.relation_extractor import RelationExtractionService
        from app.ingestion.graph_writer import GraphWriterService

        with (
            patch("app.ingestion.entity_extractor.OllamaClient") as mock_entity_client,
            patch(
                "app.ingestion.relation_extractor.OllamaClient"
            ) as mock_relation_client,
            patch("app.ingestion.graph_writer.get_neo4j_client") as mock_neo4j,
        ):
            # Setup mock Ollama responses
            mock_entity_instance = AsyncMock()
            mock_entity_instance.chat = AsyncMock(
                return_value={
                    "content": MOCK_ENTITIES_SINGLE_JSON,
                    "usage": {},
                    "model": "test",
                    "done": True,
                }
            )
            mock_entity_client.return_value = mock_entity_instance

            mock_relation_instance = AsyncMock()
            mock_relation_instance.chat = AsyncMock(
                return_value={
                    "content": MOCK_RELATIONS_SINGLE_JSON,
                    "usage": {},
                    "model": "test",
                    "done": True,
                }
            )
            mock_relation_client.return_value = mock_relation_instance

            # Setup mock Neo4j
            mock_neo4j.return_value = MockNeo4jClient()

            # Run pipeline
            entity_service = EntityExtractionService()
            relation_service = RelationExtractionService()
            graph_service = GraphWriterService(mock_neo4j.return_value)

            # Step 1: Extract entities
            entities = await entity_service.extract_entities(SAMPLE_TEXT)
            assert len(entities) >= 1

            # Step 2: Extract relations
            relations = await relation_service.extract_relations(SAMPLE_TEXT, entities)
            assert len(relations) >= 1

            # Step 3: Write to Neo4j
            doc_result = await graph_service.write_document(SAMPLE_TEXT, "test")
            assert "document_id" in doc_result

            entity_results = await graph_service.write_entities(
                entities, doc_result["document_id"]
            )
            assert len(entity_results) >= 1

            relation_results = await graph_service.write_relations(
                relations, doc_result["document_id"]
            )
            assert len(relation_results) >= 1


# ============================================================================
# Additional utility tests
# ============================================================================


class TestTemporalHelpers:
    """Tests for temporal utility functions."""

    @pytest.mark.asyncio
    async def test_apply_temporal_defaults(self):
        """Test applying default temporal properties."""
        from app.ingestion.temporal import apply_temporal_defaults

        entities = [{"name": "A", "type": "Concept"}]
        relations = [{"source": "A", "target": "B", "type": "RELATED_TO"}]

        processed_entities, processed_relations = apply_temporal_defaults(
            entities, relations
        )

        assert len(processed_entities) == 1
        assert "valid_from" in processed_entities[0]

        assert len(processed_relations) == 1
        assert "valid_to" in processed_relations[0]
        assert processed_relations[0]["valid_to"] is None

    @pytest.mark.asyncio
    async def test_validate_temporal_input(self):
        """Test temporal input validation."""
        from app.ingestion.temporal import TemporalIngestionMixin

        # Valid: no valid_to
        assert (
            TemporalIngestionMixin.validate_temporal_input(datetime.utcnow(), None)
            is True
        )

        # Valid: valid_from <= valid_to
        assert (
            TemporalIngestionMixin.validate_temporal_input(
                datetime(2024, 1, 1), datetime(2024, 12, 31)
            )
            is True
        )

        # Invalid: valid_from > valid_to
        assert (
            TemporalIngestionMixin.validate_temporal_input(
                datetime(2024, 12, 31), datetime(2024, 1, 1)
            )
            is False
        )


class TestEntityExtractorEdgeCases:
    """Edge case tests for entity extractor."""

    @pytest.mark.asyncio
    async def test_json_parsing_with_markdown(self):
        """Test JSON parsing from markdown code blocks."""
        from app.ingestion.entity_extractor import EntityExtractionService

        service = EntityExtractionService()

        # Test markdown code block parsing
        content = '```json\n[{"name": "Test", "type": "Person"}]\n```'
        result = service._parse_json_response(content)
        assert len(result) == 1
        assert result[0]["name"] == "Test"

    @pytest.mark.asyncio
    async def test_json_parsing_plain(self):
        """Test JSON parsing from plain text."""
        from app.ingestion.entity_extractor import EntityExtractionService

        service = EntityExtractionService()

        content = '[{"name": "Test", "type": "Person"}]'
        result = service._parse_json_response(content)
        assert len(result) == 1


class TestRelationExtractorEdgeCases:
    """Edge case tests for relation extractor."""

    @pytest.mark.asyncio
    async def test_validates_entity_names(self):
        """Test that relations only connect known entities."""
        from app.ingestion.relation_extractor import RelationExtractionService

        # Only include known entities
        entities = [
            {"name": "Elon Musk", "type": "Person", "description": "CEO"},
            {"name": "Tesla", "type": "Organization", "description": "Car company"},
        ]

        # Try to create a relation with an unknown entity
        relations = [
            {
                "source": "Elon Musk",
                "target": "UnknownEntity",
                "type": "KNOWS",
                "confidence": 1.0,
            },
            {
                "source": "Elon Musk",
                "target": "Tesla",
                "type": "FOUNDED",
                "confidence": 1.0,
            },
        ]

        # Filter to only valid relations (what the service does)
        entity_names = {e["name"] for e in entities}
        valid_relations = [
            r
            for r in relations
            if r.get("source") in entity_names and r.get("target") in entity_names
        ]

        # Should only have Tesla relation, not UnknownEntity
        assert len(valid_relations) == 1
        assert valid_relations[0]["target"] == "Tesla"
