"""
Graph writer service for GRAG AI.

Persists extracted entities and relationships to Neo4j with temporal metadata
and document provenance links.
"""

import uuid
from datetime import datetime
from typing import Any

import structlog

from app.database.neo4j_client import Neo4jClient, get_neo4j_client


logger = structlog.get_logger()


class GraphWriterService:
    """
    Service for writing extracted entities and relations to Neo4j.

    Handles temporal versioning and document provenance tracking.
    """

    def __init__(self, neo4j_client: Neo4jClient) -> None:
        """
        Initialize the graph writer service.

        Args:
            neo4j_client: Neo4j client instance
        """
        self.client = neo4j_client

    async def write_document(self, content: str, source: str = "api") -> dict[str, Any]:
        """
        Create a Document node in Neo4j.

        Args:
            content: Document text content
            source: Source identifier (default: "api")

        Returns:
            dict: Document node info with id, content, source
        """
        document_id = str(uuid.uuid4())

        query = """
        CREATE (d:Document {
            id: $document_id,
            content: $content,
            source: $source,
            created_at: datetime(),
            chunk_index: 0
        })
        RETURN d.id AS document_id, d.content AS content, d.source AS source
        """

        result = await self.client.execute_single(
            query,
            {
                "document_id": document_id,
                "content": content,
                "source": source,
            },
        )

        logger.info(
            "graph_writer.document_created",
            document_id=document_id,
            content_length=len(content),
        )

        return result

    async def write_entities(
        self, entities: list[dict], document_id: str
    ) -> list[dict[str, Any]]:
        """
        Write entities to Neo4j using MERGE for idempotent creation.

        Args:
            entities: List of entity dicts with name, type, description
            document_id: Document ID to link entities to

        Returns:
            list: Created entity info with neo4j_id, name, type
        """
        if not entities:
            return []

        results = []

        for entity in entities:
            entity_id = str(uuid.uuid4())
            name = entity.get("name", "")
            entity_type = entity.get("type", "Concept")
            description = entity.get("description", "")

            query = """
            MERGE (e:Entity {name: $name})
            ON CREATE SET
                e.id = $entity_id,
                e.type = $entity_type,
                e.description = $description,
                e.created_at = datetime(),
                e.updated_at = datetime()
            ON MATCH SET
                e.updated_at = datetime()
            RETURN e.id AS neo4j_id, e.name AS name, e.type AS type
            """

            result = await self.client.execute_single(
                query,
                {
                    "name": name,
                    "entity_id": entity_id,
                    "entity_type": entity_type,
                    "description": description,
                },
            )

            if result:
                results.append(result)
                logger.debug(
                    "graph_writer.entity_created",
                    name=name,
                    entity_id=result.get("neo4j_id"),
                )

                # Link document to entity for provenance
                await self.link_document_entity(document_id, result.get("neo4j_id"))

        logger.info(
            "graph_writer.entities_written",
            count=len(results),
            document_id=document_id,
        )

        return results

    async def write_relations(
        self, relations: list[dict], document_id: str
    ) -> list[dict[str, Any]]:
        """
        Write relations to Neo4j with temporal metadata.

        Relations are created using MERGE on source/target entity names,
        with valid_from set to now() and valid_to set to None (active).

        Args:
            relations: List of relation dicts with source, target, type, confidence
            document_id: Document ID for provenance

        Returns:
            list: Created relation info with relation_id, source, target, type
        """
        if not relations:
            return []

        results = []
        valid_from = datetime.utcnow()

        for relation in relations:
            source_name = relation.get("source", "")
            target_name = relation.get("target", "")
            relation_type = relation.get("type", "RELATED_TO").upper()
            confidence = relation.get("confidence", 0.5)

            # Match entities by name, create relation with temporal props
            query = """
            MATCH (s:Entity {name: $source_name})
            MATCH (t:Entity {name: $target_name})

            // Close any existing active relation of same type
            OPTIONAL MATCH (s)-[existing:RELATES_TO]->(t)
            WHERE existing.type = $relation_type AND existing.valid_to IS NULL
            SET existing.valid_to = datetime($valid_from)

            // Create new temporal relation
            CREATE (s)-[r:RELATES_TO {
                id: randomUUID(),
                type: $relation_type,
                valid_from: datetime($valid_from),
                valid_to: NULL,
                confidence: $confidence,
                source: 'api'
            }]->(t)

            RETURN r.id AS relation_id, s.name AS source_name,
                   t.name AS target_name, r.type AS relation_type,
                   r.valid_from AS valid_from
            """

            result = await self.client.execute_single(
                query,
                {
                    "source_name": source_name,
                    "target_name": target_name,
                    "relation_type": relation_type,
                    "confidence": confidence,
                    "valid_from": valid_from.isoformat(),
                },
            )

            if result:
                results.append(result)
                logger.debug(
                    "graph_writer.relation_created",
                    source=source_name,
                    target=target_name,
                    relation_type=relation_type,
                )

        logger.info(
            "graph_writer.relations_written",
            count=len(results),
            document_id=document_id,
        )

        return results

    async def link_document_entity(
        self, document_id: str, entity_id: str
    ) -> dict[str, Any]:
        """
        Create CONTAINS_ENTITY edge between Document and Entity for provenance.

        Args:
            document_id: Document node ID
            entity_id: Entity node ID

        Returns:
            dict: Link creation result
        """
        query = """
        MATCH (d:Document {id: $document_id})
        MATCH (e:Entity {id: $entity_id})
        MERGE (d)-[r:CONTAINS_ENTITY {created_at: datetime()}]->(e)
        RETURN d.id AS document_id, e.id AS entity_id
        """

        result = await self.client.execute_single(
            query,
            {"document_id": document_id, "entity_id": entity_id},
        )

        logger.debug(
            "graph_writer.document_entity_linked",
            document_id=document_id,
            entity_id=entity_id,
        )

        return result

    async def write_ingestion_result(
        self,
        text: str,
        entities: list[dict],
        relations: list[dict],
        source: str = "api",
    ) -> dict[str, Any]:
        """
        Complete workflow: write document, entities, relations to Neo4j.

        This is the main entry point for persisting ingestion results.

        Args:
            text: Document text content
            entities: List of extracted entities
            relations: List of extracted relations
            source: Source identifier (default: "api")

        Returns:
            dict: Summary with document_id, entity_ids, relation_ids
        """
        logger.info(
            "graph_writer.ingestion_start",
            entity_count=len(entities),
            relation_count=len(relations),
        )

        # Step 1: Create document node
        document_result = await self.write_document(text, source)
        document_id = document_result.get("document_id") if document_result else None

        if not document_id:
            logger.error("graph_writer.document_create_failed")
            raise RuntimeError("Failed to create document node in Neo4j")

        # Step 2: Write entities and link to document
        entity_results = await self.write_entities(entities, document_id)
        entity_ids = [e.get("neo4j_id") for e in entity_results if e.get("neo4j_id")]

        # Step 3: Write relations
        relation_results = await self.write_relations(relations, document_id)
        relation_ids = [
            r.get("relation_id") for r in relation_results if r.get("relation_id")
        ]

        logger.info(
            "graph_writer.ingestion_complete",
            document_id=document_id,
            entity_count=len(entity_ids),
            relation_count=len(relation_ids),
        )

        return {
            "document_id": document_id,
            "entity_ids": entity_ids,
            "relation_ids": relation_ids,
            "entity_count": len(entity_ids),
            "relation_count": len(relation_ids),
        }


# Singleton instance
_graph_writer_service: GraphWriterService | None = None


def get_graph_writer_service() -> GraphWriterService:
    """
    Get singleton GraphWriterService instance.

    Returns:
        GraphWriterService: Shared service instance
    """
    global _graph_writer_service
    if _graph_writer_service is None:
        neo4j_client = get_neo4j_client()
        _graph_writer_service = GraphWriterService(neo4j_client)
    return _graph_writer_service


async def write_ingestion_result(
    text: str,
    entities: list[dict],
    relations: list[dict],
    source: str = "api",
) -> dict[str, Any]:
    """
    Convenience function for writing ingestion results.

    Args:
        text: Document text content
        entities: List of extracted entities
        relations: List of extracted relations
        source: Source identifier

    Returns:
        dict: Summary with document_id, entity_ids, relation_ids
    """
    service = get_graph_writer_service()
    return await service.write_ingestion_result(text, entities, relations, source)
