"""
Graph writer service for GRAG AI.

Persists extracted entities and relationships to Neo4j with temporal metadata
and document provenance links.
"""

import uuid
from datetime import datetime
from typing import Any

import structlog

from app.config import get_settings
from app.database.neo4j_client import Neo4jClient, get_neo4j_client
from app.ingestion.chunking import split_text_into_chunks


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
        self.settings = get_settings()

    def _prepare_chunks(self, content: str) -> list[str]:
        """Split document text into chunk nodes for graph provenance."""
        chunks = split_text_into_chunks(
            content,
            max_chars=max(int(self.settings.ingestion_chunk_size_chars), 600),
            overlap_chars=max(int(self.settings.ingestion_chunk_overlap_chars), 0),
        )
        if chunks:
            return chunks
        return [content.strip()] if content.strip() else []

    async def write_document(
        self,
        content: str,
        source: str = "api",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Create a Document node in Neo4j.

        Args:
            content: Document text content
            source: Source identifier (default: "api")

        Returns:
            dict: Document node info with id, content, source
        """
        document_id = str(uuid.uuid4())

        metadata = metadata or {}
        chunk_count = len(self._prepare_chunks(content))
        content_preview = content[:4000]
        content_is_truncated = len(content) > len(content_preview)

        query = """
        CREATE (d:Document {
            id: $document_id,
            content: $content,
            source: $source,
            created_at: datetime(),
            chunk_index: 0,
            chunk_count: $chunk_count,
            char_count: $char_count,
            content_is_truncated: $content_is_truncated,
            source_file: $source_file,
            modality: $modality
        })
        RETURN d.id AS document_id, d.content AS content, d.source AS source,
               d.chunk_count AS chunk_count
        """

        result = await self.client.execute_single(
            query,
            {
                "document_id": document_id,
                "content": content_preview,
                "source": source,
                "chunk_count": chunk_count,
                "char_count": len(content),
                "content_is_truncated": content_is_truncated,
                "source_file": str(metadata.get("source_file", "")),
                "modality": str(metadata.get("modality", "text")),
            },
        )

        logger.info(
            "graph_writer.document_created",
            document_id=document_id,
            content_length=len(content),
        )

        return result

    async def write_document_chunks(
        self,
        document_id: str,
        chunks: list[str],
        source: str,
        modality: str,
    ) -> list[dict[str, Any]]:
        """Persist document chunk nodes and link them to the parent document."""
        if not chunks:
            return []

        results: list[dict[str, Any]] = []
        for idx, chunk in enumerate(chunks):
            chunk_id = str(uuid.uuid4())
            query = """
            MATCH (d:Document {id: $document_id})
            CREATE (c:DocumentChunk {
                id: $chunk_id,
                document_id: $document_id,
                chunk_index: $chunk_index,
                content: $content,
                char_count: $char_count,
                source: $source,
                modality: $modality,
                created_at: datetime()
            })
            MERGE (d)-[:HAS_CHUNK]->(c)
            RETURN c.id AS chunk_id, c.chunk_index AS chunk_index, c.content AS content
            """

            result = await self.client.execute_single(
                query,
                {
                    "document_id": document_id,
                    "chunk_id": chunk_id,
                    "chunk_index": idx,
                    "content": chunk,
                    "char_count": len(chunk),
                    "source": source,
                    "modality": modality,
                },
            )
            if result:
                results.append(result)
            else:
                results.append(
                    {
                        "chunk_id": chunk_id,
                        "chunk_index": idx,
                        "content": chunk,
                    }
                )

        logger.info(
            "graph_writer.chunks_written",
            document_id=document_id,
            chunk_count=len(results),
        )
        return results

    async def write_image_asset(
        self,
        document_id: str,
        image_metadata: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Store image metadata as a dedicated node and link to document."""
        if not image_metadata:
            return None

        query = """
        MATCH (d:Document {id: $document_id})
        MERGE (img:ImageAsset {sha256: $sha256})
        ON CREATE SET
            img.id = randomUUID(),
            img.created_at = datetime(),
            img.original_format = $original_format,
            img.original_mode = $original_mode,
            img.original_width = $original_width,
            img.original_height = $original_height,
            img.processed_format = $processed_format,
            img.processed_width = $processed_width,
            img.processed_height = $processed_height,
            img.processed_size_bytes = $processed_size_bytes
        ON MATCH SET
            img.last_seen_at = datetime(),
            img.processed_width = $processed_width,
            img.processed_height = $processed_height,
            img.processed_size_bytes = $processed_size_bytes
        MERGE (d)-[:DERIVED_FROM_IMAGE]->(img)
        RETURN img.id AS image_id, img.sha256 AS sha256
        """

        result = await self.client.execute_single(
            query,
            {
                "document_id": document_id,
                "sha256": str(image_metadata.get("sha256", "")),
                "original_format": str(image_metadata.get("original_format", "")),
                "original_mode": str(image_metadata.get("original_mode", "")),
                "original_width": int(image_metadata.get("original_width", 0)),
                "original_height": int(image_metadata.get("original_height", 0)),
                "processed_format": str(image_metadata.get("processed_format", "")),
                "processed_width": int(image_metadata.get("processed_width", 0)),
                "processed_height": int(image_metadata.get("processed_height", 0)),
                "processed_size_bytes": int(image_metadata.get("processed_size_bytes", 0)),
            },
        )
        return result

    async def write_entities(
        self,
        entities: list[dict],
        document_id: str,
        chunks: list[dict[str, Any]] | None = None,
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
                await self.link_entity_to_chunks(
                    chunks=chunks or [],
                    entity_id=result.get("neo4j_id"),
                    entity_name=name,
                )

        logger.info(
            "graph_writer.entities_written",
            count=len(results),
            document_id=document_id,
        )

        return results

    async def write_relations(
        self,
        relations: list[dict],
        document_id: str,
        chunks: list[dict[str, Any]] | None = None,
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

                relation_id = result.get("relation_id")
                if relation_id:
                    await self.write_relation_fact(
                        document_id=document_id,
                        relation_id=relation_id,
                        source_name=source_name,
                        target_name=target_name,
                        relation_type=relation_type,
                        confidence=confidence,
                        valid_from=valid_from.isoformat(),
                    )
                    await self.link_relation_to_chunks(
                        chunks=chunks or [],
                        relation_id=relation_id,
                        source_name=source_name,
                        target_name=target_name,
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
        MERGE (d)-[r:CONTAINS_ENTITY]->(e)
        ON CREATE SET r.created_at = datetime()
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

    async def link_entity_to_chunks(
        self,
        *,
        chunks: list[dict[str, Any]],
        entity_id: str,
        entity_name: str,
    ) -> None:
        """Link entities to chunk nodes where they are explicitly mentioned."""
        if not chunks or not entity_id or not entity_name:
            return

        for chunk in chunks:
            chunk_id = chunk.get("chunk_id")
            content = str(chunk.get("content", ""))
            if not chunk_id or not content:
                continue
            if entity_name.lower() not in content.lower():
                continue

            query = """
            MATCH (c:DocumentChunk {id: $chunk_id})
            MATCH (e:Entity {id: $entity_id})
            MERGE (c)-[r:MENTIONS_ENTITY]->(e)
            ON CREATE SET r.created_at = datetime()
            RETURN c.id AS chunk_id, e.id AS entity_id
            """
            await self.client.execute_single(
                query,
                {
                    "chunk_id": chunk_id,
                    "entity_id": entity_id,
                },
            )

    async def write_relation_fact(
        self,
        *,
        document_id: str,
        relation_id: str,
        source_name: str,
        target_name: str,
        relation_type: str,
        confidence: float,
        valid_from: str,
    ) -> dict[str, Any] | None:
        """Create a relation fact node to preserve node-node-node provenance."""
        query = """
        MATCH (d:Document {id: $document_id})
        MATCH (s:Entity {name: $source_name})
        MATCH (t:Entity {name: $target_name})
        MERGE (rf:RelationFact {id: $relation_id})
        ON CREATE SET
            rf.type = $relation_type,
            rf.confidence = $confidence,
            rf.valid_from = datetime($valid_from),
            rf.created_at = datetime()
        ON MATCH SET
            rf.confidence = $confidence,
            rf.type = $relation_type,
            rf.updated_at = datetime()
        MERGE (d)-[:ASSERTS_RELATION]->(rf)
        MERGE (rf)-[:FROM_ENTITY]->(s)
        MERGE (rf)-[:TO_ENTITY]->(t)
        RETURN rf.id AS relation_fact_id
        """

        return await self.client.execute_single(
            query,
            {
                "document_id": document_id,
                "relation_id": relation_id,
                "source_name": source_name,
                "target_name": target_name,
                "relation_type": relation_type,
                "confidence": confidence,
                "valid_from": valid_from,
            },
        )

    async def link_relation_to_chunks(
        self,
        *,
        chunks: list[dict[str, Any]],
        relation_id: str,
        source_name: str,
        target_name: str,
    ) -> None:
        """Link relation fact nodes to supporting chunk nodes."""
        if not chunks or not relation_id:
            return

        for chunk in chunks:
            chunk_id = chunk.get("chunk_id")
            content = str(chunk.get("content", "")).lower()
            if not chunk_id or not content:
                continue

            if source_name.lower() not in content or target_name.lower() not in content:
                continue

            query = """
            MATCH (c:DocumentChunk {id: $chunk_id})
            MATCH (rf:RelationFact {id: $relation_id})
            MERGE (c)-[r:SUPPORTS_RELATION]->(rf)
            ON CREATE SET r.created_at = datetime()
            RETURN c.id AS chunk_id, rf.id AS relation_id
            """
            await self.client.execute_single(
                query,
                {
                    "chunk_id": chunk_id,
                    "relation_id": relation_id,
                },
            )

    async def write_ingestion_result(
        self,
        text: str,
        entities: list[dict],
        relations: list[dict],
        source: str = "api",
        metadata: dict[str, Any] | None = None,
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
        metadata = metadata or {}
        modality = str(metadata.get("modality", "text"))

        document_result = await self.write_document(
            text,
            source,
            metadata=metadata,
        )
        document_id = document_result.get("document_id") if document_result else None

        if not document_id:
            logger.error("graph_writer.document_create_failed")
            raise RuntimeError("Failed to create document node in Neo4j")

        # Step 1b: Create document chunk nodes for richer graph structure.
        chunks = self._prepare_chunks(text)
        chunk_results = await self.write_document_chunks(
            document_id=document_id,
            chunks=chunks,
            source=source,
            modality=modality,
        )

        image_result = None
        image_metadata = metadata.get("image_metadata")
        if isinstance(image_metadata, dict) and image_metadata:
            image_result = await self.write_image_asset(document_id, image_metadata)

        # Step 2: Write entities and link to document/chunks
        entity_results = await self.write_entities(
            entities,
            document_id,
            chunks=chunk_results,
        )
        entity_ids = [e.get("neo4j_id") for e in entity_results if e.get("neo4j_id")]

        # Step 3: Write relations
        relation_results = await self.write_relations(
            relations,
            document_id,
            chunks=chunk_results,
        )
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
            "chunk_ids": [c.get("chunk_id") for c in chunk_results if c.get("chunk_id")],
            "image_id": image_result.get("image_id") if image_result else None,
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
