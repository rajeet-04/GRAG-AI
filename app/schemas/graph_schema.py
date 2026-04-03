"""
Graph schema definitions for GRAG AI Knowledge Graph (KR).

This schema is a READ LAYER for Phase 1 testing.
Graphiti (Phase 2+) will provide the WRITE LAYER with its own Episode and EntityFact nodes.

SCHEMA OVERVIEW:
    Entity nodes: Facts and concepts extracted from documents
    Relation nodes: Typed relationships between entities (temporal)
    Document nodes: Source documents with chunk metadata

GRAPHITI MIGRATION NOTES:
    - Phase 2+ will add: (:graphiti_Entity), (:graphiti_EntityFact), (:graphiti_Episode)
    - These coexist with our manual schema
    - Graphiti handles temporal invalidation via EntityFact nodes
"""

from datetime import datetime
from typing import Any

import structlog

from app.database.neo4j_client import get_neo4j_client


logger = structlog.get_logger()


SCHEMA_CONSTRAINTS = """
-- Entity constraint: name must be unique
CREATE CONSTRAINT entity_name IF NOT EXISTS
FOR (e:Entity) REQUIRE e.name IS UNIQUE;
"""

SCHEMA_INDEXES = """
-- Entity indexes
CREATE INDEX entity_id IF NOT EXISTS
FOR (e:Entity) ON (e.id);

CREATE INDEX entity_name IF NOT EXISTS
FOR (e:Entity) ON (e.name);

CREATE INDEX entity_type IF NOT EXISTS
FOR (e:Entity) ON (e.type);

-- Document indexes
CREATE INDEX document_id IF NOT EXISTS
FOR (d:Document) ON (d.id);

CREATE INDEX document_source IF NOT EXISTS
FOR (d:Document) ON (d.source);

-- Relation indexes for temporal queries
CREATE INDEX relation_id IF NOT EXISTS
FOR ()-[r:RELATES_TO]->() ON (r.id);

CREATE INDEX relation_type IF NOT EXISTS
FOR ()-[r:RELATES_TO]->() ON (r.type);

CREATE INDEX relation_valid_from IF NOT EXISTS
FOR ()-[r:RELATES_TO]->() ON (r.valid_from);

CREATE INDEX relation_valid_to IF NOT EXISTS
FOR ()-[r:RELATES_TO]->() ON (r.valid_to);

-- Composite temporal index for efficient time-range queries
CREATE INDEX relation_temporal IF NOT EXISTS
FOR ()-[r:RELATES_TO]->() ON (r.valid_from, r.valid_to);
"""

SCHEMA_NODE_CREATION = """
// Entity Node
// Represents a fact or concept extracted from documents
CREATE (e:Entity {
    name: $name,
    type: $type,
    description: $description,
    created_at: datetime(),
    updated_at: datetime()
});

// Document Node
// Represents a source document
CREATE (d:Document {
    content: $content,
    source: $source,
    created_at: datetime(),
    chunk_index: $chunk_index
});

// Relation Node (for temporal relationships)
// Links two entities with a typed relationship
CREATE (e1:Entity)-[r:RELATES_TO {
    type: $relation_type,
    valid_from: datetime(),
    valid_to: NULL,
    confidence: $confidence,
    source: $source
}]->(e2:Entity);

// Link entity to source document
CREATE (e)-[:EXTRACTED_FROM {source: $source, created_at: datetime()}]->(d);
"""


async def init_graph_schema() -> dict[str, Any]:
    """
    Initialize the graph schema in Neo4j.

    Creates:
    - Entity, Relation, Document node labels
    - Unique constraint on Entity.name
    - Indexes for query performance
    - Temporal indexes for valid_from/valid_to

    Returns:
        dict: Schema initialization results
    """
    client = get_neo4j_client()
    results = {
        "constraints_created": [],
        "indexes_created": [],
        "errors": [],
    }

    try:
        logger.info("schema.init.start")

        constraint_queries = [
            "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
            "CREATE CONSTRAINT entity_name IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE",
            "CREATE CONSTRAINT document_id IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
            "CREATE CONSTRAINT document_chunk_id IF NOT EXISTS FOR (c:DocumentChunk) REQUIRE c.id IS UNIQUE",
            "CREATE CONSTRAINT relation_fact_id IF NOT EXISTS FOR (rf:RelationFact) REQUIRE rf.id IS UNIQUE",
            "CREATE CONSTRAINT relation_id IF NOT EXISTS FOR ()-[r:RELATES_TO]->() REQUIRE r.id IS UNIQUE",
        ]

        index_queries = [
            "CREATE INDEX entity_id IF NOT EXISTS FOR (e:Entity) ON (e.id)",
            "CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name)",
            "CREATE INDEX entity_type IF NOT EXISTS FOR (e:Entity) ON (e.type)",
            "CREATE INDEX document_id IF NOT EXISTS FOR (d:Document) ON (d.id)",
            "CREATE INDEX document_source IF NOT EXISTS FOR (d:Document) ON (d.source)",
            "CREATE INDEX document_chunk_index IF NOT EXISTS FOR (c:DocumentChunk) ON (c.chunk_index)",
            "CREATE INDEX document_chunk_document_id IF NOT EXISTS FOR (c:DocumentChunk) ON (c.document_id)",
            "CREATE INDEX relation_fact_type IF NOT EXISTS FOR (rf:RelationFact) ON (rf.type)",
            "CREATE INDEX relation_id IF NOT EXISTS FOR ()-[r:RELATES_TO]->() ON (r.id)",
            "CREATE INDEX relation_type IF NOT EXISTS FOR ()-[r:RELATES_TO]->() ON (r.type)",
            "CREATE INDEX relation_valid_from IF NOT EXISTS FOR ()-[r:RELATES_TO]->() ON (r.valid_from)",
            "CREATE INDEX relation_valid_to IF NOT EXISTS FOR ()-[r:RELATES_TO]->() ON (r.valid_to)",
            "CREATE INDEX relation_temporal IF NOT EXISTS FOR ()-[r:RELATES_TO]->() ON (r.valid_from, r.valid_to)",
        ]

        for query in constraint_queries:
            try:
                await client.execute_write(query)
                results["constraints_created"].append(query)
                logger.info("schema.constraint.created", query=query)
            except Exception as e:
                if "already exists" not in str(e).lower():
                    results["errors"].append(str(e))
                    logger.warning("schema.constraint.error", query=query, error=str(e))

        for query in index_queries:
            try:
                await client.execute_write(query)
                results["indexes_created"].append(query)
                logger.info("schema.index.created", query=query)
            except Exception as e:
                if "already exists" not in str(e).lower():
                    results["errors"].append(str(e))
                    logger.warning("schema.index.error", query=query, error=str(e))

        logger.info("schema.init.complete", results=results)
        return results

    except Exception as e:
        logger.error("schema.init.failed", error=str(e))
        results["errors"].append(str(e))
        return results


async def verify_schema() -> dict[str, bool]:
    """
    Verify the graph schema exists in Neo4j.

    Returns:
        dict: Status of each schema component
    """
    client = get_neo4j_client()
    status = {
        "entity_id_index": False,
        "entity_name_constraint": False,
        "entity_type_index": False,
        "document_id_index": False,
        "document_source_index": False,
        "relation_id_index": False,
        "relation_type_index": False,
        "relation_valid_from_index": False,
        "relation_valid_to_index": False,
        "relation_temporal_index": False,
    }

    try:
        constraints = await client.execute("SHOW CONSTRAINTS")
        for record in constraints:
            name = record.get("name", "").lower()
            if "entity" in name and "name" in name:
                status["entity_name_constraint"] = True

        indexes = await client.execute("SHOW INDEXES")
        for record in indexes:
            name = record.get("name", "").lower()
            if "entity_id" in name:
                status["entity_id_index"] = True
            if "entity_name" in name:
                status["entity_name_constraint"] = True
            if "entity_type" in name:
                status["entity_type_index"] = True
            if "document_id" in name:
                status["document_id_index"] = True
            if "document_source" in name:
                status["document_source_index"] = True
            if "relation_id" in name:
                status["relation_id_index"] = True
            if "relation_type" in name and "temporal" not in name:
                status["relation_type_index"] = True
            if "relation_valid_from" in name:
                status["relation_valid_from_index"] = True
            if "relation_valid_to" in name:
                status["relation_valid_to_index"] = True
            if "relation_temporal" in name:
                status["relation_temporal_index"] = True

    except Exception as e:
        logger.error("schema.verify.error", error=str(e))

    return status


def create_entity_node(
    name: str,
    entity_type: str,
    description: str = "",
) -> tuple[str, dict[str, Any]]:
    """
    Generate Cypher for creating an Entity node.

    Args:
        name: Entity name (must be unique)
        entity_type: Type of entity (e.g., "Person", "Concept", "Tool")
        description: Optional description

    Returns:
        tuple: (Cypher query, parameters dict)
    """
    return (
        """
    MERGE (e:Entity {name: $name})
    ON CREATE SET
        e.id = randomUUID(),
        e.type = $type,
        e.description = $description,
        e.created_at = datetime(),
        e.updated_at = datetime()
    ON MATCH SET
        e.updated_at = datetime()
    RETURN e.id AS id, e.name AS name, e.type AS type
    """,
        {"name": name, "type": entity_type, "description": description},
    )


def create_document_node(
    title: str,
    content: str,
    source: str,
) -> tuple[str, dict[str, Any]]:
    """
    Generate Cypher for creating a Document node.

    Args:
        title: Document title
        content: Document content
        source: Source (URL, file path, etc.)

    Returns:
        tuple: (Cypher query, parameters dict)
    """
    return (
        """
    CREATE (d:Document {
        id: randomUUID(),
        title: $title,
        content: $content,
        source: $source,
        ingested_at: datetime()
    })
    RETURN d.id AS id, d.title AS title, d.source AS source
    """,
        {"title": title, "content": content, "source": source},
    )


def link_document_entity(
    document_id: str,
    entity_id: str,
) -> tuple[str, dict[str, Any]]:
    """
    Link a document to an entity it contains.

    Args:
        document_id: Document node ID
        entity_id: Entity node ID

    Returns:
        tuple: (Cypher query, parameters dict)
    """
    return (
        """
    MATCH (d:Document {id: $document_id})
    MATCH (e:Entity {id: $entity_id})
    MERGE (d)-[:CONTAINS_ENTITY {created_at: datetime()}]->(e)
    """,
        {"document_id": document_id, "entity_id": entity_id},
    )


def link_entity_mention(
    entity_id: str,
    document_id: str,
) -> tuple[str, dict[str, Any]]:
    """
    Link an entity to a document it was mentioned in.

    Args:
        entity_id: Entity node ID
        document_id: Document node ID

    Returns:
        tuple: (Cypher query, parameters dict)
    """
    return (
        """
    MATCH (e:Entity {id: $entity_id})
    MATCH (d:Document {id: $document_id})
    MERGE (e)-[:MENTIONED_IN {created_at: datetime()}]->(d)
    """,
        {"entity_id": entity_id, "document_id": document_id},
    )


def create_entity_relation(
    source_id: str,
    target_id: str,
    relation_type: str,
    confidence: float = 1.0,
    source: str = "manual",
) -> tuple[str, dict[str, Any]]:
    """
    Generate Cypher for creating a temporal RELATES_TO edge between entities.

    Args:
        source_id: Source entity ID
        target_id: Target entity ID
        relation_type: Type of relationship (e.g., WORKS_FOR, LOCATED_IN)
        confidence: Confidence score (0-1)
        source: Source of the relation

    Returns:
        tuple: (Cypher query, parameters dict)
    """
    return (
        """
    MATCH (s:Entity {id: $source_id})
    MATCH (t:Entity {id: $target_id})
    CREATE (s)-[r:RELATES_TO {
        id: randomUUID(),
        type: $relation_type,
        valid_from: datetime(),
        valid_to: NULL,
        confidence: $confidence,
        source: $source
    }]->(t)
    RETURN r.id AS relation_id, r.type AS relation_type
    """,
        {
            "source_id": source_id,
            "target_id": target_id,
            "relation_type": relation_type,
            "confidence": confidence,
            "source": source,
        },
    )


def query_valid_facts(as_of: datetime | None = None) -> tuple[str, dict[str, Any]]:
    """
    Generate Cypher for querying currently valid facts.

    Args:
        as_of: Optional datetime to query historical state

    Returns:
        tuple: (Cypher query, parameters dict)
    """
    if as_of:
        return (
            """
        MATCH path = (start:Entity)-[r:RELATES_TO]->(end:Entity)
        WHERE r.valid_from <= datetime($as_of)
          AND (r.valid_to > datetime($as_of) OR r.valid_to IS NULL)
        RETURN start.name AS source, r.type AS relation_type,
               end.name AS target, r.confidence AS confidence
        """,
            {"as_of": as_of.isoformat()},
        )
    else:
        return (
            """
        MATCH path = (start:Entity)-[r:RELATES_TO]->(end:Entity)
        WHERE r.valid_to IS NULL
        RETURN start.name AS source, r.type AS relation_type,
               end.name AS target, r.confidence AS confidence
        """,
            {},
        )
