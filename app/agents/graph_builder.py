"""
Graph Builder Agent node for LangGraph.

Wraps GraphWriterService with entity resolution and temporal versioning.
Handles persisting extracted entities and relations to Neo4j while
enforcing the KR/KB firewall (writes only to Neo4j, never ChromaDB).

Flow: entities + relations → EntityResolver → GraphWriter → Neo4j
"""

from datetime import datetime
from typing import Any

import structlog

from app.ingestion.graph_writer import get_graph_writer_service
from app.services.entity_resolver import EntityResolver, get_entity_resolver

logger = structlog.get_logger()


def get_current_timestamp() -> str:
    """
    Get current UTC timestamp for temporal versioning.

    Returns:
        ISO 8601 formatted UTC timestamp string
    """
    return datetime.utcnow().isoformat()


class GraphBuilderAgent:
    """
    Agent responsible for writing entities and relations to Neo4j.

    Pipeline:
    1. Run entity resolution to deduplicate against existing graph
    2. Write entities to Neo4j with MERGE (idempotent)
    3. Write relations with temporal valid_from/valid_to properties
    4. Preserve KR/KB firewall — Neo4j only, no ChromaDB writes
    """

    def __init__(
        self,
        entity_resolver: EntityResolver | None = None,
    ) -> None:
        """
        Initialize the graph builder agent.

        Args:
            entity_resolver: Optional pre-configured EntityResolver instance
        """
        self.entity_resolver = entity_resolver or get_entity_resolver()
        self.graph_writer = get_graph_writer_service()

    async def run(
        self,
        text: str,
        entities: list[dict[str, Any]],
        relations: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Run the full graph building pipeline.

        Args:
            text: Original source text (for document provenance)
            entities: Extracted entities from Ingestion Agent
            relations: Extracted relations from Ingestion Agent

        Returns:
            dict with graph_write_result, resolved_entities, temporal_relations
        """
        if not entities and not relations:
            logger.info("graph_builder.skip", reason="no entities or relations")
            return {
                "graph_write_result": None,
                "resolved_entities": [],
                "temporal_relations": [],
            }

        logger.info(
            "graph_builder.start",
            entity_count=len(entities),
            relation_count=len(relations),
        )

        # Step 1: Entity resolution — deduplicate against existing graph
        # Per D-38: Automatic merge for score >= 0.90
        resolution_result = await self.entity_resolver.resolve_batch(
            entity_ids=[e.get("id") for e in entities if e.get("id")]
        )
        resolved_entities = entities  # Resolution happens on Neo4j side

        if not resolution_result.success:
            logger.warning(
                "graph_builder.resolution_warnings",
                errors=resolution_result.errors,
            )

        # Step 2: Add temporal properties to relations
        # Per KR-02: valid_from/valid_to temporal versioning
        timestamp = get_current_timestamp()
        temporal_relations = []
        for rel in relations:
            temporal_rel = {
                **rel,
                "valid_from": timestamp,
                "valid_to": None,  # None = currently valid
            }
            temporal_relations.append(temporal_rel)

        # Step 3: Write to Neo4j via GraphWriterService
        # This handles: document node, entity nodes, relation edges
        write_result = await self.graph_writer.write_ingestion_result(
            text=text,
            entities=resolved_entities,
            relations=temporal_relations,
            source="agent",
        )

        logger.info(
            "graph_builder.complete",
            document_id=write_result.get("document_id"),
            entities_written=write_result.get("entity_count", 0),
            relations_written=write_result.get("relation_count", 0),
        )

        return {
            "graph_write_result": write_result,
            "resolved_entities": resolved_entities,
            "temporal_relations": temporal_relations,
        }


async def graph_builder_agent_node(state: dict[str, Any]) -> dict[str, Any]:
    """
    LangGraph node function for the Graph Builder Agent.

    Reads 'entities' and 'relations' from state (output of Ingestion Agent),
    runs entity resolution, writes to Neo4j with temporal versioning.

    Per AGNT-02 requirement: Graph Builder writes to Neo4j with temporal
    versioning. Preserves KR/KB firewall — writes only to Neo4j.

    Args:
        state: Current graph state dict

    Returns:
        Updated state dict with graph_write_result, kr_entities,
        kr_relations, and agent trace
    """
    entities = state.get("entities", [])
    relations = state.get("relations", [])
    text = state.get("ingestion_text", "")

    if not entities and not relations:
        logger.info("graph_builder.skip", reason="no entities or relations in state")
        return state

    agent = GraphBuilderAgent()
    result = await agent.run(text=text, entities=entities, relations=relations)

    existing_trace = state.get("agent_trace", [])

    return {
        **state,
        "graph_write_result": result["graph_write_result"],
        "kr_entities": result["resolved_entities"],
        "kr_relations": result["temporal_relations"],
        "agent_trace": existing_trace
        + [
            f"GraphBuilder: wrote {len(result['resolved_entities'])} entities, "
            f"{len(result['temporal_relations'])} relations to Neo4j with temporal versioning"
        ],
    }
