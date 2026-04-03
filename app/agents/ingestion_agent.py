"""
Ingestion Agent node for LangGraph.

Wraps existing Phase 4 extraction services (EntityExtractionService,
RelationExtractionService) as a LangGraph agent node. Handles the
document ingestion path — for query mode, this node is skipped.

Flow: text → EntityExtractor → RelationExtractor → entities + relations
"""

from typing import Any

import structlog

from app.ingestion.entity_extractor import EntityExtractionService
from app.ingestion.relation_extractor import RelationExtractionService

logger = structlog.get_logger()


class IngestionAgent:
    """
    Agent responsible for extracting entities and relations from raw text.

    Wraps the Phase 4 EntityExtractionService and RelationExtractionService
    into a single cohesive ingestion step for the LangGraph pipeline.
    """

    def __init__(self) -> None:
        """Initialize the ingestion agent with extraction services."""
        self.entity_extractor = EntityExtractionService()
        self.relation_extractor = RelationExtractionService()

    async def run(self, text: str) -> dict[str, Any]:
        """
        Run the full ingestion pipeline on input text.

        Args:
            text: Raw text to extract entities and relations from

        Returns:
            dict with 'entities' and 'relations' keys
        """
        if not text or not text.strip():
            logger.warning("ingestion_agent.empty_text")
            return {"entities": [], "relations": []}

        logger.info("ingestion_agent.start", text_length=len(text))

        # Step 1: Extract entities
        entities = await self.entity_extractor.extract_entities(text)
        logger.info("ingestion_agent.entities_extracted", count=len(entities))

        # Step 2: Extract relations between extracted entities
        relations = await self.relation_extractor.extract_relations(text, entities)
        logger.info("ingestion_agent.relations_extracted", count=len(relations))

        return {
            "entities": entities,
            "relations": relations,
        }


async def ingestion_agent_node(state: dict[str, Any]) -> dict[str, Any]:
    """
    LangGraph node function for the Ingestion Agent.

    Reads 'ingestion_text' from state, runs entity + relation extraction,
    and writes results back to state.

    For document ingestion mode. For query mode, this node is skipped
    (ingestion_text will be empty).

    Args:
        state: Current graph state dict

    Returns:
        Updated state dict with entities, relations, and agent trace
    """
    text = state.get("ingestion_text", "")
    if not text:
        logger.info("ingestion_agent.skip", reason="no ingestion_text in state")
        return state

    agent = IngestionAgent()
    result = await agent.run(text)

    existing_trace = state.get("agent_trace", [])

    return {
        **state,
        "entities": result["entities"],
        "relations": result["relations"],
        "agent_trace": existing_trace
        + [
            f"IngestionAgent: extracted {len(result['entities'])} entities, "
            f"{len(result['relations'])} relations"
        ],
    }


def create_ingestion_state(text: str, session_id: str = "default") -> dict[str, Any]:
    """
    Create initial graph state for document ingestion mode.

    Provides a complete state dict matching the GraphState schema
    from CONTEXT.md, populated with ingestion_text for the pipeline.

    Args:
        text: Document text to ingest
        session_id: Session identifier (default: "default")

    Returns:
        Complete GraphState dict ready for the ingestion pipeline
    """
    return {
        "user_query": "",
        "session_id": session_id,
        "ingestion_text": text,
        "search_intent": "",
        "temporal_filters": {},
        "cypher_query": "",
        "entities": [],
        "relations": [],
        "kr_entities": [],
        "kr_relations": [],
        "kr_paths": [],
        "episodic_memories": [],
        "semantic_preferences": [],
        "web_search_results": [],
        "merged_context": "",
        "token_count": 0,
        "answer": "",
        "reasoning_steps": [],
        "mermaid_path": "",
        "confidence_scores": {},
        "errors": [],
        "retry_count": 0,
        "agent_trace": ["State: initialized for ingestion"],
    }
