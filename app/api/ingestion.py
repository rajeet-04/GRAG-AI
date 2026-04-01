"""
Ingestion API endpoints for GRAG AI.

Provides REST API for document ingestion with entity and relationship extraction
and Neo4j persistence.
"""

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

import structlog
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field


logger = structlog.get_logger()


# Request/Response models
class IngestRequest(BaseModel):
    """Request model for single document ingestion."""

    text: str = Field(..., description="Text content to extract entities from")
    options: Optional[Dict[str, Any]] = Field(
        default=None, description="Optional metadata"
    )


class BatchIngestRequest(BaseModel):
    """Request model for batch document ingestion."""

    documents: List[str] = Field(..., description="List of text documents to process")


class EntityResponse(BaseModel):
    """Response model for extracted entity."""

    id: str
    name: str
    type: str
    description: Optional[str] = None
    confidence: float


class RelationResponse(BaseModel):
    """Response model for extracted relationship."""

    source: str
    target: str
    type: str
    confidence: float


class IngestResponse(BaseModel):
    """Response model for single document ingestion."""

    document_id: str
    neo4j_document_id: Optional[str] = None
    text_length: int
    entities: List[EntityResponse]
    entity_ids: Optional[List[str]] = None
    relations: List[RelationResponse]
    relation_ids: Optional[List[str]] = None
    temporal_info: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Temporal metadata for this ingestion (valid_from, valid_to, version)",
    )


class BatchIngestResponse(BaseModel):
    """Response model for batch document ingestion."""

    total_documents: int
    results: List[IngestResponse]


class DocumentRetrieveResponse(BaseModel):
    """Response model for retrieving a previously ingested document."""

    document_id: str
    text: str
    entities: List[EntityResponse]
    relations: List[RelationResponse]


# In-memory storage for document results (placeholder for now)
_document_store: Dict[str, Dict[str, Any]] = {}

# Create router
router = APIRouter(prefix="/ingest", tags=["ingestion"])


def get_graph_writer():
    """Dependency to get GraphWriterService instance."""
    from app.ingestion.graph_writer import get_graph_writer_service

    return get_graph_writer_service()


@router.post("", response_model=IngestResponse)
async def ingest_document(
    request: IngestRequest,
    graph_writer=Depends(get_graph_writer),
) -> IngestResponse:
    """
    Ingest a single document and extract entities and relationships.

    Args:
        request: IngestRequest with text content
        graph_writer: GraphWriterService for Neo4j persistence

    Returns:
        IngestResponse with extracted entities and relations
    """
    if not request.text or not request.text.strip():
        raise HTTPException(
            status_code=400,
            detail="Text content cannot be empty",
        )

    logger.info("ingest.document.start", text_length=len(request.text))

    try:
        # Import extraction services
        from app.ingestion.entity_extractor import get_entity_extraction_service
        from app.ingestion.relation_extractor import get_relation_extraction_service

        entity_service = get_entity_extraction_service()
        relation_service = get_relation_extraction_service()

        # Extract entities
        entities = await entity_service.extract_entities(request.text)

        if not entities:
            logger.warning("ingest.no_entities_extracted")

        # Extract relationships
        relations = await relation_service.extract_relations(request.text, entities)

        # Generate local document ID
        document_id = str(uuid.uuid4())

        # Write to Neo4j with temporal metadata
        neo4j_result = await graph_writer.write_ingestion_result(
            request.text, entities, relations, source="api"
        )

        # Store result in local memory store
        result = {
            "document_id": document_id,
            "neo4j_document_id": neo4j_result.get("document_id"),
            "text": request.text,
            "text_length": len(request.text),
            "entities": entities,
            "relations": relations,
            "entity_ids": neo4j_result.get("entity_ids", []),
            "relation_ids": neo4j_result.get("relation_ids", []),
        }
        _document_store[document_id] = result

        logger.info(
            "ingest.document.complete",
            document_id=document_id,
            neo4j_document_id=neo4j_result.get("document_id"),
            entity_count=len(entities),
            relation_count=len(relations),
        )

        return IngestResponse(
            document_id=document_id,
            neo4j_document_id=neo4j_result.get("document_id"),
            text_length=len(request.text),
            entities=[
                EntityResponse(
                    id=e.get("id", ""),
                    name=e.get("name", ""),
                    type=e.get("type", "Concept"),
                    description=e.get("description"),
                    confidence=e.get("confidence", 0.5),
                )
                for e in entities
            ],
            entity_ids=neo4j_result.get("entity_ids"),
            relations=[
                RelationResponse(
                    source=r.get("source", ""),
                    target=r.get("target", ""),
                    type=r.get("type", "RELATED_TO"),
                    confidence=r.get("confidence", 0.5),
                )
                for r in relations
            ],
            relation_ids=neo4j_result.get("relation_ids"),
            temporal_info={
                "valid_from": datetime.utcnow().isoformat(),
                "valid_to": None,
                "version": "v1",
            },
        )

    except Exception as e:
        logger.error("ingest.document.failed", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process document: {str(e)}",
        )


@router.post("/batch", response_model=BatchIngestResponse)
async def ingest_batch(
    request: BatchIngestRequest,
    graph_writer=Depends(get_graph_writer),
) -> BatchIngestResponse:
    """
    Ingest multiple documents in batch and extract entities and relationships.

    Args:
        request: BatchIngestRequest with list of text documents
        graph_writer: GraphWriterService for Neo4j persistence

    Returns:
        BatchIngestResponse with results for each document
    """
    if not request.documents:
        raise HTTPException(
            status_code=400,
            detail="Documents list cannot be empty",
        )

    logger.info("ingest.batch.start", document_count=len(request.documents))

    results: list[IngestResponse] = []

    try:
        # Import extraction services
        from app.ingestion.entity_extractor import get_entity_extraction_service
        from app.ingestion.relation_extractor import get_relation_extraction_service

        entity_service = get_entity_extraction_service()
        relation_service = get_relation_extraction_service()

        for idx, text in enumerate(request.documents):
            if not text or not text.strip():
                logger.warning("ingest.batch.skip_empty", index=idx)
                continue

            try:
                # Extract entities
                entities = await entity_service.extract_entities(text)

                # Extract relationships
                relations = await relation_service.extract_relations(text, entities)

                # Generate local document ID
                document_id = str(uuid.uuid4())

                # Write to Neo4j
                neo4j_result = await graph_writer.write_ingestion_result(
                    text, entities, relations, source="api"
                )

                # Store result
                result = {
                    "document_id": document_id,
                    "neo4j_document_id": neo4j_result.get("document_id"),
                    "text": text,
                    "text_length": len(text),
                    "entities": entities,
                    "relations": relations,
                    "entity_ids": neo4j_result.get("entity_ids", []),
                    "relation_ids": neo4j_result.get("relation_ids", []),
                }
                _document_store[document_id] = result

                results.append(
                    IngestResponse(
                        document_id=document_id,
                        neo4j_document_id=neo4j_result.get("document_id"),
                        text_length=len(text),
                        entities=[
                            EntityResponse(
                                id=e.get("id", ""),
                                name=e.get("name", ""),
                                type=e.get("type", "Concept"),
                                description=e.get("description"),
                                confidence=e.get("confidence", 0.5),
                            )
                            for e in entities
                        ],
                        entity_ids=neo4j_result.get("entity_ids"),
                        relations=[
                            RelationResponse(
                                source=r.get("source", ""),
                                target=r.get("target", ""),
                                type=r.get("type", "RELATED_TO"),
                                confidence=r.get("confidence", 0.5),
                            )
                            for r in relations
                        ],
                        relation_ids=neo4j_result.get("relation_ids"),
                        temporal_info={
                            "valid_from": datetime.utcnow().isoformat(),
                            "valid_to": None,
                            "version": "v1",
                        },
                    )
                )

            except Exception as e:
                logger.error("ingest.batch.item_failed", index=idx, error=str(e))
                # Continue processing other documents

        logger.info(
            "ingest.batch.complete",
            successful=len(results),
            total=len(request.documents),
        )

        return BatchIngestResponse(
            total_documents=len(request.documents),
            results=results,
        )

    except Exception as e:
        logger.error("ingest.batch.failed", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process batch: {str(e)}",
        )


@router.get("/{document_id}", response_model=DocumentRetrieveResponse)
async def get_document(document_id: str) -> DocumentRetrieveResponse:
    """
    Retrieve a previously ingested document with its extracted entities and relations.

    Args:
        document_id: The ID of the document to retrieve

    Returns:
        DocumentRetrieveResponse with document content and extraction results
    """
    if document_id not in _document_store:
        raise HTTPException(
            status_code=404,
            detail=f"Document with ID {document_id} not found",
        )

    result = _document_store[document_id]

    return DocumentRetrieveResponse(
        document_id=document_id,
        text=result["text"],
        entities=[
            EntityResponse(
                id=e.get("id", ""),
                name=e.get("name", ""),
                type=e.get("type", "Concept"),
                description=e.get("description"),
                confidence=e.get("confidence", 0.5),
            )
            for e in result["entities"]
        ],
        relations=[
            RelationResponse(
                source=r.get("source", ""),
                target=r.get("target", ""),
                type=r.get("type", "RELATED_TO"),
                confidence=r.get("confidence", 0.5),
            )
            for r in result["relations"]
        ],
    )


class TemporalInfoResponse(BaseModel):
    """Response model for temporal information of a document."""

    document_id: str
    neo4j_document_id: Optional[str] = None
    valid_from: str
    valid_to: Optional[str] = None
    version: str
    relations: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Relations with their temporal metadata",
    )


@router.get("/{document_id}/temporal", response_model=TemporalInfoResponse)
async def get_document_temporal(document_id: str) -> TemporalInfoResponse:
    """
    Get temporal details for a document's entities and relations.

    Returns temporal metadata including valid_from and valid_to for each relation.

    Args:
        document_id: The ID of the document

    Returns:
        TemporalInfoResponse with temporal details
    """
    if document_id not in _document_store:
        raise HTTPException(
            status_code=404,
            detail=f"Document with ID {document_id} not found",
        )

    result = _document_store[document_id]
    neo4j_doc_id = result.get("neo4j_document_id")

    # Build temporal info from stored relations
    relations_info = []
    for rel in result.get("relations", []):
        relations_info.append(
            {
                "source": rel.get("source", ""),
                "target": rel.get("target", ""),
                "type": rel.get("type", "RELATED_TO"),
                "confidence": rel.get("confidence", 0.5),
                "valid_from": datetime.utcnow().isoformat(),
                "valid_to": None,
            }
        )

    return TemporalInfoResponse(
        document_id=document_id,
        neo4j_document_id=neo4j_doc_id,
        valid_from=datetime.utcnow().isoformat(),
        valid_to=None,
        version="v1",
        relations=relations_info,
    )
