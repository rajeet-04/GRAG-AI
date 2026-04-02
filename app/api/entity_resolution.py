"""
Entity Resolution API endpoints for GRAG AI.

Provides REST API for:
- Evaluating entity pairs for merge decisions
- Listing and filtering merge decisions
- Manual review of pending decisions

Per D-23, D-24, D-25, D-26, D-27 from CONTEXT.md.
"""

from datetime import datetime
from typing import Any, Optional

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field


logger = structlog.get_logger()

# Create router
router = APIRouter(prefix="/entity-resolution", tags=["entity-resolution"])


# Request/Response models
class EntityPairRequest(BaseModel):
    """Request model for evaluating an entity pair."""

    entity1_id: str = Field(..., description="First entity ID")
    entity1_name: str = Field(..., description="First entity name")
    entity1_description: str = Field(default="", description="First entity description")
    entity2_id: str = Field(..., description="Second entity ID")
    entity2_name: str = Field(..., description="Second entity name")
    entity2_description: str = Field(
        default="", description="Second entity description"
    )


class MergeDecisionResponse(BaseModel):
    """Response model for a merge decision."""

    decision_id: str
    entity1_id: str
    entity2_id: str
    entity1_name: str
    entity2_name: str
    similarity_score: float
    tier: str
    created_at: str
    decision: str
    reviewer: Optional[str] = None
    reasoning: Optional[str] = None


class ReviewRequest(BaseModel):
    """Request model for manual review of a merge decision."""

    action: str = Field(..., description="Review action: 'merge' or 'reject'")
    reviewer: str = Field(..., description="Name/ID of the reviewer")
    notes: Optional[str] = Field(default=None, description="Optional review notes")


class StatusResponse(BaseModel):
    """Response model for service status."""

    status: str
    stats: dict[str, Any]


class DecisionListResponse(BaseModel):
    """Response model for listing merge decisions."""

    total: int
    decisions: list[MergeDecisionResponse]


def get_decision_service():
    """Dependency to get MergeDecisionService instance."""
    from app.services.merge_decision_service import get_merge_decision_service

    return get_merge_decision_service()


@router.get("/status", response_model=StatusResponse)
async def get_status() -> StatusResponse:
    """
    Get entity resolution service status and statistics.

    Returns:
        StatusResponse with service health and decision statistics
    """
    try:
        service = get_decision_service()
        stats = await service.get_stats()

        return StatusResponse(
            status="healthy",
            stats=stats,
        )

    except Exception as e:
        logger.error("entity_resolution.status.failed", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get status: {str(e)}",
        )


@router.post("/evaluate", response_model=MergeDecisionResponse)
async def evaluate_entity_pair(
    request: EntityPairRequest,
) -> MergeDecisionResponse:
    """
    Evaluate an entity pair and create a merge decision.

    Computes similarity score and classifies into a decision tier:
    - AUTOMATIC_MERGE: score >= 0.90
    - MANUAL_REVIEW: score 0.80-0.89
    - REJECTED: score < 0.80

    Args:
        request: EntityPairRequest with both entity details

    Returns:
        MergeDecisionResponse with tier, score, and reasoning
    """
    try:
        service = get_decision_service()

        # Build entity dictionaries
        entity1 = {
            "id": request.entity1_id,
            "name": request.entity1_name,
            "description": request.entity1_description,
        }
        entity2 = {
            "id": request.entity2_id,
            "name": request.entity2_name,
            "description": request.entity2_description,
        }

        # Evaluate pair
        decision = await service.evaluate_pair(entity1, entity2)

        logger.info(
            "entity_resolution.evaluated",
            entity1=request.entity1_name,
            entity2=request.entity2_name,
            score=decision.similarity_score,
            tier=decision.tier.value,
        )

        return MergeDecisionResponse(
            decision_id=decision.decision_id,
            entity1_id=decision.entity1_id,
            entity2_id=decision.entity2_id,
            entity1_name=decision.entity1_name,
            entity2_name=decision.entity2_name,
            similarity_score=decision.similarity_score,
            tier=decision.tier.value,
            created_at=decision.created_at.isoformat(),
            decision=decision.decision,
            reviewer=decision.reviewer,
            reasoning=decision.reasoning,
        )

    except Exception as e:
        logger.error("entity_resolution.evaluate.failed", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to evaluate entity pair: {str(e)}",
        )


@router.get("/decisions", response_model=DecisionListResponse)
async def list_decisions(
    tier: Optional[str] = Query(
        default=None,
        description="Filter by tier: AUTOMATIC_MERGE, MANUAL_REVIEW, REJECTED",
    ),
    from_date: Optional[str] = Query(
        default=None, description="Filter decisions from this date (ISO format)"
    ),
    to_date: Optional[str] = Query(
        default=None, description="Filter decisions until this date (ISO format)"
    ),
    limit: int = Query(default=100, ge=1, le=1000, description="Max results to return"),
) -> DecisionListResponse:
    """
    List merge decisions with optional filters.

    Args:
        tier: Optional tier filter
        from_date: Optional start date (ISO format)
        to_date: Optional end date (ISO format)
        limit: Maximum number of results (1-1000)

    Returns:
        DecisionListResponse with matching decisions
    """
    try:
        service = get_decision_service()

        # Parse tier if provided
        tier_enum = None
        if tier:
            from app.services.merge_decision_service import DecisionTier

            try:
                tier_enum = DecisionTier(tier)
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid tier: {tier}. Must be one of: "
                    "AUTOMATIC_MERGE, MANUAL_REVIEW, REJECTED",
                )

        # Parse dates if provided
        from_dt = None
        to_dt = None
        if from_date:
            try:
                from_dt = datetime.fromisoformat(from_date)
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid from_date format: {from_date}. Use ISO format.",
                )
        if to_date:
            try:
                to_dt = datetime.fromisoformat(to_date)
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid to_date format: {to_date}. Use ISO format.",
                )

        # Get decisions
        decisions = await service.get_decisions(
            tier=tier_enum,
            from_date=from_dt,
            to_date=to_dt,
            limit=limit,
        )

        return DecisionListResponse(
            total=len(decisions),
            decisions=[
                MergeDecisionResponse(
                    decision_id=d.decision_id,
                    entity1_id=d.entity1_id,
                    entity2_id=d.entity2_id,
                    entity1_name=d.entity1_name,
                    entity2_name=d.entity2_name,
                    similarity_score=d.similarity_score,
                    tier=d.tier.value,
                    created_at=d.created_at.isoformat(),
                    decision=d.decision,
                    reviewer=d.reviewer,
                    reasoning=d.reasoning,
                )
                for d in decisions
            ],
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("entity_resolution.list.failed", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list decisions: {str(e)}",
        )


@router.get("/decisions/{decision_id}", response_model=MergeDecisionResponse)
async def get_decision(decision_id: str) -> MergeDecisionResponse:
    """
    Get a specific merge decision by ID.

    Args:
        decision_id: The decision ID

    Returns:
        MergeDecisionResponse with full decision details
    """
    try:
        service = get_decision_service()
        decision = await service.get_decision_by_id(decision_id)

        if not decision:
            raise HTTPException(
                status_code=404,
                detail=f"Decision with ID {decision_id} not found",
            )

        return MergeDecisionResponse(
            decision_id=decision.decision_id,
            entity1_id=decision.entity1_id,
            entity2_id=decision.entity2_id,
            entity1_name=decision.entity1_name,
            entity2_name=decision.entity2_name,
            similarity_score=decision.similarity_score,
            tier=decision.tier.value,
            created_at=decision.created_at.isoformat(),
            decision=decision.decision,
            reviewer=decision.reviewer,
            reasoning=decision.reasoning,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("entity_resolution.get.failed", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get decision: {str(e)}",
        )


@router.post("/decisions/{decision_id}/review", response_model=MergeDecisionResponse)
async def review_decision(
    decision_id: str,
    request: ReviewRequest,
) -> MergeDecisionResponse:
    """
    Process a manual review of a pending merge decision.

    Per D-27: Audit trail with reviewer info.

    Args:
        decision_id: ID of the decision to review
        request: ReviewRequest with action, reviewer, and optional notes

    Returns:
        MergeDecisionResponse with updated decision
    """
    try:
        service = get_decision_service()

        # Validate action
        if request.action not in ("merge", "reject"):
            raise HTTPException(
                status_code=400,
                detail="Invalid action. Must be 'merge' or 'reject'.",
            )

        # Process review
        decision = await service.review_decision(
            decision_id=decision_id,
            action=request.action,
            reviewer=request.reviewer,
            notes=request.notes,
        )

        if not decision:
            raise HTTPException(
                status_code=404,
                detail=f"Decision with ID {decision_id} not found",
            )

        logger.info(
            "entity_resolution.reviewed",
            decision_id=decision_id,
            action=request.action,
            reviewer=request.reviewer,
        )

        return MergeDecisionResponse(
            decision_id=decision.decision_id,
            entity1_id=decision.entity1_id,
            entity2_id=decision.entity2_id,
            entity1_name=decision.entity1_name,
            entity2_name=decision.entity2_name,
            similarity_score=decision.similarity_score,
            tier=decision.tier.value,
            created_at=decision.created_at.isoformat(),
            decision=decision.decision,
            reviewer=decision.reviewer,
            reasoning=decision.reasoning,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("entity_resolution.review.failed", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to review decision: {str(e)}",
        )


@router.get("/pending", response_model=DecisionListResponse)
async def get_pending_reviews() -> DecisionListResponse:
    """
    Get all decisions in MANUAL_REVIEW tier awaiting human review.

    Returns:
        DecisionListResponse with pending decisions
    """
    try:
        service = get_decision_service()
        decisions = await service.get_pending_reviews()

        return DecisionListResponse(
            total=len(decisions),
            decisions=[
                MergeDecisionResponse(
                    decision_id=d.decision_id,
                    entity1_id=d.entity1_id,
                    entity2_id=d.entity2_id,
                    entity1_name=d.entity1_name,
                    entity2_name=d.entity2_name,
                    similarity_score=d.similarity_score,
                    tier=d.tier.value,
                    created_at=d.created_at.isoformat(),
                    decision=d.decision,
                    reviewer=d.reviewer,
                    reasoning=d.reasoning,
                )
                for d in decisions
            ],
        )

    except Exception as e:
        logger.error("entity_resolution.pending.failed", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get pending reviews: {str(e)}",
        )
