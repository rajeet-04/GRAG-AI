"""
Manual Review Queue API endpoints for GRAG AI.

Provides REST API for:
- Listing pending review items
- Getting decision details for review
- Approving or rejecting merge decisions
- Getting review queue statistics

Per D-25, D-27, D-28 from CONTEXT.md.
"""

from datetime import datetime
from typing import Any, Optional

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from app.services.review_service import (
    ReviewDetail,
    ReviewItem,
    ReviewStats,
    ReviewService,
    get_review_service,
)

logger = structlog.get_logger()

# Create router
router = APIRouter(prefix="/review-queue", tags=["review-queue"])


# Request models
class ApproveRequest(BaseModel):
    """Request model for approving a merge decision."""

    reviewer: str = Field(..., description="Name/ID of the reviewer")
    notes: str = Field(default="", description="Optional review notes")


class RejectRequest(BaseModel):
    """Request model for rejecting a merge decision."""

    reviewer: str = Field(..., description="Name/ID of the reviewer")
    notes: str = Field(default="", description="Optional rejection reason")


# Response models
class ReviewItemResponse(BaseModel):
    """Response model for a review item in the queue."""

    decision_id: str
    entity1_name: str
    entity2_name: str
    entity1_type: str
    entity2_type: str
    similarity_score: float
    tier: str
    created_at: str


class ReviewDetailResponse(BaseModel):
    """Response model for detailed review information."""

    decision_id: str
    entity1_id: str
    entity2_id: str
    entity1_name: str
    entity2_name: str
    entity1_type: str
    entity2_type: str
    entity1_description: Optional[str]
    entity2_description: Optional[str]
    similarity_score: float
    string_similarity: Optional[float]
    embedding_similarity: Optional[float]
    tier: str
    created_at: str
    reasoning: Optional[str]


class ReviewStatsResponse(BaseModel):
    """Response model for review queue statistics."""

    pending: int
    approved_today: int
    rejected_today: int
    total_reviewed: int


class ActionResponse(BaseModel):
    """Response model for approve/reject actions."""

    success: bool
    decision_id: str
    message: str
    details: Optional[dict[str, Any]] = None


# Dependency
def get_review_svc() -> ReviewService:
    """Get review service instance."""
    return get_review_service()


# Endpoints
@router.get("", response_model=list[ReviewItemResponse])
async def list_pending_reviews(
    limit: int = Query(default=50, ge=1, le=100, description="Max items to return"),
    offset: int = Query(default=0, ge=0, description="Items to skip"),
    sort_by: str = Query(default="created_at", description="Sort field"),
    review_service: ReviewService = Depends(get_review_svc),
) -> list[ReviewItemResponse]:
    """
    List pending review items from manual review queue.

    Per D-25: Returns decisions in MANUAL_REVIEW tier (score 0.80-0.89)
    Sorted by created_at (oldest first) for fairness in queue.

    Returns entity pairs with similarity scores for quick review decisions.
    """
    if sort_by not in ("created_at", "similarity_score"):
        raise HTTPException(
            status_code=400,
            detail="sort_by must be 'created_at' or 'similarity_score'",
        )

    items = await review_service.get_pending(
        limit=limit, offset=offset, sort_by=sort_by
    )

    return [
        ReviewItemResponse(
            decision_id=item.decision_id,
            entity1_name=item.entity1_name,
            entity2_name=item.entity2_name,
            entity1_type=item.entity1_type,
            entity2_type=item.entity2_type,
            similarity_score=item.similarity_score,
            tier=item.tier,
            created_at=item.created_at.isoformat() if item.created_at else "",
        )
        for item in items
    ]


@router.get("/{decision_id}", response_model=ReviewDetailResponse)
async def get_review_detail(
    decision_id: str,
    review_service: ReviewService = Depends(get_review_svc),
) -> ReviewDetailResponse:
    """
    Get detailed review information for a specific decision.

    Per D-28: Returns full similarity breakdown for review interface.
    Includes entity details, scores, and reasoning.
    """
    detail = await review_service.get_by_id(decision_id)

    if not detail:
        raise HTTPException(
            status_code=404,
            detail=f"Decision {decision_id} not found",
        )

    return ReviewDetailResponse(
        decision_id=detail.decision_id,
        entity1_id=detail.entity1_id,
        entity2_id=detail.entity2_id,
        entity1_name=detail.entity1_name,
        entity2_name=detail.entity2_name,
        entity1_type=detail.entity1_type,
        entity2_type=detail.entity2_type,
        entity1_description=detail.entity1_description,
        entity2_description=detail.entity2_description,
        similarity_score=detail.similarity_score,
        string_similarity=detail.string_similarity,
        embedding_similarity=detail.embedding_similarity,
        tier=detail.tier,
        created_at=detail.created_at.isoformat() if detail.created_at else "",
        reasoning=detail.reasoning,
    )


@router.post("/{decision_id}/approve", response_model=ActionResponse)
async def approve_merge(
    decision_id: str,
    request: ApproveRequest,
    review_service: ReviewService = Depends(get_review_svc),
) -> ActionResponse:
    """
    Approve a merge decision and execute the merge.

    Per D-27: Creates audit trail with reviewer identity and timestamp.
    Executes the actual merge in Neo4j.
    """
    result = await review_service.approve_merge(
        decision_id=decision_id,
        reviewer=request.reviewer,
        notes=request.notes,
    )

    if not result.get("success"):
        raise HTTPException(
            status_code=400,
            detail=result.get("error", "Failed to approve merge"),
        )

    return ActionResponse(
        success=True,
        decision_id=decision_id,
        message="Merge approved and executed successfully",
        details={
            "reviewer": result.get("reviewer"),
            "surviving_entity_id": result.get("surviving_entity_id"),
            "merged_entity_id": result.get("merged_entity_id"),
            "relations_updated": result.get("relations_updated"),
            "properties_merged": result.get("properties_merged"),
        },
    )


@router.post("/{decision_id}/reject", response_model=ActionResponse)
async def reject_merge(
    decision_id: str,
    request: RejectRequest,
    review_service: ReviewService = Depends(get_review_svc),
) -> ActionResponse:
    """
    Reject a merge decision without executing merge.

    Per D-27: Creates audit trail with reviewer identity and timestamp.
    """
    result = await review_service.reject_merge(
        decision_id=decision_id,
        reviewer=request.reviewer,
        notes=request.notes,
    )

    if not result.get("success"):
        raise HTTPException(
            status_code=400,
            detail=result.get("error", "Failed to reject merge"),
        )

    return ActionResponse(
        success=True,
        decision_id=decision_id,
        message="Merge rejected successfully",
        details={
            "reviewer": result.get("reviewer"),
            "action": result.get("action"),
        },
    )


@router.get("/stats", response_model=ReviewStatsResponse)
async def get_review_stats(
    review_service: ReviewService = Depends(get_review_svc),
) -> ReviewStatsResponse:
    """
    Get review queue statistics.

    Per D-28: Stats help reviewers prioritize work.
    Returns counts of pending, approved today, and rejected today.
    """
    stats = await review_service.get_statistics()

    return ReviewStatsResponse(
        pending=stats.pending_count,
        approved_today=stats.approved_today,
        rejected_today=stats.rejected_today,
        total_reviewed=stats.total_reviewed,
    )
