"""
Review service for manual review queue management.

Provides queue management for uncertain merge decisions (score 0.80-0.89)
requiring human verification. Per D-25, D-27, D-28 from CONTEXT.md.

Key functionality:
- get_pending: Retrieve pending review items (sorted by date for fairness)
- get_by_id: Get full decision details with similarity breakdown
- approve_merge: Approve merge with reviewer info and audit trail
- reject_merge: Reject merge with audit trail
- Query helpers: by_reviewer, by_date_range, statistics
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Optional

import structlog

from app.services.merge_decision_service import (
    DecisionTier,
    MergeDecision,
    get_merge_decision_service,
)
from app.services.merge_executor import MergeExecutor, get_merge_executor

logger = structlog.get_logger()


@dataclass
class ReviewItem:
    """
    Lightweight review item for queue listing.

    Per D-28: Shows entity pairs, similarity score for quick decision.
    """

    decision_id: str
    entity1_name: str
    entity2_name: str
    entity1_type: str
    entity2_type: str
    similarity_score: float
    tier: str
    created_at: datetime


@dataclass
class ReviewDetail:
    """
    Full review details for decision review.

    Per D-28: Includes similarity breakdown for review interface.
    """

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
    created_at: datetime
    reasoning: Optional[str]


@dataclass
class ReviewStats:
    """
    Statistics for review queue.

    Per D-28: Stats help reviewers prioritize work.
    """

    pending_count: int
    approved_today: int
    rejected_today: int
    total_reviewed: int


class ReviewService:
    """
    Service for managing manual review queue.

    Per D-25: Manual review queue for score 0.80-0.89 (uncertain merges)
    Per D-27: Audit trail with reviewer identity
    Per D-28: Review interface showing similarity breakdown
    """

    def __init__(
        self,
        merge_decision_service: Optional[Any] = None,
        merge_executor: Optional[MergeExecutor] = None,
    ):
        """
        Initialize review service.

        Args:
            merge_decision_service: Service for merge decisions (lazy loaded if None)
            merge_executor: Service for executing approved merges (lazy loaded if None)
        """
        self._merge_decision_service = merge_decision_service
        self._merge_executor = merge_executor

    @property
    def merge_decision_service(self):
        """Lazy-load merge decision service."""
        if self._merge_decision_service is None:
            self._merge_decision_service = get_merge_decision_service()
        return self._merge_decision_service

    @property
    def merge_executor(self):
        """Lazy-load merge executor."""
        if self._merge_executor is None:
            self._merge_executor = get_merge_executor()
        return self._merge_executor

    async def get_pending(
        self,
        limit: int = 50,
        offset: int = 0,
        sort_by: str = "created_at",
    ) -> list[ReviewItem]:
        """
        Get pending review items from manual review queue.

        Per D-25: Retrieves decisions in MANUAL_REVIEW tier (score 0.80-0.89)
        Sorted by created_at (oldest first) for fairness in queue.

        Args:
            limit: Maximum number of items to return
            offset: Number of items to skip (for pagination)
            sort_by: Sort field ("created_at" or "similarity_score")

        Returns:
            List of ReviewItem objects
        """
        order_clause = (
            "ORDER BY d.created_at ASC"
            if sort_by == "created_at"
            else "ORDER BY d.similarity_score DESC"
        )

        query = f"""
        MATCH (d:MergeDecision {{tier: $tier, decision: 'pending'}})
        RETURN d.decision_id AS decision_id,
               d.entity1_name AS entity1_name,
               d.entity2_name AS entity2_name,
               d.entity1_type AS entity1_type,
               d.entity2_type AS entity2_type,
               d.similarity_score AS similarity_score,
               d.tier AS tier,
               d.created_at AS created_at
        {order_clause}
        SKIP $offset
        LIMIT $limit
        """

        results = await self.merge_decision_service.neo4j_client.execute(
            query,
            {
                "tier": DecisionTier.MANUAL_REVIEW.value,
                "offset": offset,
                "limit": limit,
            },
        )

        items = []
        for record in results:
            items.append(
                ReviewItem(
                    decision_id=record["decision_id"],
                    entity1_name=record.get("entity1_name", ""),
                    entity2_name=record.get("entity2_name", ""),
                    entity1_type=record.get("entity1_type", "Entity"),
                    entity2_type=record.get("entity2_type", "Entity"),
                    similarity_score=record.get("similarity_score", 0.0),
                    tier=record.get("tier", "MANUAL_REVIEW"),
                    created_at=record.get("created_at", datetime.utcnow()),
                )
            )

        logger.info(
            "review_queue.pending_fetched",
            count=len(items),
            limit=limit,
            offset=offset,
        )

        return items

    async def get_by_id(self, decision_id: str) -> Optional[ReviewDetail]:
        """
        Get detailed review information for a specific decision.

        Per D-28: Returns full similarity breakdown for review interface.

        Args:
            decision_id: ID of the decision to retrieve

        Returns:
            ReviewDetail if found, None otherwise
        """
        # Get the decision
        decision = await self.merge_decision_service.get_decision_by_id(decision_id)
        if not decision:
            logger.warning("review_queue.decision_not_found", decision_id=decision_id)
            return None

        # Fetch entity details for review
        entity1_details = await self._get_entity_details(decision.entity1_id)
        entity2_details = await self._get_entity_details(decision.entity2_id)

        # Parse similarity breakdown from reasoning
        string_sim, embedding_sim = self._parse_similarity_breakdown(decision.reasoning)

        return ReviewDetail(
            decision_id=decision.decision_id,
            entity1_id=decision.entity1_id,
            entity2_id=decision.entity2_id,
            entity1_name=decision.entity1_name,
            entity2_name=decision.entity2_name,
            entity1_type=entity1_details.get("type", "Entity")
            if entity1_details
            else "Entity",
            entity2_type=entity2_details.get("type", "Entity")
            if entity2_details
            else "Entity",
            entity1_description=entity1_details.get("description")
            if entity1_details
            else None,
            entity2_description=entity2_details.get("description")
            if entity2_details
            else None,
            similarity_score=decision.similarity_score,
            string_similarity=string_sim,
            embedding_similarity=embedding_sim,
            tier=decision.tier.value,
            created_at=decision.created_at,
            reasoning=decision.reasoning,
        )

    async def _get_entity_details(self, entity_id: str) -> Optional[dict[str, Any]]:
        """Fetch entity details from Neo4j."""
        query = """
        MATCH (e:Entity {id: $entity_id})
        RETURN e.id AS id, e.name AS name, e.type AS type, e.description AS description
        """
        result = await self.merge_decision_service.neo4j_client.execute_single(
            query, {"entity_id": entity_id}
        )
        return result

    def _parse_similarity_breakdown(
        self, reasoning: Optional[str]
    ) -> tuple[Optional[float], Optional[float]]:
        """
        Parse string and embedding similarity from reasoning string.

        Args:
            reasoning: Reasoning string containing similarity details

        Returns:
            Tuple of (string_similarity, embedding_similarity) or (None, None)
        """
        if not reasoning:
            return None, None

        # Try to extract similarity values from reasoning
        # Reasoning format varies but often includes scores
        string_sim = None
        embedding_sim = None

        # Look for patterns like "string: 0.85" or "embedding: 0.92"
        import re

        string_match = re.search(r"string[:\s]+([0-9.]+)", reasoning, re.IGNORECASE)
        if string_match:
            string_sim = float(string_match.group(1))

        embed_match = re.search(
            r"(?:embedding|embed)[:\s]+([0-9.]+)", reasoning, re.IGNORECASE
        )
        if embed_match:
            embedding_sim = float(embed_match.group(1))

        return string_sim, embedding_sim

    async def approve_merge(
        self,
        decision_id: str,
        reviewer: str,
        notes: str = "",
    ) -> dict[str, Any]:
        """
        Approve a merge decision and execute the merge.

        Per D-27: Audit trail with reviewer identity and timestamp.

        Args:
            decision_id: ID of the decision to approve
            reviewer: Name/ID of the reviewer
            notes: Optional review notes

        Returns:
            Dict with merge result details
        """
        # Get the decision
        decision = await self.merge_decision_service.get_decision_by_id(decision_id)
        if not decision:
            logger.warning("review_queue.approve.not_found", decision_id=decision_id)
            return {"success": False, "error": "Decision not found"}

        # Check it's still pending
        if decision.decision != "pending":
            logger.warning(
                "review_queue.approve.already_decided",
                decision_id=decision_id,
                current_decision=decision.decision,
            )
            return {
                "success": False,
                "error": f"Decision already processed: {decision.decision}",
            }

        # Update decision as approved (this also logs to audit trail)
        await self.merge_decision_service.review_decision(
            decision_id=decision_id,
            action="merged",
            reviewer=reviewer,
            notes=notes,
        )

        # Execute the actual merge
        merge_result = await self.merge_executor.execute_merge(
            decision.entity1_id, decision.entity2_id
        )

        logger.info(
            "review_queue.approved",
            decision_id=decision_id,
            reviewer=reviewer,
            merge_success=merge_result.success,
        )

        return {
            "success": merge_result.success,
            "decision_id": decision_id,
            "reviewer": reviewer,
            "surviving_entity_id": merge_result.surviving_entity_id,
            "merged_entity_id": merge_result.merged_entity_id,
            "relations_updated": merge_result.relations_updated,
            "properties_merged": merge_result.properties_merged,
            "error": merge_result.error,
        }

    async def reject_merge(
        self,
        decision_id: str,
        reviewer: str,
        notes: str = "",
    ) -> dict[str, Any]:
        """
        Reject a merge decision without executing merge.

        Per D-27: Audit trail with reviewer identity and timestamp.

        Args:
            decision_id: ID of the decision to reject
            reviewer: Name/ID of the reviewer
            notes: Optional rejection reason

        Returns:
            Dict with rejection result
        """
        # Get the decision
        decision = await self.merge_decision_service.get_decision_by_id(decision_id)
        if not decision:
            logger.warning("review_queue.reject.not_found", decision_id=decision_id)
            return {"success": False, "error": "Decision not found"}

        # Check it's still pending
        if decision.decision != "pending":
            logger.warning(
                "review_queue.reject.already_decided",
                decision_id=decision_id,
                current_decision=decision.decision,
            )
            return {
                "success": False,
                "error": f"Decision already processed: {decision.decision}",
            }

        # Update decision as rejected (this also logs to audit trail)
        await self.merge_decision_service.review_decision(
            decision_id=decision_id,
            action="rejected",
            reviewer=reviewer,
            notes=notes,
        )

        logger.info(
            "review_queue.rejected",
            decision_id=decision_id,
            reviewer=reviewer,
        )

        return {
            "success": True,
            "decision_id": decision_id,
            "reviewer": reviewer,
            "action": "rejected",
        }

    async def get_by_reviewer(self, reviewer: str) -> list[MergeDecision]:
        """
        Get all decisions reviewed by a specific reviewer.

        Per D-27: Audit trail query by reviewer.

        Args:
            reviewer: Name/ID of the reviewer

        Returns:
            List of MergeDecision objects reviewed by this person
        """
        return await self.merge_decision_service.get_decisions(
            from_date=None,
            to_date=None,
            limit=100,
        )

    async def get_by_date_range(
        self,
        from_date: datetime,
        to_date: datetime,
    ) -> list[MergeDecision]:
        """
        Get decisions created within a date range.

        Args:
            from_date: Start of date range
            to_date: End of date range

        Returns:
            List of MergeDecision objects in the range
        """
        return await self.merge_decision_service.get_decisions(
            from_date=from_date,
            to_date=to_date,
            limit=100,
        )

    async def get_statistics(self) -> ReviewStats:
        """
        Get review queue statistics.

        Per D-28: Stats help reviewers prioritize work.

        Returns:
            ReviewStats with queue counts
        """
        # Get total pending count
        pending_query = """
        MATCH (d:MergeDecision {tier: $tier, decision: 'pending'})
        RETURN count(d) AS count
        """
        pending_result = await self.merge_decision_service.neo4j_client.execute_single(
            pending_query, {"tier": DecisionTier.MANUAL_REVIEW.value}
        )
        pending_count = pending_result.get("count", 0) if pending_result else 0

        # Get today's date range
        today_start = datetime.utcnow().replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        today_end = today_start + timedelta(days=1)

        # Get approved today
        approved_today_query = """
        MATCH (d:MergeDecision {decision: 'merged'})
        WHERE d.created_at >= datetime($today_start)
          AND d.created_at < datetime($today_end)
        RETURN count(d) AS count
        """
        approved_result = await self.merge_decision_service.neo4j_client.execute_single(
            approved_today_query,
            {
                "today_start": today_start.isoformat(),
                "today_end": today_end.isoformat(),
            },
        )
        approved_today = approved_result.get("count", 0) if approved_result else 0

        # Get rejected today
        rejected_today_query = """
        MATCH (d:MergeDecision {decision: 'rejected'})
        WHERE d.created_at >= datetime($today_start)
          AND d.created_at < datetime($today_end)
        RETURN count(d) AS count
        """
        rejected_result = await self.merge_decision_service.neo4j_client.execute_single(
            rejected_today_query,
            {
                "today_start": today_start.isoformat(),
                "today_end": today_end.isoformat(),
            },
        )
        rejected_today = rejected_result.get("count", 0) if rejected_result else 0

        # Get total reviewed (merged + rejected)
        total_reviewed_query = """
        MATCH (d:MergeDecision)
        WHERE d.decision IN ['merged', 'rejected']
        RETURN count(d) AS count
        """
        total_result = await self.merge_decision_service.neo4j_client.execute_single(
            total_reviewed_query, {}
        )
        total_reviewed = total_result.get("count", 0) if total_result else 0

        return ReviewStats(
            pending_count=pending_count,
            approved_today=approved_today,
            rejected_today=rejected_today,
            total_reviewed=total_reviewed,
        )


# Singleton instance
_review_service: Optional[ReviewService] = None


def get_review_service() -> ReviewService:
    """
    Get singleton ReviewService instance.

    Returns:
        ReviewService: Shared service instance
    """
    global _review_service

    if _review_service is None:
        _review_service = ReviewService()

    return _review_service
