"""
Merge decision engine for entity resolution.

Classifies entity pairs into three confidence tiers:
- AUTOMATIC_MERGE: score >= 0.90
- MANUAL_REVIEW: score 0.80-0.89
- REJECTED: score < 0.80

Per D-23, D-24, D-25, D-26 from CONTEXT.md.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

import structlog

from app.database.neo4j_client import Neo4jClient
from app.services.similarity_service import SimilarityService


logger = structlog.get_logger()


class DecisionTier(str, Enum):
    """
    Three-tier confidence classification for merge decisions.

    Per D-23: Three-tier confidence-based system
    """

    AUTOMATIC_MERGE = "AUTOMATIC_MERGE"
    MANUAL_REVIEW = "MANUAL_REVIEW"
    REJECTED = "REJECTED"


# Score thresholds per D-24, D-25, D-26
AUTOMATIC_MERGE_THRESHOLD = 0.90
MANUAL_REVIEW_LOWER = 0.80
MANUAL_REVIEW_UPPER = 0.90
REJECTED_THRESHOLD = 0.80


@dataclass
class MergeDecision:
    """
    Represents a merge decision for an entity pair.

    Per D-27: Audit trail with scores, reasoning, reviewer info.
    """

    entity1_id: str
    entity2_id: str
    entity1_name: str
    entity2_name: str
    similarity_score: float
    tier: DecisionTier
    created_at: datetime
    decision: str  # "merged", "pending", "rejected"
    reviewer: Optional[str] = None
    reasoning: Optional[str] = None
    decision_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage/serialization."""
        return {
            "decision_id": self.decision_id,
            "entity1_id": self.entity1_id,
            "entity2_id": self.entity2_id,
            "entity1_name": self.entity1_name,
            "entity2_name": self.entity2_name,
            "similarity_score": self.similarity_score,
            "tier": self.tier.value,
            "created_at": self.created_at.isoformat(),
            "decision": self.decision,
            "reviewer": self.reviewer,
            "reasoning": self.reasoning,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MergeDecision":
        """Create from dictionary."""
        return cls(
            decision_id=data.get("decision_id", str(uuid.uuid4())),
            entity1_id=data["entity1_id"],
            entity2_id=data["entity2_id"],
            entity1_name=data.get("entity1_name", ""),
            entity2_name=data.get("entity2_name", ""),
            similarity_score=data["similarity_score"],
            tier=DecisionTier(data["tier"]),
            created_at=datetime.fromisoformat(data["created_at"])
            if isinstance(data.get("created_at"), str)
            else data.get("created_at", datetime.utcnow()),
            decision=data.get("decision", "pending"),
            reviewer=data.get("reviewer"),
            reasoning=data.get("reasoning"),
        )


class MergeDecisionService:
    """
    Service for evaluating entity pairs and making merge decisions.

    Integrates with SimilarityService for scoring and Neo4jClient for
    audit trail storage.
    """

    def __init__(
        self,
        similarity_service: SimilarityService,
        neo4j_client: Neo4jClient,
    ):
        """
        Initialize merge decision service.

        Args:
            similarity_service: Service for computing entity similarity
            neo4j_client: Neo4j client for audit trail storage
        """
        self.similarity_service = similarity_service
        self.neo4j_client = neo4j_client

    @staticmethod
    def classify_tier(score: float) -> DecisionTier:
        """
        Classify a similarity score into a decision tier.

        Per D-24, D-25, D-26:
        - Score >= 0.90 -> AUTOMATIC_MERGE
        - Score 0.80-0.89 -> MANUAL_REVIEW
        - Score < 0.80 -> REJECTED

        Args:
            score: Similarity score (0.0-1.0)

        Returns:
            DecisionTier classification
        """
        if score >= AUTOMATIC_MERGE_THRESHOLD:
            return DecisionTier.AUTOMATIC_MERGE
        elif score >= MANUAL_REVIEW_LOWER:
            return DecisionTier.MANUAL_REVIEW
        else:
            return DecisionTier.REJECTED

    def _get_default_decision(self, tier: DecisionTier) -> str:
        """Get default decision string for a tier."""
        tier_decisions = {
            DecisionTier.AUTOMATIC_MERGE: "merged",
            DecisionTier.MANUAL_REVIEW: "pending",
            DecisionTier.REJECTED: "rejected",
        }
        return tier_decisions.get(tier, "pending")

    def _build_reasoning(
        self,
        entity1_name: str,
        entity2_name: str,
        score: float,
        tier: DecisionTier,
    ) -> str:
        """Build reasoning string for a merge decision."""
        tier_explanations = {
            DecisionTier.AUTOMATIC_MERGE: (
                f"High confidence merge: score {score:.2f} exceeds "
                f"automatic merge threshold ({AUTOMATIC_MERGE_THRESHOLD}). "
                f"Entities '{entity1_name}' and '{entity2_name}' are likely duplicates."
            ),
            DecisionTier.MANUAL_REVIEW: (
                f"Uncertain merge: score {score:.2f} falls in manual review range "
                f"({MANUAL_REVIEW_LOWER}-{MANUAL_REVIEW_UPPER}). "
                f"Entities '{entity1_name}' and '{entity2_name}' may be duplicates "
                f"but require human verification."
            ),
            DecisionTier.REJECTED: (
                f"Low confidence merge: score {score:.2f} below manual review "
                f"threshold ({REJECTED_THRESHOLD}). "
                f"Entities '{entity1_name}' and '{entity2_name}' are likely distinct."
            ),
        }
        return tier_explanations.get(
            tier,
            f"Unable to determine merge decision for score {score:.2f}",
        )

    async def evaluate_pair(
        self,
        entity1: dict[str, Any],
        entity2: dict[str, Any],
    ) -> MergeDecision:
        """
        Evaluate an entity pair and create a merge decision.

        Args:
            entity1: First entity with 'id', 'name', 'description'
            entity2: Second entity with 'id', 'name', 'description'

        Returns:
            MergeDecision with classification and reasoning
        """
        entity1_id = entity1.get("id", "")
        entity2_id = entity2.get("id", "")
        entity1_name = entity1.get("name", "")
        entity2_name = entity2.get("name", "")
        entity1_desc = entity1.get("description", "")
        entity2_desc = entity2.get("description", "")

        # Compute similarity score
        similarity_score = await self.similarity_service.compute_similarity(
            entity1_name, entity1_desc, entity2_name, entity2_desc
        )

        # Classify into tier
        tier = self.classify_tier(similarity_score)
        default_decision = self._get_default_decision(tier)
        reasoning = self._build_reasoning(
            entity1_name, entity2_name, similarity_score, tier
        )

        # Create decision record
        decision = MergeDecision(
            entity1_id=entity1_id,
            entity2_id=entity2_id,
            entity1_name=entity1_name,
            entity2_name=entity2_name,
            similarity_score=similarity_score,
            tier=tier,
            created_at=datetime.utcnow(),
            decision=default_decision,
            reasoning=reasoning,
        )

        # Store in Neo4j for audit trail
        await self._store_decision(decision)

        logger.info(
            "merge_decision.created",
            decision_id=decision.decision_id,
            entity1=entity1_name,
            entity2=entity2_name,
            score=similarity_score,
            tier=tier.value,
            decision=default_decision,
        )

        return decision

    async def batch_evaluate(
        self, entity_pairs: list[tuple[dict[str, Any], dict[str, Any]]]
    ) -> list[MergeDecision]:
        """
        Evaluate multiple entity pairs.

        Args:
            entity_pairs: List of (entity1, entity2) tuples

        Returns:
            List of MergeDecision records
        """
        decisions = []
        for entity1, entity2 in entity_pairs:
            decision = await self.evaluate_pair(entity1, entity2)
            decisions.append(decision)
        return decisions

    async def get_pending_reviews(self) -> list[MergeDecision]:
        """
        Get all decisions in MANUAL_REVIEW tier awaiting human review.

        Returns:
            List of pending merge decisions
        """
        query = """
        MATCH (d:MergeDecision {decision: 'pending', tier: $tier})
        RETURN d.decision_id AS decision_id,
               d.entity1_id AS entity1_id,
               d.entity2_id AS entity2_id,
               d.entity1_name AS entity1_name,
               d.entity2_name AS entity2_name,
               d.similarity_score AS similarity_score,
               d.tier AS tier,
               d.created_at AS created_at,
               d.decision AS decision,
               d.reviewer AS reviewer,
               d.reasoning AS reasoning
        ORDER BY d.created_at DESC
        """

        results = await self.neo4j_client.execute(
            query, {"tier": DecisionTier.MANUAL_REVIEW.value}
        )

        decisions = []
        for record in results:
            decisions.append(MergeDecision.from_dict(record))

        logger.info("merge_decision.pending_reviews", count=len(decisions))
        return decisions

    async def get_decision_by_id(self, decision_id: str) -> Optional[MergeDecision]:
        """
        Get a specific merge decision by ID.

        Args:
            decision_id: The decision ID

        Returns:
            MergeDecision if found, None otherwise
        """
        query = """
        MATCH (d:MergeDecision {decision_id: $decision_id})
        RETURN d.decision_id AS decision_id,
               d.entity1_id AS entity1_id,
               d.entity2_id AS entity2_id,
               d.entity1_name AS entity1_name,
               d.entity2_name AS entity2_name,
               d.similarity_score AS similarity_score,
               d.tier AS tier,
               d.created_at AS created_at,
               d.decision AS decision,
               d.reviewer AS reviewer,
               d.reasoning AS reasoning
        """

        result = await self.neo4j_client.execute_single(
            query, {"decision_id": decision_id}
        )

        if result:
            return MergeDecision.from_dict(result)
        return None

    async def get_decisions(
        self,
        tier: Optional[DecisionTier] = None,
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None,
        limit: int = 100,
    ) -> list[MergeDecision]:
        """
        Get merge decisions with optional filters.

        Args:
            tier: Optional tier filter
            from_date: Optional start date filter
            to_date: Optional end date filter
            limit: Maximum number of results

        Returns:
            List of matching MergeDecision records
        """
        where_clauses = []
        params: dict[str, Any] = {"limit": limit}

        if tier:
            where_clauses.append("d.tier = $tier")
            params["tier"] = tier.value

        if from_date:
            where_clauses.append("d.created_at >= datetime($from_date)")
            params["from_date"] = from_date.isoformat()

        if to_date:
            where_clauses.append("d.created_at <= datetime($to_date)")
            params["to_date"] = to_date.isoformat()

        where_clause = " AND ".join(where_clauses) if where_clauses else "1=1"

        query = f"""
        MATCH (d:MergeDecision)
        WHERE {where_clause}
        RETURN d.decision_id AS decision_id,
               d.entity1_id AS entity1_id,
               d.entity2_id AS entity2_id,
               d.entity1_name AS entity1_name,
               d.entity2_name AS entity2_name,
               d.similarity_score AS similarity_score,
               d.tier AS tier,
               d.created_at AS created_at,
               d.decision AS decision,
               d.reviewer AS reviewer,
               d.reasoning AS reasoning
        ORDER BY d.created_at DESC
        LIMIT $limit
        """

        results = await self.neo4j_client.execute(query, params)

        decisions = []
        for record in results:
            decisions.append(MergeDecision.from_dict(record))

        logger.info(
            "merge_decision.list",
            count=len(decisions),
            tier=tier.value if tier else None,
        )
        return decisions

    async def review_decision(
        self,
        decision_id: str,
        action: str,  # "merge" or "reject"
        reviewer: str,
        notes: Optional[str] = None,
    ) -> Optional[MergeDecision]:
        """
        Process a manual review of a merge decision.

        Per D-27: Audit trail with reviewer info.

        Args:
            decision_id: ID of the decision to review
            action: Review action ("merge" or "reject")
            reviewer: Name/ID of the reviewer
            notes: Optional review notes

        Returns:
            Updated MergeDecision if found, None otherwise
        """
        if action not in ("merge", "reject"):
            raise ValueError(f"Invalid action: {action}. Must be 'merge' or 'reject'.")

        decision = await self.get_decision_by_id(decision_id)
        if not decision:
            logger.warning("merge_decision.review.not_found", decision_id=decision_id)
            return None

        # Update decision based on action
        decision.decision = action
        decision.reviewer = reviewer

        # Append review notes to reasoning
        review_note = (
            f"\n\n[Review by {reviewer} at {datetime.utcnow().isoformat()}]: "
            f"Action={action.upper()}. "
            f"Notes: {notes or 'No additional notes.'}"
        )
        decision.reasoning = (decision.reasoning or "") + review_note

        # Update in Neo4j
        update_query = """
        MATCH (d:MergeDecision {decision_id: $decision_id})
        SET d.decision = $decision,
            d.reviewer = $reviewer,
            d.reasoning = $reasoning
        """

        await self.neo4j_client.execute_write(
            update_query,
            {
                "decision_id": decision_id,
                "decision": action,
                "reviewer": reviewer,
                "reasoning": decision.reasoning,
            },
        )

        logger.info(
            "merge_decision.reviewed",
            decision_id=decision_id,
            action=action,
            reviewer=reviewer,
        )

        return decision

    async def _store_decision(self, decision: MergeDecision) -> None:
        """
        Store a merge decision in Neo4j for audit trail.

        Args:
            decision: MergeDecision to store
        """
        query = """
        CREATE (d:MergeDecision {
            decision_id: $decision_id,
            entity1_id: $entity1_id,
            entity2_id: $entity2_id,
            entity1_name: $entity1_name,
            entity2_name: $entity2_name,
            similarity_score: $similarity_score,
            tier: $tier,
            created_at: datetime($created_at),
            decision: $decision,
            reviewer: $reviewer,
            reasoning: $reasoning
        })
        """

        await self.neo4j_client.execute_write(
            query,
            {
                "decision_id": decision.decision_id,
                "entity1_id": decision.entity1_id,
                "entity2_id": decision.entity2_id,
                "entity1_name": decision.entity1_name,
                "entity2_name": decision.entity2_name,
                "similarity_score": decision.similarity_score,
                "tier": decision.tier.value,
                "created_at": decision.created_at.isoformat(),
                "decision": decision.decision,
                "reviewer": decision.reviewer,
                "reasoning": decision.reasoning,
            },
        )

    async def get_stats(self) -> dict[str, Any]:
        """
        Get statistics about merge decisions.

        Returns:
            Dictionary with decision statistics
        """
        query = """
        MATCH (d:MergeDecision)
        RETURN d.tier AS tier,
               d.decision AS decision,
               count(*) AS count
        ORDER BY tier, decision
        """

        results = await self.neo4j_client.execute(query, {})

        stats = {
            "total": 0,
            "by_tier": {
                DecisionTier.AUTOMATIC_MERGE.value: 0,
                DecisionTier.MANUAL_REVIEW.value: 0,
                DecisionTier.REJECTED.value: 0,
            },
            "by_decision": {
                "merged": 0,
                "pending": 0,
                "rejected": 0,
            },
        }

        for record in results:
            tier = record.get("tier")
            decision = record.get("decision")
            count = record.get("count", 0)

            stats["total"] += count

            if tier in stats["by_tier"]:
                stats["by_tier"][tier] = count

            if decision in stats["by_decision"]:
                stats["by_decision"][decision] = count

        return stats


# Singleton instance
_merge_decision_service: Optional[MergeDecisionService] = None


def get_merge_decision_service() -> MergeDecisionService:
    """
    Get singleton MergeDecisionService instance.

    Returns:
        MergeDecisionService: Shared service instance
    """
    global _merge_decision_service

    if _merge_decision_service is None:
        from app.database.neo4j_client import get_neo4j_client
        from app.services.similarity_service import SimilarityService

        neo4j_client = get_neo4j_client()
        similarity_service = SimilarityService()
        _merge_decision_service = MergeDecisionService(
            similarity_service=similarity_service,
            neo4j_client=neo4j_client,
        )

    return _merge_decision_service
