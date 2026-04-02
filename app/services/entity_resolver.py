"""
Entity resolver orchestrator for automatic entity deduplication.

Coordinates SimilarityService, MergeDecisionService, and MergeExecutor
to provide end-to-end entity resolution with temporal handling.

Per D-35: Incremental resolution with batch size default 1000
Per D-38: Automatic merge for score >= 0.90
Per D-03: Update strategy (not delete) preserves graph connectivity
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

import structlog

from app.database.neo4j_client import Neo4jClient
from app.services.merge_decision_service import (
    DecisionTier,
    MergeDecision,
    MergeDecisionService,
)
from app.services.merge_executor import MergeExecutor, MergeResult
from app.services.similarity_service import SimilarityService

logger = structlog.get_logger(__name__)


@dataclass
class ResolutionStats:
    """Statistics from a resolution run."""

    total_entities: int = 0
    candidates_found: int = 0
    decisions_made: int = 0
    automatic_merges: int = 0
    manual_reviews: int = 0
    rejections: int = 0
    merges_executed: int = 0
    merges_failed: int = 0
    execution_time_ms: float = 0.0
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_entities": self.total_entities,
            "candidates_found": self.candidates_found,
            "decisions_made": self.decisions_made,
            "automatic_merges": self.automatic_merges,
            "manual_reviews": self.manual_reviews,
            "rejections": self.rejections,
            "merges_executed": self.merges_executed,
            "merges_failed": self.merges_failed,
            "execution_time_ms": self.execution_time_ms,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat()
            if self.completed_at
            else None,
        }


@dataclass
class ResolutionResult:
    """Result of entity resolution operation."""

    success: bool
    stats: ResolutionStats
    decisions: list[MergeDecision] = field(default_factory=list)
    merge_results: list[MergeResult] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


class EntityResolver:
    """
    Orchestrates entity resolution workflow.

    Coordinates:
    - SimilarityService for finding duplicate candidates
    - MergeDecisionService for classifying decisions
    - MergeExecutor for executing automatic merges

    Per D-35: Supports batch processing with configurable batch size
    Per D-38: Automatically executes merges for score >= 0.90
    """

    def __init__(
        self,
        similarity_service: Optional[SimilarityService] = None,
        decision_service: Optional[MergeDecisionService] = None,
        merge_executor: Optional[MergeExecutor] = None,
        neo4j_client: Optional[Neo4jClient] = None,
        similarity_threshold: float = 0.80,
        batch_size: int = 1000,
        run_automatically: bool = False,
    ):
        """
        Initialize entity resolver.

        Args:
            similarity_service: Service for computing similarity
            decision_service: Service for making merge decisions
            merge_executor: Executor for performing merges
            neo4j_client: Neo4j client (creates new if None)
            similarity_threshold: Minimum score to consider (default 0.80)
            batch_size: Entities per batch (default 1000 per D-35)
            run_automatically: Whether to run after each ingestion
        """
        self.neo4j_client = neo4j_client or Neo4jClient()
        self.similarity_service = similarity_service or SimilarityService()
        self.decision_service = decision_service or MergeDecisionService(
            similarity_service=self.similarity_service,
            neo4j_client=self.neo4j_client,
        )
        self.merge_executor = merge_executor or MergeExecutor(self.neo4j_client)
        self.similarity_threshold = similarity_threshold
        self.batch_size = batch_size
        self.run_automatically = run_automatically

    async def _find_candidate_duplicates(self, entity_id: str) -> list[dict[str, Any]]:
        """
        Find candidate duplicate entities for a given entity.

        Args:
            entity_id: Entity ID to find duplicates for

        Returns:
            List of candidate entity dictionaries
        """
        # Get the source entity
        source_query = """
        MATCH (e:Entity {id: $entity_id})
        WHERE e.merged_into IS NULL
        RETURN e.id AS id, e.name AS name, e.type AS type, e.description AS description
        """
        source = await self.neo4j_client.execute_single(
            source_query, {"entity_id": entity_id}
        )

        if not source:
            return []

        # Find candidates with similar names (blocking strategy)
        candidate_query = """
        MATCH (e:Entity {id: $entity_id})
        MATCH (candidate:Entity)
        WHERE candidate.id <> e.id
          AND candidate.merged_into IS NULL
          AND candidate.type = e.type
          AND apoc.text.similarity(e.name, candidate.name) >= $threshold
        RETURN candidate.id AS id, candidate.name AS name,
               candidate.type AS type, candidate.description AS description
        LIMIT 50
        """

        try:
            candidates = await self.neo4j_client.execute(
                candidate_query,
                {"entity_id": entity_id, "threshold": self.similarity_threshold},
            )
        except Exception:
            # Fallback without apoc if not available
            candidates = await self._find_candidates_fallback(entity_id)

        return candidates

    async def _find_candidates_fallback(self, entity_id: str) -> list[dict[str, Any]]:
        """Fallback candidate finding without apoc.text.similarity."""
        # Get source entity name for basic matching
        query = """
        MATCH (e:Entity {id: $entity_id})
        RETURN e.id AS id, e.name AS name, e.type AS type, e.description AS description
        """
        source = await self.neo4j_client.execute_single(query, {"entity_id": entity_id})

        if not source:
            return []

        # Simple name-based blocking
        name_prefix = (
            source["name"][:4].lower()
            if len(source["name"]) >= 4
            else source["name"].lower()
        )

        candidate_query = """
        MATCH (e:Entity {id: $entity_id})
        MATCH (candidate:Entity)
        WHERE candidate.id <> e.id
          AND candidate.merged_into IS NULL
          AND candidate.type = e.type
          AND toLower(candidate.name) STARTS WITH $name_prefix
        RETURN candidate.id AS id, candidate.name AS name,
               candidate.type AS type, candidate.description AS description
        LIMIT 50
        """

        return await self.neo4j_client.execute(
            candidate_query,
            {"entity_id": entity_id, "name_prefix": name_prefix},
        )

    async def _get_all_active_entities(self) -> list[dict[str, Any]]:
        """Get all active (non-merged) entities."""
        query = """
        MATCH (e:Entity)
        WHERE e.merged_into IS NULL
        RETURN e.id AS id, e.name AS name, e.type AS type, e.description AS description
        """
        return await self.neo4j_client.execute(query, {})

    async def resolve_entities(self, entity_id: str) -> ResolutionResult:
        """
        Resolve duplicates for a single entity.

        Per D-38: Automatically merges entities with score >= 0.90

        Args:
            entity_id: Entity ID to resolve

        Returns:
            ResolutionResult with decisions and merge results
        """
        stats = ResolutionStats(started_at=datetime.utcnow())
        decisions = []
        merge_results = []
        errors = []

        try:
            # Find candidate duplicates
            candidates = await self._find_candidate_duplicates(entity_id)
            stats.candidates_found = len(candidates)

            if not candidates:
                logger.info("resolve.no_candidates", entity_id=entity_id)
                return ResolutionResult(
                    success=True,
                    stats=stats,
                    decisions=[],
                    merge_results=[],
                )

            # Get source entity
            source_query = """
            MATCH (e:Entity {id: $entity_id})
            WHERE e.merged_into IS NULL
            RETURN e.id AS id, e.name AS name, e.description AS description
            """
            source = await self.neo4j_client.execute_single(
                source_query, {"entity_id": entity_id}
            )

            if not source:
                errors.append(f"Entity {entity_id} not found or already merged")
                return ResolutionResult(
                    success=False,
                    stats=stats,
                    decisions=decisions,
                    merge_results=merge_results,
                    errors=errors,
                )

            # Evaluate each candidate
            for candidate in candidates:
                decision = await self.decision_service.evaluate_pair(source, candidate)
                decisions.append(decision)
                stats.decisions_made += 1

                if decision.tier == DecisionTier.AUTOMATIC_MERGE:
                    stats.automatic_merges += 1
                    # Execute merge automatically
                    result = await self.merge_executor.execute_from_decision(decision)
                    merge_results.append(result)
                    if result.success:
                        stats.merges_executed += 1
                    else:
                        stats.merges_failed += 1
                elif decision.tier == DecisionTier.MANUAL_REVIEW:
                    stats.manual_reviews += 1
                else:
                    stats.rejections += 1

            stats.total_entities = 1
            stats.completed_at = datetime.utcnow()

            logger.info(
                "resolve.complete",
                entity_id=entity_id,
                candidates=stats.candidates_found,
                automatic=stats.automatic_merges,
                manual=stats.manual_reviews,
                rejected=stats.rejections,
            )

        except Exception as e:
            logger.error("resolve.error", entity_id=entity_id, error=str(e))
            errors.append(str(e))

        return ResolutionResult(
            success=len(errors) == 0,
            stats=stats,
            decisions=decisions,
            merge_results=merge_results,
            errors=errors,
        )

    async def resolve_batch(
        self,
        entity_ids: Optional[list[str]] = None,
        batch_size: Optional[int] = None,
    ) -> ResolutionResult:
        """
        Resolve duplicates for multiple entities in batches.

        Per D-35: Batch processing with configurable size (default 1000)

        Args:
            entity_ids: List of entity IDs (all active entities if None)
            batch_size: Entities per batch (uses instance default if None)

        Returns:
            ResolutionResult with aggregated results
        """
        batch_size = batch_size or self.batch_size
        start_time = datetime.utcnow()

        stats = ResolutionStats(started_at=start_time)
        all_decisions = []
        all_merge_results = []
        all_errors = []

        try:
            # Get entities to process
            if entity_ids:
                entities_query = """
                UNWIND $entity_ids AS entity_id
                MATCH (e:Entity {id: entity_id})
                WHERE e.merged_into IS NULL
                RETURN e.id AS id, e.name AS name, e.type AS type, e.description AS description
                """
                entities = await self.neo4j_client.execute(
                    entities_query, {"entity_ids": entity_ids}
                )
            else:
                entities = await self._get_all_active_entities()

            stats.total_entities = len(entities)
            logger.info(
                "resolve.batch_start", total=stats.total_entities, batch_size=batch_size
            )

            # Process in batches
            for i in range(0, len(entities), batch_size):
                batch = entities[i : i + batch_size]
                batch_num = (i // batch_size) + 1
                total_batches = (len(entities) + batch_size - 1) // batch_size

                logger.info(
                    "resolve.batch_processing",
                    batch=batch_num,
                    total_batches=total_batches,
                    entities_in_batch=len(batch),
                )

                for entity in batch:
                    result = await self.resolve_entities(entity["id"])
                    stats.candidates_found += result.stats.candidates_found
                    stats.decisions_made += result.stats.decisions_made
                    stats.automatic_merges += result.stats.automatic_merges
                    stats.manual_reviews += result.stats.manual_reviews
                    stats.rejections += result.stats.rejections
                    stats.merges_executed += result.stats.merges_executed
                    stats.merges_failed += result.stats.merges_failed
                    all_errors.extend(result.errors)

                    all_decisions.extend(result.decisions)
                    all_merge_results.extend(result.merge_results)

        except Exception as e:
            logger.error("resolve.batch_error", error=str(e))
            all_errors.append(str(e))

        stats.completed_at = datetime.utcnow()
        stats.execution_time_ms = (
            stats.completed_at - start_time
        ).total_seconds() * 1000

        logger.info(
            "resolve.batch_complete",
            total_entities=stats.total_entities,
            decisions=stats.decisions_made,
            automatic=stats.automatic_merges,
            manual=stats.manual_reviews,
            rejected=stats.rejections,
            executed=stats.merges_executed,
            failed=stats.merges_failed,
            time_ms=stats.execution_time_ms,
        )

        return ResolutionResult(
            success=len(all_errors) == 0,
            stats=stats,
            decisions=all_decisions,
            merge_results=all_merge_results,
            errors=all_errors,
        )

    async def get_resolution_stats(self) -> dict[str, Any]:
        """
        Get current resolution statistics.

        Returns:
            Dictionary with resolution stats
        """
        # Get counts from decision service
        decision_stats = await self.decision_service.get_stats()

        # Get merge counts
        merge_query = """
        MATCH (e:Entity)
        WHERE e.merged_into IS NOT NULL
        RETURN count(e) AS merged_count
        """
        merge_result = await self.neo4j_client.execute_single(merge_query, {})
        merged_count = merge_result.get("merged_count", 0) if merge_result else 0

        # Get active entity count
        active_query = """
        MATCH (e:Entity)
        WHERE e.merged_into IS NULL
        RETURN count(e) AS active_count
        """
        active_result = await self.neo4j_client.execute_single(active_query, {})
        active_count = active_result.get("active_count", 0) if active_result else 0

        return {
            "active_entities": active_count,
            "merged_entities": merged_count,
            "decisions": decision_stats,
            "similarity_threshold": self.similarity_threshold,
            "batch_size": self.batch_size,
            "run_automatically": self.run_automatically,
        }


# Singleton instance
_entity_resolver: Optional[EntityResolver] = None


def get_entity_resolver() -> EntityResolver:
    """
    Get singleton EntityResolver instance.

    Returns:
        EntityResolver: Shared resolver instance
    """
    global _entity_resolver

    if _entity_resolver is None:
        from app.database.neo4j_client import get_neo4j_client
        from app.services.similarity_service import SimilarityService

        neo4j_client = get_neo4j_client()
        similarity_service = SimilarityService()
        decision_service = MergeDecisionService(
            similarity_service=similarity_service,
            neo4j_client=neo4j_client,
        )
        merge_executor = MergeExecutor(neo4j_client)

        _entity_resolver = EntityResolver(
            similarity_service=similarity_service,
            decision_service=decision_service,
            merge_executor=merge_executor,
            neo4j_client=neo4j_client,
        )

    return _entity_resolver
