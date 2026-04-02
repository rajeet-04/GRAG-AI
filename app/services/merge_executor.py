"""
Merge executor for automatic entity merges with temporal handling.

Per D-03: Update strategy (not delete) preserves graph connectivity
Per D-09, D-10, D-11: Temporal union logic for valid_from/valid_to
Per D-12: Post-merge validation ensures temporal consistency
Per D-38: Automatic merge for score >= 0.90
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

import structlog

from app.database.neo4j_client import Neo4jClient
from app.services.merge_decision_service import DecisionTier, MergeDecision

logger = structlog.get_logger(__name__)


@dataclass
class MergeResult:
    """Result of a merge operation."""

    success: bool
    surviving_entity_id: str
    merged_entity_id: str
    temporal_valid_from: Optional[datetime]
    temporal_valid_to: Optional[datetime]
    relations_updated: int
    properties_merged: int
    violations: list[dict[str, Any]]
    error: Optional[str] = None


class MergeExecutor:
    """
    Executes entity merges with temporal union logic.

    Per D-03: Uses update strategy (not delete) to preserve graph connectivity.
    Per D-09, D-10, D-11: Computes temporal union of time periods.
    Per D-12: Runs post-merge temporal consistency validation.
    """

    def __init__(self, neo4j_client: Optional[Neo4jClient] = None):
        """
        Initialize merge executor.

        Args:
            neo4j_client: Neo4j client (creates new if None)
        """
        self.neo4j_client = neo4j_client or Neo4jClient()

    async def _fetch_entity(self, entity_id: str) -> Optional[dict[str, Any]]:
        """
        Fetch entity from Neo4j.

        Args:
            entity_id: Entity ID

        Returns:
            Entity dict or None if not found
        """
        query = """
        MATCH (e:Entity {id: $entity_id})
        RETURN e.id AS id, e.name AS name, e.type AS type,
               e.description AS description, e.created_at AS created_at,
               e.valid_from AS valid_from, e.valid_to AS valid_to
        """
        result = await self.neo4j_client.execute_single(query, {"entity_id": entity_id})
        return result

    def _compute_temporal_union(
        self,
        entity1: dict[str, Any],
        entity2: dict[str, Any],
    ) -> tuple[Optional[datetime], Optional[datetime]]:
        """
        Compute temporal union of two entities.

        Per D-09, D-10, D-11:
        - valid_from = earliest valid_from from either entity
        - valid_to = latest valid_to (NULL if either is NULL)

        Args:
            entity1: First entity
            entity2: Second entity

        Returns:
            Tuple of (merged_valid_from, merged_valid_to)
        """
        valid_from_1 = entity1.get("valid_from")
        valid_from_2 = entity2.get("valid_from")
        valid_to_1 = entity1.get("valid_to")
        valid_to_2 = entity2.get("valid_to")

        # Convert to datetime if needed
        if isinstance(valid_from_1, str):
            valid_from_1 = datetime.fromisoformat(valid_from_1.replace("Z", "+00:00"))
        if isinstance(valid_from_2, str):
            valid_from_2 = datetime.fromisoformat(valid_from_2.replace("Z", "+00:00"))
        if isinstance(valid_to_1, str):
            valid_to_1 = datetime.fromisoformat(valid_to_1.replace("Z", "+00:00"))
        if isinstance(valid_to_2, str):
            valid_to_2 = datetime.fromisoformat(valid_to_2.replace("Z", "+00:00"))

        # Compute earliest valid_from (per D-10)
        merged_valid_from = None
        valid_from_values = [v for v in [valid_from_1, valid_from_2] if v is not None]
        if valid_from_values:
            merged_valid_from = min(valid_from_values)

        # Compute latest valid_to, but NULL if either is NULL (per D-11)
        merged_valid_to = None
        if valid_to_1 is not None and valid_to_2 is not None:
            merged_valid_to = max(valid_to_1, valid_to_2)
        # If either is NULL, merged_valid_to remains NULL (per D-11)

        logger.debug(
            "merge.temporal_union",
            valid_from_1=valid_from_1,
            valid_from_2=valid_from_2,
            valid_to_1=valid_to_1,
            valid_to_2=valid_to_2,
            merged_valid_from=merged_valid_from,
            merged_valid_to=merged_valid_to,
        )

        return merged_valid_from, merged_valid_to

    async def execute_merge(
        self,
        entity1_id: str,
        entity2_id: str,
    ) -> MergeResult:
        """
        Execute merge of two entities using update strategy.

        Per D-03: Update rather than delete to preserve graph connectivity.
        Per D-09, D-10, D-11: Temporal union of time periods.
        Per D-12: Post-merge validation.

        Args:
            entity1_id: ID of surviving entity
            entity2_id: ID of entity to be merged into entity1

        Returns:
            MergeResult with operation status and details
        """
        # Fetch both entities
        entity1 = await self._fetch_entity(entity1_id)
        entity2 = await self._fetch_entity(entity2_id)

        if not entity1:
            return MergeResult(
                success=False,
                surviving_entity_id=entity1_id,
                merged_entity_id=entity2_id,
                temporal_valid_from=None,
                temporal_valid_to=None,
                relations_updated=0,
                properties_merged=0,
                violations=[],
                error=f"Entity {entity1_id} not found",
            )

        if not entity2:
            return MergeResult(
                success=False,
                surviving_entity_id=entity1_id,
                merged_entity_id=entity2_id,
                temporal_valid_from=None,
                temporal_valid_to=None,
                relations_updated=0,
                properties_merged=0,
                violations=[],
                error=f"Entity {entity2_id} not found",
            )

        # Compute temporal union
        merged_valid_from, merged_valid_to = self._compute_temporal_union(
            entity1, entity2
        )

        # Update surviving entity (entity1) with merged temporal metadata
        update_entity_query = """
        MATCH (e:Entity {id: $entity1_id})
        SET e.valid_from = CASE WHEN $valid_from IS NOT NULL
                                 THEN datetime($valid_from)
                                 ELSE e.valid_from END,
            e.valid_to = CASE WHEN $valid_to IS NOT NULL
                               THEN datetime($valid_to)
                               ELSE e.valid_to END,
            e.updated_at = datetime(),
            e.merged_from = $entity2_id,
            e.last_merged_at = datetime()
        """

        await self.neo4j_client.execute_write(
            update_entity_query,
            {
                "entity1_id": entity1_id,
                "entity2_id": entity2_id,
                "valid_from": merged_valid_from.isoformat()
                if merged_valid_from
                else None,
                "valid_to": merged_valid_to.isoformat() if merged_valid_to else None,
            },
        )

        # Mark entity2 as merged (per D-03: update strategy, not delete)
        merge_entity_query = """
        MATCH (e:Entity {id: $entity2_id})
        SET e.merged_into = $entity1_id,
            e.merged_at = datetime(),
            e.valid_to = datetime(),
            e.updated_at = datetime()
        """

        await self.neo4j_client.execute_write(
            merge_entity_query, {"entity1_id": entity1_id, "entity2_id": entity2_id}
        )

        # Transfer non-conflicting properties from entity2 to entity1
        # (excluding id, name, type which are primary identifiers)
        properties_merged = await self._merge_properties(entity1_id, entity2_id)

        # Update all relations pointing to entity2 to point to entity1
        relations_updated = await self._transfer_relations(
            entity1_id, entity2_id, merged_valid_from, merged_valid_to
        )

        # Validate temporal consistency (per D-12)
        violations = await self.neo4j_client.validate_temporal_consistency(entity1_id)

        logger.info(
            "merge.executed",
            surviving_entity_id=entity1_id,
            merged_entity_id=entity2_id,
            valid_from=merged_valid_from,
            valid_to=merged_valid_to,
            relations_updated=relations_updated,
            properties_merged=properties_merged,
            violations=len(violations),
        )

        return MergeResult(
            success=True,
            surviving_entity_id=entity1_id,
            merged_entity_id=entity2_id,
            temporal_valid_from=merged_valid_from,
            temporal_valid_to=merged_valid_to,
            relations_updated=relations_updated,
            properties_merged=properties_merged,
            violations=violations,
        )

    async def _merge_properties(self, entity1_id: str, entity2_id: str) -> int:
        """
        Merge non-conflicting properties from entity2 to entity1.

        Per D-03: Update strategy preserves graph connectivity.

        Args:
            entity1_id: Surviving entity ID
            entity2_id: Merged entity ID

        Returns:
            Number of properties merged
        """
        # Get all properties from both entities
        query = """
        MATCH (e1:Entity {id: $entity1_id})
        MATCH (e2:Entity {id: $entity2_id})
        RETURN properties(e1) AS props1, properties(e2) AS props2
        """

        result = await self.neo4j_client.execute_single(
            query,
            {
                "entity1_id": entity1_id,
                "entity2_id": entity2_id,
            },
        )

        if not result:
            return 0

        props1 = result.get("props1", {})
        props2 = result.get("props2", {})

        # Find non-conflicting properties (present in entity2 but not in entity1)
        excluded_keys = {
            "id",
            "name",
            "type",
            "valid_from",
            "valid_to",
            "created_at",
            "updated_at",
            "merged_into",
            "merged_at",
            "merged_from",
            "last_merged_at",
        }
        new_properties = {
            k: v
            for k, v in props2.items()
            if k not in excluded_keys and k not in props1
        }

        if not new_properties:
            return 0

        # Set new properties on entity1
        set_clause = ", ".join([f"e.{k} = ${k}" for k in new_properties.keys()])
        merge_query = f"""
        MATCH (e:Entity {{id: $entity1_id}})
        SET e.updated_at = datetime(), {set_clause}
        """

        params = {"entity1_id": entity1_id, **new_properties}
        await self.neo4j_client.execute_write(merge_query, params)

        logger.debug(
            "merge.properties_merged",
            count=len(new_properties),
            properties=list(new_properties.keys()),
        )
        return len(new_properties)

    async def _transfer_relations(
        self,
        entity1_id: str,
        entity2_id: str,
        valid_from: Optional[datetime],
        valid_to: Optional[datetime],
    ) -> int:
        """
        Transfer all relations from entity2 to entity1.

        Per D-03: Update strategy preserves graph connectivity.
        Per D-11: Update relation temporal metadata to reflect merged entity timeline.

        Args:
            entity1_id: Surviving entity ID
            entity2_id: Merged entity ID
            valid_from: Merged entity valid_from
            valid_to: Merged entity valid_to

        Returns:
            Number of relations updated
        """
        # Update outgoing relations (entity2 -> something) to (entity1 -> something)
        outgoing_query = """
        MATCH (e2:Entity {id: $entity2_id})-[r]->(target)
        WHERE e2.merged_into = $entity1_id
        WITH e2, r, target
        MERGE (e1:Entity {id: $entity1_id})-[new_r:RELATES_TO]->(target)
        ON CREATE SET new_r = properties(r)
        SET new_r.valid_from = CASE WHEN $valid_from IS NOT NULL
                                     THEN datetime($valid_from)
                                     ELSE new_r.valid_from END,
            new_r.valid_to = CASE WHEN $valid_to IS NOT NULL
                                   THEN datetime($valid_to)
                                   ELSE new_r.valid_to END,
            new_r.transferred_from = $entity2_id,
            new_r.transferred_at = datetime()
        DELETE r
        RETURN count(new_r) AS count
        """

        outgoing_result = await self.neo4j_client.execute_single(
            outgoing_query,
            {
                "entity1_id": entity1_id,
                "entity2_id": entity2_id,
                "valid_from": valid_from.isoformat() if valid_from else None,
                "valid_to": valid_to.isoformat() if valid_to else None,
            },
        )

        # Update incoming relations (something -> entity2) to (something -> entity1)
        incoming_query = """
        MATCH (source)-[r]->(e2:Entity {id: $entity2_id})
        WHERE e2.merged_into = $entity1_id
        WITH source, r, e2
        MERGE (source)-[new_r:RELATES_TO]->(e1:Entity {id: $entity1_id})
        ON CREATE SET new_r = properties(r)
        SET new_r.valid_from = CASE WHEN $valid_from IS NOT NULL
                                     THEN datetime($valid_from)
                                     ELSE new_r.valid_from END,
            new_r.valid_to = CASE WHEN $valid_to IS NOT NULL
                                   THEN datetime($valid_to)
                                   ELSE new_r.valid_to END,
            new_r.transferred_from = $entity2_id,
            new_r.transferred_at = datetime()
        DELETE r
        RETURN count(new_r) AS count
        """

        incoming_result = await self.neo4j_client.execute_single(
            incoming_query,
            {
                "entity1_id": entity1_id,
                "entity2_id": entity2_id,
                "valid_from": valid_from.isoformat() if valid_from else None,
                "valid_to": valid_to.isoformat() if valid_to else None,
            },
        )

        total_relations = (
            outgoing_result.get("count", 0) if outgoing_result else 0
        ) + (incoming_result.get("count", 0) if incoming_result else 0)

        logger.debug("merge.relations_transferred", count=total_relations)
        return total_relations

    async def execute_from_decision(
        self,
        decision: MergeDecision,
    ) -> MergeResult:
        """
        Execute merge from a MergeDecision.

        Only executes if decision tier is AUTOMATIC_MERGE.

        Per D-38: Automatic merge for score >= 0.90

        Args:
            decision: MergeDecision to execute

        Returns:
            MergeResult with operation status
        """
        if decision.tier != DecisionTier.AUTOMATIC_MERGE:
            return MergeResult(
                success=False,
                surviving_entity_id=decision.entity1_id,
                merged_entity_id=decision.entity2_id,
                temporal_valid_from=None,
                temporal_valid_to=None,
                relations_updated=0,
                properties_merged=0,
                violations=[],
                error=f"Decision tier {decision.tier.value} is not AUTOMATIC_MERGE",
            )

        result = await self.execute_merge(decision.entity1_id, decision.entity2_id)

        # Update decision with merge result
        if result.success:
            await self._mark_decision_merged(decision.decision_id)

        return result

    async def _mark_decision_merged(self, decision_id: str) -> None:
        """
        Mark a MergeDecision as merged.

        Args:
            decision_id: Decision ID
        """
        query = """
        MATCH (d:MergeDecision {decision_id: $decision_id})
        SET d.decision = 'merged',
            d.merged_at = datetime()
        """
        await self.neo4j_client.execute_write(query, {"decision_id": decision_id})

    async def execute_batch(
        self,
        decisions: list[MergeDecision],
    ) -> list[MergeResult]:
        """
        Execute merges for multiple decisions.

        Per D-35: Batch processing for efficiency.

        Args:
            decisions: List of MergeDecision objects

        Returns:
            List of MergeResult objects
        """
        results = []

        for decision in decisions:
            if decision.tier == DecisionTier.AUTOMATIC_MERGE:
                result = await self.execute_from_decision(decision)
                results.append(result)
            else:
                results.append(
                    MergeResult(
                        success=False,
                        surviving_entity_id=decision.entity1_id,
                        merged_entity_id=decision.entity2_id,
                        temporal_valid_from=None,
                        temporal_valid_to=None,
                        relations_updated=0,
                        properties_merged=0,
                        violations=[],
                        error=f"Skipped: tier {decision.tier.value} is not AUTOMATIC_MERGE",
                    )
                )

        # Log batch summary
        successful = sum(1 for r in results if r.success)
        failed = len(results) - successful

        logger.info(
            "merge.batch_complete",
            total=len(decisions),
            successful=successful,
            failed=failed,
        )

        return results


# Singleton instance
_merge_executor: Optional[MergeExecutor] = None


def get_merge_executor() -> MergeExecutor:
    """
    Get singleton MergeExecutor instance.

    Returns:
        MergeExecutor: Shared executor instance
    """
    global _merge_executor

    if _merge_executor is None:
        from app.database.neo4j_client import get_neo4j_client

        neo4j_client = get_neo4j_client()
        _merge_executor = MergeExecutor(neo4j_client)

    return _merge_executor
