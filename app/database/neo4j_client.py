"""Neo4j database client for GRAG AI."""

from contextlib import asynccontextmanager
from datetime import datetime
from typing import AsyncGenerator, Any

import structlog
from neo4j import AsyncDriver, AsyncGraphDatabase

from app.config import get_settings


logger = structlog.get_logger()


class Neo4jClient:
    """
    Async Neo4j client for executing Cypher queries.

    Provides connection management, query execution, and health checks.
    """

    def __init__(self) -> None:
        """Initialize Neo4j client with settings from config."""
        self.settings = get_settings()
        self._driver: AsyncDriver | None = None

    async def connect(self) -> AsyncDriver:
        """
        Establish connection to Neo4j.

        Returns:
            AsyncDriver: The Neo4j driver instance
        """
        if self._driver is None:
            uri = f"bolt://{self.settings.neo4j_uri.replace('bolt://', '')}"
            self._driver = AsyncGraphDatabase.driver(
                uri,
                auth=(self.settings.neo4j_user, self.settings.neo4j_password),
            )
            logger.info("neo4j.connecting", uri=uri)
        return self._driver

    async def close(self) -> None:
        """Close the Neo4j connection."""
        if self._driver is not None:
            await self._driver.close()
            self._driver = None
            logger.info("neo4j.closed")

    async def verify_connectivity(self) -> bool:
        """
        Verify connection to Neo4j is working.

        Returns:
            bool: True if connected, False otherwise
        """
        try:
            driver = await self.connect()
            await driver.verify_connectivity()
            logger.info("neo4j.connected")
            return True
        except Exception as e:
            logger.error("neo4j.connection_failed", error=str(e))
            return False

    @asynccontextmanager
    async def session(self) -> AsyncGenerator:
        """
        Async context manager for Neo4j sessions.

        Yields:
            AsyncSession: Neo4j session

        Usage:
            async with client.session() as session:
                result = await session.run("MATCH (n) RETURN n LIMIT 1")
        """
        driver = await self.connect()
        async with driver.session() as session:
            yield session

    async def execute(
        self, query: str, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """
        Execute a Cypher query and return results.

        Args:
            query: Cypher query string
            params: Optional query parameters

        Returns:
            list[dict]: Query results as list of records
        """
        async with self.session() as session:
            result = await session.run(query, params or {})
            records = await result.data()
            return records

    async def execute_single(
        self, query: str, params: dict[str, Any] | None = None
    ) -> dict[str, Any] | None:
        """
        Execute a Cypher query and return single result.

        Args:
            query: Cypher query string
            params: Optional query parameters

        Returns:
            dict | None: Single record or None if no results
        """
        records = await self.execute(query, params)
        return records[0] if records else None

    async def execute_write(
        self, query: str, params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Execute a Cypher write query with automatic commit.

        Args:
            query: Cypher query string
            params: Optional query parameters

        Returns:
            dict: Summary of write operation
        """
        async with self.session() as session:
            result = await session.run(query, params or {})
            summary = await result.consume()
            return {
                "counters": summary.counters,
                "type": summary.query_type,
            }

    async def get_valid_relations(
        self,
        source_id: str | None = None,
        target_id: str | None = None,
        relation_type: str | None = None,
        as_of: datetime | None = None,
    ) -> list[dict[str, Any]]:
        """
        Query for currently valid (non-expired) relations.

        Args:
            source_id: Optional source entity ID filter
            target_id: Optional target entity ID filter
            relation_type: Optional relation type filter
            as_of: Optional datetime to query historical state (default: now)

        Returns:
            list[dict]: Valid relations matching filters
        """
        as_of = as_of or datetime.utcnow()
        params: dict[str, Any] = {"as_of": as_of.isoformat()}

        where_clauses = [
            "r.valid_from <= datetime($as_of)",
            "(r.valid_to IS NULL OR r.valid_to >= datetime($as_of))",
        ]

        if source_id:
            where_clauses.append("s.id = $source_id")
            params["source_id"] = source_id

        if target_id:
            where_clauses.append("t.id = $target_id")
            params["target_id"] = target_id

        if relation_type:
            where_clauses.append("r.type = $relation_type")
            params["relation_type"] = relation_type

        where_clause = " AND ".join(where_clauses)

        query = f"""
            MATCH (s)-[r:RELATES_TO]->(t)
            WHERE {where_clause}
            RETURN s.id AS source_id, s.name AS source_name,
                   t.id AS target_id, t.name AS target_name,
                   r.type AS relation_type,
                   r.valid_from AS valid_from,
                   r.valid_to AS valid_to,
                   r.confidence AS confidence
            ORDER BY r.valid_from DESC
        """
        return await self.execute(query, params)

    async def get_historical_relations(
        self,
        source_id: str,
        target_id: str,
        relation_type: str | None = None,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> list[dict[str, Any]]:
        """
        Query historical relations for a specific time period.

        Args:
            source_id: Source entity ID
            target_id: Target entity ID
            relation_type: Optional relation type filter
            start_date: Optional start of time range
            end_date: Optional end of time range

        Returns:
            list[dict]: Historical relations matching filters
        """
        params: dict[str, Any] = {"source_id": source_id, "target_id": target_id}

        where_clauses = ["s.id = $source_id", "t.id = $target_id"]

        if relation_type:
            where_clauses.append("r.type = $relation_type")
            params["relation_type"] = relation_type

        if start_date:
            where_clauses.append("r.valid_from >= datetime($start_date)")
            params["start_date"] = start_date.isoformat()

        if end_date:
            where_clauses.append("r.valid_from <= datetime($end_date)")
            params["end_date"] = end_date.isoformat()

        where_clause = " AND ".join(where_clauses)

        query = f"""
            MATCH (s)-[r:RELATES_TO]->(t)
            WHERE {where_clause}
            RETURN r.type AS relation_type,
                   r.valid_from AS valid_from,
                   r.valid_to AS valid_to,
                   r.confidence AS confidence
            ORDER BY r.valid_from DESC
        """
        return await self.execute(query, params)

    async def create_temporal_relation(
        self,
        source_id: str,
        target_id: str,
        relation_type: str,
        valid_from: datetime | None = None,
        valid_to: datetime | None = None,
        confidence: float = 1.0,
        source: str = "manual",
    ) -> dict[str, Any]:
        """
        Create a temporal relation with automatic handling of overlapping relations.

        If an active relation of the same type exists between the same entities,
        it will be closed (valid_to set) before creating the new relation.

        Args:
            source_id: Source entity ID
            target_id: Target entity ID
            relation_type: Type of relationship
            valid_from: When relation becomes valid (default: now)
            valid_to: When relation expires (None = active)
            confidence: Confidence score (0-1)
            source: Source of the relation

        Returns:
            dict: Created relation info
        """
        valid_from = valid_from or datetime.utcnow()
        params = {
            "source_id": source_id,
            "target_id": target_id,
            "relation_type": relation_type,
            "valid_from": valid_from.isoformat(),
            "valid_to": valid_to.isoformat() if valid_to else None,
            "confidence": confidence,
            "source": source,
        }

        close_query = """
            MATCH (s)-[r:RELATES_TO]->(t)
            WHERE s.id = $source_id
              AND t.id = $target_id
              AND r.type = $relation_type
              AND r.valid_to IS NULL
            SET r.valid_to = datetime($valid_from)
        """
        await self.execute_write(close_query, params)

        create_query = """
            MATCH (s:Entity {id: $source_id})
            MATCH (t:Entity {id: $target_id})
            CREATE (s)-[r:RELATES_TO {
                id: randomUUID(),
                type: $relation_type,
                valid_from: datetime($valid_from),
                valid_to: CASE WHEN $valid_to IS NOT NULL THEN datetime($valid_to) ELSE NULL END,
                confidence: $confidence,
                source: $source
            }]->(t)
            RETURN r.id AS relation_id, r.type AS relation_type,
                   r.valid_from AS valid_from, r.valid_to AS valid_to
        """
        result = await self.execute_single(create_query, params)
        logger.info(
            "relation.created",
            source_id=source_id,
            target_id=target_id,
            relation_type=relation_type,
        )
        return result

    async def validate_temporal_consistency(
        self, entity_id: str | None = None
    ) -> list[dict[str, Any]]:
        """
        Validate temporal consistency for relations.

        Checks:
        1. No relations have valid_from > valid_to
        2. No overlapping active relations of same type

        Args:
            entity_id: Optional entity ID to limit scope

        Returns:
            list[dict]: List of violations found (empty if consistent)
        """
        violations = []

        invalid_ordering_query = """
            MATCH (s)-[r:RELATES_TO]->(t)
            WHERE r.valid_to IS NOT NULL
              AND r.valid_from > r.valid_to
        """
        params: dict[str, Any] = {}
        if entity_id:
            invalid_ordering_query += " AND (s.id = $entity_id OR t.id = $entity_id)"
            params["entity_id"] = entity_id

        invalid_ordering_query += """
            RETURN 'invalid_ordering' AS violation_type,
                   s.id AS source_id, t.id AS target_id,
                   r.valid_from AS valid_from, r.valid_to AS valid_to,
                   r.type AS relation_type
        """

        result = await self.execute(invalid_ordering_query, params)
        violations.extend(result)

        overlap_query = """
            MATCH (s)-[r1:RELATES_TO]->(t)
            WHERE r1.valid_to IS NULL
            MATCH (s)-[r2:RELATES_TO]->(t)
            WHERE r1 <> r2
              AND r2.valid_to IS NULL
              AND r1.type = r2.type
        """
        overlap_params: dict[str, Any] = {}
        if entity_id:
            overlap_query += " AND (s.id = $entity_id OR t.id = $entity_id)"
            overlap_params["entity_id"] = entity_id

        overlap_query += """
            RETURN 'overlapping' AS violation_type,
                   s.id AS source_id, t.id AS target_id,
                   r1.id AS relation1_id, r2.id AS relation2_id,
                   r1.type AS relation_type
        """

        result = await self.execute(overlap_query, overlap_params)
        violations.extend(result)

        if violations:
            logger.warning("temporal.violations_found", count=len(violations))
        else:
            logger.info("temporal.consistency_valid")

        return violations


_neo4j_client: Neo4jClient | None = None


def get_neo4j_client() -> Neo4jClient:
    """
    Get singleton Neo4j client instance.

    Returns:
        Neo4jClient: Shared client instance
    """
    global _neo4j_client
    if _neo4j_client is None:
        _neo4j_client = Neo4jClient()
    return _neo4j_client
