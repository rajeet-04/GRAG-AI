"""Neo4j database client for GRAG AI."""

import re
from contextlib import asynccontextmanager
from datetime import datetime
from typing import AsyncGenerator, Any, List, Optional

import structlog
from neo4j import AsyncDriver, AsyncGraphDatabase

from app.config import get_settings
from app.schemas.retrieval import TraversalResult


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

    # ── BFS Depth Validation ─────────────────────────────────────────

    def _validate_bfs_depth(self, cypher: str, max_depth: int = 4) -> bool:
        """Validate Cypher doesn't exceed max traversal depth.

        Returns True if valid, False if depth would exceed max_depth.
        Blocks queries like (a)-[*]->(b) without bounds.

        Args:
            cypher: Cypher query string to validate
            max_depth: Maximum allowed traversal depth (default 4)

        Returns:
            bool: True if query passes depth validation
        """
        # Check for completely unbounded patterns: [*] with no numbers
        if re.search(r"\[\*\](?!\d)", cypher):
            return False

        # Check for range without max: [1,] or [2,] etc.
        if re.search(r"\[\d+,\s*\]", cypher):
            return False

        # Check explicit bounds against max_depth: *5, *6 etc.
        depth_patterns = re.findall(r"\*(\d+)", cypher)
        for depth_str in depth_patterns:
            if int(depth_str) > max_depth:
                return False

        return True

    # ── Temporal Filter ──────────────────────────────────────────────

    def _inject_temporal_filter(
        self, cypher: str, as_of: Optional[datetime] = None
    ) -> str:
        """Inject temporal validity filter into Cypher WHERE clause.

        Adds: AND r.valid_from <= datetime() AND (r.valid_to IS NULL OR r.valid_to >= datetime())

        Must be applied to ALL relation traversals to satisfy RETR-03.

        Args:
            cypher: Cypher query string
            as_of: Optional datetime for historical query (default: now)

        Returns:
            str: Modified Cypher with temporal filter injected
        """
        as_of = as_of or datetime.utcnow()
        as_of_str = as_of.isoformat()
        temporal_constraint = (
            f"AND r.valid_from <= datetime('{as_of_str}') "
            f"AND (r.valid_to IS NULL OR r.valid_to >= datetime('{as_of_str}'))"
        )

        # Find WHERE clause position
        cypher_upper = cypher.upper()
        where_idx = cypher_upper.find("WHERE")
        if where_idx == -1:
            # No WHERE — add before RETURN/ORDER/LIMIT
            for keyword in ["RETURN", "ORDER BY", "LIMIT"]:
                idx = cypher_upper.find(keyword)
                if idx != -1:
                    return (
                        cypher[:idx]
                        + "WHERE "
                        + temporal_constraint
                        + "\n"
                        + cypher[idx:]
                    )
            # No keyword found — append at end
            return cypher + "\n" + temporal_constraint
        else:
            # Append to existing WHERE
            return (
                cypher[: where_idx + 5]
                + " "
                + temporal_constraint
                + " AND "
                + cypher[where_idx + 5 :]
            )

    # ── Multi-hop BFS Traversal ──────────────────────────────────────

    async def multi_hop_traverse(
        self,
        start_entity_id: str,
        max_depth: int = 4,
        relation_types: Optional[List[str]] = None,
        as_of: Optional[datetime] = None,
        max_results: int = 50,
    ) -> List[TraversalResult]:
        """Execute BFS traversal from start entity up to max_depth hops.

        Args:
            start_entity_id: Starting entity ID (must exist in Neo4j)
            max_depth: Maximum traversal depth (1-4), defaults to 4
            relation_types: Optional list of relation types to filter
            as_of: Optional datetime for historical query (default: now)
            max_results: Maximum total results (LIMIT per hop)

        Returns:
            list[TraversalResult]: Traversal results with paths and confidences

        Raises:
            ValueError: If max_depth exceeds 4 or is less than 1

        Key constraints (enforced):
        - LIMIT max_results per hop (prevents BFS explosion)
        - Temporal filter: valid_from <= now <= valid_to
        - Depth capped at max_depth
        """
        if max_depth < 1 or max_depth > 4:
            raise ValueError(f"max_depth must be 1-4, got {max_depth}")

        as_of = as_of or datetime.utcnow()

        # Optional relation type filter clause
        rel_filter = ""
        params: dict[str, Any] = {
            "start_id": start_entity_id,
            "as_of": as_of.isoformat(),
            "max_results": max_results,
        }

        if relation_types:
            rel_filter = "AND all(rr IN r WHERE rr.type IN $relation_types)"
            params["relation_types"] = relation_types

        # BFS query with temporal filtering and depth cap
        # Uses WHERE ALL(rel IN r ...) to check every relation in the path
        query = f"""
            MATCH path = (start {{id: $start_id}})-[r:RELATES_TO*1..{max_depth}]-(end)
            WHERE ALL(rel IN r WHERE rel.valid_from <= datetime($as_of))
              AND ALL(rel IN r WHERE rel.valid_to IS NULL OR rel.valid_to >= datetime($as_of))
              {rel_filter}
            WITH path, r,
                 start.id AS start_id, start.name AS start_name,
                 end.id AS end_id, end.name AS end_name, end.type AS end_type
            RETURN
                end_id AS entity_id,
                end_name AS entity_name,
                end_type AS entity_type,
                avg(reduce(conf = 0.0, rel IN r | CASE WHEN rel.confidence > conf THEN rel.confidence ELSE conf END)) AS confidence,
                length(path) AS depth,
                [rel IN r | {{
                    source_id: startNode(rel).id,
                    source_name: startNode(rel).name,
                    relation_type: rel.type,
                    relation_id: rel.id,
                    target_id: endNode(rel).id,
                    target_name: endNode(rel).name,
                    valid_from: toString(rel.valid_from),
                    valid_to: CASE WHEN rel.valid_to IS NULL THEN null ELSE toString(rel.valid_to) END,
                    confidence: rel.confidence
                }}] AS path
            ORDER BY depth ASC, confidence DESC
            LIMIT $max_results
        """

        # Validate query depth before execution
        if not self._validate_bfs_depth(query, max_depth):
            logger.warning("multi_hop_traverse.depth_exceeded", max_depth=max_depth)
            return []

        try:
            results = await self.execute(query, params)
            return [self._parse_traversal_result(r) for r in results]
        except Exception as e:
            logger.error("multi_hop_traverse.failed", error=str(e))
            return []

    def _parse_traversal_result(self, record: dict) -> TraversalResult:
        """Parse Neo4j record into TraversalResult.

        Args:
            record: Raw Neo4j result record

        Returns:
            TraversalResult: Typed traversal result
        """
        return TraversalResult(
            entity_id=record.get("entity_id", ""),
            entity_name=record.get("entity_name", ""),
            entity_type=record.get("entity_type", ""),
            confidence=float(record.get("confidence", 0.0)),
            depth=int(record.get("depth", 0)),
            path=record.get("path", []),
        )

    # ── Index Management ─────────────────────────────────────────────

    async def ensure_indexes(self) -> List[dict[str, Any]]:
        """Create required indexes for retrieval performance.

        Indexes required for RETR-01 performance (<3s latency):
        - Entity.name: Fast entity lookup by name
        - Entity.type: Fast filtering by entity type
        - Relation temporal: Fast temporal queries

        Returns:
            list: Index creation results
        """
        indexes = [
            ("entity_name", "FOR (e:Entity) ON (e.name)"),
            ("entity_type", "FOR (e:Entity) ON (e.type)"),
            ("relation_valid_from", "FOR ()-[r:RELATES_TO]-() ON (r.valid_from)"),
            ("relation_valid_to", "FOR ()-[r:RELATES_TO]-() ON (r.valid_to)"),
        ]

        results = []
        for name, schema in indexes:
            try:
                query = "CREATE INDEX {} IF NOT EXISTS {}".format(name, schema)
                await self.execute(query)
                results.append({"index": name, "status": "created"})
                logger.info("index.created", index=name)
            except Exception as e:
                # Index may already exist — not an error
                results.append({"index": name, "status": "exists", "error": str(e)})
                logger.debug("index.exists", index=name)

        return results


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
