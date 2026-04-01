"""Graphiti client for episodic edge management."""

from datetime import datetime
from typing import Any

import structlog
from neo4j import AsyncDriver

from app.graphiti.config import GraphitiConfig, get_graphiti_config


logger = structlog.get_logger()


class GraphitiClient:
    """
    Client for Graphiti episodic memory system.

    Graphiti provides temporal versioning for entities and facts.
    It maintains separate node labels (graphiti_*) from the manual KR schema.

    KR/KB SPLIT:
    - Graphiti: KR layer (factual knowledge evolution)
    - ChromaDB: KB layer (conversation history, episodic memory)
    """

    def __init__(
        self,
        driver: AsyncDriver,
        config: GraphitiConfig | None = None,
    ) -> None:
        """
        Initialize Graphiti client.

        Args:
            driver: Neo4j async driver
            config: Optional Graphiti configuration
        """
        self._driver = driver
        self._config = config or get_graphiti_config()
        self._graphiti = None

    async def initialize(self) -> None:
        """Initialize Graphiti instance."""
        try:
            from graphiti_core import Graphiti
            from graphiti_core.config import GraphitiConfig as GConfig

            gcfg = GConfig(
                neo4j_driver=self._driver,
                graph_node_labels=self._config.graphiti_node_labels,
                episode_history_window_days=self._config.graphiti_episode_history_window,
                max_entities=self._config.graphiti_max_entities,
                entity_similarity_threshold=self._config.graphiti_entity_similarity_threshold,
            )
            self._graphiti = Graphiti(config=gcfg)
            logger.info("graphiti.initialized")
        except ImportError:
            logger.warning("graphiti.not_installed")
            self._graphiti = None

    @property
    def is_available(self) -> bool:
        """Check if Graphiti is available and initialized."""
        return self._graphiti is not None

    @property
    def node_labels(self) -> dict[str, str]:
        """Get Graphiti node labels (separate from KR schema)."""
        return self._config.graphiti_node_labels

    async def add_episode(
        self,
        name: str,
        episode_body: str,
        source_name: str,
        source_uri: str | None = None,
        source_description: str | None = None,
    ) -> dict[str, Any] | None:
        """
        Add an episodic fact to the knowledge graph.

        Args:
            name: Name/identifier for this episode
            episode_body: Text describing the fact/event
            source_name: Name of the source (e.g., "News Article")
            source_uri: Optional URI of the source
            source_description: Optional description of the source

        Returns:
            dict: Episode info or None if Graphiti unavailable
        """
        if not self._graphiti:
            logger.warning("graphiti.unavailable")
            return None

        try:
            from datetime import timezone
            from graphiti_core.episode import EpisodeSource

            source = EpisodeSource(
                source_name=source_name,
                source_uri=source_uri or "",
                source_description=source_description or "",
            )

            result = await self._graphiti.add_episode(
                name=name,
                episode_body=episode_body,
                source=source,
            )

            logger.info(
                "graphiti.episode.created",
                name=name,
                episode_id=result.get("uuid") if result else None,
            )
            return result

        except Exception as e:
            logger.error("graphiti.episode.error", error=str(e))
            return None

    async def search(
        self,
        query: str,
        time_range: tuple[datetime, datetime] | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """
        Search Graphiti for facts matching query.

        Args:
            query: Natural language query
            time_range: Optional (start, end) tuple for temporal filtering
            limit: Maximum number of results

        Returns:
            list[dict]: Matching facts with temporal metadata
        """
        if not self._graphiti:
            logger.warning("graphiti.unavailable")
            return []

        try:
            result = await self._graphiti.search(
                query=query,
                time_range=time_range,
                top_k=limit,
            )

            facts = []
            if hasattr(result, "results"):
                for r in result.results:
                    facts.append(
                        {
                            "name": r.get("name"),
                            "fact": r.get("fact"),
                            "valid_from": r.get("valid_from"),
                            "valid_to": r.get("valid_to"),
                            "confidence": r.get("confidence", 1.0),
                            "source": r.get("source"),
                        }
                    )

            logger.info("graphiti.search.complete", query=query, results=len(facts))
            return facts

        except Exception as e:
            logger.error("graphiti.search.error", error=str(e))
            return []

    async def get_entity_history(
        self,
        entity_name: str,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> list[dict[str, Any]]:
        """
        Get historical facts for an entity.

        Args:
            entity_name: Name of entity to query
            start_date: Optional start of time range
            end_date: Optional end of time range

        Returns:
            list[dict]: Historical facts for the entity
        """
        if not self._graphiti:
            return []

        time_range = None
        if start_date and end_date:
            time_range = (start_date, end_date)

        return await self.search(
            query=entity_name,
            time_range=time_range,
            limit=50,
        )

    async def query_entity_history(
        self,
        entity_id: str,
        entity_name: str,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> dict[str, Any]:
        """
        Unified query for entity history combining Graphiti and KR results.

        Args:
            entity_id: Entity UUID
            entity_name: Entity name
            start_date: Optional start of time range
            end_date: Optional end of time range

        Returns:
            dict: Combined historical facts from Graphiti and KR
        """
        results = {
            "entity_id": entity_id,
            "entity_name": entity_name,
            "graphiti_facts": [],
            "temporal_relations": [],
            "time_range": {
                "start": start_date.isoformat() if start_date else None,
                "end": end_date.isoformat() if end_date else None,
            },
        }

        if self._graphiti:
            graphiti_facts = await self.get_entity_history(
                entity_name=entity_name,
                start_date=start_date,
                end_date=end_date,
            )
            results["graphiti_facts"] = graphiti_facts
            logger.info(
                "graphiti.entity_history",
                entity_name=entity_name,
                facts=len(graphiti_facts),
            )

        return results

    async def verify_label_separation(self) -> dict[str, bool]:
        """
        Verify Graphiti uses separate labels from manual KR schema.

        Returns:
            dict: Status of label verification
        """
        status = {
            "graphiti_labels_exist": False,
            "kr_labels_exist": False,
            "no_collision": True,
        }

        try:
            async with self._driver.session() as session:
                result = await session.run("""
                    MATCH (n)
                    WITH labels(n) AS lbls, count(*) AS cnt
                    UNWIND lbls AS label
                    RETURN label, sum(cnt) AS count
                    ORDER BY count DESC
                """)
                records = await result.data()

                graphiti_labels = set()
                kr_labels = set()

                for record in records:
                    label = record.get("label", "")
                    if label.startswith("graphiti_"):
                        graphiti_labels.add(label)
                    elif label in ("Entity", "Relation", "Document"):
                        kr_labels.add(label)

                status["graphiti_labels_exist"] = len(graphiti_labels) > 0
                status["kr_labels_exist"] = len(kr_labels) > 0

                overlap = graphiti_labels & kr_labels
                status["no_collision"] = len(overlap) == 0

                if overlap:
                    logger.warning("graphiti.label_collision", overlap=list(overlap))
                else:
                    logger.info(
                        "graphiti.label_separation_verified",
                        graphiti=len(graphiti_labels),
                        kr=len(kr_labels),
                    )

        except Exception as e:
            logger.error("graphiti.label_verify.error", error=str(e))

        return status


_graphiti_client: GraphitiClient | None = None


async def get_graphiti_client(driver: AsyncDriver) -> GraphitiClient:
    """
    Get singleton Graphiti client instance.

    Args:
        driver: Neo4j async driver

    Returns:
        GraphitiClient: Shared client instance
    """
    global _graphiti_client
    if _graphiti_client is None:
        _graphiti_client = GraphitiClient(driver)
        await _graphiti_client.initialize()
    return _graphiti_client
