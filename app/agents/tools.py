"""LangChain tools wrapping GRAG AI services for agent use.

Provides tools for:
- Neo4j knowledge graph search (KR layer)
- ChromaDB episodic memory search (KB layer)
- ChromaDB semantic preference search (KB layer)
"""

import json
from datetime import datetime
from typing import Any

import structlog
from langchain_core.tools import tool

from app.database.neo4j_client import get_neo4j_client
from app.database.chroma_client import get_chromadb_client

logger = structlog.get_logger()


@tool
def neo4j_search(query: str, temporal_filters: str = "") -> str:
    """Search Neo4j knowledge graph with a Cypher query.

    Use this tool to find entities, relationships, and paths in the
    knowledge graph. Supports temporal filtering via valid_from/valid_to.

    Args:
        query: Cypher query string to execute against Neo4j
        temporal_filters: JSON string with optional 'valid_from', 'valid_to'
            ISO datetime keys. Example: '{"valid_to": "2024-01-01T00:00:00"}'

    Returns:
        JSON string of query results or error message
    """
    import asyncio

    async def _run() -> str:
        client = get_neo4j_client()
        try:
            # Parse temporal filters if provided
            filters: dict[str, Any] = {}
            if temporal_filters:
                try:
                    filters = json.loads(temporal_filters)
                except json.JSONDecodeError:
                    logger.warning(
                        "neo4j_search.invalid_temporal_filters", raw=temporal_filters
                    )

            # Inject temporal constraints into query if filters present
            final_query = query
            params: dict[str, Any] = {}

            if filters.get("valid_from"):
                params["temporal_valid_from"] = filters["valid_from"]
            if filters.get("valid_to"):
                params["temporal_valid_to"] = filters["valid_to"]

            # If the query uses temporal filter params, wrap with WHERE
            if params and "$temporal_" in final_query:
                # Query already references temporal params — pass them through
                pass
            elif params:
                # Auto-inject temporal filter into MATCH patterns
                temporal_where = []
                if "temporal_valid_from" in params:
                    temporal_where.append(
                        "r.valid_from >= datetime($temporal_valid_from)"
                    )
                if "temporal_valid_to" in params:
                    temporal_where.append(
                        "(r.valid_to IS NULL OR r.valid_to <= datetime($temporal_valid_to))"
                    )
                if temporal_where:
                    inject_point = final_query.upper().find("WHERE")
                    if inject_point == -1:
                        # No WHERE clause — add one before RETURN/ORDER/LIMIT
                        for keyword in ["RETURN", "ORDER BY", "LIMIT"]:
                            idx = final_query.upper().find(keyword)
                            if idx != -1:
                                final_query = (
                                    final_query[:idx]
                                    + f"WHERE {' AND '.join(temporal_where)}\n"
                                    + final_query[idx:]
                                )
                                break
                    else:
                        # Append to existing WHERE
                        final_query = (
                            final_query[: inject_point + 5]
                            + (" " + " AND ".join(temporal_where) + " AND ")
                            + final_query[inject_point + 5 :]
                        )

            results = await client.execute(final_query, params)
            logger.info(
                "neo4j_search.executed",
                query_preview=query[:100],
                result_count=len(results),
            )
            return json.dumps(results, default=str)

        except Exception as e:
            logger.error("neo4j_search.error", error=str(e), query=query[:200])
            return json.dumps({"error": str(e), "query": query[:200]})

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor() as pool:
            return pool.submit(asyncio.run, _run()).result()
    return asyncio.run(_run())


@tool
def chroma_episodic_search(query: str, limit: int = 5) -> str:
    """Search episodic memory in ChromaDB.

    Use this to find past interaction summaries, session history,
    and conversational context. Episodic memories capture what happened
    in previous sessions with temporal metadata.

    Args:
        query: Natural language query to search episodic memories
        limit: Maximum number of results to return (default 5)

    Returns:
        JSON string of matching episodic memories with metadata
    """
    import asyncio

    async def _run() -> str:
        try:
            client = get_chromadb_client()
            collection = client.get_episodic_collection()
            embedding_service = client._embedding_service

            embedding = await embedding_service.embed_text(query)

            results = collection.query(
                query_embeddings=[embedding],
                n_results=limit,
                include=["documents", "metadatas", "distances"],
            )

            episodes = []
            if results["ids"] and results["ids"][0]:
                for i, ep_id in enumerate(results["ids"][0]):
                    metadata = results["metadatas"][0][i]
                    episodes.append(
                        {
                            "id": ep_id,
                            "content": results["documents"][0][i],
                            "summary": metadata.get("summary", ""),
                            "session_id": metadata.get("session_id", ""),
                            "user_id": metadata.get("user_id", ""),
                            "timestamp": metadata.get("timestamp", ""),
                            "distance": results["distances"][0][i]
                            if "distances" in results
                            else None,
                        }
                    )

            logger.info(
                "chroma_episodic_search.executed",
                query=query[:100],
                result_count=len(episodes),
            )
            return json.dumps(episodes, default=str)

        except Exception as e:
            logger.error("chroma_episodic_search.error", error=str(e))
            return json.dumps({"error": str(e)})

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor() as pool:
            return pool.submit(asyncio.run, _run()).result()
    return asyncio.run(_run())


@tool
def chroma_semantic_search(query: str, limit: int = 3) -> str:
    """Search semantic preferences in ChromaDB.

    Use this to find user preferences, communication styles,
    and learned behavioral patterns. Semantic memory captures
    long-term user traits and preferences.

    Args:
        query: Natural language query to search semantic preferences
        limit: Maximum number of results to return (default 3)

    Returns:
        JSON string of matching semantic preferences with metadata
    """
    import asyncio

    async def _run() -> str:
        try:
            client = get_chromadb_client()
            collection = client.get_semantic_collection()
            embedding_service = client._embedding_service

            embedding = await embedding_service.embed_text(query)

            results = collection.query(
                query_embeddings=[embedding],
                n_results=limit,
                include=["documents", "metadatas", "distances"],
            )

            preferences = []
            if results["ids"] and results["ids"][0]:
                for i, pref_id in enumerate(results["ids"][0]):
                    metadata = results["metadatas"][0][i]
                    preferences.append(
                        {
                            "id": pref_id,
                            "content": results["documents"][0][i],
                            "preference_type": metadata.get("preference_type", ""),
                            "source": metadata.get("source", ""),
                            "confidence": metadata.get("confidence", 0.0),
                            "distance": results["distances"][0][i]
                            if "distances" in results
                            else None,
                        }
                    )

            logger.info(
                "chroma_semantic_search.executed",
                query=query[:100],
                result_count=len(preferences),
            )
            return json.dumps(preferences, default=str)

        except Exception as e:
            logger.error("chroma_semantic_search.error", error=str(e))
            return json.dumps({"error": str(e)})

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor() as pool:
            return pool.submit(asyncio.run, _run()).result()
    return asyncio.run(_run())


# Tool registry for LangGraph agent binding
AGENT_TOOLS = [neo4j_search, chroma_episodic_search, chroma_semantic_search]
