"""Ollama web-search fallback helpers.

Used when local KR/KB context is insufficient or the query explicitly
requires fresh/real-time information.
"""

from __future__ import annotations

from typing import Any, Dict, List

import structlog

from app.config import get_settings
from app.llm.ollama_client import OllamaClient


logger = structlog.get_logger()

LATEST_QUERY_HINTS = (
    "latest",
    "today",
    "right now",
    "current",
    "new",
    "news",
    "recent",
    "updated",
    "update",
    "this week",
    "this month",
    "this year",
)


def is_latest_data_query(query: str) -> bool:
    """Detect if query likely needs fresh external data."""
    text = (query or "").strip().lower()
    if not text:
        return False
    return any(token in text for token in LATEST_QUERY_HINTS)


def should_trigger_web_search(
    *,
    query: str,
    kr_entities: List[dict[str, Any]],
    episodic_memories: List[dict[str, Any]],
    semantic_preferences: List[dict[str, Any]],
) -> bool:
    """Decide whether web search fallback should be attempted."""
    has_local_context = bool(kr_entities or episodic_memories or semantic_preferences)
    needs_latest = is_latest_data_query(query)

    # Trigger when local context is empty OR query asks for latest/current data.
    return (not has_local_context) or needs_latest


async def execute_web_search(query: str) -> List[Dict[str, str]]:
    """Execute Ollama cloud web search and return normalized results."""
    settings = get_settings()

    if not bool(getattr(settings, "query_enable_web_search_fallback", True)):
        return []

    if not bool(getattr(settings, "ollama_enable_web_search", True)):
        return []

    if not (query or "").strip():
        return []

    if not settings.ollama_cloud_api_key:
        logger.info("web_search.skipped", reason="missing OLLAMA_CLOUD_API_KEY")
        return []

    max_results = max(1, min(int(getattr(settings, "ollama_web_search_max_results", 5)), 10))

    try:
        client = OllamaClient(use_cloud=True)
        payload = await client.web_search(query=query, max_results=max_results)
        raw_results = payload.get("results", []) or []

        normalized: List[Dict[str, str]] = []
        for item in raw_results:
            normalized.append(
                {
                    "title": str(item.get("title", "")).strip(),
                    "url": str(item.get("url", "")).strip(),
                    "content": str(item.get("content", "")).strip(),
                }
            )

        logger.info("web_search.complete", result_count=len(normalized))
        return normalized
    except Exception as e:
        logger.warning("web_search.failed", error=str(e))
        return []


def format_web_search_results(results: List[Dict[str, str]]) -> str:
    """Format web search results for context builder merging."""
    if not results:
        return ""

    lines = ["## Web Search Results\n"]
    for idx, item in enumerate(results, start=1):
        title = item.get("title", "Untitled").strip() or "Untitled"
        url = item.get("url", "").strip()
        snippet = item.get("content", "").strip()

        lines.append(f"- [{idx}] {title}")
        if snippet:
            lines.append(f"  {snippet[:300]}")
        if url:
            lines.append(f"  Source: {url}")

    return "\n".join(lines).strip()
