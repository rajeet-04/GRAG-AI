"""Query Agent — NL → Cypher translation with temporal filters.

Extracts search intent from natural language queries, generates Cypher
queries for Neo4j knowledge graph traversal, and applies temporal filters
based on query time context.

Uses configurable Ollama routing (local by default, cloud optional).
"""

import json
import asyncio
import re
from time import perf_counter
from datetime import datetime
from typing import Any

import structlog

from app.config import get_settings
from app.llm.ollama_client import OllamaClient

logger = structlog.get_logger()


INTENT_EXTRACTION_PROMPT = """Given the user query below, extract:
1. Search intent (what they're looking for — a clear, concise description)
2. Temporal filters if any (e.g., 'when I was at X', 'before 2024', 'last year')
3. Entity types to find (e.g., Person, Organization, Location, Concept)

User Query: {query}

Respond ONLY with valid JSON in this exact format — no markdown fences, no explanation:
{{"intent": "...", "temporal_filters": {{"valid_from": null, "valid_to": null, "temporal_hint": "..."}}, "entity_types": ["..."]}}

Rules:
- If no temporal context, set temporal_filters to {{"valid_from": null, "valid_to": null, "temporal_hint": null}}
- temporal_hint captures the raw temporal phrase from the query (e.g., "before 2024")
- entity_types should list entity TYPE NAMES (e.g., Person, Organization) not specific entities
"""

CYPHER_GENERATION_PROMPT = """Given the search intent and temporal filters, generate a Cypher query for a Neo4j knowledge graph.

The graph schema uses:
- Entity nodes with properties: id, name, type, description
- RELATES_TO relationships with properties: id, type, valid_from (datetime), valid_to (datetime), confidence (float), source

Search Intent: {intent}
Temporal Filters: {temporal_filters}
Entity Types: {entity_types}

Generate a Cypher query that:
1. Searches for entities matching the intent
2. If temporal filters exist, filter relations by temporal validity:
   - valid_from <= target_datetime
   - valid_to IS NULL OR valid_to >= target_datetime
3. Limits traversal to 2-4 hops maximum
4. Returns entities, relations, and paths for explanation

Constraints:
- Use MATCH patterns, never CREATE/MERGE/DELETE
- Always include LIMIT 50 to prevent runaway queries
- Return meaningful columns: entity names, relation types, paths

Respond with ONLY the Cypher query — no explanation, no markdown fences."""


async def _chat_with_backend(
    *,
    backend: str,
    client: OllamaClient,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
    think: bool,
    timeout_sec: float,
) -> dict[str, Any]:
    """Execute one chat call and capture timing/health metadata."""
    started = perf_counter()
    try:
        response = await asyncio.wait_for(
            client.chat(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                think=think,
            ),
            timeout=max(1.0, float(timeout_sec)),
        )
        content = (response.get("content") or "").strip()
        if not content:
            return {
                "backend": backend,
                "ok": False,
                "error": "empty_content",
                "duration_ms": round((perf_counter() - started) * 1000, 1),
            }
        return {
            "backend": backend,
            "ok": True,
            "response": response,
            "duration_ms": round((perf_counter() - started) * 1000, 1),
        }
    except asyncio.TimeoutError:
        return {
            "backend": backend,
            "ok": False,
            "error": "timeout",
            "duration_ms": round((perf_counter() - started) * 1000, 1),
        }
    except Exception as e:
        return {
            "backend": backend,
            "ok": False,
            "error": str(e),
            "duration_ms": round((perf_counter() - started) * 1000, 1),
        }


async def _query_llm_call(
    *,
    settings,
    stage: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
    think: bool,
    stage_timeout_sec: float,
) -> dict[str, Any]:
    """Route query-stage LLM calls with local/cloud serial or parallel behavior.

    Modes:
    - Local only: QUERY_USE_CLOUD=false
    - Serial: QUERY_USE_CLOUD=true, QUERY_PARALLEL_LLM=false (cloud primary, local fallback)
    - Parallel race: QUERY_USE_CLOUD=true, QUERY_PARALLEL_LLM=true
    """
    local_client = OllamaClient(use_cloud=False)
    cloud_enabled = bool(settings.query_use_cloud)
    can_use_cloud = cloud_enabled and bool(settings.ollama_cloud_api_key)

    if cloud_enabled and not can_use_cloud:
        logger.warning(
            "query_agent.cloud_unavailable",
            stage=stage,
            reason="missing OLLAMA_CLOUD_API_KEY",
        )

    # Local only mode
    if not can_use_cloud:
        result = await _chat_with_backend(
            backend="local",
            client=local_client,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            think=think,
            timeout_sec=stage_timeout_sec,
        )
        if not result["ok"]:
            raise RuntimeError(
                f"{stage}: local call failed ({result.get('error', 'unknown')})"
            )
        logger.info(
            "query_agent.llm_selected",
            stage=stage,
            mode="local_only",
            backend="local",
            duration_ms=result["duration_ms"],
        )
        return result["response"]

    # Cloud+Local parallel race mode
    if settings.query_parallel_llm:
        cloud_client = OllamaClient(use_cloud=True)
        timeout_s = max(
            min(float(settings.query_parallel_timeout_sec), float(stage_timeout_sec)),
            1.0,
        )

        logger.info(
            "query_agent.parallel_race.start",
            stage=stage,
            timeout_sec=timeout_s,
            local_model=local_client.model,
            cloud_model=cloud_client.model,
        )

        tasks = {
            asyncio.create_task(
                _chat_with_backend(
                    backend="local",
                    client=local_client,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    think=think,
                    timeout_sec=stage_timeout_sec,
                )
            ),
            asyncio.create_task(
                _chat_with_backend(
                    backend="cloud",
                    client=cloud_client,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    think=think,
                    timeout_sec=stage_timeout_sec,
                )
            ),
        }

        deadline = perf_counter() + timeout_s
        failures: list[dict[str, Any]] = []

        while tasks and perf_counter() < deadline:
            remaining = max(deadline - perf_counter(), 0.05)
            done, pending = await asyncio.wait(
                tasks,
                timeout=remaining,
                return_when=asyncio.FIRST_COMPLETED,
            )

            if not done:
                break

            for task in done:
                result = task.result()
                if result.get("ok") and result.get("response", {}).get("content", ""):
                    for p in pending:
                        p.cancel()
                    if pending:
                        await asyncio.gather(*pending, return_exceptions=True)

                    logger.info(
                        "query_agent.llm_selected",
                        stage=stage,
                        mode="parallel_race",
                        backend=result["backend"],
                        duration_ms=result["duration_ms"],
                    )
                    return result["response"]

                failures.append(result)
                logger.warning(
                    "query_agent.parallel_race.backend_failed",
                    stage=stage,
                    backend=result.get("backend"),
                    error=result.get("error"),
                    duration_ms=result.get("duration_ms"),
                )

            tasks = pending

        # Cleanup any remaining tasks
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        # Fallback to deterministic local retry if race returned no usable content
        local_retry = await _chat_with_backend(
            backend="local",
            client=local_client,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            think=think,
            timeout_sec=stage_timeout_sec,
        )
        if local_retry["ok"]:
            logger.info(
                "query_agent.llm_selected",
                stage=stage,
                mode="parallel_race_fallback_local",
                backend="local",
                duration_ms=local_retry["duration_ms"],
            )
            return local_retry["response"]

        raise RuntimeError(
            f"{stage}: parallel race failed; failures={len(failures) + 1}"
        )

    # Serial mode: cloud primary, local fallback
    cloud_client = OllamaClient(use_cloud=True)
    logger.info(
        "query_agent.serial.start",
        stage=stage,
        primary_backend="cloud",
        fallback_backend="local",
        cloud_model=cloud_client.model,
        local_model=local_client.model,
    )
    cloud_result = await _chat_with_backend(
        backend="cloud",
        client=cloud_client,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        think=think,
        timeout_sec=stage_timeout_sec,
    )
    if cloud_result["ok"] and cloud_result.get("response", {}).get("content", ""):
        logger.info(
            "query_agent.llm_selected",
            stage=stage,
            mode="serial_cloud_primary",
            backend="cloud",
            duration_ms=cloud_result["duration_ms"],
        )
        return cloud_result["response"]

    local_result = await _chat_with_backend(
        backend="local",
        client=local_client,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        think=think,
        timeout_sec=stage_timeout_sec,
    )
    if local_result["ok"]:
        logger.warning(
            "query_agent.serial_cloud_failed_local_used",
            stage=stage,
            cloud_error=cloud_result.get("error"),
            local_duration_ms=local_result["duration_ms"],
        )
        return local_result["response"]

    raise RuntimeError(
        f"{stage}: cloud and local failed (cloud={cloud_result.get('error')}, local={local_result.get('error')})"
    )


def _parse_json_response(response: str) -> dict[str, Any]:
    """Parse JSON from LLM response, handling common formatting issues.

    Strips markdown fences, leading/trailing text, and extracts the
    first valid JSON object.
    """
    text = response.strip()

    # Strip markdown code fences
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first line (```json or ```) and last line (```)
        text = (
            "\n".join(lines[1:-1])
            if lines[-1].strip() == "```"
            else "\n".join(lines[1:])
        )
        text = text.strip()

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Extract JSON object via regex
    match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    logger.warning("query_agent.json_parse_failed", response_preview=text[:200])
    return {}


def _extract_temporal_from_query(query: str) -> dict[str, Any]:
    """Extract temporal information from natural language query.

    Uses keyword-based detection for common temporal phrases.
    Returns dict with temporal filter information.
    """
    query_lower = query.lower()
    temporal: dict[str, Any] = {
        "valid_from": None,
        "valid_to": None,
        "temporal_hint": None,
    }

    # Year-based patterns
    year_match = re.search(r"(?:before|prior to|until)\s+(\d{4})", query_lower)
    if year_match:
        year = int(year_match.group(1))
        temporal["valid_to"] = f"{year}-01-01T00:00:00"
        temporal["temporal_hint"] = f"before {year}"
        return temporal

    year_match = re.search(r"(?:after|since|from)\s+(\d{4})", query_lower)
    if year_match:
        year = int(year_match.group(1))
        temporal["valid_from"] = f"{year}-01-01T00:00:00"
        temporal["temporal_hint"] = f"after {year}"
        return temporal

    year_range = re.search(r"(\d{4})\s*(?:to|-)\s*(\d{4})", query_lower)
    if year_range:
        temporal["valid_from"] = f"{year_range.group(1)}-01-01T00:00:00"
        temporal["valid_to"] = f"{year_range.group(2)}-12-31T23:59:59"
        temporal["temporal_hint"] = f"{year_range.group(1)} to {year_range.group(2)}"
        return temporal

    # Relative time patterns
    current_year = datetime.utcnow().year

    if "last year" in query_lower:
        temporal["valid_from"] = f"{current_year - 1}-01-01T00:00:00"
        temporal["valid_to"] = f"{current_year - 1}-12-31T23:59:59"
        temporal["temporal_hint"] = "last year"
        return temporal

    if "this year" in query_lower:
        temporal["valid_from"] = f"{current_year}-01-01T00:00:00"
        temporal["temporal_hint"] = "this year"
        return temporal

    if "recently" in query_lower or "recent" in query_lower:
        temporal["valid_from"] = f"{current_year}-01-01T00:00:00"
        temporal["temporal_hint"] = "recent"
        return temporal

    # "when I was at X" — captures location-based temporal context
    when_match = re.search(r"when\s+(?:I|i)\s+was\s+(?:at|in)\s+(\w+)", query_lower)
    if when_match:
        temporal["temporal_hint"] = f"when at {when_match.group(1)}"
        return temporal

    # "during" patterns
    during_match = re.search(r"during\s+(\w+)", query_lower)
    if during_match:
        temporal["temporal_hint"] = f"during {during_match.group(1)}"
        return temporal

    return temporal


def _build_safe_cypher(search_term: str) -> str:
    """Build a conservative Cypher query that is syntactically valid.

    Uses a simple case-insensitive name match with optional neighbor expansion.
    """
    cleaned = re.sub(r"[^A-Za-z0-9 _:-]", " ", search_term)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()[:80]

    if not cleaned:
        return (
            "MATCH (e:Entity) "
            "OPTIONAL MATCH (e)-[r:RELATES_TO]-(related:Entity) "
            "RETURN e, r, related LIMIT 50"
        )

    escaped = cleaned.replace("\\", "\\\\").replace("'", "\\'")
    return (
        "MATCH (e:Entity) "
        f"WHERE toLower(e.name) CONTAINS toLower('{escaped}') "
        "OPTIONAL MATCH (e)-[r:RELATES_TO]-(related:Entity) "
        "RETURN e, r, related LIMIT 50"
    )


async def _validate_cypher_with_explain(cypher: str) -> bool:
    """Validate Cypher syntax using Neo4j EXPLAIN without executing traversal."""
    try:
        from app.database.neo4j_client import get_neo4j_client

        client = get_neo4j_client()
        await client.execute(f"EXPLAIN {cypher}")
        return True
    except Exception as e:
        logger.warning(
            "query_agent.cypher_explain_failed",
            error=str(e),
            cypher_preview=cypher[:200],
        )
        return False


async def query_agent_node(state: dict[str, Any]) -> dict[str, Any]:
    """Query Agent LangGraph node: NL → intent + Cypher + temporal filters.

    Steps:
    1. Extract search intent from user query using local Ollama
    2. Generate Cypher query from extracted intent
    3. Apply temporal filters from query context

    Args:
        state: GraphState dict containing at minimum 'user_query'

    Returns:
        Partial GraphState update with search_intent, cypher_query,
        temporal_filters, and updated agent_trace
    """
    user_query = state.get("user_query", "")
    if not user_query:
        logger.warning("query_agent.empty_query")
        return {
            "search_intent": "",
            "cypher_query": "",
            "temporal_filters": {},
            "agent_trace": state.get("agent_trace", [])
            + ["QueryAgent: ERROR — empty user query"],
        }

    logger.info("query_agent.starting", query=user_query[:200])
    settings = get_settings()
    logger.info(
        "query_agent.routing",
        mode=(
            "parallel_race"
            if settings.query_use_cloud and settings.query_parallel_llm
            else "serial_cloud_primary"
            if settings.query_use_cloud
            else "local_only"
        ),
        query_use_cloud=settings.query_use_cloud,
        query_parallel_llm=settings.query_parallel_llm,
        intent_timeout_sec=settings.query_intent_timeout_sec,
        cypher_timeout_sec=settings.query_cypher_timeout_sec,
    )

    try:
        # Step 1: Extract intent
        intent_prompt = INTENT_EXTRACTION_PROMPT.format(query=user_query)
        intent_response = await _query_llm_call(
            settings=settings,
            stage="intent",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a query analysis assistant. "
                        "Extract structured intent from natural language queries. "
                        "Always respond with valid JSON only."
                    ),
                },
                {"role": "user", "content": intent_prompt},
            ],
            temperature=0.1,
            max_tokens=settings.query_intent_max_tokens,
            think=False,  # JSON output — disable thinking trace
            stage_timeout_sec=settings.query_intent_timeout_sec,
        )

        intent_content = intent_response.get("content", "")
        parsed = _parse_json_response(intent_content)

        if not parsed:
            logger.warning(
                "query_agent.intent_parse_failed", raw_response=intent_content[:300]
            )
            # Fallback: use the raw query as intent
            parsed = {
                "intent": user_query,
                "temporal_filters": {},
                "entity_types": [],
            }

        search_intent = parsed.get("intent", user_query)
        entity_types = parsed.get("entity_types", [])
        llm_temporal = parsed.get("temporal_filters", {})

        logger.info(
            "query_agent.intent_extracted",
            intent=search_intent[:200],
            entity_types=entity_types,
        )

        # Step 2: Generate Cypher from intent
        cypher_prompt = CYPHER_GENERATION_PROMPT.format(
            intent=search_intent,
            temporal_filters=json.dumps(llm_temporal, default=str),
            entity_types=json.dumps(entity_types),
        )
        cypher_response = await _query_llm_call(
            settings=settings,
            stage="cypher",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a Cypher query generator for a Neo4j knowledge graph. "
                        "Generate only valid Cypher queries — no explanation, no markdown."
                    ),
                },
                {"role": "user", "content": cypher_prompt},
            ],
            temperature=0.1,
            max_tokens=settings.query_cypher_max_tokens,
            think=False,  # Cypher output — disable thinking trace
            stage_timeout_sec=settings.query_cypher_timeout_sec,
        )

        cypher_content = cypher_response.get("content", "").strip()

        # Clean Cypher — strip markdown fences if LLM wraps them
        if cypher_content.startswith("```"):
            lines = cypher_content.split("\n")
            cypher_content = "\n".join(
                lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
            )
            cypher_content = cypher_content.strip()

        fallback_cypher = _build_safe_cypher(search_intent or user_query)

        # Basic Cypher validation — must contain MATCH
        if "MATCH" not in cypher_content.upper():
            logger.warning(
                "query_agent.cypher_invalid", cypher_preview=cypher_content[:200]
            )
            cypher_content = fallback_cypher

        # Strong validation using Neo4j parser; fallback if invalid.
        if not await _validate_cypher_with_explain(cypher_content):
            logger.warning(
                "query_agent.cypher_fallback_applied",
                original_preview=cypher_content[:200],
            )
            cypher_content = fallback_cypher

        logger.info(
            "query_agent.cypher_generated",
            cypher_preview=cypher_content[:200],
        )

        # Step 3: Extract temporal filters from query (keyword-based fallback)
        temporal_filters = _extract_temporal_from_query(user_query)

        # Merge LLM-extracted temporal with keyword-extracted temporal
        # LLM hints take priority, keyword extraction fills gaps
        if llm_temporal.get("valid_from") and not temporal_filters.get("valid_from"):
            temporal_filters["valid_from"] = llm_temporal["valid_from"]
        if llm_temporal.get("valid_to") and not temporal_filters.get("valid_to"):
            temporal_filters["valid_to"] = llm_temporal["valid_to"]
        if llm_temporal.get("temporal_hint") and not temporal_filters.get(
            "temporal_hint"
        ):
            temporal_filters["temporal_hint"] = llm_temporal["temporal_hint"]

        # Build agent trace
        trace = state.get("agent_trace", [])
        trace.append(
            f"QueryAgent: extracted intent='{search_intent[:80]}', "
            f"cypher={len(cypher_content)}chars, "
            f"temporal={temporal_filters.get('temporal_hint', 'none')}"
        )

        return {
            "search_intent": search_intent,
            "cypher_query": cypher_content,
            "temporal_filters": temporal_filters,
            "agent_trace": trace,
        }

    except Exception as e:
        logger.error("query_agent.error", error=str(e), query=user_query[:200])
        errors = state.get("errors", [])
        errors.append(
            {
                "agent": "query_agent",
                "error": str(e),
                "query": user_query[:200],
                "timestamp": datetime.utcnow().isoformat(),
            }
        )
        trace = state.get("agent_trace", [])
        trace.append(f"QueryAgent: ERROR — {str(e)[:100]}")

        return {
            "search_intent": user_query,
            "cypher_query": "",
            "temporal_filters": {},
            "errors": errors,
            "agent_trace": trace,
        }
