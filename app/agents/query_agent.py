"""Query Agent — NL → Cypher translation with temporal filters.

Extracts search intent from natural language queries, generates Cypher
queries for Neo4j knowledge graph traversal, and applies temporal filters
based on query time context.

Uses local Ollama (qwen2.5:7b-q4_k_m) for fast intent extraction
per CONTEXT.md Decision 4: Hybrid Local/Cloud pattern.
"""

import json
import re
from datetime import datetime
from typing import Any

import structlog

from app.llm.ollama_client import get_ollama_client

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
    ollama = get_ollama_client()

    try:
        # Step 1: Extract intent using local Ollama
        intent_prompt = INTENT_EXTRACTION_PROMPT.format(query=user_query)
        intent_response = await ollama.chat(
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
            max_tokens=512,
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
        cypher_response = await ollama.chat(
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
            max_tokens=1024,
        )

        cypher_content = cypher_response.get("content", "").strip()

        # Clean Cypher — strip markdown fences if LLM wraps them
        if cypher_content.startswith("```"):
            lines = cypher_content.split("\n")
            cypher_content = "\n".join(
                lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
            )
            cypher_content = cypher_content.strip()

        # Basic Cypher validation — must contain MATCH
        if "MATCH" not in cypher_content.upper():
            logger.warning(
                "query_agent.cypher_invalid", cypher_preview=cypher_content[:200]
            )
            # Generate a safe fallback query
            cypher_content = (
                "MATCH (e:Entity) "
                "WHERE e.name CONTAINS $search_term "
                "OPTIONAL MATCH (e)-[r:RELATES_TO]-(related) "
                "RETURN e, r, related LIMIT 50"
            )

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
