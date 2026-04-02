"""
Context Builder Agent node for LangGraph.

Merges KR (Knowledge Graph) and KB (Episodic + Semantic) retrieval results
into a coherent context string, enforcing a strict token budget with priority-
based truncation. KR facts are NEVER truncated — only KB content is trimmed.

Priority order (highest to lowest):
1. KR Graph facts — never truncated
2. KB Episodic memories — drop oldest first
3. KB Semantic preferences — lowest priority, trimmed first

Requirements: AGNT-04, RETR-04, RETR-05
"""

from typing import Any

import structlog
import tiktoken

from app.retrieval.ranker import RankingConfig, rank_retrieval_results

logger = structlog.get_logger()

# Token budget constants (per CONTEXT.md Decision 7)
DEFAULT_TOKEN_BUDGET = 8192
WARNING_THRESHOLD = 6553  # 80% of 8192


class PriorityLevel:
    """Priority levels for context truncation."""

    KR_GRAPH_FACTS = 1  # NEVER truncate
    EPISODIC_MEMORIES = 2  # Drop oldest first
    SEMANTIC_PREFERENCES = 3  # Lowest priority


def _get_encoding(model: str = "gpt-4") -> tiktoken.Encoding:
    """
    Get tiktoken encoding for a model.

    Uses encoding_for_model to map model name to encoding.
    Falls back to cl100k_base if model is unknown.

    Args:
        model: Model name (e.g., "gpt-4", "gpt-3.5-turbo")

    Returns:
        tiktoken.Encoding instance
    """
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        logger.debug(
            "context_builder.unknown_model_fallback",
            model=model,
            fallback="cl100k_base",
        )
        return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str, model: str = "gpt-4") -> int:
    """
    Count tokens using tiktoken.

    Args:
        text: Text to count tokens for
        model: Model name for tokenizer selection (default: gpt-4)

    Returns:
        int: Number of tokens in the text
    """
    enc = _get_encoding(model)
    return len(enc.encode(text))


def _format_kr_section(
    kr_entities: list[dict[str, Any]],
    kr_relations: list[dict[str, Any]],
) -> str:
    """
    Format KR results into a context section.

    KR facts are highest priority and NEVER truncated.

    Args:
        kr_entities: List of entity dicts with 'name' and optional 'description'
        kr_relations: List of relation dicts with 'source', 'type', 'target'

    Returns:
        Formatted string section
    """
    parts: list[str] = ["## Knowledge Graph Results\n"]

    if kr_entities:
        for entity in kr_entities:
            name = entity.get("name", entity.get("content", "Unknown"))
            description = entity.get("description", "")
            if description:
                parts.append(f"- {name}: {description}")
            else:
                parts.append(f"- {name}")

    if kr_relations:
        for rel in kr_relations:
            source = rel.get("source", "?")
            rel_type = rel.get("type", rel.get("relation_type", "RELATED"))
            target = rel.get("target", "?")
            parts.append(f"- {source} --[{rel_type}]--> {target}")

    if len(parts) == 1:
        parts.append("- No graph results found")

    return "\n".join(parts)


def _format_episodic_section(
    episodic_memories: list[dict[str, Any]],
) -> str:
    """
    Format episodic memories into a context section.

    Episodic memories are sorted oldest-first so that when truncation
    occurs, the oldest memories are dropped first.

    Args:
        episodic_memories: List of episodic memory dicts

    Returns:
        Formatted string section
    """
    parts: list[str] = ["\n## Episodic Memories\n"]

    if not episodic_memories:
        parts.append("- No episodic memories available")
        return "\n".join(parts)

    # Sort by timestamp ascending (oldest first) for truncation priority
    sorted_memories = sorted(
        episodic_memories,
        key=lambda x: x.get("timestamp", ""),
    )

    for memory in sorted_memories:
        summary = memory.get("summary", "")
        if not summary:
            summary = memory.get("content", "")
        if summary:
            parts.append(f"- {summary}")

    if len(parts) == 1:
        parts.append("- No episodic memories available")

    return "\n".join(parts)


def _format_semantic_section(
    semantic_preferences: list[dict[str, Any]],
) -> str:
    """
    Format semantic preferences into a context section.

    Semantic preferences are the lowest priority and trimmed first.

    Args:
        semantic_preferences: List of semantic preference dicts

    Returns:
        Formatted string section
    """
    parts: list[str] = ["\n## User Preferences\n"]

    if not semantic_preferences:
        parts.append("- No user preferences available")
        return "\n".join(parts)

    for pref in semantic_preferences:
        content = pref.get("content", "")
        pref_type = pref.get("preference_type", "")
        confidence = pref.get("confidence", 0.0)

        if content:
            if pref_type:
                parts.append(
                    f"- [{pref_type}] {content} (confidence: {confidence:.0%})"
                )
            else:
                parts.append(f"- {content}")

    if len(parts) == 1:
        parts.append("- No user preferences available")

    return "\n".join(parts)


def merge_with_budget(
    kr_entities: list[dict[str, Any]],
    kr_relations: list[dict[str, Any]],
    episodic_memories: list[dict[str, Any]],
    semantic_preferences: list[dict[str, Any]],
    budget: int = DEFAULT_TOKEN_BUDGET,
) -> dict[str, Any]:
    """
    Merge context by priority, truncating lowest priority first.

    Priority order (highest to lowest):
    1. KR Graph facts — NEVER truncated
    2. KB Episodic memories — drop oldest first
    3. KB Semantic preferences — lowest priority

    When the budget is exceeded, semantic preferences are trimmed first,
    then episodic memories (oldest first). KR facts are always included
    in full, even if they alone exceed the budget (with a warning).

    Args:
        kr_entities: KR entity results from Neo4j
        kr_relations: KR relation results from Neo4j
        episodic_memories: KB episodic memory results from ChromaDB
        semantic_preferences: KB semantic preference results from ChromaDB
        budget: Maximum token budget (default: 8192)

    Returns:
        dict with keys:
            - context: Merged context string
            - token_count: Total tokens in merged context
            - truncated: List of sections that were truncated
            - budget: Token budget used
            - budget_pct: Percentage of budget used
    """
    truncated: list[str] = []

    # Step 1: Format KR section (NEVER truncated)
    kr_section = _format_kr_section(kr_entities, kr_relations)
    kr_tokens = count_tokens(kr_section)

    if kr_tokens >= budget:
        logger.warning(
            "context_builder.kr_exceeds_budget",
            kr_tokens=kr_tokens,
            budget=budget,
        )

    remaining_budget = max(0, budget - kr_tokens)

    # Step 2: Add episodic memories (drop oldest first)
    episodic_parts: list[str] = ["\n## Episodic Memories\n"]

    if episodic_memories:
        # Sort oldest first so truncation drops oldest
        sorted_episodic = sorted(
            episodic_memories,
            key=lambda x: x.get("timestamp", ""),
        )

        for memory in sorted_episodic:
            summary = memory.get("summary", "") or memory.get("content", "")
            if not summary:
                continue

            candidate_text = f"- {summary}"
            candidate_tokens = count_tokens(candidate_text)

            if candidate_tokens <= remaining_budget:
                episodic_parts.append(candidate_text)
                remaining_budget -= candidate_tokens
            else:
                truncated.append("episodic_memories")
                break

        if len(episodic_parts) == 1:
            episodic_parts = [
                "\n## Episodic Memories\n- No episodic memories available"
            ]
    else:
        episodic_parts.append("- No episodic memories available")

    # Step 3: Add semantic preferences (lowest priority)
    semantic_parts: list[str] = ["\n## User Preferences\n"]

    if semantic_preferences:
        for pref in semantic_preferences:
            content = pref.get("content", "")
            pref_type = pref.get("preference_type", "")
            confidence = pref.get("confidence", 0.0)

            if not content:
                continue

            if pref_type:
                candidate_text = (
                    f"- [{pref_type}] {content} (confidence: {confidence:.0%})"
                )
            else:
                candidate_text = f"- {content}"

            candidate_tokens = count_tokens(candidate_text)

            if candidate_tokens <= remaining_budget:
                semantic_parts.append(candidate_text)
                remaining_budget -= candidate_tokens
            else:
                truncated.append("semantic_preferences")
                break

        if len(semantic_parts) == 1:
            semantic_parts = ["\n## User Preferences\n- No user preferences available"]
    else:
        semantic_parts.append("- No user preferences available")

    # Build final merged context
    merged = kr_section + "\n".join(episodic_parts) + "\n".join(semantic_parts)
    total_tokens = count_tokens(merged)
    budget_pct = (total_tokens / budget * 100) if budget > 0 else 0

    # Log warning if near budget limit
    if total_tokens > WARNING_THRESHOLD:
        logger.warning(
            "context_builder.budget_warning",
            token_count=total_tokens,
            budget=budget,
            budget_pct=f"{budget_pct:.1f}%",
        )

    logger.info(
        "context_builder.merged",
        token_count=total_tokens,
        budget=budget,
        budget_pct=f"{budget_pct:.1f}%",
        kr_entities=len(kr_entities),
        kr_relations=len(kr_relations),
        episodic_count=len(episodic_memories),
        semantic_count=len(semantic_preferences),
        truncated=truncated,
    )

    return {
        "context": merged,
        "token_count": total_tokens,
        "truncated": truncated,
        "budget": budget,
        "budget_pct": round(budget_pct, 1),
    }


async def context_builder_node(state: dict[str, Any]) -> dict[str, Any]:
    """
    LangGraph node function for the Context Builder Agent.

    Reads KR and KB retrieval results from state, ranks them with
    confidence-weighted late fusion, then merges with strict
    priority-based token budget enforcement.

    Per CONTEXT.md Decision 4:
    - Normalize graph confidence and vector similarity to 0-1 scale
    - Graph gets 0.6 weight, vector gets 0.4 weight
    - Deduplicate by Entity ID, rank by unified confidence score
    - Clean arrays for downstream processing

    Per CONTEXT.md Decision 7:
    - KR facts: highest priority, NEVER truncated
    - Episodic memories: drop oldest first
    - Semantic preferences: lowest priority

    Per RETR-04: Token counting via tiktoken before LLM calls.
    Per RETR-05: 8192 token default budget with 80% warning.

    Args:
        state: Current graph state dict with kr_entities, kr_relations,
               episodic_memories, semantic_preferences

    Returns:
        Updated state dict with ranked_entities, ranked_relations,
        merged_context, token_count, and agent trace
    """
    kr_entities = state.get("kr_entities", [])
    kr_relations = state.get("kr_relations", [])
    episodic = state.get("episodic_memories", [])
    semantic = state.get("semantic_preferences", [])

    # Skip if no results to process
    if not kr_entities and not episodic and not semantic:
        existing_trace = state.get("agent_trace", [])
        return {
            **state,
            "ranked_entities": [],
            "ranked_relations": [],
            "merged_context": "",
            "token_count": 0,
            "agent_trace": existing_trace + ["ContextBuilder: No results to merge"],
        }

    # ── Rank results with late fusion ────────────────────────────────
    ranking_config = RankingConfig(
        graph_weight=0.6,  # From CONTEXT.md Decision 4
        vector_weight=0.4,
        max_results=50,
    )

    ranked_entities, ranked_relations = rank_retrieval_results(
        kr_entities=kr_entities,
        kr_relations=kr_relations,
        episodic_memories=episodic,
        semantic_preferences=semantic,
        config=ranking_config,
    )

    # Convert ScoredResult dicts to plain dicts for merge_with_budget
    # (which expects 'name' key for kr_entities format)
    ranked_kr_entities = []
    for entity in ranked_entities:
        ranked_kr_entities.append(
            {
                "id": entity["entity_id"],
                "name": entity.get("entity_name", ""),
                "type": entity.get("entity_type", ""),
                "confidence": entity["unified_score"],
                "source": entity["source"],
                "description": f"[{entity['source']}] score: {entity['unified_score']:.2f}",
            }
        )

    # ── Merge with token budget ──────────────────────────────────────
    merged = merge_with_budget(
        kr_entities=ranked_kr_entities,
        kr_relations=kr_relations,
        episodic_memories=episodic,
        semantic_preferences=semantic,
    )

    existing_trace = state.get("agent_trace", [])

    trace_msg = (
        f"ContextBuilder: ranked {len(ranked_entities)} entities "
        f"({len([e for e in ranked_entities if e['source'] == 'merged'])} merged), "
        f"merged with token budget "
        f"({merged['token_count']}/{merged['budget']} tokens, "
        f"{merged['budget_pct']}%)"
    )
    if merged["truncated"]:
        trace_msg += f" — truncated: {', '.join(merged['truncated'])}"

    return {
        **state,
        "ranked_entities": ranked_entities,
        "ranked_relations": ranked_relations,
        "merged_context": merged["context"],
        "token_count": merged["token_count"],
        "agent_trace": existing_trace + [trace_msg],
    }
