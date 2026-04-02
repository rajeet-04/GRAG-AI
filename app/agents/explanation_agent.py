"""
Explanation Agent node for LangGraph.

Generates human-readable explainable AI (xAI) output from graph traversal
paths and merged context. Produces a strict three-section Markdown format:
1. Natural Language Answer
2. Step-by-Step Reasoning Path (A -> B -> C)
3. Mermaid.js diagram of graph traversal

Uses cloud Ollama model (qwen2.5:14b-q4_k_m) for large context handling
without OOM crashes on 8GB VRAM systems.
"""

from __future__ import annotations

import re
from typing import Any

import structlog

logger = structlog.get_logger()


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

EXPLANATION_SYSTEM_PROMPT = """You are an explanation agent that generates human-readable \
answers from graph traversal paths. Your output MUST follow this THREE-SECTION format:

## Section 1: Natural Language Answer
Answer the user's question based on the retrieved context.

## Section 2: Step-by-Step Reasoning Path
Explain your reasoning as a chain: A → B → C → ...
Show how each piece of evidence leads to the next.

## Section 3: Mermaid Diagram
Generate a Mermaid.js graph showing the traversal path.
Format:
```mermaid
graph TD
    A[User Query] --> B[Entity A]
    B --> C[Relation Type]
    C --> D[Entity B]
```

CRITICAL: 
- Output ONLY these three sections, nothing else
- Use valid Mermaid.js syntax
- Reasoning must explicitly connect each step to the next with "→" arrows"""


EXPLANATION_USER_PROMPT = """User Query: {query}

Retrieved Context:
{context}

Graph Paths:
{paths}

Confidence Scores:
{confidence}

Generate your response in the three-section format."""


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


def extract_mermaid_block(text: str) -> str:
    """Extract the mermaid code block from a markdown section.

    Handles both fenced (```mermaid ... ```) and raw graph TD blocks.
    Returns the inner mermaid source, or empty string if not found.
    """
    # Try fenced code block first
    fenced = re.search(r"```mermaid\s*\n(.*?)```", text, re.DOTALL)
    if fenced:
        return fenced.group(1).strip()

    # Fallback: look for graph TD / LR / RL / TB directly
    graph_match = re.search(r"(graph\s+(?:TD|LR|RL|TB).*?)(?:\n\n|\Z)", text, re.DOTALL)
    if graph_match:
        return graph_match.group(1).strip()

    return ""


def parse_three_section_response(response: str) -> dict[str, Any]:
    """Parse LLM response into answer, reasoning_steps, and mermaid.

    Expects the strict three-section format produced by the system prompt.
    Falls back gracefully if sections are missing.

    Args:
        response: Raw LLM response text

    Returns:
        dict with keys: answer (str), reasoning_steps (list[str]), mermaid (str)
    """
    answer = ""
    reasoning: list[str] = []
    mermaid = ""

    # Split on ## headings
    sections = re.split(r"^## ", response, flags=re.MULTILINE)

    for section in sections:
        lower = section.lower()

        if "natural language answer" in lower:
            # Everything after the heading, before next section
            body = section.split("\n", 1)
            answer = body[1].strip() if len(body) > 1 else ""

        elif "step-by-step reasoning" in lower or "reasoning path" in lower:
            # Extract lines containing the → arrow
            reasoning = [
                line.strip().lstrip("-•").strip()
                for line in section.split("\n")
                if "→" in line or "->" in line
            ]
            # If no arrow lines, grab all non-heading lines as one step
            if not reasoning:
                lines = [
                    line.strip()
                    for line in section.split("\n")
                    if line.strip() and not line.strip().startswith("#")
                ]
                if lines:
                    reasoning = [f"Step: {lines[0]}"]

        elif "mermaid" in lower or "diagram" in lower:
            mermaid = extract_mermaid_block(section)

    return {"answer": answer, "reasoning_steps": reasoning, "mermaid": mermaid}


# ---------------------------------------------------------------------------
# LLM invocation
# ---------------------------------------------------------------------------


def _build_llm():
    """Use lightweight cloud client path for explanation generation.

    Returning None here forces explanation_agent_node to use the project's
    async OllamaClient(use_cloud=True), which is more memory efficient in
    constrained Docker environments.
    """
    return None


def _format_paths(paths: list[dict]) -> str:
    """Format graph paths for the prompt."""
    if not paths:
        return "No graph paths available."

    lines = []
    for i, path in enumerate(paths, 1):
        if isinstance(path, dict):
            src = path.get("source", "?")
            rel = path.get("relation_type", path.get("type", "RELATED"))
            tgt = path.get("target", "?")
            conf = path.get("confidence", "N/A")
            lines.append(f"  {i}. {src} --[{rel}]--> {tgt} (confidence: {conf})")
        else:
            lines.append(f"  {i}. {path}")
    return "\n".join(lines)


def _format_confidence(confidence: dict) -> str:
    """Format confidence scores for the prompt."""
    if not confidence:
        return "No confidence scores available."
    return "\n".join(f"  - {k}: {v}" for k, v in confidence.items())


def format_confidence_percent(confidence: float) -> str:
    """Format confidence as percentage string.

    Args:
        confidence: Float between 0.0 and 1.0

    Returns:
        Formatted string like "(Confidence: 87%)"
    """
    if confidence is None:
        return "(Confidence: N/A)"
    percentage = round(confidence * 100)
    return f"(Confidence: {percentage}%)"


def _annotate_reasoning_with_confidence(
    steps: list[str], paths: list[dict]
) -> list[str]:
    """Annotate reasoning steps with confidence percentages.

    Matches each step index to the corresponding path's confidence score
    and appends the formatted percentage string.

    Args:
        steps: Raw reasoning step strings from LLM
        paths: Original graph path dicts with confidence scores

    Returns:
        Annotated reasoning steps with "(Confidence: XX%)" appended
    """
    annotated = []
    for i, step in enumerate(steps):
        conf = None
        if i < len(paths) and isinstance(paths[i], dict):
            conf = paths[i].get("confidence")
        elif i < len(paths) and isinstance(paths[i], (int, float)):
            conf = paths[i]

        conf_str = format_confidence_percent(conf)
        annotated.append(f"{step} {conf_str}")

    return annotated


def select_top_n_paths(paths: list[dict], n: int = 10) -> tuple[list[dict], list[dict]]:
    """Select top-N paths by confidence score.

    Args:
        paths: List of path dicts with 'confidence' key
        n: Maximum number of paths to select (default 10, minimum 5)

    Returns:
        Tuple of (selected_paths, excluded_paths)
    """
    if not paths:
        return [], []

    # Sort by confidence descending
    sorted_paths = sorted(paths, key=lambda p: p.get("confidence", 0.0), reverse=True)

    # Enforce minimum 5, maximum 10
    n = max(5, min(n, 10))

    return sorted_paths[:n], sorted_paths[n:]


def _summarize_excluded_paths(excluded: list[dict]) -> str:
    """Generate a summary sentence for excluded lower-confidence paths.

    Args:
        excluded: List of path dicts that were not selected for reasoning

    Returns:
        A summary string describing excluded paths conceptually
    """
    if not excluded:
        return ""

    avg_conf = sum(p.get("confidence", 0.0) for p in excluded) / len(excluded)
    avg_pct = round(avg_conf * 100)
    return (
        f"*Additionally, {len(excluded)} lower-confidence paths "
        f"(average {avg_pct}% confidence) supported this conclusion "
        f"but are omitted from the detailed reasoning for brevity.*"
    )


def validate_mermaid(mermaid: str) -> bool:
    """Validate Mermaid syntax lightly.

    Checks:
    - Matching brackets for node definitions
    - Valid graph direction (TD, LR, RL, BT)
    - Arrow syntax (-->, ---, ==>, etc.)

    Args:
        mermaid: Mermaid diagram source

    Returns:
        True if valid, False otherwise
    """
    if not mermaid or not mermaid.strip():
        return False

    # Must start with graph direction
    if not re.match(r"^\s*(graph|flowchart)\s+(TD|LR|RL|BT)", mermaid, re.IGNORECASE):
        return False

    # Check balanced brackets in node definitions
    open_brackets = mermaid.count("[")
    close_brackets = mermaid.count("]")
    if open_brackets != close_brackets:
        return False

    # Check for valid arrows
    valid_arrows = re.findall(r"(-->|---|==>|===|-->|.-)", mermaid)
    if not valid_arrows:
        return False

    return True


# ---------------------------------------------------------------------------
# LangGraph node function
# ---------------------------------------------------------------------------


async def explanation_agent_node(state: dict[str, Any]) -> dict[str, Any]:
    """LangGraph node function for the Explanation Agent.

    Reads merged_context, kr_paths, and confidence_scores from state.
    Invokes the cloud LLM to produce three-section xAI output:
    answer, reasoning_steps, mermaid_path.

    Args:
        state: Current graph state dict

    Returns:
        Updated state dict with answer, reasoning_steps, mermaid_path,
        confidence_scores, and appended agent_trace.
    """
    user_query = state.get("user_query", "")
    context = state.get("merged_context", "")
    paths = state.get("kr_paths", [])
    confidence = state.get("confidence_scores", {})

    # Extract context truncation state
    context_truncated = state.get("context_truncated", False)
    truncation_warning = state.get("truncation_warning")

    logger.info(
        "explanation_agent.start",
        query_length=len(user_query),
        context_length=len(context),
        path_count=len(paths),
    )

    # Select top-N highest-confidence paths for reasoning
    selected_paths, excluded_paths = select_top_n_paths(paths)

    logger.info(
        "explanation_agent.path_selection",
        selected=len(selected_paths),
        excluded=len(excluded_paths),
    )

    # Build prompt
    user_msg = EXPLANATION_USER_PROMPT.format(
        query=user_query,
        context=context if context else "No context available.",
        paths=_format_paths(selected_paths),
        confidence=_format_confidence(confidence),
    )

    # Try LangChain ChatOllama first, fall back to direct OllamaClient
    llm = _build_llm()

    if llm is not None:
        logger.info("explanation_agent.using", backend="langchain_chatollama")
        response = llm.invoke(
            [
                {"role": "system", "content": EXPLANATION_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ]
        )
        raw_text = response.content if hasattr(response, "content") else str(response)
    else:
        from app.llm.ollama_client import OllamaClient

        # Try cloud first, fall back to local if cloud fails
        raw_text = ""
        try:
            logger.info("explanation_agent.using", backend="ollama_client_cloud")
            cloud_client = OllamaClient(use_cloud=True)
            result = await cloud_client.chat(
                messages=[
                    {"role": "system", "content": EXPLANATION_SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.3,
                max_tokens=4096,
            )
            raw_text = result.get("content", "")
            logger.info("explanation_agent.cloud_success", content_length=len(raw_text))
        except Exception as cloud_err:
            logger.warning(
                "explanation_agent.cloud_failed_using_local",
                error=str(cloud_err),
            )
            # Fall back to local Ollama (qwen3.5:9b)
            local_client = OllamaClient(use_cloud=False)
            result = await local_client.chat(
                messages=[
                    {"role": "system", "content": EXPLANATION_SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.3,
                max_tokens=2048,
            )
            raw_text = result.get("content", "")
            logger.info("explanation_agent.local_fallback_success", content_length=len(raw_text))

    # Parse three-section response
    parsed = parse_three_section_response(raw_text)
    answer = parsed["answer"]

    # Annotate reasoning steps with confidence percentages
    reasoning_steps = _annotate_reasoning_with_confidence(
        parsed["reasoning_steps"], selected_paths
    )

    # Add excluded paths summary to answer if any were dropped
    if excluded_paths:
        excluded_summary = _summarize_excluded_paths(excluded_paths)
        answer = answer + "\n\n" + excluded_summary

    # Add truncation disclaimer if applicable
    if context_truncated:
        disclaimer = "\n\n**Note:** This answer is based on incomplete context data as some information was truncated to fit the token budget. The reasoning below may not reflect all available knowledge."
        answer = answer + disclaimer

        logger.info(
            "explanation_agent.truncation_disclaimer",
            context_truncated=context_truncated,
            warning=truncation_warning,
        )

    # Validate Mermaid syntax — graceful degradation if invalid
    mermaid_raw = parsed["mermaid"]
    if validate_mermaid(mermaid_raw):
        mermaid_path = mermaid_raw
        logger.info("explanation_agent.mermaid_validated")
    else:
        mermaid_path = ""
        logger.warning(
            "explanation_agent.mermaid_invalid",
            reason="syntax validation failed, dropping diagram",
        )

    # Build citation map: "[n]" -> step index
    citation_map = {f"[{i + 1}]": i for i in range(len(reasoning_steps))}

    logger.info(
        "explanation_agent.complete",
        answer_length=len(answer),
        reasoning_steps=len(reasoning_steps),
        has_mermaid=bool(mermaid_path),
        citation_count=len(citation_map),
    )

    existing_trace = state.get("agent_trace", [])

    return {
        **state,
        "answer": answer,
        "reasoning_steps": reasoning_steps,
        "mermaid_path": mermaid_path,
        "confidence_scores": confidence,
        "citation_map": citation_map,
        "agent_trace": existing_trace + ["ExplanationAgent: generated xAI output"],
    }
