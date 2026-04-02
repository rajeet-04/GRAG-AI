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

import os
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
    """Create a ChatOllama instance pointing at the cloud/base URL.

    Uses the OLLAMA_BASE_URL env var (defaults to localhost). The plan
    specifies cloud Ollama for large-context handling.
    """
    try:
        from langchain_ollama import ChatOllama
    except ImportError:
        # Fallback: return None — caller should use ollama_client directly
        return None

    base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    # Use a model with larger context window for explanation tasks
    model = os.environ.get("EXPLANATION_MODEL", "qwen2.5:14b-q4_k_m")

    return ChatOllama(model=model, base_url=base_url)


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

    # Build prompt
    user_msg = EXPLANATION_USER_PROMPT.format(
        query=user_query,
        context=context if context else "No context available.",
        paths=_format_paths(paths),
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
        # Fallback: use project's async OllamaClient
        logger.info("explanation_agent.using", backend="ollama_client")
        from app.llm.ollama_client import get_ollama_client

        client = get_ollama_client()
        result = await client.chat(
            messages=[
                {"role": "system", "content": EXPLANATION_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.3,
            max_tokens=4096,
        )
        raw_text = result.get("content", "")

    # Parse three-section response
    parsed = parse_three_section_response(raw_text)
    answer = parsed["answer"]

    # Add truncation disclaimer if applicable
    if context_truncated:
        disclaimer = "\n\n**Note:** This answer is based on incomplete context data as some information was truncated to fit the token budget. The reasoning below may not reflect all available knowledge."
        answer = answer + disclaimer

        logger.info(
            "explanation_agent.truncation_disclaimer",
            context_truncated=context_truncated,
            warning=truncation_warning,
        )

    logger.info(
        "explanation_agent.complete",
        answer_length=len(parsed["answer"]),
        reasoning_steps=len(parsed["reasoning_steps"]),
        has_mermaid=bool(parsed["mermaid"]),
    )

    existing_trace = state.get("agent_trace", [])

    return {
        **state,
        "answer": answer,
        "reasoning_steps": parsed["reasoning_steps"],
        "mermaid_path": parsed["mermaid"],
        "confidence_scores": confidence,
        "agent_trace": existing_trace + ["ExplanationAgent: generated xAI output"],
    }
