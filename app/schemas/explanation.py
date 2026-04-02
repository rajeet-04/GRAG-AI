"""xAI response schemas for the Explanation Agent.

Provides structured Pydantic models for parsing and validating
the Explanation Agent's output before it enters GraphState.
"""

from pydantic import BaseModel
from typing import Optional


class ReasoningStep(BaseModel):
    """Single reasoning step with confidence."""

    step: int
    content: str
    confidence: float


class ExplanationResponse(BaseModel):
    """Structured response from Explanation Agent."""

    answer: str
    reasoning_steps: list[ReasoningStep]
    mermaid_path: Optional[str] = None
    citation_map: dict[str, int] = {}
