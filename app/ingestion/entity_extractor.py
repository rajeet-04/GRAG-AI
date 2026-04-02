"""
Entity extraction service for GRAG AI.

Uses LLM to extract structured entities from unstructured text.
"""

import json
import uuid
from typing import Any, Dict, List, Optional

import structlog

from app.llm.ollama_client import OllamaClient


logger = structlog.get_logger()

ENTITY_TYPES = [
    "Person",
    "Organization",
    "Concept",
    "Location",
    "Tool",
    "Event",
    "Product",
]

EXTRACTION_PROMPT = """You are an entity extraction system. Extract all named entities from the given text.

Return a JSON list of entities with the following structure:
[
  {"name": "EntityName", "type": "Person|Organization|Concept|Location|Tool|Event|Product", "description": "Brief description", "confidence": 0.0-1.0}
]

Guidelines:
- Only extract entities that are explicitly mentioned in the text
- Assign the most appropriate type from: Person, Organization, Concept, Location, Tool, Event, Product
- Provide a brief description based on context
- Assign confidence based on how certain the extraction is (1.0 = very certain, 0.5 = uncertain)

Text to process:
"""

EXAMPLE_OUTPUT = """Example output for "Elon Musk founded Tesla in 2003.":
[{"name": "Elon Musk", "type": "Person", "description": "CEO and founder of SpaceX and Tesla", "confidence": 1.0}, {"name": "Tesla", "type": "Organization", "description": "Electric vehicle and clean energy company", "confidence": 1.0}, {"name": "2003", "type": "Event", "description": "Year Tesla was founded", "confidence": 0.9}]
"""


class EntityExtractionService:
    """
    Service for extracting entities from text using LLM.

    Extracts named entities with type, description, and confidence.
    """

    def __init__(self) -> None:
        """Initialize the entity extraction service using Ollama cloud."""
        self.ollama_client = OllamaClient(use_cloud=True)
        self._in_memory_store: Dict[str, Dict[str, Any]] = {}

    async def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract entities from the given text.

        Args:
            text: Input text to extract entities from

        Returns:
            list: List of extracted entities with id, name, type, description, confidence
        """
        if not text or not text.strip():
            logger.warning("entity_extraction.empty_text")
            return []

        try:
            logger.info("entity_extraction.start", text_length=len(text))

            prompt = f"{EXTRACTION_PROMPT}\n{text}\n\n{EXAMPLE_OUTPUT}"

            response = await self.ollama_client.chat(
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert entity extraction system. Return ONLY valid JSON, no explanation.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=1024,
                think=False,  # Disable thinking for structured JSON output
            )

            content = response.get("content", "").strip()

            # Try to parse JSON from the response
            entities = self._parse_json_response(content)

            # Add UUIDs to each entity
            for entity in entities:
                entity["id"] = str(uuid.uuid4())

            logger.info(
                "entity_extraction.complete",
                entity_count=len(entities),
                entity_names=[e.get("name") for e in entities],
            )

            return entities

        except Exception as e:
            logger.error("entity_extraction.failed", error=str(e))
            return []

    def _parse_json_response(self, content: str) -> List[Dict[str, Any]]:
        """
        Parse JSON from LLM response, handling various formats.

        Args:
            content: Raw response content

        Returns:
            list: Parsed entity list
        """
        # Try direct parse first
        try:
            data = json.loads(content)
            if isinstance(data, list):
                return data
            return []
        except json.JSONDecodeError:
            pass

        # Try to find JSON in markdown code block
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]

        try:
            data = json.loads(content.strip())
            if isinstance(data, list):
                return data
            return []
        except json.JSONDecodeError:
            pass

        # Try to extract array from text
        import re

        match = re.search(r"\[.*\]", content, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
                if isinstance(data, list):
                    return data
            except json.JSONDecodeError:
                pass

        logger.warning("entity_extraction.parse_failed", content=content[:200])
        return []

    def store_result(self, document_id: str, result: Dict[str, Any]) -> None:
        """Store extraction result in memory for retrieval."""
        self._in_memory_store[document_id] = result

    def get_result(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve stored extraction result by document ID."""
        return self._in_memory_store.get(document_id)


# Singleton instance
_entity_extraction_service: Optional[EntityExtractionService] = None


def get_entity_extraction_service() -> EntityExtractionService:
    """Get singleton entity extraction service instance."""
    global _entity_extraction_service
    if _entity_extraction_service is None:
        _entity_extraction_service = EntityExtractionService()
    return _entity_extraction_service
