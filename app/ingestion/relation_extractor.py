"""
Relationship extraction service for GRAG AI.

Extracts typed relationships between entities from text.
"""

import json
from typing import Any

import structlog

from app.llm.ollama_client import OllamaClient


logger = structlog.get_logger()

RELATION_TYPES = [
    "FOUNDED",
    "WORKS_FOR",
    "OWNS",
    "CREATED",
    "LOCATED_IN",
    "BASED_IN",
    "PART_OF",
    "USES",
    "DEPENDS_ON",
    "INTEGRATES_WITH",
    "RELATED_TO",
    "KNOWS",
    "MARRIED_TO",
    "LEADS",
]

EXTRACTION_PROMPT = """You are a relationship extraction system. Given entities extracted from text, identify the relationships between them.

Return a JSON list of relationships with the following structure:
[
  {"source": "EntityName1", "target": "EntityName2", "type": "RELATION_TYPE", "confidence": 0.0-1.0}
]

Relationship types to use:
- FOUNDED: Entity founded another entity
- WORKS_FOR: Person works for organization
- OWNS: Entity owns another entity
- CREATED: Entity created another entity
- LOCATED_IN: Entity is located in a place
- BASED_IN: Organization based in a location
- PART_OF: Entity is part of another entity
- USES: Entity uses another entity
- DEPENDS_ON: Entity depends on another
- INTEGRATES_WITH: Entity integrates with another
- RELATED_TO: Generic relationship
- KNOWS: Person knows another person
- LEADS: Person leads an organization

Only create relationships between entities that appear in the provided entity list.
"""

EXAMPLE_OUTPUT = """Example output for entities ["Elon Musk", "Tesla", "SpaceX"]:
[{"source": "Elon Musk", "target": "Tesla", "type": "FOUNDED", "confidence": 1.0}, {"source": "Elon Musk", "target": "SpaceX", "type": "FOUNDED", "confidence": 1.0}]
"""


class RelationExtractionService:
    """
    Service for extracting relationships between entities using LLM.

    Identifies typed relationships with source, target, type, and confidence.
    """

    def __init__(self) -> None:
        """Initialize the relationship extraction service."""
        self.ollama_client = OllamaClient()

    async def extract_relations(
        self,
        text: str,
        entities: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Extract relationships between entities from text.

        Args:
            text: Original input text
            entities: List of extracted entities from the text

        Returns:
            list: List of relationships with source, target, type, confidence
        """
        if not text or not text.strip():
            logger.warning("relation_extraction.empty_text")
            return []

        if not entities:
            logger.warning("relation_extraction.no_entities")
            return []

        try:
            logger.info(
                "relation_extraction.start",
                text_length=len(text),
                entity_count=len(entities),
            )

            entity_list = ", ".join([f'"{e["name"]}"' for e in entities])
            prompt = f"""{EXTRACTION_PROMPT}

Entities extracted: [{entity_list}]

Text: {text}

{EXAMPLE_OUTPUT}
"""

            response = await self.ollama_client.chat(
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert relationship extraction system. Return ONLY valid JSON, no explanation.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=1024,
            )

            content = response.get("content", "").strip()

            # Try to parse JSON from the response
            relations = self._parse_json_response(content)

            # Validate that relations are between known entities
            entity_names = {e["name"] for e in entities}
            valid_relations = []
            for rel in relations:
                if (
                    rel.get("source") in entity_names
                    and rel.get("target") in entity_names
                ):
                    # Normalize type to uppercase
                    rel["type"] = rel.get("type", "RELATED_TO").upper()
                    valid_relations.append(rel)
                else:
                    logger.warning(
                        "relation_extraction.invalid_entity",
                        source=rel.get("source"),
                        target=rel.get("target"),
                    )

            logger.info(
                "relation_extraction.complete",
                relation_count=len(valid_relations),
                relation_types=[r.get("type") for r in valid_relations],
            )

            return valid_relations

        except Exception as e:
            logger.error("relation_extraction.failed", error=str(e))
            return []

    def _parse_json_response(self, content: str) -> list[dict[str, Any]]:
        """
        Parse JSON from LLM response, handling various formats.

        Args:
            content: Raw response content

        Returns:
            list: Parsed relationship list
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

        logger.warning("relation_extraction.parse_failed", content=content[:200])
        return []


# Singleton instance
_relation_extraction_service: RelationExtractionService | None = None


def get_relation_extraction_service() -> RelationExtractionService:
    """Get singleton relationship extraction service instance."""
    global _relation_extraction_service
    if _relation_extraction_service is None:
        _relation_extraction_service = RelationExtractionService()
    return _relation_extraction_service
