"""
Entity extraction service for GRAG AI.

Uses LLM to extract structured entities from unstructured text.
"""

import asyncio
import json
import re
import uuid
from typing import Any, Dict, List, Optional

import structlog

from app.config import get_settings
from app.ingestion.chunking import split_text_into_chunks
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

EXTRACTION_PROMPT = """You are an entity extraction system. Extract all high-value entities from the given text.

Return a JSON list of entities with the following structure:
[
  {"name": "EntityName", "type": "Person|Organization|Concept|Location|Tool|Event|Product", "description": "Brief description", "confidence": 0.0-1.0}
]

Guidelines:
- Only extract entities explicitly grounded in the text
- Assign the most appropriate type from: Person, Organization, Concept, Location, Tool, Event, Product
- Include technologies, standards, products, projects, protocols, and important time events
- Normalize obvious aliases and formatting variants to a single canonical name
- Provide a brief description based on nearby context
- Assign confidence based on certainty (1.0 = very certain, 0.5 = uncertain)

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
        """Initialize the entity extraction service."""
        self.settings = get_settings()
        self.ollama_client = OllamaClient(use_cloud=self.settings.ollama_use_cloud)
        self._in_memory_store: Dict[str, Dict[str, Any]] = {}

    def _normalize_name(self, value: str) -> str:
        """Normalize an entity name for deduplication."""
        name = re.sub(r"\s+", " ", str(value or "")).strip(" \t\n\r,;:.()\"'")
        return name.strip()

    def _sanitize_entities(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Sanitize and validate raw entity payloads."""
        sanitized: List[Dict[str, Any]] = []

        for entity in entities:
            name = self._normalize_name(entity.get("name", ""))
            if len(name) < 2:
                continue

            # Keep 4-digit years as Event entities.
            if name.isdigit() and len(name) == 4:
                entity_type = "Event"
            else:
                entity_type = str(entity.get("type", "Concept")).strip().title()
                if entity_type not in ENTITY_TYPES:
                    entity_type = "Concept"

            try:
                confidence = float(entity.get("confidence", 0.5))
            except (TypeError, ValueError):
                confidence = 0.5
            confidence = max(0.0, min(1.0, confidence))

            description = str(entity.get("description", "")).strip()[:240]

            sanitized.append(
                {
                    "name": name,
                    "type": entity_type,
                    "description": description,
                    "confidence": confidence,
                }
            )

        return sanitized

    def _merge_entities(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge duplicate entities by normalized name with confidence-aware selection."""
        merged: Dict[str, Dict[str, Any]] = {}

        for entity in self._sanitize_entities(entities):
            key = entity["name"].lower()
            existing = merged.get(key)

            if existing is None:
                merged[key] = entity
                continue

            if entity["confidence"] > existing["confidence"]:
                merged[key] = entity
                continue

            if not existing.get("description") and entity.get("description"):
                existing["description"] = entity["description"]

            if existing.get("type") == "Concept" and entity.get("type") != "Concept":
                existing["type"] = entity["type"]

        return sorted(merged.values(), key=lambda x: x.get("confidence", 0.0), reverse=True)

    def _heuristic_entities(self, text: str) -> List[Dict[str, Any]]:
        """Lightweight heuristic fallback when LLM extraction is sparse."""
        if not text:
            return []

        candidates: set[str] = set()

        # Title Case entities
        for match in re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z0-9]+){0,3}\b", text):
            candidates.add(match.strip())

        # Acronyms and protocol-style tokens
        for match in re.findall(r"\b[A-Z]{2,}(?:[-_][A-Z0-9]{2,})*\b", text):
            candidates.add(match.strip())

        # Year events
        for match in re.findall(r"\b(19\d{2}|20\d{2})\b", text):
            candidates.add(match)

        heuristic_entities: List[Dict[str, Any]] = []
        for name in sorted(candidates)[:40]:
            heuristic_entities.append(
                {
                    "name": name,
                    "type": "Event" if name.isdigit() and len(name) == 4 else "Concept",
                    "description": "Heuristic extraction fallback",
                    "confidence": 0.45,
                }
            )

        return heuristic_entities

    async def _extract_entities_single_pass(
        self,
        text: str,
        *,
        chunk_index: int,
    ) -> List[Dict[str, Any]]:
        """Run one LLM extraction pass for a text segment."""
        if not text.strip():
            return []

        prompt = f"{EXTRACTION_PROMPT}\n{text}\n\n{EXAMPLE_OUTPUT}"

        max_tokens = 900 if len(text) > 1400 else 640
        response = await self.ollama_client.chat(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert entity extraction system. "
                        "Return ONLY valid JSON with no explanation."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=max_tokens,
            think=False,
        )

        content = response.get("content", "").strip()
        parsed = self._parse_json_response(content)

        logger.debug(
            "entity_extraction.chunk.complete",
            chunk_index=chunk_index,
            chunk_length=len(text),
            entity_count=len(parsed),
        )

        return parsed

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
            min_chunking_chars = max(int(self.settings.ingestion_min_chunking_chars), 1)
            should_chunk = len(text) >= min_chunking_chars

            if should_chunk:
                chunks = split_text_into_chunks(
                    text,
                    max_chars=max(int(self.settings.ingestion_chunk_size_chars), 800),
                    overlap_chars=max(int(self.settings.ingestion_chunk_overlap_chars), 0),
                )
            else:
                chunks = [text.strip()]

            logger.info(
                "entity_extraction.start",
                text_length=len(text),
                chunk_count=len(chunks),
                chunked=should_chunk,
            )

            semaphore = asyncio.Semaphore(max(int(self.settings.ingestion_chunk_parallelism), 1))

            async def run_chunk(idx: int, chunk_text: str) -> List[Dict[str, Any]]:
                async with semaphore:
                    return await self._extract_entities_single_pass(
                        chunk_text,
                        chunk_index=idx,
                    )

            tasks = [
                asyncio.create_task(run_chunk(idx, chunk_text))
                for idx, chunk_text in enumerate(chunks)
                if chunk_text.strip()
            ]

            extracted: List[Dict[str, Any]] = []
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for idx, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.warning(
                        "entity_extraction.chunk.failed",
                        chunk_index=idx,
                        error=str(result),
                    )
                    continue
                extracted.extend(result)

            if not extracted:
                # Recovery pass on a bounded prefix to avoid giving up on difficult PDFs.
                recovery_text = text[: max(int(self.settings.ingestion_chunk_size_chars) * 2, 4000)]
                extracted.extend(
                    await self._extract_entities_single_pass(
                        recovery_text,
                        chunk_index=-1,
                    )
                )

            entities = self._merge_entities(extracted)

            # If extraction is still sparse, enrich with heuristic candidates.
            if len(entities) < 3:
                entities = self._merge_entities(entities + self._heuristic_entities(text))

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
