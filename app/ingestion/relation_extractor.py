"""
Relationship extraction service for GRAG AI.

Extracts typed relationships between entities from text.
"""

import asyncio
import json
import re
from typing import Any, Dict, List, Optional

import structlog

from app.config import get_settings
from app.ingestion.chunking import split_text_into_chunks
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
        self.settings = get_settings()
        use_cloud_for_ingestion = bool(
            getattr(self.settings, "ingestion_use_cloud", False)
        )
        self.ollama_client = OllamaClient(use_cloud=use_cloud_for_ingestion)

    def _normalize_name(self, value: str) -> str:
        """Normalize an entity name for relation matching."""
        return re.sub(r"\s+", " ", str(value or "")).strip(" \t\n\r,;:.()\"'")

    def _resolve_entity_name(
        self,
        value: str,
        entity_lookup: Dict[str, str],
    ) -> str | None:
        """Resolve a possibly-variant entity name to a canonical known entity."""
        normalized = self._normalize_name(value).lower()
        if not normalized:
            return None

        if normalized in entity_lookup:
            return entity_lookup[normalized]

        # Fallback: substring match for minor formatting differences.
        for key, canonical in entity_lookup.items():
            if normalized in key or key in normalized:
                return canonical

        return None

    def _build_entity_lookup(self, entities: List[Dict[str, Any]]) -> Dict[str, str]:
        """Build lowercase -> canonical entity name lookup."""
        lookup: Dict[str, str] = {}
        for entity in entities:
            name = self._normalize_name(entity.get("name", ""))
            if name:
                lookup[name.lower()] = name
        return lookup

    def _merge_relations(self, relations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Deduplicate relations and keep highest-confidence version."""
        merged: Dict[tuple[str, str, str], Dict[str, Any]] = {}

        for rel in relations:
            source = self._normalize_name(rel.get("source", ""))
            target = self._normalize_name(rel.get("target", ""))
            relation_type = str(rel.get("type", "RELATED_TO")).upper().strip() or "RELATED_TO"

            if relation_type not in RELATION_TYPES:
                relation_type = "RELATED_TO"

            if not source or not target or source == target:
                continue

            try:
                confidence = float(rel.get("confidence", 0.5))
            except (TypeError, ValueError):
                confidence = 0.5
            confidence = max(0.0, min(1.0, confidence))

            key = (source.lower(), target.lower(), relation_type)
            existing = merged.get(key)
            if existing is None or confidence > existing.get("confidence", 0.0):
                merged[key] = {
                    "source": source,
                    "target": target,
                    "type": relation_type,
                    "confidence": confidence,
                }

        return sorted(merged.values(), key=lambda x: x.get("confidence", 0.0), reverse=True)

    def _entities_for_chunk(
        self,
        chunk_text: str,
        entities: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Select entities explicitly mentioned in a chunk."""
        lowered = chunk_text.lower()
        selected: List[Dict[str, Any]] = []

        for entity in entities:
            name = self._normalize_name(entity.get("name", ""))
            if name and name.lower() in lowered:
                selected.append(entity)

        return selected

    def _heuristic_relations(
        self,
        text: str,
        entities: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Keyword-based relation fallback for recall improvement."""
        entity_names = [self._normalize_name(e.get("name", "")) for e in entities]
        entity_names = [name for name in entity_names if name]
        if len(entity_names) < 2:
            return []

        sorted_names = sorted(entity_names, key=len, reverse=True)
        heuristic: List[Dict[str, Any]] = []
        sentences = re.split(r"(?<=[.!?])\s+", text)

        for sentence in sentences:
            sentence_lower = sentence.lower()
            mentions = []
            for name in sorted_names:
                idx = sentence_lower.find(name.lower())
                if idx >= 0:
                    mentions.append((idx, name))

            if len(mentions) < 2:
                continue

            mentions.sort(key=lambda x: x[0])
            source = mentions[0][1]
            target = mentions[1][1]

            relation_type = "RELATED_TO"
            if any(keyword in sentence_lower for keyword in ("founded", "co-founded", "founder")):
                relation_type = "FOUNDED"
            elif any(keyword in sentence_lower for keyword in ("works for", "worked for", "employee of")):
                relation_type = "WORKS_FOR"
            elif any(keyword in sentence_lower for keyword in ("leads", "lead", "ceo of", "heads")):
                relation_type = "LEADS"
            elif any(keyword in sentence_lower for keyword in ("created", "built", "developed", "invented")):
                relation_type = "CREATED"
            elif any(keyword in sentence_lower for keyword in ("located in", "based in", "headquartered in")):
                relation_type = "LOCATED_IN"
            elif any(keyword in sentence_lower for keyword in ("uses", "using", "utilizes")):
                relation_type = "USES"

            heuristic.append(
                {
                    "source": source,
                    "target": target,
                    "type": relation_type,
                    "confidence": 0.55,
                }
            )

        return heuristic

    async def _extract_relations_single_pass(
        self,
        *,
        text: str,
        entities: List[Dict[str, Any]],
        phase: str,
    ) -> List[Dict[str, Any]]:
        """Run one LLM relation extraction pass."""
        if not text.strip() or len(entities) < 2:
            return []

        entity_lookup = self._build_entity_lookup(entities)
        entity_list = ", ".join([f'"{entity_lookup[key]}"' for key in entity_lookup])
        prompt = f"""{EXTRACTION_PROMPT}

Entities extracted: [{entity_list}]

Text: {text}

{EXAMPLE_OUTPUT}
"""

        max_tokens = 900 if len(text) > 1500 else 640
        response = await self.ollama_client.chat(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert relationship extraction system. "
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

        validated: List[Dict[str, Any]] = []
        for rel in parsed:
            source = self._resolve_entity_name(rel.get("source", ""), entity_lookup)
            target = self._resolve_entity_name(rel.get("target", ""), entity_lookup)

            if not source or not target:
                continue

            relation_type = str(rel.get("type", "RELATED_TO")).upper().strip() or "RELATED_TO"
            if relation_type not in RELATION_TYPES:
                relation_type = "RELATED_TO"

            try:
                confidence = float(rel.get("confidence", 0.5))
            except (TypeError, ValueError):
                confidence = 0.5

            validated.append(
                {
                    "source": source,
                    "target": target,
                    "type": relation_type,
                    "confidence": max(0.0, min(1.0, confidence)),
                }
            )

        logger.debug(
            "relation_extraction.pass.complete",
            phase=phase,
            relation_count=len(validated),
            entity_count=len(entities),
            text_length=len(text),
        )

        return validated

    async def extract_relations(
        self,
        text: str,
        entities: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
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
            min_chunking_chars = max(int(self.settings.ingestion_min_chunking_chars), 1)
            should_chunk = len(text) >= min_chunking_chars

            if should_chunk:
                chunks = split_text_into_chunks(
                    text,
                    max_chars=max(int(self.settings.ingestion_chunk_size_chars), 900),
                    overlap_chars=max(int(self.settings.ingestion_chunk_overlap_chars), 0),
                )
            else:
                chunks = [text.strip()]

            logger.info(
                "relation_extraction.start",
                text_length=len(text),
                entity_count=len(entities),
                chunk_count=len(chunks),
                chunked=should_chunk,
            )

            semaphore = asyncio.Semaphore(max(int(self.settings.ingestion_chunk_parallelism), 1))

            async def run_chunk(idx: int, chunk_text: str) -> List[Dict[str, Any]]:
                chunk_entities = self._entities_for_chunk(chunk_text, entities)
                if len(chunk_entities) < 2:
                    return []
                async with semaphore:
                    return await self._extract_relations_single_pass(
                        text=chunk_text,
                        entities=chunk_entities,
                        phase=f"chunk_{idx}",
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
                        "relation_extraction.chunk.failed",
                        chunk_index=idx,
                        error=str(result),
                    )
                    continue
                extracted.extend(result)

            # Secondary global pass for cross-chunk relationships.
            if should_chunk and bool(self.settings.ingestion_relation_global_pass):
                global_entities = entities[:80]
                if len(global_entities) >= 2:
                    global_context = "\n\n".join(chunks[:2] + chunks[-1:])
                    extracted.extend(
                        await self._extract_relations_single_pass(
                            text=global_context,
                            entities=global_entities,
                            phase="global",
                        )
                    )

            extracted.extend(self._heuristic_relations(text, entities))
            valid_relations = self._merge_relations(extracted)

            logger.info(
                "relation_extraction.complete",
                relation_count=len(valid_relations),
                relation_types=[r.get("type") for r in valid_relations],
            )

            return valid_relations

        except Exception as e:
            logger.error("relation_extraction.failed", error=str(e))
            return self._merge_relations(self._heuristic_relations(text, entities))

    def _parse_json_response(self, content: str) -> List[Dict[str, Any]]:
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
