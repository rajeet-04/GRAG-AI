"""
Temporal metadata utilities for GRAG AI ingestion pipeline.

Provides helpers for adding temporal properties to entities and relations.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


class TemporalIngestionMixin:
    """
    Mixin class providing temporal metadata handling for ingestion operations.

    Methods:
        prepare_temporal_entity: Add temporal properties to an entity dict
        prepare_temporal_relation: Add temporal properties to a relation dict
        validate_temporal_input: Validate temporal input parameters
        get_temporal_query_context: Generate context for temporal queries
        apply_temporal_defaults: Apply defaults to all entities/relations
        validate_ingestion_temporal: Validate temporal metadata
    """

    @staticmethod
    def prepare_temporal_entity(
        entity: Dict[str, Any],
        valid_from: Optional[datetime] = None,
        valid_to: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Add temporal properties to an entity dictionary.

        Args:
            entity: Entity dictionary to add temporal properties to
            valid_from: When entity becomes valid (default: now)
            valid_to: When entity expires (None = active, default)

        Returns:
            dict: Entity with added temporal properties
        """
        now = datetime.utcnow()

        temporal_entity = entity.copy()
        temporal_entity["valid_from"] = (valid_from or now).isoformat()
        temporal_entity["valid_to"] = valid_to.isoformat() if valid_to else None
        temporal_entity["version"] = "v1"

        return temporal_entity

    @staticmethod
    def prepare_temporal_relation(
        relation: Dict[str, Any],
        valid_from: Optional[datetime] = None,
        valid_to: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Add temporal properties to a relation dictionary.

        Handles confidence propagation from the relation to temporal metadata.

        Args:
            relation: Relation dictionary to add temporal properties to
            valid_from: When relation becomes valid (default: now)
            valid_to: When relation expires (None = active, default)

        Returns:
            dict: Relation with added temporal properties
        """
        now = datetime.utcnow()

        temporal_relation = relation.copy()
        temporal_relation["valid_from"] = (valid_from or now).isoformat()
        temporal_relation["valid_to"] = valid_to.isoformat() if valid_to else None
        temporal_relation["version"] = "v1"

        # Ensure confidence is present (propagate from relation if not already set)
        if "confidence" not in temporal_relation:
            temporal_relation["confidence"] = 1.0

        return temporal_relation

    @staticmethod
    def validate_temporal_input(
        valid_from: Optional[datetime],
        valid_to: Optional[datetime],
    ) -> bool:
        """
        Validate temporal input parameters.

        Checks:
        - If valid_to is provided, valid_from must be <= valid_to

        Args:
            valid_from: Start of validity period
            valid_to: End of validity period

        Returns:
            bool: True if valid, False if validation fails
        """
        if valid_to is None:
            return True

        if valid_from is None:
            valid_from = datetime.utcnow()

        return valid_from <= valid_to

    @staticmethod
    def get_temporal_query_context(
        as_of: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Generate query context for temporal-aware retrieval.

        Useful for generating query parameters for Neo4j temporal queries.

        Args:
            as_of: Point in time to query (default: now)

        Returns:
            dict: Query context with temporal parameters
        """
        as_of = as_of or datetime.utcnow()

        return {
            "as_of": as_of.isoformat(),
            "as_of_formatted": as_of.strftime("%Y-%m-%d %H:%M:%S"),
            "is_current": True,  # Current query means active relations (valid_to is None)
        }

    @staticmethod
    def apply_temporal_defaults(
        entities: List[Dict[str, Any]],
        relations: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Apply default temporal properties to all entities and relations.

        Applies:
        - valid_from: current UTC time
        - valid_to: None (active)
        - version: "v1"

        Args:
            entities: List of entity dictionaries
            relations: List of relation dictionaries

        Returns:
            tuple: (processed_entities, processed_relations)
        """
        now = datetime.utcnow()

        processed_entities = [
            TemporalIngestionMixin.prepare_temporal_entity(e, now, None)
            for e in entities
        ]

        processed_relations = [
            TemporalIngestionMixin.prepare_temporal_relation(r, now, None)
            for r in relations
        ]

        return processed_entities, processed_relations

    @staticmethod
    def validate_ingestion_temporal(
        entities: List[Dict[str, Any]],
        relations: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Validate all temporal metadata in entities and relations.

        Returns a list of validation warnings (non-blocking).

        Checks:
        - Temporal ordering (valid_from <= valid_to)
        - Confidence values in valid range

        Args:
            entities: List of entity dictionaries
            relations: List of relation dictionaries

        Returns:
            list[dict]: List of validation warnings (empty if no issues)
        """
        warnings: list[dict[str, Any]] = []

        for idx, entity in enumerate(entities):
            # Check confidence
            confidence = entity.get("confidence")
            if confidence is not None:
                if not isinstance(confidence, (int, float)):
                    warnings.append(
                        {
                            "type": "invalid_confidence_type",
                            "context": "entity",
                            "index": idx,
                            "entity_name": entity.get("name", "unknown"),
                            "message": f"Confidence should be numeric, got {type(confidence).__name__}",
                        }
                    )
                elif confidence < 0 or confidence > 1:
                    warnings.append(
                        {
                            "type": "confidence_out_of_range",
                            "context": "entity",
                            "index": idx,
                            "entity_name": entity.get("name", "unknown"),
                            "message": f"Confidence {confidence} not in [0, 1]",
                        }
                    )

        for idx, relation in enumerate(relations):
            # Check temporal ordering
            valid_from = relation.get("valid_from")
            valid_to = relation.get("valid_to")

            if valid_from and valid_to:
                # Parse if string
                if isinstance(valid_from, str):
                    valid_from = datetime.fromisoformat(
                        valid_from.replace("Z", "+00:00")
                    )
                if isinstance(valid_to, str):
                    valid_to = datetime.fromisoformat(valid_to.replace("Z", "+00:00"))

                if valid_from > valid_to:
                    warnings.append(
                        {
                            "type": "invalid_temporal_order",
                            "context": "relation",
                            "index": idx,
                            "source": relation.get("source", "unknown"),
                            "target": relation.get("target", "unknown"),
                            "message": f"valid_from {valid_from} > valid_to {valid_to}",
                        }
                    )

            # Check confidence
            confidence = relation.get("confidence")
            if confidence is not None:
                if not isinstance(confidence, (int, float)):
                    warnings.append(
                        {
                            "type": "invalid_confidence_type",
                            "context": "relation",
                            "index": idx,
                            "source": relation.get("source", "unknown"),
                            "message": f"Confidence should be numeric, got {type(confidence).__name__}",
                        }
                    )
                elif confidence < 0 or confidence > 1:
                    warnings.append(
                        {
                            "type": "confidence_out_of_range",
                            "context": "relation",
                            "index": idx,
                            "source": relation.get("source", "unknown"),
                            "message": f"Confidence {confidence} not in [0, 1]",
                        }
                    )

        return warnings


# Module-level convenience functions
def apply_temporal_defaults(
    entities: List[Dict[str, Any]],
    relations: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Apply default temporal properties to all entities and relations."""
    return TemporalIngestionMixin.apply_temporal_defaults(entities, relations)


def validate_ingestion_temporal(
    entities: List[Dict[str, Any]],
    relations: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Validate temporal metadata in entities and relations."""
    return TemporalIngestionMixin.validate_ingestion_temporal(entities, relations)
