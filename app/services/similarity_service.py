"""Entity similarity computation service for deduplication."""

import logging
import math
from typing import List, Optional, Tuple

import jellyfish  # For Jaro-Winkler distance

from app.llm.embedding import EmbeddingService
from app.utils.text_utils import normalize_entity_name, normalize_entity_description

logger = logging.getLogger(__name__)


class SimilarityService:
    """
    Service for computing similarity between entities for deduplication.

    Combines Jaro-Winkler string similarity and embedding cosine similarity
    with weighted scoring to achieve ≥85% accuracy threshold.
    """

    def __init__(self, embedding_service: Optional[EmbeddingService] = None):
        """
        Initialize similarity service.

        Args:
            embedding_service: EmbeddingService instance (creates new if None)
        """
        self.embedding_service = embedding_service or EmbeddingService()
        # Weights for hybrid scoring (from CONTEXT.md decision)
        self.string_weight = 0.4
        self.embedding_weight = 0.6

    def compute_jaro_winkler(self, str1: str, str2: str) -> float:
        """
        Compute Jaro-Winkler similarity between two strings.

        Args:
            str1: First string
            str2: Second string

        Returns:
            Similarity score between 0.0 and 1.0
        """
        if not str1 and not str2:
            return 1.0
        if not str1 or not str2:
            return 0.0

        # jellyfish.jaro_winkler_similarity returns 0.0-1.0
        similarity = jellyfish.jaro_winkler_similarity(str1, str2)
        return max(0.0, min(1.0, similarity))

    async def compute_embedding_similarity(self, text1: str, text2: str) -> float:
        """
        Compute cosine similarity between embeddings of two texts.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Similarity score between 0.0 and 1.0
        """
        if not text1 and not text2:
            return 1.0
        if not text1 or not text2:
            return 0.0

        try:
            # Get embeddings for both texts
            embedding1 = await self.embedding_service.embed_text(text1)
            embedding2 = await self.embedding_service.embed_text(text2)

            # Compute cosine similarity
            return self._cosine_similarity(embedding1, embedding2)
        except Exception as e:
            logger.warning(
                "embedding_similarity.failed",
                error=str(e),
                text1_length=len(text1),
                text2_length=len(text2),
            )
            # Fallback to string similarity if embedding fails
            return self.compute_jaro_winkler(text1, text2)

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Compute cosine similarity between two vectors.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Similarity score between 0.0 and 1.0
        """
        if len(vec1) != len(vec2):
            raise ValueError("Vectors must have same length")

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(b * b for b in vec2))

        if magnitude1 == 0.0 or magnitude2 == 0.0:
            return 0.0

        similarity = dot_product / (magnitude1 * magnitude2)
        # Cosine similarity ranges from -1 to 1, shift to 0-1
        return (similarity + 1.0) / 2.0

    async def compute_similarity(
        self,
        entity1_name: str,
        entity1_description: str,
        entity2_name: str,
        entity2_description: str,
    ) -> float:
        """
        Compute hybrid similarity between two entities.

        Args:
            entity1_name: First entity name
            entity1_description: First entity description
            entity2_name: Second entity name
            entity2_description: Second entity description

        Returns:
            Weighted hybrid similarity score (0.0-1.0)
        """
        # Normalize inputs
        norm_name1 = normalize_entity_name(entity1_name)
        norm_name2 = normalize_entity_name(entity2_name)
        norm_desc1 = normalize_entity_description(entity1_description)
        norm_desc2 = normalize_entity_description(entity2_description)

        # Compute string similarity (using names)
        string_sim = self.compute_jaro_winkler(norm_name1, norm_name2)

        # Compute embedding similarity (using descriptions, fallback to names)
        if norm_desc1 or norm_desc2:
            embedding_sim = await self.compute_embedding_similarity(
                norm_desc1, norm_desc2
            )
        else:
            # If no descriptions, use name similarity for embedding component
            embedding_sim = string_sim

        # Apply weighted hybrid formula: 0.4 × string + 0.6 × embedding
        hybrid_score = (
            self.string_weight * string_sim + self.embedding_weight * embedding_sim
        )

        logger.debug(
            "similarity.computed",
            entity1_name=entity1_name,
            entity2_name=entity2_name,
            string_sim=string_sim,
            embedding_sim=embedding_sim,
            hybrid_score=hybrid_score,
        )

        return max(0.0, min(1.0, hybrid_score))

    async def handle_entity_pair(
        self, entity1: dict, entity2: dict
    ) -> Tuple[float, dict]:
        """
        Compute similarity for a pair of entity dictionaries.

        Args:
            entity1: First entity dictionary with 'name' and 'description' keys
            entity2: Second entity dictionary with 'name' and 'description' keys

        Returns:
            Tuple of (similarity_score, breakdown_dict)
        """
        name1 = entity1.get("name", "")
        desc1 = entity1.get("description", "")
        name2 = entity2.get("name", "")
        desc2 = entity2.get("description", "")

        similarity = await self.compute_similarity(name1, desc1, name2, desc2)

        breakdown = {
            "name1": name1,
            "name2": name2,
            "desc1_length": len(desc1),
            "desc2_length": len(desc2),
            "similarity_score": similarity,
        }

        return similarity, breakdown
