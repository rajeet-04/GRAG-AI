"""Tests for similarity service."""

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from app.services.similarity_service import SimilarityService


class TestSimilarityService:
    """Test cases for SimilarityService."""

    def test_jaro_winkler_identical(self):
        """Test Jaro-Winkler with identical strings."""
        service = SimilarityService()
        score = service.compute_jaro_winkler("hello", "hello")
        assert score == 1.0

    def test_jaro_winkler_completely_different(self):
        """Test Jaro-Winkler with completely different strings."""
        service = SimilarityService()
        score = service.compute_jaro_winkler("hello", "world")
        # Jaro-Winkler of "hello" and "world" is approximately 0.422
        assert 0.4 <= score <= 0.5

    def test_jaro_winkler_empty_strings(self):
        """Test Jaro-Winkler with empty strings."""
        service = SimilarityService()
        score = service.compute_jaro_winkler("", "")
        assert score == 1.0

    def test_jaro_winkler_one_empty(self):
        """Test Jaro-Winkler with one empty string."""
        service = SimilarityService()
        score = service.compute_jaro_winkler("hello", "")
        assert score == 0.0

    def test_jaro_winkler_typos(self):
        """Test Jaro-Winkler handles typos well."""
        service = SimilarityService()
        # Common typos
        score1 = service.compute_jaro_winkler("john", "jon")
        score2 = service.compute_jaro_winkler("smith", "smit")
        assert score1 > 0.8  # Should be high for minor typo
        assert score2 > 0.8  # Should be high for minor typo

    def test_cosine_similarity_identical(self):
        """Test cosine similarity with identical vectors."""
        service = SimilarityService()
        vec = [1.0, 2.0, 3.0]
        similarity = service._cosine_similarity(vec, vec)
        assert similarity == 1.0

    def test_cosine_similarity_opposite(self):
        """Test cosine similarity with opposite vectors."""
        service = SimilarityService()
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [-1.0, 0.0, 0.0]
        similarity = service._cosine_similarity(vec1, vec2)
        assert similarity == 0.0  # Opposite vectors map to 0 in our 0-1 scaling

    def test_cosine_similarity_orthogonal(self):
        """Test cosine similarity with orthogonal vectors."""
        service = SimilarityService()
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [0.0, 1.0, 0.0]
        similarity = service._cosine_similarity(vec1, vec2)
        assert similarity == 0.5  # Orthogonal vectors map to 0.5 in our 0-1 scaling

    @pytest.mark.asyncio
    async def test_embedding_similarity_both_empty(self):
        """Test embedding similarity with both texts empty."""
        service = SimilarityService()
        similarity = await service.compute_embedding_similarity("", "")
        assert similarity == 1.0

    @pytest.mark.asyncio
    async def test_embedding_similarity_one_empty(self):
        """Test embedding similarity with one text empty."""
        service = SimilarityService()
        similarity = await service.compute_embedding_similarity("hello", "")
        assert similarity == 0.0

    @pytest.mark.asyncio
    async def test_compute_similarity_weights(self):
        """Test that compute_similarity applies correct weights."""
        service = SimilarityService()

        # Mock the embedding service to return fixed similarity
        with patch.object(
            service.embedding_service, "compute_embedding_similarity"
        ) as mock_emb:
            mock_emb.return_value = 0.8

            # With string_sim=1.0 and embedding_sim=0.8
            # Expected: 0.4*1.0 + 0.6*0.8 = 0.4 + 0.48 = 0.88
            similarity = await service.compute_similarity(
                "exact match", "desc1", "exact match", "desc2"
            )

            # Should be close to expected value
            assert abs(similarity - 0.88) < 0.01

    @pytest.mark.asyncio
    async def test_compute_similarity_fallback_to_names(self):
        """Test fallback to name similarity when descriptions missing."""
        service = SimilarityService()

        # Mock embedding service to raise exception (simulating failure)
        with patch.object(
            service.embedding_service, "compute_embedding_similarity"
        ) as mock_emb:
            mock_emb.side_effect = Exception("Embedding failed")

            # Should fall back to name similarity
            similarity = await service.compute_similarity(
                "john smith",
                "",  # First entity
                "jon smith",
                "",  # Second entity (typo in first name)
            )

            # Should be high due to Jaro-Winkler handling typos well
            assert similarity > 0.8

    @pytest.mark.asyncio
    async def test_handle_entity_pair(self):
        """Test handling entity pair dictionaries."""
        service = SimilarityService()

        entity1 = {"name": "Apple Inc", "description": "Technology company"}
        entity2 = {"name": "Apple Corporation", "description": "Technology corporation"}

        similarity, breakdown = await service.handle_entity_pair(entity1, entity2)

        # Should return similarity score and breakdown
        assert isinstance(similarity, float)
        assert 0.0 <= similarity <= 1.0
        assert isinstance(breakdown, dict)
        assert "name1" in breakdown
        assert "name2" in breakdown
        assert "similarity_score" in breakdown
        assert breakdown["similarity_score"] == similarity

    @pytest.mark.asyncio
    async def test_handle_entity_pair_missing_fields(self):
        """Test handling entity pair with missing fields."""
        service = SimilarityService()

        entity1 = {"name": "Test"}  # Missing description
        entity2 = {"description": "Test description"}  # Missing name

        similarity, breakdown = await service.handle_entity_pair(entity1, entity2)

        # Should handle missing fields gracefully
        assert isinstance(similarity, float)
        assert 0.0 <= similarity <= 1.0
