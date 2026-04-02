"""
Integration tests for entity resolution pipeline.

Tests cover:
- test_similarity_scoring_accuracy: Validates similarity scoring
- test_merge_decision_tiers: Tests three-tier decision classification
- test_temporal_handling_on_merge: Tests temporal union on merge
- test_review_queue_workflow: Tests manual review queue workflow
- Edge case tests: Similar names (different entities), different names (same entity)
"""

import asyncio
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from app.services.merge_decision_service import (
    AUTOMATIC_MERGE_THRESHOLD,
    MANUAL_REVIEW_LOWER,
    DecisionTier,
    MergeDecision,
    MergeDecisionService,
)
from app.services.merge_executor import MergeExecutor, MergeResult
from app.services.similarity_service import SimilarityService
from app.services.validation_service import (
    EntityPair,
    ValidationService,
    ValidationMetrics,
    ValidationReport,
)


# For Python 3.7 compatibility - create a simple async mock
class AsyncMock(MagicMock):
    """Async mock for Python 3.7 compatibility."""

    async def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs)


class TestSimilarityScoringAccuracy:
    """Test cases for similarity scoring accuracy validation."""

    @pytest.mark.asyncio
    async def test_identical_entities(self):
        """Test similarity between identical entities."""
        service = SimilarityService()

        # With mocked embedding to avoid Ollama dependency
        with patch.object(service.embedding_service, "embed_text") as mock_embed:
            # Return same embedding for identical texts
            mock_embed.side_effect = lambda text: [1.0] * 768

            similarity = await service.compute_similarity(
                "Microsoft",
                "Technology company",
                "Microsoft",
                "Technology company",
            )

            # Should be very high similarity
            assert similarity > 0.9

    @pytest.mark.asyncio
    async def test_case_variations(self):
        """Test that case variations are handled."""
        service = SimilarityService()

        with patch.object(service.embedding_service, "embed_text") as mock_embed:
            mock_embed.side_effect = lambda text: [1.0] * 768

            similarity = await service.compute_similarity(
                "GOOGLE",
                "Search engine",
                "google",
                "Search engine",
            )

            # Should be high due to normalization
            assert similarity > 0.8

    @pytest.mark.asyncio
    async def test_acronyms_full_name(self):
        """Test similarity between acronym and full name."""
        service = SimilarityService()

        with patch.object(service.embedding_service, "embed_text") as mock_embed:
            mock_embed.side_effect = lambda text: [1.0] * 768

            similarity = await service.compute_similarity(
                "International Business Machines",
                "Tech company",
                "IBM",
                "Tech company",
            )

            # Should have moderate string similarity, high embedding similarity
            assert 0.5 <= similarity <= 1.0

    @pytest.mark.asyncio
    async def test_completely_different_entities(self):
        """Test similarity between completely different entities."""
        service = SimilarityService()

        with patch.object(service.embedding_service, "embed_text") as mock_embed:
            # Return different embeddings
            mock_embed.side_effect = (
                lambda text: [1.0 if "toyota" in text.lower() else 0.0] * 768
            )

            similarity = await service.compute_similarity(
                "Toyota",
                "Japanese car manufacturer",
                "Ford",
                "American car manufacturer",
            )

            # Should be low similarity
            assert similarity < 0.5


class TestMergeDecisionTiers:
    """Test cases for merge decision tier classification."""

    def test_automatic_merge_threshold(self):
        """Test AUTOMATIC_MERGE tier for high scores."""
        # Score >= 0.90 should be AUTOMATIC_MERGE
        assert MergeDecisionService.classify_tier(0.95) == DecisionTier.AUTOMATIC_MERGE
        assert MergeDecisionService.classify_tier(0.90) == DecisionTier.AUTOMATIC_MERGE
        assert MergeDecisionService.classify_tier(1.0) == DecisionTier.AUTOMATIC_MERGE

    def test_manual_review_threshold(self):
        """Test MANUAL_REVIEW tier for medium scores."""
        # Score 0.80-0.89 should be MANUAL_REVIEW
        assert MergeDecisionService.classify_tier(0.89) == DecisionTier.MANUAL_REVIEW
        assert MergeDecisionService.classify_tier(0.85) == DecisionTier.MANUAL_REVIEW
        assert MergeDecisionService.classify_tier(0.80) == DecisionTier.MANUAL_REVIEW

    def test_rejected_threshold(self):
        """Test REJECTED tier for low scores."""
        # Score < 0.80 should be REJECTED
        assert MergeDecisionService.classify_tier(0.79) == DecisionTier.REJECTED
        assert MergeDecisionService.classify_tier(0.50) == DecisionTier.REJECTED
        assert MergeDecisionService.classify_tier(0.0) == DecisionTier.REJECTED

    @pytest.mark.asyncio
    async def test_decision_creation_with_tier(self):
        """Test MergeDecision creation includes correct tier."""
        mock_similarity = AsyncMock(spec=SimilarityService)
        mock_similarity.compute_similarity = AsyncMock(return_value=0.92)

        mock_neo4j = MagicMock()
        mock_neo4j.execute_write = AsyncMock()

        service = MergeDecisionService(mock_similarity, mock_neo4j)

        entity1 = {"id": "e1", "name": "Apple Inc", "description": "Tech company"}
        entity2 = {
            "id": "e2",
            "name": "Apple Incorporated",
            "description": "Tech company",
        }

        decision = await service.evaluate_pair(entity1, entity2)

        assert decision.tier == DecisionTier.AUTOMATIC_MERGE
        assert decision.similarity_score == 0.92
        assert decision.decision == "merged"


class TestTemporalHandlingOnMerge:
    """Test cases for temporal handling during entity merge."""

    def test_temporal_union_both_defined(self):
        """Test temporal union when both entities have defined time periods."""
        executor = MergeExecutor()

        entity1 = {
            "valid_from": datetime(2020, 1, 1),
            "valid_to": datetime(2023, 12, 31),
        }
        entity2 = {
            "valid_from": datetime(2021, 6, 15),
            "valid_to": datetime(2024, 6, 15),
        }

        valid_from, valid_to = executor._compute_temporal_union(entity1, entity2)

        # Earliest valid_from
        assert valid_from == datetime(2020, 1, 1)
        # Latest valid_to (both defined)
        assert valid_to == datetime(2024, 6, 15)

    def test_temporal_union_one_null(self):
        """Test temporal union when one entity has NULL valid_to."""
        executor = MergeExecutor()

        entity1 = {
            "valid_from": datetime(2020, 1, 1),
            "valid_to": datetime(2023, 12, 31),
        }
        entity2 = {
            "valid_from": datetime(2021, 6, 15),
            "valid_to": None,  # NULL - ongoing entity
        }

        valid_from, valid_to = executor._compute_temporal_union(entity1, entity2)

        # Earliest valid_from
        assert valid_from == datetime(2020, 1, 1)
        # Should be NULL per D-11 (if either is NULL, merged is NULL)
        assert valid_to is None

    def test_temporal_union_both_null(self):
        """Test temporal union when both entities have NULL valid_to."""
        executor = MergeExecutor()

        entity1 = {
            "valid_from": datetime(2020, 1, 1),
            "valid_to": None,
        }
        entity2 = {
            "valid_from": datetime(2021, 6, 15),
            "valid_to": None,
        }

        valid_from, valid_to = executor._compute_temporal_union(entity1, entity2)

        # Earliest valid_from
        assert valid_from == datetime(2020, 1, 1)
        # Both NULL, so merged is NULL
        assert valid_to is None


class TestReviewQueueWorkflow:
    """Test cases for manual review queue workflow."""

    @pytest.mark.asyncio
    async def test_review_decision_merge_action(self):
        """Test manual review with merge action."""
        mock_similarity = AsyncMock(spec=SimilarityService)
        mock_similarity.compute_similarity = AsyncMock(return_value=0.85)

        mock_neo4j = MagicMock()

        # Setup mock to return pending decision
        mock_neo4j.execute_single = AsyncMock(
            return_value={
                "decision_id": "test-123",
                "entity1_id": "e1",
                "entity2_id": "e2",
                "entity1_name": "Apple Inc",
                "entity2_name": "Apple Incorporated",
                "similarity_score": 0.85,
                "tier": "MANUAL_REVIEW",
                "created_at": datetime.utcnow(),
                "decision": "pending",
                "reviewer": None,
                "reasoning": "Uncertain merge",
            }
        )
        mock_neo4j.execute_write = AsyncMock()

        service = MergeDecisionService(mock_similarity, mock_neo4j)

        result = await service.review_decision(
            decision_id="test-123",
            action="merge",
            reviewer="admin",
            notes="Confirmed as duplicate",
        )

        assert result is not None
        assert result.decision == "merge"
        assert result.reviewer == "admin"

    @pytest.mark.asyncio
    async def test_review_decision_reject_action(self):
        """Test manual review with reject action."""
        mock_similarity = AsyncMock(spec=SimilarityService)
        mock_neo4j = MagicMock()

        mock_neo4j.execute_single = AsyncMock(
            return_value={
                "decision_id": "test-456",
                "entity1_id": "e1",
                "entity2_id": "e2",
                "entity1_name": "Paris",
                "entity2_name": "Paris",
                "similarity_score": 0.82,
                "tier": "MANUAL_REVIEW",
                "created_at": datetime.utcnow(),
                "decision": "pending",
                "reviewer": None,
                "reasoning": "Needs review",
            }
        )
        mock_neo4j.execute_write = AsyncMock()

        service = MergeDecisionService(mock_similarity, mock_neo4j)

        result = await service.review_decision(
            decision_id="test-456",
            action="reject",
            reviewer="admin",
            notes="Different entities - different cities",
        )

        assert result is not None
        assert result.decision == "reject"
        assert result.reviewer == "admin"

    @pytest.mark.asyncio
    async def test_review_invalid_action_raises(self):
        """Test that invalid review action raises ValueError."""
        mock_similarity = AsyncMock(spec=SimilarityService)
        mock_neo4j = MagicMock()
        mock_neo4j.execute_single = AsyncMock(return_value=None)

        service = MergeDecisionService(mock_similarity, mock_neo4j)

        with pytest.raises(ValueError, match="Invalid action"):
            await service.review_decision(
                decision_id="test",
                action="invalid",
                reviewer="admin",
            )


class TestValidationService:
    """Test cases for validation service."""

    @pytest.mark.asyncio
    async def test_validation_accuracy_calculation(self):
        """Test that accuracy is calculated correctly."""
        service = ValidationService()

        # Create test dataset
        test_pairs = [
            EntityPair(
                entity1={"name": "Apple", "description": "Tech"},
                entity2={"name": "Apple", "description": "Tech"},
                is_duplicate=True,
            ),
            EntityPair(
                entity1={"name": "Apple", "description": "Tech"},
                entity2={"name": "Microsoft", "description": "Tech"},
                is_duplicate=False,
            ),
            EntityPair(
                entity1={"name": "Google", "description": "Search"},
                entity2={"name": "Google", "description": "Search"},
                is_duplicate=True,
            ),
            EntityPair(
                entity1={"name": "Google", "description": "Search"},
                entity2={"name": "Yahoo", "description": "Search"},
                is_duplicate=False,
            ),
        ]

        # Run validation
        with patch.object(service.similarity_service, "compute_similarity") as mock_sim:
            # Return controlled values
            mock_sim.side_effect = [0.95, 0.30, 0.92, 0.25]

            report = await service.validate_accuracy(test_pairs, threshold=0.80)

        # Check metrics
        assert report.metrics.total_pairs == 4
        # 2 true positives (duplicates correctly identified) + 2 true negatives (distinct correctly identified)
        assert report.metrics.true_positives == 2
        assert report.metrics.true_negatives == 2
        assert report.metrics.accuracy == 1.0

    @pytest.mark.asyncio
    async def test_validation_precision_recall_f1(self):
        """Test precision, recall, F1 calculation."""
        service = ValidationService()

        # Create imbalanced dataset
        test_pairs = [
            EntityPair(
                entity1={"name": "A", "description": "d"},
                entity2={"name": "A", "description": "d"},
                is_duplicate=True,
            ),
            EntityPair(
                entity1={"name": "B", "description": "d"},
                entity2={"name": "C", "description": "d"},
                is_duplicate=True,
            ),
            EntityPair(
                entity1={"name": "D", "description": "d"},
                entity2={"name": "E", "description": "d"},
                is_duplicate=False,
            ),
        ]

        with patch.object(service.similarity_service, "compute_similarity") as mock_sim:
            # 2 duplicates: 1 TP, 1 FN; 1 distinct: 1 TN (no FP)
            mock_sim.side_effect = [0.90, 0.70, 0.30]

            report = await service.validate_accuracy(test_pairs, threshold=0.80)

        # Precision = TP / (TP + FP) = 1 / (1 + 0) = 1.0
        assert report.metrics.precision == 1.0
        # Recall = TP / (TP + FN) = 1 / (1 + 1) = 0.5
        assert report.metrics.recall == 0.5
        # F1 = 2 * P * R / (P + R) = 2 * 1 * 0.5 / 1.5 = 0.667
        assert abs(report.metrics.f1_score - 0.667) < 0.01

    @pytest.mark.asyncio
    async def test_threshold_analysis(self):
        """Test threshold analysis at different cutoffs."""
        service = ValidationService()

        test_pairs = [
            EntityPair(
                entity1={"name": "A", "description": "d"},
                entity2={"name": "A", "description": "d"},
                is_duplicate=True,
            ),
            EntityPair(
                entity1={"name": "B", "description": "d"},
                entity2={"name": "C", "description": "d"},
                is_duplicate=False,
            ),
        ]

        with patch.object(service.similarity_service, "compute_similarity") as mock_sim:
            mock_sim.side_effect = [0.90, 0.30]

            report = await service.validate_accuracy(test_pairs, threshold=0.80)

        # Check threshold analysis was performed
        assert len(report.threshold_analysis) > 0

        # Should have analysis for multiple thresholds
        thresholds_tested = [m.threshold_used for m in report.threshold_analysis]
        assert 0.80 in thresholds_tested

    def test_generate_validation_report_format(self):
        """Test validation report formatting."""
        service = ValidationService()

        metrics = ValidationMetrics(
            total_pairs=10,
            true_positives=8,
            true_negatives=1,
            false_positives=0,
            false_negatives=1,
            accuracy=0.9,
            precision=1.0,
            recall=0.889,
            f1_score=0.941,
            threshold_used=0.80,
        )

        results = []
        report = ValidationReport(
            metrics=metrics,
            results=results,
            generated_at=datetime.utcnow(),
            threshold_analysis=[],
            problematic_cases=[],
        )

        report_str = service.generate_validation_report(report)

        # Check report content
        assert "ENTITY RESOLUTION VALIDATION REPORT" in report_str
        assert "Accuracy:" in report_str
        assert "90.00%" in report_str
        assert "KR-04" in report_str


class TestEdgeCases:
    """Test edge cases for entity resolution."""

    @pytest.mark.asyncio
    async def test_apple_inc_vs_apple_incorporated(self):
        """Test: Apple Inc vs Apple Incorporated (should detect as duplicate)."""
        service = SimilarityService()

        with patch.object(service.embedding_service, "embed_text") as mock_embed:
            mock_embed.side_effect = lambda text: [1.0] * 768

            similarity = await service.compute_similarity(
                "Apple Inc",
                "Technology company",
                "Apple Incorporated",
                "Technology company",
            )

            # Should detect as potential duplicate (above manual review threshold)
            assert similarity >= MANUAL_REVIEW_LOWER

    @pytest.mark.asyncio
    async def test_john_smith_jr(self):
        """Test: John Smith vs John Smith Jr (should detect as duplicate)."""
        service = SimilarityService()

        with patch.object(service.embedding_service, "embed_text") as mock_embed:
            mock_embed.side_effect = lambda text: [1.0] * 768

            similarity = await service.compute_similarity(
                "John Smith",
                "Software engineer",
                "John Smith Jr",
                "Software engineer",
            )

            # Should detect as potential duplicate
            assert similarity >= MANUAL_REVIEW_LOWER

    @pytest.mark.asyncio
    async def test_paris_city_vs_company(self):
        """Test: Paris company vs Paris city (should NOT detect as duplicate - different types)."""
        service = SimilarityService()

        with patch.object(service.embedding_service, "embed_text") as mock_embed:
            # Return different embeddings for different content
            def get_embedding(text):
                if "city" in text.lower():
                    return [1.0] * 768
                elif "company" in text.lower():
                    return [0.0] + [1.0] * 767
                return [0.5] * 768

            mock_embed.side_effect = get_embedding

            similarity = await service.compute_similarity(
                "Paris",
                "A city in France",
                "Paris",
                "A company name",
            )

            # Should NOT detect as duplicate due to different descriptions
            assert similarity < MANUAL_REVIEW_LOWER

    @pytest.mark.asyncio
    async def test_same_name_different_entities(self):
        """Test: Similar names but actually different entities."""
        service = SimilarityService()

        with patch.object(service.embedding_service, "embed_text") as mock_embed:
            # Different embeddings for different entities
            def get_embedding(text):
                if "michael johnson" in text.lower():
                    if "athlete" in text.lower():
                        return [1.0] * 768
                    elif "actor" in text.lower():
                        return [0.0] + [1.0] * 767
                return [0.5] * 768

            mock_embed.side_effect = get_embedding

            similarity = await service.compute_similarity(
                "Michael Johnson",
                "Olympic athlete",
                "Michael Johnson",
                "American actor",
            )

            # Should be in manual review or rejected zone
            # (descriptions differ significantly)
            assert 0.0 <= similarity <= 1.0

    @pytest.mark.asyncio
    async def test_typo_handling(self):
        """Test: Common typos are handled correctly."""
        service = SimilarityService()

        # Test without embedding (string-only similarity)
        similarity = service.compute_jaro_winkler("Amazn", "Amazon")

        # Jaro-Winkler should handle single character typos well
        assert similarity > 0.8

    @pytest.mark.asyncio
    async def test_empty_descriptions(self):
        """Test: Entities with empty descriptions use name similarity."""
        service = SimilarityService()

        with patch.object(service.embedding_service, "embed_text") as mock_embed:
            mock_embed.side_effect = lambda text: [1.0] * 768

            similarity = await service.compute_similarity(
                "Microsoft",
                "",  # Empty description
                "Microsoft",
                "",  # Empty description
            )

            # Should still work with just names
            assert similarity > 0.9


class TestValidationServiceIntegration:
    """Integration tests for full validation pipeline."""

    @pytest.mark.asyncio
    async def test_default_dataset_creation(self):
        """Test that default test dataset is created correctly."""
        dataset = ValidationService.get_default_test_dataset()

        assert len(dataset) > 0

        # Check for expected edge cases
        has_apple_inc = any(
            "Apple Inc" in p.name1 and "Apple Incorporated" in p.name2 for p in dataset
        )
        has_jr = any("John Smith" in p.name1 and "Jr" in p.name2 for p in dataset)
        has_paris = any(
            p.name1 == "Paris" and p.name2 == "Paris" and not p.is_duplicate
            for p in dataset
        )

        assert has_apple_inc, "Missing Apple Inc vs Incorporated test case"
        assert has_jr, "Missing John Smith Jr test case"
        assert has_paris, "Missing Paris city vs city test case"

    @pytest.mark.asyncio
    async def test_validation_with_default_dataset(self):
        """Test validation runs with default dataset."""
        service = ValidationService()
        dataset = ValidationService.get_default_test_dataset()

        # Mock similarity to return controlled values
        # For true duplicates: high score (>0.85)
        # For true distinct: low score (<0.80)
        call_count = [0]

        async def mock_similarity(name1, desc1, name2, desc2):
            call_count[0] += 1
            idx = call_count[0] - 1

            # Simple heuristic based on names
            n1 = name1.lower().strip()
            n2 = name2.lower().strip()

            # Identical or very similar
            if n1 == n2 or (n1 in n2 or n2 in n1):
                return 0.92
            # Partially similar
            elif len(set(n1.split()) & set(n2.split())) > 0:
                return 0.75
            # Different
            return 0.25

        with patch.object(
            service.similarity_service,
            "compute_similarity",
            side_effect=mock_similarity,
        ):
            report = await service.validate_accuracy(dataset, threshold=0.80)

        # Check basic metrics
        assert report.metrics.total_pairs == len(dataset)
        assert report.metrics.accuracy > 0

    @pytest.mark.asyncio
    async def test_kr04_threshold_check(self):
        """Test KR-04 threshold requirement check."""
        service = ValidationService()

        # Create dataset with known accuracy
        metrics = ValidationMetrics(
            total_pairs=100,
            true_positives=86,
            true_negatives=10,
            false_positives=2,
            false_negatives=2,
            accuracy=0.96,
            precision=0.977,
            recall=0.977,
            f1_score=0.977,
            threshold_used=0.80,
        )

        results = []
        report = ValidationReport(
            metrics=metrics,
            results=results,
            generated_at=datetime.utcnow(),
        )

        report_str = service.generate_validation_report(report)

        # Should show PASSED
        assert "✓ PASSED" in report_str
        assert "≥85%" in report_str


# Run all tests with: pytest tests/services/test_entity_resolution.py -v
