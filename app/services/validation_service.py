"""
Validation service for measuring entity resolution accuracy.

Provides accuracy measurement against labeled test sets and generates
validation reports with precision, recall, and F1 metrics.

Per KR-04: Entity deduplication logic with ≥85% accuracy
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

import structlog

from app.services.similarity_service import SimilarityService
from app.services.merge_decision_service import (
    AUTOMATIC_MERGE_THRESHOLD,
    MANUAL_REVIEW_LOWER,
    DecisionTier,
)

logger = structlog.get_logger()


@dataclass
class EntityPair:
    """
    A labeled entity pair for validation testing.

    Attributes:
        entity1: First entity dict with 'name' and 'description'
        entity2: Second entity dict with 'name' and 'description'
        is_duplicate: Ground truth - True if entities are duplicates
        notes: Optional notes about this pair
    """

    entity1: dict[str, Any]
    entity2: dict[str, Any]
    is_duplicate: bool
    notes: Optional[str] = None

    @property
    def name1(self) -> str:
        return self.entity1.get("name", "")

    @property
    def name2(self) -> str:
        return self.entity2.get("name", "")


@dataclass
class ValidationResult:
    """
    Result of validation on a single entity pair.

    Attributes:
        pair: The entity pair tested
        predicted_is_duplicate: Model's prediction
        similarity_score: Raw similarity score
        predicted_tier: Classification tier
        correct: Whether prediction matched ground truth
    """

    pair: EntityPair
    predicted_is_duplicate: bool
    similarity_score: float
    predicted_tier: DecisionTier
    correct: bool


@dataclass
class ValidationMetrics:
    """
    Aggregated validation metrics.

    Attributes:
        total_pairs: Total number of test pairs
        true_positives: Correctly identified duplicates
        true_negatives: Correctly identified distinct entities
        false_positives: Predicted duplicate but actually distinct
        false_negatives: Predicted distinct but actually duplicates
        accuracy: Overall accuracy (TP+TN)/Total
        precision: TP/(TP+FP)
        recall: TP/(TP+FN)
        f1_score: Harmonic mean of precision and recall
        threshold_used: Similarity threshold for classification
    """

    total_pairs: int = 0
    true_positives: int = 0
    true_negatives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    threshold_used: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for reporting."""
        return {
            "total_pairs": self.total_pairs,
            "true_positives": self.true_positives,
            "true_negatives": self.true_negatives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "accuracy": round(self.accuracy, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1_score": round(self.f1_score, 4),
            "threshold_used": self.threshold_used,
        }


@dataclass
class ValidationReport:
    """
    Complete validation report with metrics and analysis.

    Attributes:
        metrics: Aggregated metrics
        results: Individual test results
        generated_at: Report generation timestamp
        threshold_analysis: Metrics at different thresholds
    """

    metrics: ValidationMetrics
    results: list[ValidationResult]
    generated_at: datetime
    threshold_analysis: list[ValidationMetrics] = field(default_factory=list)
    problematic_cases: list[ValidationResult] = field(default_factory=list)


class ValidationService:
    """
    Service for validating entity resolution accuracy.

    Provides:
    - Accuracy measurement against labeled test sets
    - Precision, recall, F1 metrics
    - Threshold analysis for optimal performance
    - Edge case testing and reporting

    Per KR-04: Validates ≥85% accuracy requirement
    """

    def __init__(self, similarity_service: Optional[SimilarityService] = None):
        """
        Initialize validation service.

        Args:
            similarity_service: SimilarityService for computing scores
        """
        self.similarity_service = similarity_service or SimilarityService()

    async def validate_accuracy(
        self,
        test_dataset: list[EntityPair],
        threshold: float = MANUAL_REVIEW_LOWER,
    ) -> ValidationReport:
        """
        Validate accuracy against a labeled test set.

        Args:
            test_dataset: List of EntityPair with ground truth labels
            threshold: Similarity threshold for duplicate classification

        Returns:
            ValidationReport with metrics and detailed results
        """
        logger.info("validation.start", pairs=len(test_dataset), threshold=threshold)

        results: list[ValidationResult] = []

        # Evaluate each pair
        for pair in test_dataset:
            # Compute similarity score
            score = await self.similarity_service.compute_similarity(
                pair.name1,
                pair.entity1.get("description", ""),
                pair.name2,
                pair.entity2.get("description", ""),
            )

            # Classify as duplicate if score >= threshold
            predicted_is_duplicate = score >= threshold

            # Determine tier
            predicted_tier = self._classify_tier(score)

            # Check correctness
            correct = predicted_is_duplicate == pair.is_duplicate

            result = ValidationResult(
                pair=pair,
                predicted_is_duplicate=predicted_is_duplicate,
                similarity_score=score,
                predicted_tier=predicted_tier,
                correct=correct,
            )
            results.append(result)

        # Aggregate metrics
        metrics = self._compute_metrics(results, threshold)

        # Identify problematic cases
        problematic = [r for r in results if not r.correct]

        # Run threshold analysis
        threshold_analysis = await self._analyze_thresholds(test_dataset)

        logger.info(
            "validation.complete",
            accuracy=metrics.accuracy,
            precision=metrics.precision,
            recall=metrics.recall,
            f1=metrics.f1_score,
            threshold=threshold,
        )

        return ValidationReport(
            metrics=metrics,
            results=results,
            generated_at=datetime.utcnow(),
            threshold_analysis=threshold_analysis,
            problematic_cases=problematic,
        )

    def _classify_tier(self, score: float) -> DecisionTier:
        """Classify score into decision tier."""
        if score >= AUTOMATIC_MERGE_THRESHOLD:
            return DecisionTier.AUTOMATIC_MERGE
        elif score >= MANUAL_REVIEW_LOWER:
            return DecisionTier.MANUAL_REVIEW
        else:
            return DecisionTier.REJECTED

    def _compute_metrics(
        self, results: list[ValidationResult], threshold: float
    ) -> ValidationMetrics:
        """Compute aggregated metrics from results."""
        tp = sum(1 for r in results if r.correct and r.pair.is_duplicate)
        tn = sum(1 for r in results if r.correct and not r.pair.is_duplicate)
        fp = sum(1 for r in results if not r.correct and not r.pair.is_duplicate)
        fn = sum(1 for r in results if not r.correct and r.pair.is_duplicate)

        total = len(results)
        accuracy = (tp + tn) / total if total > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        return ValidationMetrics(
            total_pairs=total,
            true_positives=tp,
            true_negatives=tn,
            false_positives=fp,
            false_negatives=fn,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            threshold_used=threshold,
        )

    async def _analyze_thresholds(
        self, test_dataset: list[EntityPair]
    ) -> list[ValidationMetrics]:
        """Analyze performance at different thresholds."""
        thresholds = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
        analysis = []

        for threshold in thresholds:
            # Compute metrics at this threshold
            results = []
            for pair in test_dataset:
                score = await self.similarity_service.compute_similarity(
                    pair.name1,
                    pair.entity1.get("description", ""),
                    pair.name2,
                    pair.entity2.get("description", ""),
                )
                predicted = score >= threshold
                correct = predicted == pair.is_duplicate

                results.append(
                    ValidationResult(
                        pair=pair,
                        predicted_is_duplicate=predicted,
                        similarity_score=score,
                        predicted_tier=self._classify_tier(score),
                        correct=correct,
                    )
                )

            metrics = self._compute_metrics(results, threshold)
            analysis.append(metrics)

        return analysis

    def generate_validation_report(self, report: ValidationReport) -> str:
        """
        Generate human-readable validation report.

        Args:
            report: ValidationReport to format

        Returns:
            Formatted report string
        """
        m = report.metrics

        lines = [
            "=" * 60,
            "ENTITY RESOLUTION VALIDATION REPORT",
            "=" * 60,
            "",
            f"Generated: {report.generated_at.isoformat()}",
            "",
            "OVERALL METRICS",
            "-" * 40,
            f"Total Test Pairs:     {m.total_pairs}",
            f"Threshold Used:       {m.threshold_used}",
            "",
            f"Accuracy:   {m.accuracy:.2%}",
            f"Precision:  {m.precision:.2%}",
            f"Recall:     {m.recall:.2%}",
            f"F1 Score:   {m.f1_score:.2%}",
            "",
            "CONFUSION MATRIX",
            "-" * 40,
            f"True Positives:  {m.true_positives:4d}  (correctly identified duplicates)",
            f"True Negatives:  {m.true_negatives:4d}  (correctly identified distinct)",
            f"False Positives: {m.false_positives:4d}  (predicted duplicate, actually distinct)",
            f"False Negatives: {m.false_negatives:4d}  (predicted distinct, actually duplicate)",
            "",
        ]

        # Add threshold analysis
        if report.threshold_analysis:
            lines.extend(
                [
                    "THRESHOLD ANALYSIS",
                    "-" * 40,
                    f"{'Threshold':<12} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12}",
                    "-" * 60,
                ]
            )
            for metrics in report.threshold_analysis:
                lines.append(
                    f"{metrics.threshold_used:<12.2f} "
                    f"{metrics.accuracy:<12.2%} "
                    f"{metrics.precision:<12.2%} "
                    f"{metrics.recall:<12.2%} "
                    f"{metrics.f1_score:<12.2%}"
                )
            lines.append("")

        # Add problematic cases
        if report.problematic_cases:
            lines.extend(
                [
                    "PROBLEMATIC CASES",
                    "-" * 40,
                ]
            )
            for i, result in enumerate(report.problematic_cases, 1):
                pair = result.pair
                lines.append(f"{i}. {pair.name1} vs {pair.name2}")
                lines.append(
                    f"   Ground Truth:    {'Duplicate' if pair.is_duplicate else 'Distinct'}"
                )
                lines.append(
                    f"   Predicted:       {'Duplicate' if result.predicted_is_duplicate else 'Distinct'}"
                )
                lines.append(f"   Similarity:      {result.similarity_score:.3f}")
                if pair.notes:
                    lines.append(f"   Notes:           {pair.notes}")
                lines.append("")

        # KR-04 Check
        kr04_passed = m.accuracy >= 0.85
        lines.extend(
            [
                "=" * 60,
                "KR-04 REQUIREMENT CHECK",
                "=" * 60,
                f"Required: ≥85% accuracy",
                f"Actual:   {m.accuracy:.2%}",
                f"Status:   {'✓ PASSED' if kr04_passed else '✗ FAILED'}",
                "=" * 60,
            ]
        )

        return "\n".join(lines)

    @staticmethod
    def get_default_test_dataset() -> list[EntityPair]:
        """
        Get default synthetic labeled test dataset.

        Covers edge cases per plan requirements:
        - Test 4: "Apple Inc" vs "Apple Incorporated" (should detect as duplicate)
        - Test 5: "John Smith" vs "John Smith Jr" (should detect as duplicate)
        - Test 6: "Paris" company vs "Paris" city (should NOT detect as duplicate - different types)

        Returns:
            List of EntityPair with ground truth labels
        """
        return [
            # === TRUE DUPLICATES (should be detected as duplicates) ===
            # Identical names
            EntityPair(
                entity1={"name": "Microsoft", "description": "Technology company"},
                entity2={"name": "Microsoft", "description": "Tech company"},
                is_duplicate=True,
                notes="Identical names",
            ),
            # Minor variations - typos
            EntityPair(
                entity1={
                    "name": "Apple Inc",
                    "description": "Technology company based in Cupertino",
                },
                entity2={"name": "Apple Inc.", "description": "Tech company Cupertino"},
                is_duplicate=True,
                notes="With period",
            ),
            # Case variations
            EntityPair(
                entity1={"name": "GOOGLE", "description": "Search engine company"},
                entity2={"name": "google", "description": "Search engine company"},
                is_duplicate=True,
                notes="Case variation",
            ),
            # Abbreviations
            EntityPair(
                entity1={
                    "name": "International Business Machines",
                    "description": "Big Blue tech company",
                },
                entity2={"name": "IBM", "description": "Big Blue tech company"},
                is_duplicate=True,
                notes="Full name vs abbreviation",
            ),
            # Spelling variations
            EntityPair(
                entity1={"name": "Flavour", "description": "British food brand"},
                entity2={"name": "Flavor", "description": "American food brand"},
                is_duplicate=True,
                notes="British vs American spelling",
            ),
            # Organization suffixes
            EntityPair(
                entity1={
                    "name": "Apple Inc",
                    "description": "Consumer electronics company",
                },
                entity2={
                    "name": "Apple Incorporated",
                    "description": "Consumer electronics company",
                },
                is_duplicate=True,
                notes="Inc vs Incorporated",
            ),
            # Jr suffix
            EntityPair(
                entity1={
                    "name": "John Smith",
                    "description": "Software engineer at Google",
                },
                entity2={"name": "John Smith Jr", "description": "Software engineer"},
                is_duplicate=True,
                notes="With Jr suffix",
            ),
            # Typos
            EntityPair(
                entity1={"name": "Amazon", "description": "E-commerce giant"},
                entity2={"name": "Amazn", "description": "E-commerce giant"},
                is_duplicate=True,
                notes="Typo in name",
            ),
            # === TRUE DISTINCT ENTITIES (should NOT be detected as duplicates) ===
            # Same name, different type
            EntityPair(
                entity1={
                    "name": "Paris",
                    "description": "French city, capital of France",
                },
                entity2={
                    "name": "Paris",
                    "description": "Texas city in the United States",
                },
                is_duplicate=False,
                notes="Same name, different city (type matters)",
            ),
            # Different entities with similar names
            EntityPair(
                entity1={"name": "Mercury", "description": "Planet closest to the sun"},
                entity2={
                    "name": "Mercury",
                    "description": "Element with atomic number 80",
                },
                is_duplicate=False,
                notes="Planet vs element - different types",
            ),
            # Same person name, different people
            EntityPair(
                entity1={
                    "name": "Michael Johnson",
                    "description": "Athlete, Olympic gold medalist",
                },
                entity2={"name": "Michael Johnson", "description": "American actor"},
                is_duplicate=False,
                notes="Different people with same name",
            ),
            # Similar but distinct companies
            EntityPair(
                entity1={"name": "Apple", "description": "Fruit, edible"},
                entity2={"name": "Apple", "description": "Technology company"},
                is_duplicate=False,
                notes="Fruit vs company",
            ),
            # Completely different
            EntityPair(
                entity1={
                    "name": "Toyota",
                    "description": "Japanese automotive manufacturer",
                },
                entity2={
                    "name": "Ford",
                    "description": "American automotive manufacturer",
                },
                is_duplicate=False,
                notes="Different car companies",
            ),
            # Edge case: similar sounding but different
            EntityPair(
                entity1={
                    "name": "Bank of America",
                    "description": "US financial institution",
                },
                entity2={
                    "name": "Bank of America",
                    "description": "Fictional bank in movies",
                },
                is_duplicate=False,
                notes="Real vs fictional (different source)",
            ),
            # Similar but clearly distinct organizations
            EntityPair(
                entity1={
                    "name": "Harvard University",
                    "description": "Ivy League university in Massachusetts",
                },
                entity2={
                    "name": "Harvard College",
                    "description": "Undergraduate college within Harvard University",
                },
                is_duplicate=True,
                notes="University vs college within (related entities)",
            ),
        ]


# Singleton instance
_validation_service: Optional[ValidationService] = None


def get_validation_service() -> ValidationService:
    """
    Get singleton ValidationService instance.

    Returns:
        ValidationService: Shared validation instance
    """
    global _validation_service

    if _validation_service is None:
        _validation_service = ValidationService()

    return _validation_service
