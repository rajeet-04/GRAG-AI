"""Tests for the result ranking module with late fusion."""

from __future__ import annotations

import pytest

# Import directly from the module to avoid __init__.py pulling in chromadb
from app.retrieval.ranker import (
    RankingConfig,
    ScoredResult,
    _enrich_with_relations,
    merge_graph_and_vector,
    normalize_distance_to_similarity,
    rank_retrieval_results,
)


class TestDistanceToSimilarity:
    """Tests for distance-to-similarity conversion."""

    def test_zero_distance_is_perfect_similarity(self):
        assert normalize_distance_to_similarity(0.0) == 1.0

    def test_perfect_distance_is_zero_similarity(self):
        assert normalize_distance_to_similarity(1.0) == 0.0

    def test_half_distance(self):
        assert normalize_distance_to_similarity(0.5) == 0.5

    def test_quarter_distance(self):
        assert normalize_distance_to_similarity(0.25) == 0.75

    def test_clamped_above_one(self):
        assert normalize_distance_to_similarity(1.5) == 0.0

    def test_clamped_below_zero(self):
        # Negative distance → similarity > 1.0 → clamped to 1.0
        assert normalize_distance_to_similarity(-0.5) == 1.0

    def test_small_distance_near_one(self):
        assert normalize_distance_to_similarity(0.01) == pytest.approx(0.99)


class TestMergeGraphAndVector:
    """Tests for graph and vector result merging."""

    def test_empty_inputs(self):
        result = merge_graph_and_vector([], [])
        assert result == []

    def test_graph_only(self):
        graph = [
            {"entity_id": "e1", "confidence": 0.9},
            {"entity_id": "e2", "confidence": 0.8},
        ]
        result = merge_graph_and_vector(graph, [])
        assert len(result) == 2
        assert result[0]["entity_id"] == "e1"  # Higher confidence first
        assert result[0]["source"] == "graph"
        assert result[0]["unified_score"] == 0.9
        assert result[1]["entity_id"] == "e2"

    def test_vector_only(self):
        vector = [
            {"id": "e1", "distance": 0.2},  # similarity = 0.8
            {"id": "e2", "distance": 0.4},  # similarity = 0.6
        ]
        result = merge_graph_and_vector([], vector)
        assert len(result) == 2
        assert result[0]["entity_id"] == "e1"  # Higher similarity first
        assert result[0]["source"] == "vector"

    def test_late_fusion_merge(self):
        """Entity in both sources gets merged score."""
        graph = [{"entity_id": "e1", "confidence": 0.9}]
        vector = [{"id": "e1", "distance": 0.2}]  # similarity = 0.8

        result = merge_graph_and_vector(graph, vector)
        assert len(result) == 1
        assert result[0]["source"] == "merged"
        # 0.6 * 0.9 + 0.4 * 0.8 = 0.54 + 0.32 = 0.86
        assert result[0]["unified_score"] == pytest.approx(0.86)

    def test_deduplication_by_entity_id(self):
        """Same entity in both sources should merge, not duplicate."""
        graph = [{"entity_id": "e1", "confidence": 0.9}]
        vector = [{"id": "e1", "distance": 0.2}]

        result = merge_graph_and_vector(graph, vector)
        assert len(result) == 1  # Not 2!

    def test_vector_only_penalized(self):
        """Vector-only results get penalty multiplier."""
        vector = [{"id": "e1", "distance": 0.0}]  # Perfect similarity = 1.0

        result = merge_graph_and_vector([], vector)
        # 1.0 * 0.7 = 0.7
        assert result[0]["unified_score"] == pytest.approx(0.7)

    def test_custom_weights(self):
        """Custom weight configuration."""
        graph = [{"entity_id": "e1", "confidence": 0.8}]
        vector = [{"id": "e1", "distance": 0.0}]  # similarity = 1.0

        config = RankingConfig(graph_weight=0.8, vector_weight=0.2)
        result = merge_graph_and_vector(graph, vector, config=config)

        # 0.8 * 0.8 + 0.2 * 1.0 = 0.64 + 0.2 = 0.84
        assert result[0]["unified_score"] == pytest.approx(0.84)

    def test_min_score_filter(self):
        """Results below min_unified_score are filtered out."""
        graph = [
            {"entity_id": "e1", "confidence": 0.9},
            {"entity_id": "e2", "confidence": 0.1},
        ]
        config = RankingConfig(min_unified_score=0.5)
        result = merge_graph_and_vector(graph, [], config=config)
        assert len(result) == 1
        assert result[0]["entity_id"] == "e1"

    def test_max_results_limit(self):
        """Results are capped at max_results."""
        graph = [
            {"entity_id": f"e{i}", "confidence": 0.5 + i * 0.01} for i in range(100)
        ]
        config = RankingConfig(max_results=10)
        result = merge_graph_and_vector(graph, [], config=config)
        assert len(result) == 10

    def test_missing_entity_id_skipped(self):
        """Results without entity_id are skipped."""
        graph = [{"confidence": 0.9}]  # No entity_id or id
        vector = [{"distance": 0.2}]  # No id or entity_id

        result = merge_graph_and_vector(graph, vector)
        assert len(result) == 0

    def test_mixed_entity_id_keys(self):
        """Handles both entity_id and id keys."""
        graph = [{"entity_id": "e1", "confidence": 0.9}]
        vector = [{"id": "e2", "distance": 0.1}]

        result = merge_graph_and_vector(graph, vector)
        assert len(result) == 2
        ids = {r["entity_id"] for r in result}
        assert ids == {"e1", "e2"}

    def test_vector_similarity_field(self):
        """Handles similarity field directly (not just distance)."""
        vector = [{"id": "e1", "similarity": 0.8}]

        result = merge_graph_and_vector([], vector)
        assert len(result) == 1
        # 0.8 * 0.7 (penalty) = 0.56
        assert result[0]["unified_score"] == pytest.approx(0.56)

    def test_multiple_graph_and_vector(self):
        """Complex scenario with multiple results from both sources."""
        graph = [
            {"entity_id": "alice", "confidence": 0.95},
            {"entity_id": "bob", "confidence": 0.80},
            {"entity_id": "charlie", "confidence": 0.70},
        ]
        vector = [
            {"id": "bob", "distance": 0.1},  # similarity = 0.9
            {"id": "diana", "distance": 0.3},  # similarity = 0.7
            {"id": "charlie", "distance": 0.5},  # similarity = 0.5
        ]

        result = merge_graph_and_vector(graph, vector)
        assert len(result) == 4  # alice, bob (merged), charlie (merged), diana (vector)

        # Check bob is merged
        bob = next(r for r in result if r["entity_id"] == "bob")
        assert bob["source"] == "merged"
        # 0.6 * 0.80 + 0.4 * 0.9 = 0.48 + 0.36 = 0.84
        assert bob["unified_score"] == pytest.approx(0.84)

        # Check diana is vector-only
        diana = next(r for r in result if r["entity_id"] == "diana")
        assert diana["source"] == "vector"
        assert diana["unified_score"] == pytest.approx(0.7 * 0.7)


class TestRankRetrievalResults:
    """Tests for the full ranking pipeline."""

    def test_empty_results(self):
        ranked, relations = rank_retrieval_results([], [], [], [])
        assert ranked == []
        assert relations == []

    def test_combines_episodic_and_semantic(self):
        """Episodic and semantic are combined as vector results."""
        kr_entities = [{"entity_id": "e1", "confidence": 0.9}]
        kr_relations = []
        episodic = [{"id": "ep1", "distance": 0.1}]
        semantic = [{"id": "sp1", "distance": 0.2}]

        ranked, _ = rank_retrieval_results(
            kr_entities, kr_relations, episodic, semantic
        )
        # e1 (graph) + ep1 (vector) + sp1 (vector) = 3 unique
        assert len(ranked) == 3

    def test_relations_enriched(self):
        """Relations are enriched with reasoning paths."""
        kr_entities = [{"entity_id": "e1", "confidence": 0.9}]
        kr_relations = [
            {"source_id": "e1", "target_id": "e2", "relation_type": "KNOWS"},
        ]
        episodic = []
        semantic = []

        ranked, enriched_relations = rank_retrieval_results(
            kr_entities, kr_relations, episodic, semantic
        )
        assert len(ranked) == 1
        # Enriched relations have reasoning_path populated
        assert "reasoning_path" in enriched_relations[0]
        assert len(enriched_relations[0]["reasoning_path"]) == 1
        assert enriched_relations[0]["reasoning_path"][0]["source_id"] == "e1"


class TestEnrichWithRelations:
    """Tests for relation enrichment."""

    def test_empty_relations(self):
        entities = [
            ScoredResult(
                entity_id="e1",
                entity_name="Entity 1",
                entity_type="Person",
                unified_score=0.9,
                source="graph",
                confidence_graph=0.9,
                similarity_vector=None,
                path=None,
                reasoning_path=[],
            )
        ]
        result = _enrich_with_relations(entities, [])
        assert result[0]["reasoning_path"] == []

    def test_relations_matched_by_source_and_target(self):
        entities = [
            ScoredResult(
                entity_id="e1",
                entity_name="Entity 1",
                entity_type="Person",
                unified_score=0.9,
                source="graph",
                confidence_graph=0.9,
                similarity_vector=None,
                path=None,
                reasoning_path=[],
            )
        ]
        relations = [
            {"source_id": "e1", "target_id": "e2", "relation_type": "KNOWS"},
            {"source_id": "e3", "target_id": "e1", "relation_type": "WORKS_WITH"},
        ]
        result = _enrich_with_relations(entities, relations)
        assert len(result[0]["reasoning_path"]) == 2
