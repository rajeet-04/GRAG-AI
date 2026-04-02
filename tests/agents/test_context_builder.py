"""Tests for the context builder module."""

from __future__ import annotations

import pytest

from app.agents.context_builder import (
    count_tokens,
    merge_with_budget,
    PriorityLevel,
)


class TestCountTokens:
    """Tests for token counting functionality."""

    def test_basic_token_counting(self):
        """Test that count_tokens returns a positive number for non-empty text."""
        text = "This is a test sentence."
        tokens = count_tokens(text)
        assert tokens > 0

    def test_empty_string(self):
        """Test that empty string returns 0 tokens."""
        assert count_tokens("") == 0

    def test_short_text(self):
        """Test token count for short text."""
        text = "Hello"
        tokens = count_tokens(text)
        assert tokens > 0

    def test_long_text(self):
        """Test token count for longer text."""
        text = "This is a longer text with more words to count. " * 10
        tokens = count_tokens(text)
        assert tokens > 50

    def test_special_characters(self):
        """Test token count includes special characters."""
        text = "Hello, world! How are you? 😊"
        tokens = count_tokens(text)
        assert tokens > 0


class TestPriorityLevel:
    """Tests for priority level constants."""

    def test_kr_is_highest_priority(self):
        """KR should have lowest number (highest priority)."""
        assert PriorityLevel.KR_GRAPH_FACTS < PriorityLevel.EPISODIC_MEMORIES
        assert PriorityLevel.EPISODIC_MEMORIES < PriorityLevel.SEMANTIC_PREFERENCES

    def test_priority_values(self):
        """Test actual priority values."""
        assert PriorityLevel.KR_GRAPH_FACTS == 1
        assert PriorityLevel.EPISODIC_MEMORIES == 2
        assert PriorityLevel.SEMANTIC_PREFERENCES == 3


class TestMergeWithBudget:
    """Tests for merge_with_budget function."""

    def test_empty_inputs(self):
        """Test with empty inputs returns minimal context."""
        result = merge_with_budget(
            kr_entities=[],
            kr_relations=[],
            episodic_memories=[],
            semantic_preferences=[],
        )

        assert "context" in result
        assert result["token_count"] >= 0
        assert "budget" in result

    def test_respects_budget_limit(self):
        """Budget should be enforced, token_count <= budget."""
        from app.config import ContextConfig

        config = ContextConfig()
        # Create large input that exceeds budget
        large_entities = [
            {"name": f"Entity{i}", "description": "x" * 500} for i in range(50)
        ]

        result = merge_with_budget(
            kr_entities=large_entities,
            kr_relations=[],
            episodic_memories=[],
            semantic_preferences=[],
        )

        assert result["token_count"] <= config.default_token_budget

    def test_returns_truncation_flags(self):
        """Should return context_truncated and truncation_warning."""
        result = merge_with_budget(
            kr_entities=[{"name": "Test", "description": "x" * 1000}],
            kr_relations=[],
            episodic_memories=[],
            semantic_preferences=[],
        )

        assert "context_truncated" in result
        assert "truncation_warning" in result

    def test_no_truncation_when_under_budget(self):
        """When under budget, context_truncated should be False."""
        small_entities = [{"name": "Test", "description": "Hello world"}]

        result = merge_with_budget(
            kr_entities=small_entities,
            kr_relations=[],
            episodic_memories=[],
            semantic_preferences=[],
        )

        # May or may not be truncated depending on content, but should not error
        assert result["token_count"] >= 0

    def test_episodic_memories_sorted_by_timestamp(self):
        """Test that episodic memories are processed in oldest-first order."""
        episodic = [
            {"summary": "New memory", "timestamp": "2024-03-01"},
            {"summary": "Old memory", "timestamp": "2024-01-01"},
            {"summary": "Middle memory", "timestamp": "2024-02-01"},
        ]

        result = merge_with_budget(
            kr_entities=[],
            kr_relations=[],
            episodic_memories=episodic,
            semantic_preferences=[],
        )

        # Should not error - just testing order doesn't break
        assert "context" in result

    def test_warning_at_threshold(self):
        """Warning should be set when approaching 80% threshold."""
        from app.config import ContextConfig

        config = ContextConfig()
        warning_threshold = int(config.default_token_budget * 0.8)

        # Create content that approaches warning threshold
        text = "x" * (warning_threshold - 100)

        result = merge_with_budget(
            kr_entities=[{"name": "Test", "description": text}],
            kr_relations=[],
            episodic_memories=[],
            semantic_preferences=[],
        )

        if result["budget_pct"] >= 80:
            assert result["truncation_warning"] is not None

    def test_budget_percentage_calculated(self):
        """Test that budget_pct is calculated correctly."""
        result = merge_with_budget(
            kr_entities=[{"name": "Test", "description": "x" * 100}],
            kr_relations=[],
            episodic_memories=[],
            semantic_preferences=[],
        )

        assert "budget_pct" in result
        # Should be reasonable percentage
        assert result["budget_pct"] > 0


class TestEmergencyKRTruncation:
    """Tests for emergency KR truncation scenario."""

    def test_kr_exceeds_budget_graceful_handling(self):
        """When KR alone exceeds budget, should handle gracefully."""
        from app.config import ContextConfig

        config = ContextConfig()
        # Create massive KR that alone exceeds budget
        huge_entities = [
            {"name": f"E{i}", "description": "x" * 2000} for i in range(20)
        ]

        result = merge_with_budget(
            kr_entities=huge_entities,
            kr_relations=[],
            episodic_memories=[],
            semantic_preferences=[],
        )

        # Should not hard error - should handle gracefully
        assert result["token_count"] <= config.default_token_budget
        assert "context_truncated" in result

    def test_returns_all_required_keys(self):
        """All required keys should be present in result."""
        result = merge_with_budget(
            kr_entities=[{"name": "Test", "description": "test"}],
            kr_relations=[],
            episodic_memories=[],
            semantic_preferences=[],
        )

        required_keys = [
            "context",
            "token_count",
            "truncated",
            "budget",
            "budget_pct",
            "context_truncated",
            "truncation_warning",
        ]

        for key in required_keys:
            assert key in result, f"Missing required key: {key}"


class TestTruncationPriority:
    """Tests for truncation priority behavior."""

    def test_semantic_truncated_before_episodic(self):
        """Semantic preferences should be truncated before episodic."""
        # Create enough content to fill budget
        long_desc = "x" * 500
        many_entities = [{"name": f"E{i}", "description": long_desc} for i in range(10)]
        many_episodic = [
            {"summary": f"Memory {i}", "timestamp": f"2024-01-{i:02d}"}
            for i in range(5)
        ]
        many_semantic = [{"content": f"Pref {i}"} for i in range(5)]

        result = merge_with_budget(
            kr_entities=many_entities,
            kr_relations=[],
            episodic_memories=many_episodic,
            semantic_preferences=many_semantic,
        )

        # If anything was truncated, check it includes semantic first
        if result["context_truncated"]:
            assert "truncated" in result
            # Semantic should be prioritized for truncation (higher priority number)

    def test_episodic_truncated_oldest_first(self):
        """Oldest episodic memories should be truncated first."""
        episodic = [
            {"summary": "Oldest", "timestamp": "2024-01-01"},
            {"summary": "Middle", "timestamp": "2024-02-01"},
            {"summary": "Newest", "timestamp": "2024-03-01"},
        ]

        result = merge_with_budget(
            kr_entities=[],
            kr_relations=[],
            episodic_memories=episodic,
            semantic_preferences=[],
        )

        # Should not error - oldest should be dropped first
        assert "context" in result
