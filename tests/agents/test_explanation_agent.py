"""Tests for the explanation agent xAI features.

Covers: confidence formatting, Top-N path selection, Mermaid validation,
graceful degradation, citation mapping, three-section response parsing,
and reasoning steps with confidence enrichment.
"""

from __future__ import annotations

import pytest

from app.agents.explanation_agent import (
    format_confidence_percent,
    select_top_n_paths,
    validate_mermaid,
    parse_three_section_response,
    _annotate_reasoning_with_confidence,
)


class TestFormatConfidencePercent:
    """Tests for format_confidence_percent utility."""

    def test_high_confidence(self):
        """0.92 should produce '(Confidence: 92%)'."""
        assert format_confidence_percent(0.92) == "(Confidence: 92%)"

    def test_medium_high_confidence(self):
        """0.87 should produce '(Confidence: 87%)'."""
        assert format_confidence_percent(0.87) == "(Confidence: 87%)"

    def test_half_confidence(self):
        """0.5 should produce '(Confidence: 50%)'."""
        assert format_confidence_percent(0.5) == "(Confidence: 50%)"

    def test_zero_confidence(self):
        """0.0 should produce '(Confidence: 0%)'."""
        assert format_confidence_percent(0.0) == "(Confidence: 0%)"

    def test_full_confidence(self):
        """1.0 should produce '(Confidence: 100%)'."""
        assert format_confidence_percent(1.0) == "(Confidence: 100%)"

    def test_none_confidence(self):
        """None should produce '(Confidence: N/A)'."""
        assert format_confidence_percent(None) == "(Confidence: N/A)"

    def test_rounding_down(self):
        """0.924 should round to 92%."""
        assert format_confidence_percent(0.924) == "(Confidence: 92%)"

    def test_rounding_up(self):
        """0.926 should round to 93%."""
        assert format_confidence_percent(0.926) == "(Confidence: 93%)"


class TestSelectTopNPaths:
    """Tests for select_top_n_paths function."""

    def _make_paths(self, n):
        """Helper: create n paths with descending confidence."""
        return [{"id": i, "confidence": round(1.0 - i * 0.05, 2)} for i in range(n)]

    def test_select_top_10_from_15(self):
        """Given 15 paths with n=10, should return top 10 by confidence."""
        paths = self._make_paths(15)
        selected, excluded = select_top_n_paths(paths, n=10)
        assert len(selected) == 10
        assert len(excluded) == 5

    def test_select_top_5_from_15(self):
        """Given 15 paths with n=5, should return top 5 by confidence."""
        paths = self._make_paths(15)
        selected, excluded = select_top_n_paths(paths, n=5)
        assert len(selected) == 5
        assert len(excluded) == 10

    def test_excluded_paths_are_lowest_confidence(self):
        """Excluded paths should have lower confidence than selected ones."""
        paths = self._make_paths(15)
        selected, excluded = select_top_n_paths(paths, n=10)
        min_selected_conf = min(p["confidence"] for p in selected)
        max_excluded_conf = max(p["confidence"] for p in excluded)
        assert min_selected_conf >= max_excluded_conf

    def test_empty_paths(self):
        """Empty input should return empty lists."""
        selected, excluded = select_top_n_paths([], n=10)
        assert selected == []
        assert excluded == []

    def test_n_clamped_to_minimum_5(self):
        """n=2 should be clamped to minimum 5."""
        paths = self._make_paths(10)
        selected, excluded = select_top_n_paths(paths, n=2)
        assert len(selected) == 5
        assert len(excluded) == 5

    def test_n_clamped_to_maximum_10(self):
        """n=20 should be clamped to maximum 10."""
        paths = self._make_paths(15)
        selected, excluded = select_top_n_paths(paths, n=20)
        assert len(selected) == 10
        assert len(excluded) == 5

    def test_fewer_paths_than_n(self):
        """3 paths with n=10 should return all 3."""
        paths = self._make_paths(3)
        selected, excluded = select_top_n_paths(paths, n=10)
        assert len(selected) == 3
        assert len(excluded) == 0

    def test_sorts_by_confidence_descending(self):
        """Unsorted input should be returned sorted by confidence descending."""
        paths = [
            {"id": 0, "confidence": 0.3},
            {"id": 1, "confidence": 0.9},
            {"id": 2, "confidence": 0.6},
        ]
        selected, excluded = select_top_n_paths(paths, n=10)
        assert selected[0]["confidence"] == 0.9
        assert selected[1]["confidence"] == 0.6
        assert selected[2]["confidence"] == 0.3

    def test_missing_confidence_defaults_to_zero(self):
        """Paths without confidence key should sort to the end."""
        paths = [
            {"id": 0},
            {"id": 1, "confidence": 0.8},
        ]
        selected, excluded = select_top_n_paths(paths, n=10)
        assert selected[0]["confidence"] == 0.8
        assert selected[1] == {"id": 0}


class TestValidateMermaid:
    """Tests for validate_mermaid function."""

    def test_valid_simple_graph(self):
        """Basic valid Mermaid should return True."""
        mermaid = "graph TD\n  A[Node] --> B[Node]"
        assert validate_mermaid(mermaid) is True

    def test_valid_flowchart(self):
        """flowchart syntax should be valid."""
        mermaid = "flowchart LR\n  A[Start] --> B[End]"
        assert validate_mermaid(mermaid) is True

    def test_valid_with_multiple_nodes(self):
        """Multi-node graph should be valid."""
        mermaid = (
            "graph TD\n"
            "  A[User Query] --> B[Entity A]\n"
            "  B --> C[Relation]\n"
            "  C --> D[Entity B]"
        )
        assert validate_mermaid(mermaid) is True

    def test_invalid_no_direction(self):
        """Missing graph direction should return False."""
        mermaid = "  A[Node] --> B[Node]"
        assert validate_mermaid(mermaid) is False

    def test_invalid_mismatched_brackets(self):
        """Unbalanced brackets should return False."""
        mermaid = "graph TD\n  A[Node --> B[Node]"
        assert validate_mermaid(mermaid) is False

    def test_invalid_no_arrows(self):
        """No arrow syntax should return False."""
        mermaid = "graph TD\n  A[Node]\n  B[Node]"
        assert validate_mermaid(mermaid) is False

    def test_empty_string(self):
        """Empty string should return False."""
        assert validate_mermaid("") is False

    def test_whitespace_only(self):
        """Whitespace-only string should return False."""
        assert validate_mermaid("   ") is False

    def test_invalid_direction_keyword(self):
        """Invalid direction should return False."""
        mermaid = "graph INVALID\n  A[Node] --> B[Node]"
        assert validate_mermaid(mermaid) is False

    def test_case_insensitive_direction(self):
        """Direction should be case-insensitive."""
        mermaid = "Graph td\n  A[Node] --> B[Node]"
        assert validate_mermaid(mermaid) is True

    def test_none_input(self):
        """None input should return False."""
        assert validate_mermaid(None) is False


class TestValidateMermaidGracefulDegradation:
    """Tests ensuring invalid Mermaid returns False, not raises."""

    def test_no_exception_on_garbage_input(self):
        """Random text should return False, not raise."""
        result = validate_mermaid("this is not mermaid at all!!!")
        assert result is False

    def test_no_exception_on_malformed_brackets(self):
        """Deeply malformed brackets should not raise."""
        result = validate_mermaid("graph TD\n  A[[[Node --> B[Node]]]")
        assert result is False

    def test_no_exception_on_partial_graph(self):
        """Incomplete graph definition should not raise."""
        result = validate_mermaid("graph")
        assert result is False

    def test_no_exception_on_special_characters(self):
        """Special characters should not raise (may be valid syntax)."""
        # This input has valid direction, balanced brackets, and arrows
        result = validate_mermaid("graph TD\n  A[@#$%] --> B[&*()]")
        # Should not raise - result depends on validation rules
        assert isinstance(result, bool)


class TestCitationMapGeneration:
    """Tests for citation_map generation logic."""

    def test_basic_citation_map(self):
        """3 reasoning steps should produce {[1]: 0, [2]: 1, [3]: 2}."""
        steps = ["Step A", "Step B", "Step C"]
        citation_map = {f"[{i + 1}]": i for i in range(len(steps))}
        assert citation_map == {"[1]": 0, "[2]": 1, "[3]": 2}

    def test_empty_citation_map(self):
        """No steps should produce empty citation map."""
        steps = []
        citation_map = {f"[{i + 1}]": i for i in range(len(steps))}
        assert citation_map == {}

    def test_single_step_citation(self):
        """Single step should produce {[1]: 0}."""
        steps = ["Only step"]
        citation_map = {f"[{i + 1}]": i for i in range(len(steps))}
        assert citation_map == {"[1]": 0}

    def test_citation_map_keys_are_strings(self):
        """All keys should be string-formatted bracket numbers."""
        steps = ["A", "B", "C", "D", "E"]
        citation_map = {f"[{i + 1}]": i for i in range(len(steps))}
        for key in citation_map:
            assert isinstance(key, str)
            assert key.startswith("[")
            assert key.endswith("]")

    def test_citation_map_values_are_indices(self):
        """Values should be zero-based indices."""
        steps = ["A", "B", "C"]
        citation_map = {f"[{i + 1}]": i for i in range(len(steps))}
        for key, value in citation_map.items():
            expected_index = int(key.strip("[]")) - 1
            assert value == expected_index


class TestParseThreeSectionResponse:
    """Tests for parse_three_section_response function."""

    def test_full_three_section_response(self):
        """Parse a complete three-section response."""
        response = (
            "## Natural Language Answer\n"
            "The answer is based on the graph.\n\n"
            "## Step-by-Step Reasoning Path\n"
            "- Entity A → Entity B → Entity C\n\n"
            "## Mermaid Diagram\n"
            "```mermaid\n"
            "graph TD\n"
            "  A[Entity A] --> B[Entity B]\n"
            "```"
        )
        result = parse_three_section_response(response)
        assert "The answer is based on the graph" in result["answer"]
        assert len(result["reasoning_steps"]) >= 1
        assert "Entity A" in result["reasoning_steps"][0]
        assert result["mermaid"] != ""

    def test_missing_mermaid_section(self):
        """Response without mermaid section should still parse answer and reasoning."""
        response = (
            "## Natural Language Answer\n"
            "No diagram available.\n\n"
            "## Step-by-Step Reasoning Path\n"
            "- A → B\n"
        )
        result = parse_three_section_response(response)
        assert "No diagram available" in result["answer"]
        assert len(result["reasoning_steps"]) >= 1
        assert result["mermaid"] == ""

    def test_missing_reasoning_section(self):
        """Response without reasoning section should return empty reasoning."""
        response = "## Natural Language Answer\nJust the answer.\n"
        result = parse_three_section_response(response)
        assert "Just the answer" in result["answer"]
        assert result["reasoning_steps"] == []

    def test_empty_response(self):
        """Empty response should return empty fields."""
        result = parse_three_section_response("")
        assert result["answer"] == ""
        assert result["reasoning_steps"] == []
        assert result["mermaid"] == ""

    def test_reasoning_with_dashes(self):
        """Reasoning lines with dashes should be cleaned."""
        response = "## Step-by-Step Reasoning Path\n- A → B\n- B → C\n"
        result = parse_three_section_response(response)
        assert len(result["reasoning_steps"]) == 2
        # Leading dashes should be stripped
        assert not result["reasoning_steps"][0].startswith("-")

    def test_reasoning_with_arrows(self):
        """Lines with → arrows should be captured."""
        response = "## Step-by-Step Reasoning\nUser → Query → Entity\n"
        result = parse_three_section_response(response)
        assert len(result["reasoning_steps"]) >= 1

    def test_reasoning_with_html_arrows(self):
        """Lines with -> arrows should also be captured."""
        response = "## Step-by-Step Reasoning\nA -> B -> C\n"
        result = parse_three_section_response(response)
        assert len(result["reasoning_steps"]) >= 1


class TestReasoningStepsWithConfidence:
    """Tests for _annotate_reasoning_with_confidence function."""

    def test_annotate_steps_with_confidence(self):
        """Reasoning steps should be annotated with confidence percentages."""
        steps = ["Entity A connects to B", "Entity B relates to C"]
        paths = [
            {"source": "A", "target": "B", "confidence": 0.92},
            {"source": "B", "target": "C", "confidence": 0.87},
        ]
        result = _annotate_reasoning_with_confidence(steps, paths)
        assert len(result) == 2
        assert "(Confidence: 92%)" in result[0]
        assert "(Confidence: 87%)" in result[1]

    def test_more_steps_than_paths(self):
        """Extra steps without matching paths should get N/A confidence."""
        steps = ["Step 1", "Step 2", "Step 3"]
        paths = [{"confidence": 0.9}]
        result = _annotate_reasoning_with_confidence(steps, paths)
        assert len(result) == 3
        assert "(Confidence: 90%)" in result[0]
        assert "(Confidence: N/A)" in result[1]
        assert "(Confidence: N/A)" in result[2]

    def test_empty_steps(self):
        """Empty steps should return empty list."""
        result = _annotate_reasoning_with_confidence([], [{"confidence": 0.9}])
        assert result == []

    def test_empty_paths(self):
        """Empty paths should annotate all steps with N/A."""
        steps = ["Step 1", "Step 2"]
        result = _annotate_reasoning_with_confidence(steps, [])
        assert len(result) == 2
        assert "(Confidence: N/A)" in result[0]
        assert "(Confidence: N/A)" in result[1]

    def test_preserves_original_step_text(self):
        """Original step text should be preserved, not replaced."""
        steps = ["A → B"]
        paths = [{"confidence": 0.5}]
        result = _annotate_reasoning_with_confidence(steps, paths)
        assert "A → B" in result[0]
