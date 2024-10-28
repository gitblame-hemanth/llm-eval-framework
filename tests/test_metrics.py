"""Tests for built-in metrics and custom metric registry."""

from __future__ import annotations

import numpy as np
import pytest

from src.metrics.builtin import (
    bleu_score,
    cosine_similarity,
    exact_match,
    f1_token_overlap,
    rouge_l,
)
from src.metrics.custom import MetricRegistry, register_metric

# ===================================================================
# BLEU Score
# ===================================================================


class TestBleuScore:
    def test_perfect_match(self):
        text = "The quick brown fox jumps over the lazy dog"
        assert bleu_score(text, text) == pytest.approx(1.0)

    def test_no_match(self):
        ref = "The quick brown fox"
        hyp = "completely unrelated sentence here"
        assert bleu_score(ref, hyp) == pytest.approx(0.0)

    def test_partial_match(self):
        ref = "The quick brown fox jumps over the lazy dog"
        hyp = "The quick brown cat jumps over the lazy frog"
        score = bleu_score(ref, hyp)
        assert 0.0 < score < 1.0

    def test_empty_reference(self):
        assert bleu_score("", "some text") == 0.0

    def test_empty_hypothesis(self):
        assert bleu_score("some text", "") == 0.0


# ===================================================================
# ROUGE-L
# ===================================================================


class TestRougeL:
    def test_perfect_match(self):
        text = "The quick brown fox jumps over the lazy dog"
        assert rouge_l(text, text) == pytest.approx(1.0)

    def test_no_match(self):
        ref = "alpha beta gamma"
        hyp = "delta epsilon zeta"
        assert rouge_l(ref, hyp) == pytest.approx(0.0)

    def test_partial_match(self):
        ref = "The cat sat on the mat"
        hyp = "The cat is on the mat"
        score = rouge_l(ref, hyp)
        assert 0.0 < score < 1.0

    def test_empty_strings(self):
        assert rouge_l("", "something") == 0.0
        assert rouge_l("something", "") == 0.0


# ===================================================================
# Cosine Similarity
# ===================================================================


class TestCosineSimilarity:
    def test_identical_vectors(self):
        vec = np.array([1.0, 2.0, 3.0])
        assert cosine_similarity(vec, vec) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])
        assert cosine_similarity(a, b) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        a = np.array([1.0, 0.0])
        b = np.array([-1.0, 0.0])
        assert cosine_similarity(a, b) == pytest.approx(-1.0)

    def test_different_lengths_raises(self):
        a = np.array([1.0, 2.0])
        b = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="dimensions must match"):
            cosine_similarity(a, b)

    def test_zero_vector(self):
        a = np.array([0.0, 0.0])
        b = np.array([1.0, 2.0])
        assert cosine_similarity(a, b) == pytest.approx(0.0)


# ===================================================================
# Exact Match
# ===================================================================


class TestExactMatch:
    def test_true_match(self):
        assert exact_match("Paris", "Paris") == 1.0

    def test_false_match(self):
        assert exact_match("Paris", "London") == 0.0

    def test_case_insensitive(self):
        assert exact_match("PARIS", "paris") == 1.0

    def test_whitespace_stripped(self):
        assert exact_match("  Paris  ", "Paris") == 1.0

    def test_none_reference(self):
        assert exact_match(None, "Paris") == 0.0


# ===================================================================
# F1 Token Overlap
# ===================================================================


class TestF1TokenOverlap:
    def test_perfect_overlap(self):
        text = "the quick brown fox"
        assert f1_token_overlap(text, text) == pytest.approx(1.0)

    def test_partial_overlap(self):
        ref = "the quick brown fox"
        hyp = "the slow brown cat"
        score = f1_token_overlap(ref, hyp)
        # "the" and "brown" overlap -> 2 overlap tokens
        # precision = 2/4, recall = 2/4, f1 = 0.5
        assert score == pytest.approx(0.5)

    def test_no_overlap(self):
        ref = "alpha beta gamma"
        hyp = "delta epsilon zeta"
        assert f1_token_overlap(ref, hyp) == pytest.approx(0.0)

    def test_empty_strings(self):
        assert f1_token_overlap("", "something") == 0.0
        assert f1_token_overlap("something", "") == 0.0


# ===================================================================
# Metric Registry
# ===================================================================


class TestMetricRegistry:
    def setup_method(self):
        """Reset singleton before each test."""
        MetricRegistry.reset()

    def teardown_method(self):
        MetricRegistry.reset()

    def test_register_and_get(self):
        registry = MetricRegistry()
        fn = lambda r, h: 1.0
        registry.register("test_metric", fn)
        assert registry.get("test_metric") is fn

    def test_list_registered(self):
        registry = MetricRegistry()
        registry.register("metric_b", lambda r, h: 0.5)
        registry.register("metric_a", lambda r, h: 0.5)
        names = registry.list()
        assert names == ["metric_a", "metric_b"]  # sorted

    def test_get_missing_raises(self):
        registry = MetricRegistry()
        with pytest.raises(KeyError, match="not found"):
            registry.get("nonexistent")

    def test_register_decorator(self):
        @register_metric("decorated_metric")
        def my_metric(response: str, reference: str, **kwargs) -> float:
            return 0.42

        registry = MetricRegistry()
        assert registry.has("decorated_metric")
        assert registry.get("decorated_metric")("a", "b") == 0.42

    def test_unregister(self):
        registry = MetricRegistry()
        registry.register("temp", lambda r, h: 0.0)
        assert registry.has("temp")
        registry.unregister("temp")
        assert not registry.has("temp")

    def test_clear(self):
        registry = MetricRegistry()
        registry.register("m1", lambda r, h: 0.0)
        registry.register("m2", lambda r, h: 0.0)
        registry.clear()
        assert registry.list() == []
