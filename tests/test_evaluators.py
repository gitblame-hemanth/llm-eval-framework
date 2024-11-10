"""Tests for evaluator base classes, results, and summary."""

from __future__ import annotations

import pytest

from src.evaluators.base import EvalResult, EvaluationSummary
from src.evaluators.comparative import ComparisonResult
from src.evaluators.llm import LLMEvaluator
from src.evaluators.rag import RAGEvaluator

# ===================================================================
# EvalResult
# ===================================================================


class TestEvalResult:
    def test_creation(self, sample_eval_result):
        r = sample_eval_result
        assert r.test_id == "tc-001"
        assert r.model_used == "gpt-4o"
        assert r.latency_ms == 142.5
        assert "bleu_score" in r.scores

    def test_passed_all_above_threshold(self):
        r = EvalResult(
            test_id="t1",
            scores={"a": 0.8, "b": 0.6},
            details={"pass_threshold": 0.5},
            latency_ms=10.0,
            model_used="test",
        )
        assert r.passed is True

    def test_failed_below_threshold(self):
        r = EvalResult(
            test_id="t1",
            scores={"a": 0.8, "b": 0.3},
            details={"pass_threshold": 0.5},
            latency_ms=10.0,
            model_used="test",
        )
        assert r.passed is False

    def test_passed_default_threshold(self):
        """Default threshold is 0.5 when not specified in details."""
        r = EvalResult(
            test_id="t1",
            scores={"a": 0.5},
            details={},
            latency_ms=10.0,
            model_used="test",
        )
        assert r.passed is True

    def test_timestamp_auto_set(self):
        r = EvalResult(
            test_id="t1",
            scores={},
            details={},
            latency_ms=0.0,
            model_used="test",
        )
        assert r.timestamp  # should be non-empty ISO string


# ===================================================================
# EvaluationSummary
# ===================================================================


class TestEvaluationSummary:
    def test_from_results(self, sample_results):
        summary = EvaluationSummary.from_results(sample_results)
        assert summary.total == 3
        # tc-001 passes (0.9>=0.5, 0.95>=0.5), tc-002 passes, tc-003 fails (0.3<0.5)
        assert summary.passed == 2
        assert summary.failed == 1

    def test_avg_scores(self, sample_results):
        summary = EvaluationSummary.from_results(sample_results)
        # bleu: (0.9+0.6+0.3)/3 = 0.6
        assert summary.avg_scores["bleu_score"] == pytest.approx(0.6, abs=0.001)
        # rouge_l: (0.95+0.7+0.4)/3 = 0.6833
        assert summary.avg_scores["rouge_l"] == pytest.approx(0.6833, abs=0.001)

    def test_min_max_scores(self, sample_results):
        summary = EvaluationSummary.from_results(sample_results)
        assert summary.min_scores["bleu_score"] == pytest.approx(0.3)
        assert summary.max_scores["bleu_score"] == pytest.approx(0.9)

    def test_empty_results(self):
        summary = EvaluationSummary.from_results([])
        assert summary.total == 0
        assert summary.passed == 0
        assert summary.avg_scores == {}

    def test_per_metric_stats_keys(self, sample_results):
        summary = EvaluationSummary.from_results(sample_results)
        stats = summary.per_metric_stats["bleu_score"]
        assert "mean" in stats
        assert "median" in stats
        assert "std_dev" in stats
        assert "min" in stats
        assert "max" in stats
        assert "count" in stats
        assert "pass_rate" in stats


# ===================================================================
# RAGEvaluator init
# ===================================================================


class TestRAGEvaluator:
    def test_init_defaults(self):
        evaluator = RAGEvaluator(model_name="test-model")
        assert evaluator.model_name == "test-model"

    def test_init_with_providers(self):
        llm_fn = lambda prompt: "response"
        embed_fn = lambda text: [0.1, 0.2, 0.3]
        evaluator = RAGEvaluator(
            model_name="test-model",
            llm_provider=llm_fn,
            embed_provider=embed_fn,
        )
        assert evaluator.model_name == "test-model"


# ===================================================================
# LLMEvaluator init
# ===================================================================


class TestLLMEvaluator:
    def test_init_defaults(self):
        evaluator = LLMEvaluator(model_name="test-model")
        assert evaluator.model_name == "test-model"

    def test_init_with_providers(self):
        llm_fn = lambda prompt: "response"
        embed_fn = lambda text: [0.1, 0.2, 0.3]
        model_fn = lambda question: "answer"
        evaluator = LLMEvaluator(
            model_name="test-model",
            llm_provider=llm_fn,
            embed_provider=embed_fn,
            model_provider=model_fn,
            metrics=["factual_accuracy", "hallucination_score"],
        )
        assert evaluator.model_name == "test-model"


# ===================================================================
# ComparisonResult
# ===================================================================


class TestComparisonResult:
    def test_structure(self):
        cr = ComparisonResult(
            model_a_scores={"bleu": [0.8, 0.9]},
            model_b_scores={"bleu": [0.6, 0.7]},
            wins={"bleu": 2},
            losses={"bleu": 0},
            ties={"bleu": 0},
            statistical_significance={"bleu": 0.03},
            effect_size={"bleu": 0.8},
            overall_winner="model_a",
        )
        assert cr.overall_winner == "model_a"
        assert cr.wins["bleu"] == 2

    def test_summary_property(self):
        cr = ComparisonResult(
            model_a_scores={"bleu": [0.8, 0.9]},
            model_b_scores={"bleu": [0.6, 0.7]},
            wins={"bleu": 2},
            losses={"bleu": 0},
            ties={"bleu": 0},
            statistical_significance={"bleu": 0.03},
            effect_size={"bleu": 0.8},
            overall_winner="model_a",
        )
        summary = cr.summary
        assert summary["bleu"]["significant"] is True
        assert summary["overall_winner"] == "model_a"
