"""Comparative evaluator for A/B testing models and prompts with statistical analysis."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import structlog
from scipy import stats as scipy_stats

from src.config import SuiteConfig, TestCase
from src.evaluators.base import BaseEvaluator, EvalResult

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

ModelCallable = Callable[[str], str]
"""question -> model response."""

PromptFormatter = Callable[[str, str], str]
"""(template, question) -> formatted prompt."""


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class ComparisonResult:
    """Result of comparing two models or prompts across a test suite."""

    model_a_scores: dict[str, list[float]]
    model_b_scores: dict[str, list[float]]
    wins: dict[str, int]
    losses: dict[str, int]
    ties: dict[str, int]
    statistical_significance: dict[str, float]
    effect_size: dict[str, float]
    overall_winner: str | None
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def summary(self) -> dict[str, Any]:
        """Human-readable summary of the comparison."""
        result: dict[str, Any] = {}
        for metric in self.wins:
            a_mean = _safe_mean(self.model_a_scores.get(metric, []))
            b_mean = _safe_mean(self.model_b_scores.get(metric, []))
            result[metric] = {
                "model_a_mean": round(a_mean, 4),
                "model_b_mean": round(b_mean, 4),
                "wins": self.wins[metric],
                "losses": self.losses[metric],
                "ties": self.ties[metric],
                "p_value": round(self.statistical_significance.get(metric, 1.0), 6),
                "effect_size": round(self.effect_size.get(metric, 0.0), 4),
                "significant": self.statistical_significance.get(metric, 1.0) < 0.05,
            }
        result["overall_winner"] = self.overall_winner
        return result


def _safe_mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _cohens_d(a: list[float], b: list[float]) -> float:
    """Compute Cohen's d effect size between two samples."""
    if len(a) < 2 or len(b) < 2:
        return 0.0

    mean_a = _safe_mean(a)
    mean_b = _safe_mean(b)

    var_a = sum((x - mean_a) ** 2 for x in a) / (len(a) - 1)
    var_b = sum((x - mean_b) ** 2 for x in b) / (len(b) - 1)

    # Pooled standard deviation
    pooled_std = math.sqrt(((len(a) - 1) * var_a + (len(b) - 1) * var_b) / (len(a) + len(b) - 2))

    if pooled_std == 0:
        return 0.0

    return (mean_a - mean_b) / pooled_std


def _paired_ttest(a: list[float], b: list[float]) -> float:
    """Paired t-test, returns p-value. Falls back to 1.0 on error."""
    if len(a) != len(b) or len(a) < 2:
        return 1.0

    # Check for zero variance in differences
    diffs = [x - y for x, y in zip(a, b)]
    if all(d == diffs[0] for d in diffs):
        return 1.0 if diffs[0] == 0 else 0.0

    try:
        _, p_value = scipy_stats.ttest_rel(a, b)
        if math.isnan(p_value):
            return 1.0
        return float(p_value)
    except Exception:
        return 1.0


# ---------------------------------------------------------------------------
# Comparative Evaluator
# ---------------------------------------------------------------------------


class ComparativeEvaluator:
    """A/B comparison of models or prompts with statistical significance testing.

    This evaluator runs a base evaluator against two variants (models or prompts)
    and produces a ComparisonResult with per-metric wins/losses/ties, p-values
    from paired t-tests, and Cohen's d effect sizes.

    Parameters
    ----------
    evaluator:
        A BaseEvaluator subclass instance used to score each response.
    significance_level:
        Threshold for statistical significance (default 0.05).
    tie_margin:
        Score difference below this is counted as a tie (default 0.01).
    """

    def __init__(
        self,
        evaluator: BaseEvaluator,
        significance_level: float = 0.05,
        tie_margin: float = 0.01,
    ) -> None:
        self._evaluator = evaluator
        self._significance = significance_level
        self._tie_margin = tie_margin
        self._log = logger.bind(evaluator="ComparativeEvaluator")

    # ------------------------------------------------------------------
    # Model comparison
    # ------------------------------------------------------------------

    def compare_models(
        self,
        test_suite: SuiteConfig | list[TestCase],
        model_a: ModelCallable,
        model_b: ModelCallable,
        metrics: list[str] | None = None,
    ) -> ComparisonResult:
        """Compare two models on the same test suite.

        Parameters
        ----------
        test_suite:
            A SuiteConfig or list of TestCase instances.
        model_a:
            Callable that generates responses for model A.
        model_b:
            Callable that generates responses for model B.
        metrics:
            Optional list of metric names to compare. If None, uses all
            metrics returned by the evaluator.

        Returns
        -------
        ComparisonResult
        """
        test_cases = test_suite.test_cases if isinstance(test_suite, SuiteConfig) else test_suite

        self._log.info(
            "compare_models_start",
            n_cases=len(test_cases),
        )

        results_a: list[EvalResult] = []
        results_b: list[EvalResult] = []

        for tc in test_cases:
            question = tc.question
            context = tc.context

            # Generate responses
            try:
                resp_a = model_a(question)
            except Exception as exc:
                self._log.error("model_a_error", test_id=tc.id, error=str(exc))
                resp_a = ""

            try:
                resp_b = model_b(question)
            except Exception as exc:
                self._log.error("model_b_error", test_id=tc.id, error=str(exc))
                resp_b = ""

            # Evaluate both
            result_a = self._evaluator.evaluate(tc, resp_a, context)
            result_b = self._evaluator.evaluate(tc, resp_b, context)

            results_a.append(result_a)
            results_b.append(result_b)

        return self._build_comparison(results_a, results_b, metrics)

    # ------------------------------------------------------------------
    # Prompt comparison
    # ------------------------------------------------------------------

    def compare_prompts(
        self,
        test_suite: SuiteConfig | list[TestCase],
        prompt_a: str,
        prompt_b: str,
        model: ModelCallable,
        metrics: list[str] | None = None,
    ) -> ComparisonResult:
        """Compare two prompt templates using the same model.

        Prompt templates should contain ``{question}`` as a placeholder.

        Parameters
        ----------
        test_suite:
            A SuiteConfig or list of TestCase instances.
        prompt_a:
            First prompt template with ``{question}`` placeholder.
        prompt_b:
            Second prompt template with ``{question}`` placeholder.
        model:
            Callable that generates responses given a formatted prompt.
        metrics:
            Optional list of metric names to compare.

        Returns
        -------
        ComparisonResult
        """
        test_cases = test_suite.test_cases if isinstance(test_suite, SuiteConfig) else test_suite

        self._log.info(
            "compare_prompts_start",
            n_cases=len(test_cases),
        )

        results_a: list[EvalResult] = []
        results_b: list[EvalResult] = []

        for tc in test_cases:
            question = tc.question
            context = tc.context

            # Format prompts
            formatted_a = prompt_a.format(question=question)
            formatted_b = prompt_b.format(question=question)

            # Generate responses
            try:
                resp_a = model(formatted_a)
            except Exception as exc:
                self._log.error("prompt_a_error", test_id=tc.id, error=str(exc))
                resp_a = ""

            try:
                resp_b = model(formatted_b)
            except Exception as exc:
                self._log.error("prompt_b_error", test_id=tc.id, error=str(exc))
                resp_b = ""

            result_a = self._evaluator.evaluate(tc, resp_a, context)
            result_b = self._evaluator.evaluate(tc, resp_b, context)

            results_a.append(result_a)
            results_b.append(result_b)

        return self._build_comparison(results_a, results_b, metrics)

    # ------------------------------------------------------------------
    # Internal — build ComparisonResult from paired eval results
    # ------------------------------------------------------------------

    def _build_comparison(
        self,
        results_a: list[EvalResult],
        results_b: list[EvalResult],
        metrics: list[str] | None = None,
    ) -> ComparisonResult:
        """Aggregate paired results into a ComparisonResult with stats."""
        # Collect all metric names if not specified
        all_metrics: set[str] = set()
        for r in results_a + results_b:
            all_metrics.update(r.scores.keys())

        if metrics:
            all_metrics = all_metrics & set(metrics)

        model_a_scores: dict[str, list[float]] = {m: [] for m in all_metrics}
        model_b_scores: dict[str, list[float]] = {m: [] for m in all_metrics}
        wins: dict[str, int] = dict.fromkeys(all_metrics, 0)
        losses: dict[str, int] = dict.fromkeys(all_metrics, 0)
        ties: dict[str, int] = dict.fromkeys(all_metrics, 0)

        for ra, rb in zip(results_a, results_b):
            for metric in all_metrics:
                score_a = ra.scores.get(metric)
                score_b = rb.scores.get(metric)
                if score_a is None or score_b is None:
                    continue

                model_a_scores[metric].append(score_a)
                model_b_scores[metric].append(score_b)

                diff = score_a - score_b
                if abs(diff) < self._tie_margin:
                    ties[metric] += 1
                elif diff > 0:
                    wins[metric] += 1
                else:
                    losses[metric] += 1

        # Statistical analysis
        significance: dict[str, float] = {}
        effect_sizes: dict[str, float] = {}

        for metric in all_metrics:
            a_vals = model_a_scores[metric]
            b_vals = model_b_scores[metric]
            significance[metric] = _paired_ttest(a_vals, b_vals)
            effect_sizes[metric] = _cohens_d(a_vals, b_vals)

        # Overall winner determination
        overall_winner = self._determine_winner(
            wins, losses, ties, significance, model_a_scores, model_b_scores
        )

        result = ComparisonResult(
            model_a_scores=model_a_scores,
            model_b_scores=model_b_scores,
            wins=wins,
            losses=losses,
            ties=ties,
            statistical_significance=significance,
            effect_size=effect_sizes,
            overall_winner=overall_winner,
            details={
                "n_test_cases": len(results_a),
                "metrics_compared": sorted(all_metrics),
                "significance_level": self._significance,
                "tie_margin": self._tie_margin,
            },
        )

        self._log.info(
            "comparison_complete",
            overall_winner=overall_winner,
            n_metrics=len(all_metrics),
        )
        return result

    def _determine_winner(
        self,
        wins: dict[str, int],
        losses: dict[str, int],
        ties: dict[str, int],
        significance: dict[str, float],
        a_scores: dict[str, list[float]],
        b_scores: dict[str, list[float]],
    ) -> str | None:
        """Determine overall winner using significant metric wins.

        Counts metrics where one model is significantly better (p < threshold)
        with a positive effect size. Returns "model_a", "model_b", or None (tie).
        """
        a_significant_wins = 0
        b_significant_wins = 0

        for metric in wins:
            p = significance.get(metric, 1.0)
            if p >= self._significance:
                continue  # Not statistically significant

            a_mean = _safe_mean(a_scores.get(metric, []))
            b_mean = _safe_mean(b_scores.get(metric, []))

            if a_mean > b_mean:
                a_significant_wins += 1
            elif b_mean > a_mean:
                b_significant_wins += 1

        if a_significant_wins > b_significant_wins:
            return "model_a"
        elif b_significant_wins > a_significant_wins:
            return "model_b"

        # Fallback: use total wins across all metrics
        total_a_wins = sum(wins.values())
        total_b_wins = sum(losses.values())

        if total_a_wins > total_b_wins:
            return "model_a"
        elif total_b_wins > total_a_wins:
            return "model_b"

        return None
