"""Base evaluator ABC and shared result dataclasses."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class EvalResult:
    """Result of evaluating a single test case."""

    test_id: str
    scores: dict[str, float]
    details: dict[str, Any]
    latency_ms: float
    model_used: str
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    @property
    def passed(self) -> bool:
        """True if all scores meet a 0.5 threshold (configurable via details)."""
        threshold = self.details.get("pass_threshold", 0.5)
        return all(s >= threshold for s in self.scores.values())


@dataclass
class EvaluationSummary:
    """Aggregate statistics across multiple EvalResults."""

    total: int
    passed: int
    failed: int
    avg_scores: dict[str, float]
    min_scores: dict[str, float]
    max_scores: dict[str, float]
    per_metric_stats: dict[str, dict[str, float]]

    @classmethod
    def from_results(
        cls, results: list[EvalResult], pass_threshold: float = 0.5
    ) -> EvaluationSummary:
        """Build a summary from a list of EvalResult instances."""
        if not results:
            return cls(
                total=0,
                passed=0,
                failed=0,
                avg_scores={},
                min_scores={},
                max_scores={},
                per_metric_stats={},
            )

        total = len(results)
        passed = sum(1 for r in results if r.passed)
        failed = total - passed

        # Collect all metric names
        all_metrics: set[str] = set()
        for r in results:
            all_metrics.update(r.scores.keys())

        avg_scores: dict[str, float] = {}
        min_scores: dict[str, float] = {}
        max_scores: dict[str, float] = {}
        per_metric_stats: dict[str, dict[str, float]] = {}

        for metric in sorted(all_metrics):
            values = [r.scores[metric] for r in results if metric in r.scores]
            if not values:
                continue

            avg_val = sum(values) / len(values)
            min_val = min(values)
            max_val = max(values)

            # Variance and std dev
            variance = sum((v - avg_val) ** 2 for v in values) / len(values)
            std_dev = variance**0.5

            # Median
            sorted_vals = sorted(values)
            mid = len(sorted_vals) // 2
            if len(sorted_vals) % 2 == 0:
                median = (sorted_vals[mid - 1] + sorted_vals[mid]) / 2
            else:
                median = sorted_vals[mid]

            avg_scores[metric] = round(avg_val, 4)
            min_scores[metric] = round(min_val, 4)
            max_scores[metric] = round(max_val, 4)
            per_metric_stats[metric] = {
                "mean": round(avg_val, 4),
                "median": round(median, 4),
                "std_dev": round(std_dev, 4),
                "min": round(min_val, 4),
                "max": round(max_val, 4),
                "count": len(values),
                "pass_rate": round(sum(1 for v in values if v >= pass_threshold) / len(values), 4),
            }

        return cls(
            total=total,
            passed=passed,
            failed=failed,
            avg_scores=avg_scores,
            min_scores=min_scores,
            max_scores=max_scores,
            per_metric_stats=per_metric_stats,
        )


# ---------------------------------------------------------------------------
# Abstract base evaluator
# ---------------------------------------------------------------------------


class BaseEvaluator(ABC):
    """Abstract base for all evaluators in the framework.

    Subclasses implement ``_evaluate_impl`` with metric-specific logic.
    The public ``evaluate`` method handles timing, logging, and error wrapping.
    """

    def __init__(self, model_name: str = "unknown", **kwargs: Any) -> None:
        self.model_name = model_name
        self._log = logger.bind(evaluator=self.__class__.__name__, model=model_name)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        test_case: Any,
        response: str,
        context: str | None = None,
    ) -> EvalResult:
        """Evaluate a response against a test case.

        Parameters
        ----------
        test_case:
            A ``TestCase`` instance (or any object with ``id``, ``question``,
            ``expected_answer`` attributes).
        response:
            The model-generated response to evaluate.
        context:
            Optional retrieval context (for RAG evaluations).

        Returns
        -------
        EvalResult
        """
        test_id = getattr(test_case, "id", str(test_case))
        self._log.info("evaluate_start", test_id=test_id)

        start = time.perf_counter()
        try:
            scores, details = self._evaluate_impl(test_case, response, context)
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start) * 1000
            self._log.error("evaluate_error", test_id=test_id, error=str(exc))
            return EvalResult(
                test_id=test_id,
                scores={},
                details={"error": str(exc)},
                latency_ms=round(elapsed_ms, 2),
                model_used=self.model_name,
            )
        elapsed_ms = (time.perf_counter() - start) * 1000

        result = EvalResult(
            test_id=test_id,
            scores=scores,
            details=details,
            latency_ms=round(elapsed_ms, 2),
            model_used=self.model_name,
        )
        self._log.info(
            "evaluate_done",
            test_id=test_id,
            latency_ms=result.latency_ms,
            scores=scores,
        )
        return result

    def evaluate_batch(
        self, test_cases: list[Any], responses: list[str], contexts: list[str | None] | None = None
    ) -> list[EvalResult]:
        """Evaluate a batch of test cases sequentially."""
        if contexts is None:
            contexts = [None] * len(test_cases)
        if len(test_cases) != len(responses):
            raise ValueError("test_cases and responses must have the same length")
        return [
            self.evaluate(tc, resp, ctx) for tc, resp, ctx in zip(test_cases, responses, contexts)
        ]

    def summarize(
        self, results: list[EvalResult], pass_threshold: float = 0.5
    ) -> EvaluationSummary:
        """Build an EvaluationSummary from results."""
        return EvaluationSummary.from_results(results, pass_threshold=pass_threshold)

    # ------------------------------------------------------------------
    # Abstract
    # ------------------------------------------------------------------

    @abstractmethod
    def _evaluate_impl(
        self,
        test_case: Any,
        response: str,
        context: str | None = None,
    ) -> tuple[dict[str, float], dict[str, Any]]:
        """Return (scores, details) for a single evaluation.

        Must be implemented by subclasses.
        """
        ...
