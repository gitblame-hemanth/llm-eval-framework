"""Async evaluation runner with concurrency control, rate limiting, and cost estimation."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from aiolimiter import AsyncLimiter
from tqdm.asyncio import tqdm

from src.config import EvalConfig, TestCase

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class EvalResult:
    """Result of evaluating a single test case."""

    test_case_id: str
    question: str
    expected_answer: str
    actual_answer: str
    scores: dict[str, float] = field(default_factory=dict)
    passed: bool = False
    error: str | None = None
    latency_seconds: float = 0.0
    model: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "test_case_id": self.test_case_id,
            "question": self.question,
            "expected_answer": self.expected_answer,
            "actual_answer": self.actual_answer,
            "scores": self.scores,
            "passed": self.passed,
            "error": self.error,
            "latency_seconds": self.latency_seconds,
            "model": self.model,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvalResult:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class CostEstimate:
    """Estimated cost for running a test suite."""

    estimated_input_tokens: int
    estimated_output_tokens: int
    estimated_cost_usd: float

    def __str__(self) -> str:
        return (
            f"Estimated tokens: {self.estimated_input_tokens:,} in / "
            f"{self.estimated_output_tokens:,} out | "
            f"Estimated cost: ${self.estimated_cost_usd:.4f}"
        )


# ---------------------------------------------------------------------------
# Evaluator protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class Evaluator(Protocol):
    """Protocol that evaluator implementations must satisfy."""

    async def evaluate(self, test_case: TestCase, model: str | None = None) -> EvalResult: ...


# ---------------------------------------------------------------------------
# Pricing table (USD per 1K tokens)
# ---------------------------------------------------------------------------

_PRICING: dict[str, dict[str, float]] = {
    # OpenAI
    "gpt-4o": {"input": 0.0025, "output": 0.01},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "gpt-4-turbo": {"input": 0.01, "output": 0.03},
    "gpt-4": {"input": 0.03, "output": 0.06},
    "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
    # Anthropic
    "claude-3-opus-20240229": {"input": 0.015, "output": 0.075},
    "claude-3-sonnet-20240229": {"input": 0.003, "output": 0.015},
    "claude-3-haiku-20240307": {"input": 0.00025, "output": 0.00125},
    "claude-3-5-sonnet-20241022": {"input": 0.003, "output": 0.015},
    "claude-sonnet-4-20250514": {"input": 0.003, "output": 0.015},
    "claude-opus-4-20250514": {"input": 0.015, "output": 0.075},
}

_DEFAULT_PRICING = {"input": 0.005, "output": 0.015}


# ---------------------------------------------------------------------------
# AsyncRunner
# ---------------------------------------------------------------------------


class AsyncRunner:
    """Run evaluation suites asynchronously with concurrency and rate limiting."""

    def __init__(
        self,
        evaluator: Evaluator,
        config: EvalConfig,
        max_concurrent: int = 10,
        rate_limit_rpm: int = 60,
    ) -> None:
        self.evaluator = evaluator
        self.config = config
        self.max_concurrent = max_concurrent
        self.rate_limit_rpm = rate_limit_rpm
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._limiter = AsyncLimiter(rate_limit_rpm, 60)
        self._checkpoint_callback: Any = None

    def set_checkpoint_callback(self, callback: Any) -> None:
        """Set a callback invoked after each result: callback(result, all_results)."""
        self._checkpoint_callback = callback

    async def run_suite(
        self,
        test_cases: list[TestCase],
        model: str | None = None,
        *,
        skip_ids: set[str] | None = None,
    ) -> list[EvalResult]:
        """Run all test cases concurrently, returning a list of EvalResult.

        Parameters
        ----------
        test_cases:
            The test cases to evaluate.
        model:
            Override model name (falls back to config.model_name).
        skip_ids:
            Set of test-case IDs to skip (used for checkpoint resume).
        """
        effective_model = model or self.config.model_name
        cases_to_run = test_cases
        if skip_ids:
            cases_to_run = [tc for tc in test_cases if tc.id not in skip_ids]
            logger.info(
                "Skipping %d already-completed test cases", len(test_cases) - len(cases_to_run)
            )

        if not cases_to_run:
            logger.info("No test cases to run.")
            return []

        results: list[EvalResult] = []
        lock = asyncio.Lock()

        async def _wrapped(tc: TestCase) -> EvalResult:
            result = await self._run_single(tc, effective_model)
            async with lock:
                results.append(result)
                if self._checkpoint_callback:
                    self._checkpoint_callback(result, results)
            return result

        tasks = [_wrapped(tc) for tc in cases_to_run]

        completed = []
        for coro in tqdm(
            asyncio.as_completed(tasks),
            total=len(tasks),
            desc=f"Evaluating ({effective_model})",
            unit="case",
        ):
            result = await coro
            completed.append(result)

        return results

    async def _run_single(self, test_case: TestCase, model: str | None = None) -> EvalResult:
        """Evaluate a single test case with semaphore and rate-limit guards."""
        async with self._semaphore:
            await self._limiter.acquire()
            start = time.perf_counter()
            try:
                result = await self.evaluator.evaluate(test_case, model=model)
                result.latency_seconds = time.perf_counter() - start
                result.model = model or self.config.model_name
                return result
            except Exception as exc:
                elapsed = time.perf_counter() - start
                logger.error("Test case %s failed: %s", test_case.id, exc)
                return EvalResult(
                    test_case_id=test_case.id,
                    question=test_case.question,
                    expected_answer=test_case.expected_answer,
                    actual_answer="",
                    error=str(exc),
                    latency_seconds=elapsed,
                    model=model or self.config.model_name,
                )

    def _estimate_cost(
        self,
        test_cases: list[TestCase],
        model: str | None = None,
    ) -> CostEstimate:
        """Estimate token usage and USD cost for a suite run.

        Heuristic: ~4 chars per token for English text.
        """
        effective_model = model or self.config.model_name
        pricing = _PRICING.get(effective_model, _DEFAULT_PRICING)

        total_input_tokens = 0
        total_output_tokens = 0

        for tc in test_cases:
            # Input: system prompt overhead + question + optional context
            input_chars = len(tc.question) + 200  # system prompt estimate
            if tc.context:
                input_chars += len(tc.context)
            input_tokens = input_chars // 4

            # Output: estimate based on expected answer length (with margin)
            output_tokens = max(len(tc.expected_answer) // 4, 50) * 2

            total_input_tokens += input_tokens
            total_output_tokens += output_tokens

        cost = (total_input_tokens / 1000) * pricing["input"] + (
            total_output_tokens / 1000
        ) * pricing["output"]

        return CostEstimate(
            estimated_input_tokens=total_input_tokens,
            estimated_output_tokens=total_output_tokens,
            estimated_cost_usd=round(cost, 6),
        )
