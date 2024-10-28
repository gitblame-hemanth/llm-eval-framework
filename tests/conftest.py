"""Shared fixtures for llm-eval-framework tests."""

from __future__ import annotations

import pytest

from src.config import TestCase
from src.evaluators.base import EvalResult


@pytest.fixture
def sample_test_case() -> TestCase:
    """A single TestCase with question, expected_answer, and context."""
    return TestCase(
        id="tc-001",
        question="What is the capital of France?",
        expected_answer="The capital of France is Paris.",
        context="France is a country in Western Europe. Its capital city is Paris.",
        tags=["geography", "factual"],
        metadata={"difficulty": "easy"},
    )


@pytest.fixture
def sample_eval_result() -> EvalResult:
    """An EvalResult with realistic scores."""
    return EvalResult(
        test_id="tc-001",
        scores={
            "bleu_score": 0.72,
            "rouge_l": 0.85,
            "exact_match": 0.0,
            "f1_token_overlap": 0.78,
        },
        details={"pass_threshold": 0.5, "model_response": "Paris is the capital of France."},
        latency_ms=142.5,
        model_used="gpt-4o",
    )


@pytest.fixture
def sample_results() -> list[EvalResult]:
    """List of 3 EvalResults with varying scores."""
    return [
        EvalResult(
            test_id="tc-001",
            scores={"bleu_score": 0.9, "rouge_l": 0.95},
            details={"pass_threshold": 0.5},
            latency_ms=120.0,
            model_used="gpt-4o",
        ),
        EvalResult(
            test_id="tc-002",
            scores={"bleu_score": 0.6, "rouge_l": 0.7},
            details={"pass_threshold": 0.5},
            latency_ms=200.0,
            model_used="gpt-4o",
        ),
        EvalResult(
            test_id="tc-003",
            scores={"bleu_score": 0.3, "rouge_l": 0.4},
            details={"pass_threshold": 0.5},
            latency_ms=150.0,
            model_used="gpt-4o",
        ),
    ]


@pytest.fixture
def tmp_output_dir(tmp_path):
    """Temporary directory for test output files."""
    return tmp_path
