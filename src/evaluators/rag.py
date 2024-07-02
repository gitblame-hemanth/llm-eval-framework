"""RAG (Retrieval-Augmented Generation) evaluator with four core metrics."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any

import structlog

from src.evaluators.base import BaseEvaluator

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Default LLM provider protocol
# ---------------------------------------------------------------------------

LLMCallable = Callable[[str], str]
"""Signature for the LLM judge function: prompt -> completion text."""


def _default_llm_provider(prompt: str) -> str:
    """Placeholder that raises — must be replaced with a real LLM provider."""
    raise NotImplementedError("No LLM provider configured. Pass llm_provider= to RAGEvaluator.")


# ---------------------------------------------------------------------------
# Embedding helper
# ---------------------------------------------------------------------------

EmbedCallable = Callable[[str], list[float]]


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# JSON extraction helper
# ---------------------------------------------------------------------------

_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\[.*?\]|\{.*?\})\s*```", re.DOTALL)
_BARE_JSON_RE = re.compile(r"(\[.*\])", re.DOTALL)


def _extract_json_list(text: str) -> list[str]:
    """Best-effort extraction of a JSON string list from LLM output."""
    # Try fenced code block first
    m = _JSON_BLOCK_RE.search(text)
    raw = m.group(1) if m else None

    if raw is None:
        m2 = _BARE_JSON_RE.search(text)
        raw = m2.group(1) if m2 else None

    if raw is None:
        # Fallback: split by numbered lines
        lines = [
            ln.strip().lstrip("0123456789.-) ").strip('"').strip("'")
            for ln in text.splitlines()
            if ln.strip() and not ln.strip().startswith("{")
        ]
        return [ln for ln in lines if len(ln) > 5]

    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
    except json.JSONDecodeError:
        pass
    return []


# ---------------------------------------------------------------------------
# RAG Evaluator
# ---------------------------------------------------------------------------


class RAGEvaluator(BaseEvaluator):
    """Evaluates RAG pipeline quality across four dimensions.

    Metrics
    -------
    - **faithfulness**: fraction of claims in the answer grounded in context.
    - **answer_relevance**: cosine similarity between original question and
      hypothetical questions generated from the answer.
    - **context_precision**: precision@k — how early relevant context chunks appear.
    - **context_recall**: fraction of expected-answer sentences supported by context.

    Parameters
    ----------
    model_name:
        Name of the model being evaluated (for bookkeeping).
    llm_provider:
        Callable ``(prompt: str) -> str`` used for LLM-as-judge calls.
    embed_provider:
        Callable ``(text: str) -> list[float]`` used for embedding text.
    n_hypothetical_questions:
        Number of hypothetical questions to generate for answer_relevance.
    """

    def __init__(
        self,
        model_name: str = "unknown",
        llm_provider: LLMCallable | None = None,
        embed_provider: EmbedCallable | None = None,
        n_hypothetical_questions: int = 3,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_name=model_name, **kwargs)
        self._llm = llm_provider or _default_llm_provider
        self._embed = embed_provider
        self._n_hyp = n_hypothetical_questions

    # ------------------------------------------------------------------
    # Metric: Faithfulness
    # ------------------------------------------------------------------

    def faithfulness(self, answer: str, context: str) -> float:
        """Fraction of answer claims that are grounded in the context.

        1. LLM extracts atomic claims from the answer.
        2. For each claim, LLM checks if it is supported by the context.
        3. Returns supported / total.
        """
        # Step 1 — extract claims
        extract_prompt = (
            "Extract all factual claims from the following answer as a JSON list of strings.\n"
            "Each claim should be a short, self-contained factual statement.\n"
            "Return ONLY a JSON array.\n\n"
            f"Answer:\n{answer}"
        )
        raw_claims = self._llm(extract_prompt)
        claims = _extract_json_list(raw_claims)

        if not claims:
            logger.warning("faithfulness_no_claims", answer_snippet=answer[:80])
            return 1.0  # No claims to refute → vacuously faithful

        # Step 2 — verify each claim
        supported = 0
        claim_verdicts: list[dict[str, Any]] = []
        for claim in claims:
            verify_prompt = (
                "Given the context below, determine if the following claim is supported.\n"
                "Respond with ONLY 'yes' or 'no'.\n\n"
                f"Context:\n{context}\n\n"
                f"Claim: {claim}"
            )
            verdict_raw = self._llm(verify_prompt).strip().lower()
            is_supported = verdict_raw.startswith("yes")
            if is_supported:
                supported += 1
            claim_verdicts.append({"claim": claim, "supported": is_supported})

        score = supported / len(claims)
        logger.debug(
            "faithfulness_result",
            total_claims=len(claims),
            supported=supported,
            score=score,
        )
        return round(score, 4)

    # ------------------------------------------------------------------
    # Metric: Answer Relevance
    # ------------------------------------------------------------------

    def answer_relevance(self, question: str, answer: str) -> float:
        """Cosine similarity between original question and hypothetical questions.

        1. LLM generates N hypothetical questions that the answer could address.
        2. Embed original question and each hypothetical question.
        3. Compute average cosine similarity.
        """
        if self._embed is None:
            raise ValueError(
                "answer_relevance requires an embed_provider. Pass embed_provider= to RAGEvaluator."
            )

        gen_prompt = (
            f"Given the following answer, generate {self._n_hyp} different questions "
            "that this answer could be responding to. Return them as a JSON list of strings.\n\n"
            f"Answer:\n{answer}"
        )
        raw = self._llm(gen_prompt)
        hyp_questions = _extract_json_list(raw)

        if not hyp_questions:
            logger.warning("answer_relevance_no_questions", answer_snippet=answer[:80])
            return 0.0

        q_embedding = self._embed(question)
        similarities: list[float] = []
        for hq in hyp_questions:
            hq_embedding = self._embed(hq)
            sim = _cosine_similarity(q_embedding, hq_embedding)
            similarities.append(sim)

        score = sum(similarities) / len(similarities)
        logger.debug(
            "answer_relevance_result",
            n_hypothetical=len(hyp_questions),
            avg_similarity=score,
        )
        return round(max(0.0, min(1.0, score)), 4)

    # ------------------------------------------------------------------
    # Metric: Context Precision
    # ------------------------------------------------------------------

    def context_precision(self, question: str, context_chunks: list[str], expected: str) -> float:
        """Precision@k — measures how early relevant chunks appear in the ranked context.

        For each chunk at position k, the LLM judges if it is relevant to the
        question+expected answer.  precision@k = (relevant in top-k) / k.
        Final score = mean of precision@k at each relevant position.
        """
        if not context_chunks:
            return 0.0

        relevance_flags: list[bool] = []
        for chunk in context_chunks:
            prompt = (
                "Is the following context chunk relevant to answering the question?\n"
                "Consider the expected answer when judging relevance.\n"
                "Respond with ONLY 'yes' or 'no'.\n\n"
                f"Question: {question}\n"
                f"Expected answer: {expected}\n\n"
                f"Context chunk:\n{chunk}"
            )
            verdict = self._llm(prompt).strip().lower()
            relevance_flags.append(verdict.startswith("yes"))

        if not any(relevance_flags):
            return 0.0

        # Compute precision@k at each relevant position
        precision_at_k_values: list[float] = []
        cumulative_relevant = 0
        for k_idx, is_relevant in enumerate(relevance_flags):
            if is_relevant:
                cumulative_relevant += 1
                precision_at_k = cumulative_relevant / (k_idx + 1)
                precision_at_k_values.append(precision_at_k)

        score = sum(precision_at_k_values) / len(precision_at_k_values)
        logger.debug(
            "context_precision_result",
            n_chunks=len(context_chunks),
            n_relevant=sum(relevance_flags),
            score=score,
        )
        return round(score, 4)

    # ------------------------------------------------------------------
    # Metric: Context Recall
    # ------------------------------------------------------------------

    def context_recall(self, expected_answer: str, context: str) -> float:
        """Fraction of expected-answer sentences that are supported by the context.

        1. Split expected answer into sentences.
        2. For each sentence, LLM checks if it can be attributed to the context.
        3. Returns supported / total.
        """
        # Split into sentences (simple heuristic)
        sentences = [
            s.strip()
            for s in re.split(r"(?<=[.!?])\s+", expected_answer)
            if s.strip() and len(s.strip()) > 5
        ]

        if not sentences:
            return 1.0  # Vacuously true

        supported = 0
        for sentence in sentences:
            prompt = (
                "Can the following sentence be attributed to (i.e., is it supported by) "
                "the given context? Respond with ONLY 'yes' or 'no'.\n\n"
                f"Context:\n{context}\n\n"
                f"Sentence: {sentence}"
            )
            verdict = self._llm(prompt).strip().lower()
            if verdict.startswith("yes"):
                supported += 1

        score = supported / len(sentences)
        logger.debug(
            "context_recall_result",
            total_sentences=len(sentences),
            supported=supported,
            score=score,
        )
        return round(score, 4)

    # ------------------------------------------------------------------
    # Core evaluate implementation
    # ------------------------------------------------------------------

    def _evaluate_impl(
        self,
        test_case: Any,
        response: str,
        context: str | None = None,
    ) -> tuple[dict[str, float], dict[str, Any]]:
        """Run all four RAG metrics and return scores + details."""
        question: str = getattr(test_case, "question", "")
        expected: str = getattr(test_case, "expected_answer", "")
        ctx: str = context or getattr(test_case, "context", "") or ""

        if not ctx:
            logger.warning(
                "rag_no_context",
                test_id=getattr(test_case, "id", "?"),
                msg="No context provided; faithfulness and context metrics may be unreliable.",
            )

        # Parse context chunks if separated by double newlines
        context_chunks = [c.strip() for c in ctx.split("\n\n") if c.strip()]

        scores: dict[str, float] = {}
        details: dict[str, Any] = {}

        # 1. Faithfulness
        try:
            scores["faithfulness"] = self.faithfulness(response, ctx)
        except Exception as exc:
            scores["faithfulness"] = 0.0
            details["faithfulness_error"] = str(exc)

        # 2. Answer relevance
        try:
            scores["answer_relevance"] = self.answer_relevance(question, response)
        except Exception as exc:
            scores["answer_relevance"] = 0.0
            details["answer_relevance_error"] = str(exc)

        # 3. Context precision
        try:
            scores["context_precision"] = self.context_precision(question, context_chunks, expected)
        except Exception as exc:
            scores["context_precision"] = 0.0
            details["context_precision_error"] = str(exc)

        # 4. Context recall
        try:
            scores["context_recall"] = self.context_recall(expected, ctx)
        except Exception as exc:
            scores["context_recall"] = 0.0
            details["context_recall_error"] = str(exc)

        details["question"] = question
        details["context_chunks_count"] = len(context_chunks)
        details["response_length"] = len(response)

        return scores, details
