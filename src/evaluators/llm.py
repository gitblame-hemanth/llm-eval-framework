"""LLM output quality evaluator with factual accuracy, hallucination,
instruction-following, and consistency metrics."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any

import structlog

from src.evaluators.base import BaseEvaluator

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

LLMCallable = Callable[[str], str]
"""prompt -> completion text."""

EmbedCallable = Callable[[str], list[float]]
"""text -> embedding vector."""

ModelCallable = Callable[[str], str]
"""question -> model response (for consistency runs)."""


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\[.*?\]|\{.*?\})\s*```", re.DOTALL)
_BARE_JSON_RE = re.compile(r"(\[.*\])", re.DOTALL)


def _extract_json_list(text: str) -> list[str]:
    m = _JSON_BLOCK_RE.search(text)
    raw = m.group(1) if m else None
    if raw is None:
        m2 = _BARE_JSON_RE.search(text)
        raw = m2.group(1) if m2 else None
    if raw is None:
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


def _extract_score(text: str, max_val: float = 5.0) -> float:
    """Extract a numeric score from LLM judge output and normalize to 0-1."""
    # Look for patterns like "Score: 4/5", "4 out of 5", "4/5", just "4"
    patterns = [
        r"(\d+(?:\.\d+)?)\s*/\s*(\d+)",  # 4/5
        r"(\d+(?:\.\d+)?)\s*out of\s*(\d+)",  # 4 out of 5
        r"[Ss]core[:\s]+(\d+(?:\.\d+)?)",  # Score: 4
        r"(\d+(?:\.\d+)?)",  # bare number
    ]
    for pattern in patterns:
        m = re.search(pattern, text)
        if m:
            groups = m.groups()
            if len(groups) == 2:
                return float(groups[0]) / float(groups[1])
            val = float(groups[0])
            if val <= 1.0:
                return val
            return min(val / max_val, 1.0)
    return 0.0


# ---------------------------------------------------------------------------
# LLM Evaluator
# ---------------------------------------------------------------------------


class LLMEvaluator(BaseEvaluator):
    """Evaluates LLM output quality across multiple dimensions.

    Metrics
    -------
    - **factual_accuracy**: semantic similarity + exact fact extraction vs ground truth.
    - **hallucination_score**: percentage of claims NOT grounded in context.
    - **instruction_following**: LLM-as-judge rubric scoring for instruction adherence.
    - **consistency**: pairwise similarity across N repeated generations.

    Parameters
    ----------
    model_name:
        Name of the model being evaluated.
    llm_provider:
        ``(prompt) -> str`` for LLM-as-judge calls.
    embed_provider:
        ``(text) -> list[float]`` for embedding-based similarity.
    model_provider:
        ``(question) -> str`` for generating multiple responses (consistency).
    metrics:
        List of metric names to run. Defaults to all four.
    n_consistency_runs:
        Number of responses to generate for consistency metric.
    """

    DEFAULT_METRICS = [
        "factual_accuracy",
        "hallucination_score",
        "instruction_following",
        "consistency",
    ]

    def __init__(
        self,
        model_name: str = "unknown",
        llm_provider: LLMCallable | None = None,
        embed_provider: EmbedCallable | None = None,
        model_provider: ModelCallable | None = None,
        metrics: list[str] | None = None,
        n_consistency_runs: int = 5,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_name=model_name, **kwargs)
        self._llm = llm_provider
        self._embed = embed_provider
        self._model = model_provider
        self._metrics = metrics or self.DEFAULT_METRICS
        self._n_consistency = n_consistency_runs

    def _require_llm(self) -> LLMCallable:
        if self._llm is None:
            raise ValueError("This metric requires llm_provider.")
        return self._llm

    def _require_embed(self) -> EmbedCallable:
        if self._embed is None:
            raise ValueError("This metric requires embed_provider.")
        return self._embed

    # ------------------------------------------------------------------
    # Metric: Factual Accuracy
    # ------------------------------------------------------------------

    def factual_accuracy(self, answer: str, ground_truth: str) -> float:
        """Combine semantic similarity with fact-level extraction to score accuracy.

        1. Compute cosine similarity between answer and ground truth embeddings.
        2. LLM extracts key facts from both, computes overlap.
        3. Final score = weighted average (0.4 semantic + 0.6 fact overlap).
        """
        embed = self._require_embed()
        llm = self._require_llm()

        # Semantic similarity
        answer_emb = embed(answer)
        truth_emb = embed(ground_truth)
        semantic_sim = _cosine_similarity(answer_emb, truth_emb)

        # Fact extraction and overlap
        extract_prompt = (
            "Extract the key factual claims from BOTH texts below as two JSON arrays.\n"
            "Return a JSON object with keys 'answer_facts' and 'truth_facts'.\n\n"
            f"Answer:\n{answer}\n\n"
            f"Ground Truth:\n{ground_truth}"
        )
        raw = llm(extract_prompt)

        fact_overlap = 0.0
        try:
            # Try to parse JSON object
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            if m:
                data = json.loads(m.group())
                answer_facts = data.get("answer_facts", [])
                truth_facts = data.get("truth_facts", [])

                if truth_facts:
                    # For each truth fact, check if any answer fact covers it
                    matched = 0
                    for tf in truth_facts:
                        check_prompt = (
                            "Does any of the following answer facts cover the same information "
                            f"as this truth fact?\n\nTruth fact: {tf}\n\n"
                            f"Answer facts: {json.dumps(answer_facts)}\n\n"
                            "Respond with ONLY 'yes' or 'no'."
                        )
                        verdict = llm(check_prompt).strip().lower()
                        if verdict.startswith("yes"):
                            matched += 1
                    fact_overlap = matched / len(truth_facts)
        except (json.JSONDecodeError, KeyError, TypeError):
            # Fallback: use LLM to directly score
            fallback_prompt = (
                "On a scale of 0 to 5, how factually accurate is the answer compared "
                "to the ground truth? Consider completeness and correctness.\n"
                "Respond with ONLY the score (e.g., '4/5').\n\n"
                f"Answer:\n{answer}\n\n"
                f"Ground Truth:\n{ground_truth}"
            )
            fact_overlap = _extract_score(llm(fallback_prompt))

        score = 0.4 * semantic_sim + 0.6 * fact_overlap
        logger.debug(
            "factual_accuracy_result",
            semantic_sim=round(semantic_sim, 4),
            fact_overlap=round(fact_overlap, 4),
            score=round(score, 4),
        )
        return round(max(0.0, min(1.0, score)), 4)

    # ------------------------------------------------------------------
    # Metric: Hallucination Score
    # ------------------------------------------------------------------

    def hallucination_score(self, answer: str, context: str) -> float:
        """Percentage of claims in the answer NOT grounded in context.

        Returns a score where 0.0 = no hallucination, 1.0 = fully hallucinated.
        """
        llm = self._require_llm()

        extract_prompt = (
            "Extract all factual claims from the following answer as a JSON list of strings.\n"
            "Each claim should be a short, self-contained factual statement.\n"
            "Return ONLY a JSON array.\n\n"
            f"Answer:\n{answer}"
        )
        claims = _extract_json_list(llm(extract_prompt))

        if not claims:
            return 0.0  # No claims → no hallucination

        unsupported = 0
        for claim in claims:
            prompt = (
                "Is the following claim supported by the context below?\n"
                "Respond with ONLY 'yes' or 'no'.\n\n"
                f"Context:\n{context}\n\n"
                f"Claim: {claim}"
            )
            verdict = llm(prompt).strip().lower()
            if not verdict.startswith("yes"):
                unsupported += 1

        score = unsupported / len(claims)
        logger.debug(
            "hallucination_result",
            total_claims=len(claims),
            unsupported=unsupported,
            score=score,
        )
        return round(score, 4)

    # ------------------------------------------------------------------
    # Metric: Instruction Following
    # ------------------------------------------------------------------

    def instruction_following(self, instruction: str, response: str) -> float:
        """LLM-as-judge rubric scoring for how well the response follows the instruction.

        Uses a structured rubric with 5 dimensions:
        1. Completeness — all parts of instruction addressed
        2. Accuracy — response is factually correct given instruction
        3. Format compliance — response matches requested format
        4. Constraint adherence — respects any constraints/restrictions
        5. Relevance — no off-topic content

        Returns a normalized score in [0, 1].
        """
        llm = self._require_llm()

        rubric_prompt = (
            "You are an expert evaluator. Score the following response against the instruction "
            "on these 5 dimensions (each 0-5):\n\n"
            "1. **Completeness** — Does the response address all parts of the instruction?\n"
            "2. **Accuracy** — Is the response factually correct given the instruction?\n"
            "3. **Format** — Does the response match the requested format/structure?\n"
            "4. **Constraints** — Does the response respect all constraints and restrictions?\n"
            "5. **Relevance** — Is the response focused without off-topic content?\n\n"
            f"Instruction:\n{instruction}\n\n"
            f"Response:\n{response}\n\n"
            "Return a JSON object with keys 'completeness', 'accuracy', 'format', "
            "'constraints', 'relevance' — each an integer 0-5. Include a brief 'reasoning' field.\n"
            "Return ONLY the JSON object."
        )
        raw = llm(rubric_prompt)

        # Parse rubric scores
        try:
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            if m:
                data = json.loads(m.group())
                dimensions = ["completeness", "accuracy", "format", "constraints", "relevance"]
                total = 0.0
                count = 0
                for dim in dimensions:
                    val = data.get(dim, 0)
                    if isinstance(val, (int, float)):
                        total += min(float(val), 5.0)
                        count += 1
                if count > 0:
                    score = total / (count * 5.0)
                    return round(max(0.0, min(1.0, score)), 4)
        except (json.JSONDecodeError, TypeError):
            pass

        # Fallback: simple score extraction
        return round(_extract_score(raw), 4)

    # ------------------------------------------------------------------
    # Metric: Consistency
    # ------------------------------------------------------------------

    def consistency(self, question: str, n_runs: int | None = None) -> float:
        """Generate N responses to the same question and measure pairwise similarity.

        Returns average pairwise cosine similarity across all generated responses.
        A high score indicates consistent, deterministic outputs.
        """
        if self._model is None:
            raise ValueError("consistency requires model_provider.")
        embed = self._require_embed()
        n = n_runs or self._n_consistency

        # Generate N responses
        responses: list[str] = []
        for i in range(n):
            try:
                resp = self._model(question)
                responses.append(resp)
            except Exception as exc:
                logger.warning("consistency_generation_error", run=i, error=str(exc))

        if len(responses) < 2:
            logger.warning("consistency_insufficient_responses", count=len(responses))
            return 0.0

        # Embed all responses
        embeddings = [embed(r) for r in responses]

        # Compute all pairwise cosine similarities
        similarities: list[float] = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sim = _cosine_similarity(embeddings[i], embeddings[j])
                similarities.append(sim)

        score = sum(similarities) / len(similarities)
        logger.debug(
            "consistency_result",
            n_responses=len(responses),
            n_pairs=len(similarities),
            avg_similarity=round(score, 4),
        )
        return round(max(0.0, min(1.0, score)), 4)

    # ------------------------------------------------------------------
    # Core evaluate implementation
    # ------------------------------------------------------------------

    def _evaluate_impl(
        self,
        test_case: Any,
        response: str,
        context: str | None = None,
    ) -> tuple[dict[str, float], dict[str, Any]]:
        """Run configured metrics and return scores + details."""
        question: str = getattr(test_case, "question", "")
        expected: str = getattr(test_case, "expected_answer", "")
        ctx: str = context or getattr(test_case, "context", "") or ""

        scores: dict[str, float] = {}
        details: dict[str, Any] = {}

        for metric_name in self._metrics:
            try:
                if metric_name == "factual_accuracy":
                    scores[metric_name] = self.factual_accuracy(response, expected)

                elif metric_name == "hallucination_score":
                    if not ctx:
                        details["hallucination_skipped"] = "no context provided"
                        continue
                    scores[metric_name] = self.hallucination_score(response, ctx)

                elif metric_name == "instruction_following":
                    scores[metric_name] = self.instruction_following(question, response)

                elif metric_name == "consistency":
                    scores[metric_name] = self.consistency(question)

                else:
                    details[f"{metric_name}_error"] = f"Unknown metric: {metric_name}"

            except Exception as exc:
                scores[metric_name] = 0.0
                details[f"{metric_name}_error"] = str(exc)
                logger.warning("metric_error", metric=metric_name, error=str(exc))

        details["metrics_run"] = list(scores.keys())
        details["response_length"] = len(response)

        return scores, details
