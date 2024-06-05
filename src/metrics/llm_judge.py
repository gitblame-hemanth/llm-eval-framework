"""LLM-as-a-judge evaluation with multi-provider support and structured rubric scoring."""

from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

DEFAULT_RUBRIC: dict[str, dict[str, Any]] = {
    "relevance": {
        "description": "How well the response addresses the question asked.",
        "weight": 1.0,
    },
    "coherence": {
        "description": "Logical flow, clarity, and readability of the response.",
        "weight": 1.0,
    },
    "groundedness": {
        "description": "Whether claims are supported by provided context or verifiable facts.",
        "weight": 1.0,
    },
    "completeness": {
        "description": "How thoroughly the response covers the question without omissions.",
        "weight": 1.0,
    },
}


@dataclass
class JudgeResult:
    """Result of an LLM judge evaluation."""

    score: float  # Weighted aggregate score normalized to [0, 1]
    reasoning: str
    rubric_scores: dict[str, float] = field(default_factory=dict)  # criterion -> score (1-5)


# ---------------------------------------------------------------------------
# Provider abstraction
# ---------------------------------------------------------------------------


def call_openai(messages: list[dict[str, str]], model: str = "gpt-4") -> str:
    """Call OpenAI Chat Completions API.

    Requires the ``openai`` package and ``OPENAI_API_KEY`` env var.
    """
    import openai

    client = openai.OpenAI()
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.0,
        max_tokens=2048,
    )
    return response.choices[0].message.content or ""


def call_anthropic(messages: list[dict[str, str]], model: str = "claude-sonnet-4-20250514") -> str:
    """Call Anthropic Messages API.

    Requires the ``anthropic`` package and ``ANTHROPIC_API_KEY`` env var.
    """
    import anthropic

    client = anthropic.Anthropic()

    # Anthropic expects system message separately
    system_msg = ""
    api_messages = []
    for m in messages:
        if m["role"] == "system":
            system_msg = m["content"]
        else:
            api_messages.append(m)

    kwargs: dict[str, Any] = {
        "model": model,
        "max_tokens": 2048,
        "messages": api_messages,
    }
    if system_msg:
        kwargs["system"] = system_msg

    response = client.messages.create(**kwargs)
    return response.content[0].text


def call_bedrock(
    messages: list[dict[str, str]], model: str = "anthropic.claude-3-sonnet-20240229-v1:0"
) -> str:
    """Call AWS Bedrock Converse API.

    Requires ``boto3`` and configured AWS credentials.
    """
    import boto3

    client = boto3.client("bedrock-runtime")

    # Bedrock converse expects a specific message format
    system_parts: list[dict[str, str]] = []
    converse_messages: list[dict[str, Any]] = []

    for m in messages:
        if m["role"] == "system":
            system_parts.append({"text": m["content"]})
        else:
            converse_messages.append(
                {
                    "role": m["role"],
                    "content": [{"text": m["content"]}],
                }
            )

    kwargs: dict[str, Any] = {
        "modelId": model,
        "messages": converse_messages,
        "inferenceConfig": {"maxTokens": 2048, "temperature": 0.0},
    }
    if system_parts:
        kwargs["system"] = system_parts

    response = client.converse(**kwargs)
    return response["output"]["message"]["content"][0]["text"]


_PROVIDER_FNS: dict[str, Callable] = {
    "openai": call_openai,
    "anthropic": call_anthropic,
    "bedrock": call_bedrock,
}

# ---------------------------------------------------------------------------
# Retry / rate-limit helpers
# ---------------------------------------------------------------------------


def _call_with_retry(
    fn: Callable,
    messages: list[dict[str, str]],
    model: str,
    max_retries: int = 3,
    base_delay: float = 1.0,
) -> str:
    """Call a provider function with exponential backoff retry on transient errors."""
    last_exc: Exception | None = None

    for attempt in range(max_retries):
        try:
            return fn(messages, model=model)
        except Exception as exc:
            last_exc = exc
            err_str = str(exc).lower()
            # Retry on rate-limit or transient server errors
            is_retryable = any(
                kw in err_str
                for kw in ("rate", "429", "500", "502", "503", "529", "overloaded", "timeout")
            )
            if not is_retryable or attempt == max_retries - 1:
                raise

            delay = base_delay * (2**attempt)
            logger.warning(
                "Provider call failed (attempt %d/%d), retrying in %.1fs: %s",
                attempt + 1,
                max_retries,
                delay,
                exc,
            )
            time.sleep(delay)

    # Should never reach here, but just in case
    raise last_exc  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Score parsing
# ---------------------------------------------------------------------------


def _build_rubric_text(rubric: dict[str, dict[str, Any]]) -> str:
    """Format rubric criteria into prompt text."""
    lines = []
    for name, info in rubric.items():
        desc = info.get("description", "No description provided.")
        lines.append(f"- **{name}** (weight {info.get('weight', 1.0)}): {desc}")
    return "\n".join(lines)


def _parse_judge_response(text: str, rubric: dict[str, dict[str, Any]]) -> JudgeResult:
    """Extract structured scores and reasoning from the judge LLM response.

    Expected response format (flexible parsing):
        criterion_name: <int 1-5>
        ...
        REASONING: <free text>
    Also tries JSON extraction as a fallback.
    """
    rubric_scores: dict[str, float] = {}
    reasoning = ""

    # --- Attempt 1: look for JSON block ---
    json_match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if not json_match:
        json_match = re.search(r"(\{[^{}]*\"score\"[^{}]*\})", text, re.DOTALL)

    if json_match:
        try:
            parsed = json.loads(json_match.group(1))
            if "scores" in parsed and isinstance(parsed["scores"], dict):
                for k, v in parsed["scores"].items():
                    rubric_scores[k.lower()] = float(v)
            if "reasoning" in parsed:
                reasoning = str(parsed["reasoning"])
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

    # --- Attempt 2: line-by-line regex ---
    if not rubric_scores:
        for criterion in rubric:
            pattern = re.compile(
                rf"{re.escape(criterion)}\s*[:\-]\s*(\d(?:\.\d+)?)",
                re.IGNORECASE,
            )
            match = pattern.search(text)
            if match:
                rubric_scores[criterion] = float(match.group(1))

    # --- Extract reasoning if not found yet ---
    if not reasoning:
        reasoning_match = re.search(
            r"(?:REASONING|Reasoning|reasoning|Explanation|EXPLANATION)[:\s]*(.+)",
            text,
            re.DOTALL,
        )
        if reasoning_match:
            reasoning = reasoning_match.group(1).strip()
        else:
            # Use full text as reasoning fallback
            reasoning = text.strip()

    # --- Compute weighted aggregate score (normalize 1-5 → 0-1) ---
    total_weight = 0.0
    weighted_sum = 0.0

    for criterion, info in rubric.items():
        weight = info.get("weight", 1.0)
        raw_score = rubric_scores.get(criterion, 3.0)  # default mid-range
        # Clamp to [1, 5]
        raw_score = max(1.0, min(5.0, raw_score))
        rubric_scores[criterion] = raw_score
        weighted_sum += (raw_score - 1.0) / 4.0 * weight  # normalize to [0, 1]
        total_weight += weight

    aggregate = weighted_sum / total_weight if total_weight > 0 else 0.0

    return JudgeResult(
        score=round(aggregate, 4),
        reasoning=reasoning,
        rubric_scores=rubric_scores,
    )


# ---------------------------------------------------------------------------
# LLMJudge
# ---------------------------------------------------------------------------


class LLMJudge:
    """Evaluate LLM responses using another LLM as a judge.

    Args:
        provider: One of ``"openai"``, ``"anthropic"``, ``"bedrock"``, or a
            callable ``(messages, model) -> str``.
        model: Model identifier to pass to the provider.
        rubric: Optional custom rubric dict. Keys are criterion names, values
            are dicts with ``description`` (str) and ``weight`` (float).
            Defaults to relevance / coherence / groundedness / completeness.
        max_retries: Maximum number of retry attempts on transient failures.
    """

    def __init__(
        self,
        provider: str | Callable,
        model: str,
        rubric: dict[str, dict[str, Any]] | None = None,
        max_retries: int = 3,
    ) -> None:
        if isinstance(provider, str):
            if provider not in _PROVIDER_FNS:
                raise ValueError(
                    f"Unknown provider '{provider}'. "
                    f"Choose from {list(_PROVIDER_FNS.keys())} or pass a callable."
                )
            self._call_fn = _PROVIDER_FNS[provider]
        elif callable(provider):
            self._call_fn = provider
        else:
            raise TypeError("provider must be a string name or callable")

        self.model = model
        self.rubric = rubric or DEFAULT_RUBRIC
        self.max_retries = max_retries

    # ---- public API ----

    def judge(
        self,
        question: str,
        response: str,
        context: str | None = None,
        reference: str | None = None,
    ) -> JudgeResult:
        """Evaluate a response using the configured LLM judge.

        Args:
            question: The original question / prompt.
            response: The LLM-generated response to evaluate.
            context: Optional retrieval context provided to the LLM.
            reference: Optional gold-standard reference answer.

        Returns:
            A ``JudgeResult`` with aggregate score, per-criterion scores, and reasoning.
        """
        messages = self._build_messages(question, response, context, reference)
        raw_output = _call_with_retry(
            self._call_fn,
            messages,
            model=self.model,
            max_retries=self.max_retries,
        )
        return _parse_judge_response(raw_output, self.rubric)

    # ---- internals ----

    def _build_messages(
        self,
        question: str,
        response: str,
        context: str | None,
        reference: str | None,
    ) -> list[dict[str, str]]:
        rubric_text = _build_rubric_text(self.rubric)

        system_prompt = (
            "You are an expert evaluation judge. Score the given response on each criterion "
            "listed below using a scale of 1 (worst) to 5 (best). Be strict and objective.\n\n"
            "Rubric criteria:\n"
            f"{rubric_text}\n\n"
            "Respond with EXACTLY this format:\n"
            "```json\n"
            "{\n"
            '  "scores": {"criterion_name": <int 1-5>, ...},\n'
            '  "reasoning": "<brief justification>"\n'
            "}\n"
            "```"
        )

        user_parts = [f"**Question:** {question}", f"**Response to evaluate:** {response}"]
        if context:
            user_parts.append(f"**Context provided:** {context}")
        if reference:
            user_parts.append(f"**Reference answer:** {reference}")

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "\n\n".join(user_parts)},
        ]
