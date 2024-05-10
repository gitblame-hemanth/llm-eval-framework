"""Built-in evaluation metrics for text comparison and similarity."""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Callable, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tokenize(text: str) -> list[str]:
    """Lowercase, strip, and split on whitespace + punctuation boundaries."""
    return re.findall(r"\w+", text.lower().strip())


def _ngrams(tokens: Sequence[str], n: int) -> list[tuple]:
    """Extract n-grams from a token list."""
    if len(tokens) < n:
        return []
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


# ---------------------------------------------------------------------------
# BLEU Score (1-4 gram, brevity penalty)
# ---------------------------------------------------------------------------


def bleu_score(reference: str, hypothesis: str) -> float:
    """Compute corpus-style BLEU (1-4 gram) between a single reference and hypothesis.

    Implements modified precision for n-grams 1..4 with brevity penalty.
    Returns a float in [0, 1].
    """
    if not reference or not hypothesis:
        return 0.0

    ref_tokens = _tokenize(reference)
    hyp_tokens = _tokenize(hypothesis)

    if not ref_tokens or not hyp_tokens:
        return 0.0

    # Brevity penalty
    bp = (
        math.exp(1 - len(ref_tokens) / len(hyp_tokens))
        if len(hyp_tokens) < len(ref_tokens)
        else 1.0
    )

    log_avg = 0.0
    max_n = min(4, len(hyp_tokens), len(ref_tokens))

    if max_n == 0:
        return 0.0

    for n in range(1, max_n + 1):
        ref_ng = Counter(_ngrams(ref_tokens, n))
        hyp_ng = Counter(_ngrams(hyp_tokens, n))

        # Clipped counts
        clipped = {ng: min(count, ref_ng.get(ng, 0)) for ng, count in hyp_ng.items()}
        numerator = sum(clipped.values())
        denominator = sum(hyp_ng.values())

        if denominator == 0 or numerator == 0:
            return 0.0  # Any zero n-gram precision → BLEU = 0

        log_avg += math.log(numerator / denominator)

    log_avg /= max_n
    return bp * math.exp(log_avg)


# ---------------------------------------------------------------------------
# ROUGE-L (LCS-based F1)
# ---------------------------------------------------------------------------


def _lcs_length(x: Sequence[str], y: Sequence[str]) -> int:
    """Compute length of longest common subsequence via DP."""
    m, n = len(x), len(y)
    if m == 0 or n == 0:
        return 0

    # Space-optimized: two rows
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev, curr = curr, [0] * (n + 1)

    return prev[n]


def rouge_l(reference: str, hypothesis: str) -> float:
    """Compute ROUGE-L F1 score between reference and hypothesis.

    Uses longest common subsequence on tokenized text.
    Returns a float in [0, 1].
    """
    if not reference or not hypothesis:
        return 0.0

    ref_tokens = _tokenize(reference)
    hyp_tokens = _tokenize(hypothesis)

    if not ref_tokens or not hyp_tokens:
        return 0.0

    lcs_len = _lcs_length(ref_tokens, hyp_tokens)
    if lcs_len == 0:
        return 0.0

    precision = lcs_len / len(hyp_tokens)
    recall = lcs_len / len(ref_tokens)

    f1 = 2 * precision * recall / (precision + recall)
    return f1


# ---------------------------------------------------------------------------
# Cosine Similarity
# ---------------------------------------------------------------------------


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors.

    Args:
        vec_a: First vector (1-D array-like).
        vec_b: Second vector (1-D array-like).

    Returns:
        Cosine similarity in [-1, 1]. Returns 0.0 if either vector has zero norm.
    """
    a = np.asarray(vec_a, dtype=np.float64).ravel()
    b = np.asarray(vec_b, dtype=np.float64).ravel()

    if a.shape != b.shape:
        raise ValueError(f"Vector dimensions must match: {a.shape} vs {b.shape}")

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return float(np.dot(a, b) / (norm_a * norm_b))


# ---------------------------------------------------------------------------
# Semantic Similarity
# ---------------------------------------------------------------------------


def semantic_similarity(
    text_a: str,
    text_b: str,
    embeddings_fn: Callable[[str], np.ndarray],
) -> float:
    """Compute semantic similarity by embedding both texts and returning cosine similarity.

    Args:
        text_a: First text string.
        text_b: Second text string.
        embeddings_fn: A callable that takes a string and returns a numpy array embedding.

    Returns:
        Cosine similarity of the two embeddings in [-1, 1].
    """
    if not text_a or not text_b:
        return 0.0

    vec_a = embeddings_fn(text_a)
    vec_b = embeddings_fn(text_b)
    return cosine_similarity(vec_a, vec_b)


# ---------------------------------------------------------------------------
# Exact Match
# ---------------------------------------------------------------------------


def exact_match(reference: str, hypothesis: str) -> float:
    """Normalized exact match: lowercase, strip, then compare.

    Returns 1.0 if match, 0.0 otherwise.
    """
    if reference is None or hypothesis is None:
        return 0.0

    ref_norm = reference.strip().lower()
    hyp_norm = hypothesis.strip().lower()
    return 1.0 if ref_norm == hyp_norm else 0.0


# ---------------------------------------------------------------------------
# F1 Token Overlap
# ---------------------------------------------------------------------------


def f1_token_overlap(reference: str, hypothesis: str) -> float:
    """Compute token-level F1 between reference and hypothesis.

    Treats each text as a bag of (lowercased) tokens, then computes
    precision, recall, and their harmonic mean.

    Returns a float in [0, 1].
    """
    if not reference or not hypothesis:
        return 0.0

    ref_tokens = _tokenize(reference)
    hyp_tokens = _tokenize(hypothesis)

    if not ref_tokens or not hyp_tokens:
        return 0.0

    ref_counts = Counter(ref_tokens)
    hyp_counts = Counter(hyp_tokens)

    # Number of overlapping tokens (clipped by reference count)
    overlap = 0
    for token, count in hyp_counts.items():
        overlap += min(count, ref_counts.get(token, 0))

    if overlap == 0:
        return 0.0

    precision = overlap / sum(hyp_counts.values())
    recall = overlap / sum(ref_counts.values())

    f1 = 2 * precision * recall / (precision + recall)
    return f1
