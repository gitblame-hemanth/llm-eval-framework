"""JSON and CSV report generation for evaluation results."""

from __future__ import annotations

import csv
import json
import logging
import statistics
from pathlib import Path
from typing import Any

import pandas as pd

from src.runners.async_runner import EvalResult

logger = logging.getLogger(__name__)


def _compute_summary(results: list[EvalResult]) -> dict[str, Any]:
    """Compute aggregate summary statistics from a list of results."""
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    failed = total - passed
    errors = sum(1 for r in results if r.error)

    # Aggregate per-metric scores
    metric_scores: dict[str, list[float]] = {}
    for r in results:
        for metric, score in r.scores.items():
            metric_scores.setdefault(metric, []).append(score)

    metric_stats: dict[str, dict[str, float]] = {}
    for metric, scores in metric_scores.items():
        metric_stats[metric] = {
            "mean": round(statistics.mean(scores), 4),
            "median": round(statistics.median(scores), 4),
            "stdev": round(statistics.stdev(scores), 4) if len(scores) > 1 else 0.0,
            "min": round(min(scores), 4),
            "max": round(max(scores), 4),
        }

    latencies = [r.latency_seconds for r in results if r.latency_seconds > 0]
    latency_stats = {}
    if latencies:
        latency_stats = {
            "mean_seconds": round(statistics.mean(latencies), 3),
            "median_seconds": round(statistics.median(latencies), 3),
            "p95_seconds": round(sorted(latencies)[int(len(latencies) * 0.95)], 3),
            "total_seconds": round(sum(latencies), 3),
        }

    # Overall average score across all metrics
    all_scores = [s for r in results for s in r.scores.values()]
    avg_score = round(statistics.mean(all_scores), 4) if all_scores else 0.0

    return {
        "total": total,
        "passed": passed,
        "failed": failed,
        "errors": errors,
        "pass_rate": round(passed / total, 4) if total else 0.0,
        "average_score": avg_score,
        "metric_stats": metric_stats,
        "latency": latency_stats,
    }


class JSONReporter:
    """Generate JSON and CSV evaluation reports."""

    def generate(self, results: list[EvalResult], output_path: str | Path) -> Path:
        """Write a full JSON report with per-test results and summary stats.

        Returns the path to the written file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        report = {
            "summary": _compute_summary(results),
            "results": [r.to_dict() for r in results],
        }

        output_path.write_text(
            json.dumps(report, indent=2, default=str),
            encoding="utf-8",
        )
        logger.info("JSON report written to %s", output_path)
        return output_path

    def to_csv(self, results: list[EvalResult], output_path: str | Path) -> Path:
        """Write a flattened CSV with one row per test case.

        Returns the path to the written file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Collect all metric names across results
        all_metrics: list[str] = []
        seen: set[str] = set()
        for r in results:
            for m in r.scores:
                if m not in seen:
                    all_metrics.append(m)
                    seen.add(m)

        fieldnames = [
            "test_case_id",
            "question",
            "expected_answer",
            "actual_answer",
            "passed",
            "error",
            "latency_seconds",
            "model",
        ] + [f"score_{m}" for m in all_metrics]

        with open(output_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for r in results:
                row: dict[str, Any] = {
                    "test_case_id": r.test_case_id,
                    "question": r.question,
                    "expected_answer": r.expected_answer,
                    "actual_answer": r.actual_answer,
                    "passed": r.passed,
                    "error": r.error or "",
                    "latency_seconds": round(r.latency_seconds, 3),
                    "model": r.model,
                }
                for m in all_metrics:
                    row[f"score_{m}"] = round(r.scores.get(m, 0.0), 4)
                writer.writerow(row)

        logger.info("CSV report written to %s (%d rows)", output_path, len(results))
        return output_path

    def to_dataframe(self, results: list[EvalResult]) -> pd.DataFrame:
        """Convert results to a pandas DataFrame for interactive analysis."""
        rows = []
        for r in results:
            row: dict[str, Any] = {
                "test_case_id": r.test_case_id,
                "question": r.question,
                "expected_answer": r.expected_answer,
                "actual_answer": r.actual_answer,
                "passed": r.passed,
                "error": r.error,
                "latency_seconds": r.latency_seconds,
                "model": r.model,
            }
            for metric, score in r.scores.items():
                row[f"score_{metric}"] = score
            rows.append(row)

        return pd.DataFrame(rows)
