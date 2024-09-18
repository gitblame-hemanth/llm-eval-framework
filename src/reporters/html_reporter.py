"""HTML report generation with inline CSS/SVG charts via Jinja2 templates."""

from __future__ import annotations

import logging
import statistics
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader

from src.runners.async_runner import EvalResult

logger = logging.getLogger(__name__)

_TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"


def _build_report_data(
    results: list[EvalResult],
    title: str = "Evaluation Report",
) -> dict[str, Any]:
    """Build template context from evaluation results."""
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    failed = total - passed
    errors = sum(1 for r in results if r.error)

    # Per-metric aggregates
    metric_scores: dict[str, list[float]] = {}
    for r in results:
        for m, s in r.scores.items():
            metric_scores.setdefault(m, []).append(s)

    metric_stats: dict[str, dict[str, float]] = {}
    for m, scores in metric_scores.items():
        metric_stats[m] = {
            "mean": round(statistics.mean(scores), 4),
            "median": round(statistics.median(scores), 4),
            "min": round(min(scores), 4),
            "max": round(max(scores), 4),
        }

    all_scores = [s for r in results for s in r.scores.values()]
    avg_score = round(statistics.mean(all_scores), 4) if all_scores else 0.0

    latencies = [r.latency_seconds for r in results if r.latency_seconds > 0]
    avg_latency = round(statistics.mean(latencies), 3) if latencies else 0.0

    return {
        "title": title,
        "total": total,
        "passed": passed,
        "failed": failed,
        "errors": errors,
        "pass_rate": round(passed / total * 100, 1) if total else 0.0,
        "avg_score": avg_score,
        "avg_latency": avg_latency,
        "metric_stats": metric_stats,
        "results": [r.to_dict() for r in results],
    }


def _build_comparison_data(
    results_by_model: dict[str, list[EvalResult]],
    title: str = "Model Comparison Report",
) -> dict[str, Any]:
    """Build template context for side-by-side model comparison."""
    models = list(results_by_model.keys())
    model_summaries: list[dict[str, Any]] = []

    # Collect all metrics across all models
    all_metrics: set[str] = set()
    for results in results_by_model.values():
        for r in results:
            all_metrics.update(r.scores.keys())

    for model_name, results in results_by_model.items():
        total = len(results)
        passed = sum(1 for r in results if r.passed)
        all_scores = [s for r in results for s in r.scores.values()]
        avg_score = round(statistics.mean(all_scores), 4) if all_scores else 0.0
        latencies = [r.latency_seconds for r in results if r.latency_seconds > 0]
        avg_latency = round(statistics.mean(latencies), 3) if latencies else 0.0

        per_metric: dict[str, float] = {}
        for metric in all_metrics:
            scores = [r.scores.get(metric, 0.0) for r in results if metric in r.scores]
            per_metric[metric] = round(statistics.mean(scores), 4) if scores else 0.0

        model_summaries.append(
            {
                "name": model_name,
                "total": total,
                "passed": passed,
                "pass_rate": round(passed / total * 100, 1) if total else 0.0,
                "avg_score": avg_score,
                "avg_latency": avg_latency,
                "metrics": per_metric,
            }
        )

    return {
        "title": title,
        "comparison": True,
        "models": models,
        "model_summaries": model_summaries,
        "all_metrics": sorted(all_metrics),
    }


class HTMLReporter:
    """Generate self-contained HTML reports with inline CSS and SVG charts."""

    def __init__(self) -> None:
        self._env = Environment(
            loader=FileSystemLoader(str(_TEMPLATE_DIR)),
            autoescape=True,
        )

    def generate(
        self,
        results: list[EvalResult],
        output_path: str | Path,
        title: str = "Evaluation Report",
    ) -> Path:
        """Render a single-model evaluation report to an HTML file.

        Returns the path to the written file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        ctx = _build_report_data(results, title=title)
        ctx["comparison"] = False

        template = self._env.get_template("report.html")
        html = template.render(**ctx)
        output_path.write_text(html, encoding="utf-8")
        logger.info("HTML report written to %s", output_path)
        return output_path

    def generate_comparison(
        self,
        results_by_model: dict[str, list[EvalResult]],
        output_path: str | Path,
        title: str = "Model Comparison Report",
    ) -> Path:
        """Render a side-by-side model comparison report.

        Parameters
        ----------
        results_by_model:
            Mapping of model name -> list of EvalResult.
        output_path:
            Destination HTML file path.
        title:
            Report title.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        ctx = _build_comparison_data(results_by_model, title=title)

        template = self._env.get_template("report.html")
        html = template.render(**ctx)
        output_path.write_text(html, encoding="utf-8")
        logger.info("Comparison report written to %s", output_path)
        return output_path
