"""Click CLI for the LLM Evaluation Framework."""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any

import click

from src.config import EvalConfig, TestCase, load_suite
from src.reporters.html_reporter import HTMLReporter
from src.reporters.json_reporter import JSONReporter
from src.runners.async_runner import AsyncRunner, EvalResult
from src.runners.checkpoint import CheckpointManager

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _load_results_from_file(path: str | Path) -> list[EvalResult]:
    """Load EvalResult list from a JSON report file."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    raw_results = data.get("results", data) if isinstance(data, dict) else data
    return [EvalResult.from_dict(r) for r in raw_results]


class _DummyEvaluator:
    """Placeholder evaluator — real implementations live in src.evaluators."""

    def __init__(self, config: EvalConfig, metrics: list[str] | None = None) -> None:
        self.config = config
        self.metrics = metrics or []

    async def evaluate(self, test_case: TestCase, model: str | None = None) -> EvalResult:
        raise NotImplementedError(
            "No evaluator backend configured. "
            "Implement an Evaluator in src/evaluators/ and wire it here."
        )


def _create_evaluator(config: EvalConfig, metrics: list[str] | None = None) -> Any:
    """Create the appropriate evaluator based on config.

    Override this function once real evaluator implementations exist.
    """
    try:
        from src.evaluators import create_evaluator  # type: ignore[import-untyped]

        return create_evaluator(config, metrics=metrics)
    except (ImportError, AttributeError):
        return _DummyEvaluator(config, metrics=metrics)


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging.")
def cli(verbose: bool) -> None:
    """LLM Eval Framework — evaluate LLM outputs against test suites."""
    _setup_logging(verbose)


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


@cli.command()
@click.option(
    "--suite",
    "-s",
    required=True,
    type=click.Path(exists=True),
    help="Path to test suite YAML file.",
)
@click.option("--model", "-m", default="gpt-4o", help="Model name to evaluate.")
@click.option(
    "--provider",
    "-p",
    default="openai",
    type=click.Choice(["openai", "anthropic", "bedrock"]),
    help="LLM provider.",
)
@click.option("--output", "-o", default="reports/results.json", help="Output file path.")
@click.option("--metrics", default=None, help="Comma-separated metric names to evaluate.")
@click.option("--max-concurrent", default=10, type=int, help="Max concurrent requests.")
@click.option("--checkpoint/--no-checkpoint", default=False, help="Enable checkpoint/resume.")
@click.option(
    "--api-key", envvar="LLM_API_KEY", default="", help="API key (or set LLM_API_KEY env var)."
)
def run(
    suite: str,
    model: str,
    provider: str,
    output: str,
    metrics: str | None,
    max_concurrent: int,
    checkpoint: bool,
    api_key: str,
) -> None:
    """Run an evaluation suite against a model."""
    suite_config = load_suite(suite)
    metric_list = [m.strip() for m in metrics.split(",")] if metrics else None

    config = EvalConfig(
        model_name=model,
        provider=provider,
        api_key=api_key,
    )

    evaluator = _create_evaluator(config, metrics=metric_list)
    runner = AsyncRunner(
        evaluator=evaluator,
        config=config,
        max_concurrent=max_concurrent,
        rate_limit_rpm=config.rate_limit_rpm,
    )

    # Cost estimation
    cost = runner._estimate_cost(suite_config.test_cases, model)
    click.echo(click.style(f"\n{cost}", fg="cyan"))
    click.echo(f"Suite: {suite_config.name} ({len(suite_config.test_cases)} test cases)")
    click.echo(f"Model: {model} ({provider})\n")

    if not click.confirm("Proceed with evaluation?", default=True):
        click.echo("Aborted.")
        return

    # Checkpoint setup
    skip_ids: set[str] = set()
    prior_results: list[EvalResult] = []
    ckpt_mgr: CheckpointManager | None = None

    if checkpoint:
        ckpt_mgr = CheckpointManager()
        state = ckpt_mgr.load(suite_config.name)
        if state:
            prior_results = state.completed
            skip_ids = {r.test_case_id for r in prior_results}
            click.echo(
                click.style(
                    f"Resuming from checkpoint: {len(prior_results)} completed, "
                    f"{len(state.remaining_ids)} remaining",
                    fg="yellow",
                )
            )

        cb = ckpt_mgr.create_callback(
            suite_config.name,
            [tc.id for tc in suite_config.test_cases],
        )
        runner.set_checkpoint_callback(cb)

    # Run
    new_results = asyncio.run(
        runner.run_suite(suite_config.test_cases, model=model, skip_ids=skip_ids)
    )
    all_results = prior_results + new_results

    # Generate report
    reporter = JSONReporter()
    reporter.generate(all_results, output)

    # Clear checkpoint on success
    if ckpt_mgr:
        ckpt_mgr.clear(suite_config.name)

    # Summary
    passed = sum(1 for r in all_results if r.passed)
    total = len(all_results)
    errors = sum(1 for r in all_results if r.error)

    click.echo(
        click.style(
            f"\nResults: {passed}/{total} passed", fg="green" if passed == total else "yellow"
        )
    )
    if errors:
        click.echo(click.style(f"Errors: {errors}", fg="red"))
    click.echo(f"Report: {output}")


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------


@cli.command()
@click.option(
    "--suite",
    "-s",
    required=True,
    type=click.Path(exists=True),
    help="Path to test suite YAML file.",
)
@click.option("--models", "-m", required=True, help="Comma-separated model names to compare.")
@click.option(
    "--provider", "-p", default="openai", type=click.Choice(["openai", "anthropic", "bedrock"])
)
@click.option("--output", "-o", default="reports/comparison.html", help="Output HTML file.")
@click.option("--metrics", default=None, help="Comma-separated metric names.")
@click.option("--max-concurrent", default=10, type=int)
@click.option("--api-key", envvar="LLM_API_KEY", default="")
def compare(
    suite: str,
    models: str,
    provider: str,
    output: str,
    metrics: str | None,
    max_concurrent: int,
    api_key: str,
) -> None:
    """Compare multiple models on the same test suite."""
    suite_config = load_suite(suite)
    model_list = [m.strip() for m in models.split(",")]
    metric_list = [m.strip() for m in metrics.split(",")] if metrics else None

    click.echo(
        click.style(f"\nComparing {len(model_list)} models on '{suite_config.name}'", fg="cyan")
    )
    click.echo(f"Models: {', '.join(model_list)}")
    click.echo(f"Test cases: {len(suite_config.test_cases)}\n")

    results_by_model: dict[str, list[EvalResult]] = {}

    for model_name in model_list:
        click.echo(click.style(f"\n--- Evaluating: {model_name} ---", fg="blue"))

        config = EvalConfig(model_name=model_name, provider=provider, api_key=api_key)
        evaluator = _create_evaluator(config, metrics=metric_list)
        runner = AsyncRunner(evaluator=evaluator, config=config, max_concurrent=max_concurrent)

        cost = runner._estimate_cost(suite_config.test_cases, model_name)
        click.echo(f"  {cost}")

        model_results = asyncio.run(runner.run_suite(suite_config.test_cases, model=model_name))
        results_by_model[model_name] = model_results

        passed = sum(1 for r in model_results if r.passed)
        click.echo(click.style(f"  {model_name}: {passed}/{len(model_results)} passed", fg="green"))

    # Generate comparison report
    reporter = HTMLReporter()
    reporter.generate_comparison(results_by_model, output)

    # Also save raw JSON
    json_path = Path(output).with_suffix(".json")
    all_results = [r for results in results_by_model.values() for r in results]
    JSONReporter().generate(all_results, json_path)

    click.echo(click.style(f"\nComparison report: {output}", fg="green"))
    click.echo(f"JSON data: {json_path}")


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


@cli.command()
@click.option(
    "--input",
    "-i",
    "input_path",
    required=True,
    type=click.Path(exists=True),
    help="Input JSON results file.",
)
@click.option(
    "--format",
    "-f",
    "fmt",
    type=click.Choice(["json", "csv", "html"]),
    default="html",
    help="Output format.",
)
@click.option("--output", "-o", default=None, help="Output file path (auto-generated if omitted).")
def report(input_path: str, fmt: str, output: str | None) -> None:
    """Generate a report from existing evaluation results."""
    results = _load_results_from_file(input_path)
    click.echo(f"Loaded {len(results)} results from {input_path}")

    if output is None:
        stem = Path(input_path).stem
        ext = {"json": ".json", "csv": ".csv", "html": ".html"}[fmt]
        output = f"reports/{stem}_report{ext}"

    if fmt == "json":
        JSONReporter().generate(results, output)
    elif fmt == "csv":
        JSONReporter().to_csv(results, output)
    elif fmt == "html":
        HTMLReporter().generate(results, output)

    click.echo(click.style(f"Report written to {output}", fg="green"))


# ---------------------------------------------------------------------------
# ci
# ---------------------------------------------------------------------------


@cli.command()
@click.option(
    "--suite", "-s", required=True, type=click.Path(exists=True), help="Path to test suite YAML."
)
@click.option("--model", "-m", default="gpt-4o", help="Model to evaluate.")
@click.option(
    "--provider", "-p", default="openai", type=click.Choice(["openai", "anthropic", "bedrock"])
)
@click.option(
    "--baseline",
    "-b",
    default=None,
    type=click.Path(exists=True),
    help="Baseline results JSON for regression check.",
)
@click.option(
    "--threshold", "-t", default=0.8, type=float, help="Minimum pass rate threshold (0-1)."
)
@click.option("--output", "-o", default="reports/ci_results.json", help="Output file path.")
@click.option("--metrics", default=None, help="Comma-separated metric names.")
@click.option("--max-concurrent", default=10, type=int)
@click.option("--api-key", envvar="LLM_API_KEY", default="")
def ci(
    suite: str,
    model: str,
    provider: str,
    baseline: str | None,
    threshold: float,
    output: str,
    metrics: str | None,
    max_concurrent: int,
    api_key: str,
) -> None:
    """Run evaluation in CI mode — exits with code 0 (pass) or 1 (fail) based on threshold."""
    suite_config = load_suite(suite)
    metric_list = [m.strip() for m in metrics.split(",")] if metrics else None

    config = EvalConfig(model_name=model, provider=provider, api_key=api_key)
    evaluator = _create_evaluator(config, metrics=metric_list)
    runner = AsyncRunner(evaluator=evaluator, config=config, max_concurrent=max_concurrent)

    click.echo(click.style(f"CI Evaluation: {suite_config.name}", fg="cyan"))
    click.echo(f"Model: {model} | Threshold: {threshold:.0%}")

    results = asyncio.run(runner.run_suite(suite_config.test_cases, model=model))
    JSONReporter().generate(results, output)

    total = len(results)
    passed = sum(1 for r in results if r.passed)
    pass_rate = passed / total if total else 0.0
    errors = sum(1 for r in results if r.error)

    click.echo(f"\nResults: {passed}/{total} passed ({pass_rate:.1%})")
    if errors:
        click.echo(click.style(f"Errors: {errors}", fg="red"))

    # Baseline regression check
    if baseline:
        baseline_results = _load_results_from_file(baseline)
        baseline_passed = sum(1 for r in baseline_results if r.passed)
        baseline_rate = baseline_passed / len(baseline_results) if baseline_results else 0.0

        regression = pass_rate < baseline_rate
        if regression:
            click.echo(
                click.style(
                    f"REGRESSION: pass rate dropped from {baseline_rate:.1%} to {pass_rate:.1%}",
                    fg="red",
                    bold=True,
                )
            )
        else:
            delta = pass_rate - baseline_rate
            click.echo(
                click.style(
                    f"No regression: {pass_rate:.1%} vs baseline {baseline_rate:.1%} "
                    f"(+{delta:.1%})",
                    fg="green",
                )
            )

    # Threshold gate
    if pass_rate >= threshold:
        click.echo(
            click.style(
                f"\nPASSED: {pass_rate:.1%} >= {threshold:.0%} threshold", fg="green", bold=True
            )
        )
        sys.exit(0)
    else:
        click.echo(
            click.style(
                f"\nFAILED: {pass_rate:.1%} < {threshold:.0%} threshold", fg="red", bold=True
            )
        )
        sys.exit(1)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cli()
