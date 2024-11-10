"""Tests for JSON, CSV, and HTML reporters."""

from __future__ import annotations

import csv
import json

from src.reporters.json_reporter import JSONReporter
from src.runners.async_runner import EvalResult


def _make_results() -> list[EvalResult]:
    """Create a list of EvalResult instances for reporter tests."""
    return [
        EvalResult(
            test_case_id="tc-001",
            question="What is Python?",
            expected_answer="A programming language",
            actual_answer="Python is a programming language",
            scores={"bleu_score": 0.85, "rouge_l": 0.9},
            passed=True,
            error=None,
            latency_seconds=0.5,
            model="gpt-4o",
        ),
        EvalResult(
            test_case_id="tc-002",
            question="What is 2+2?",
            expected_answer="4",
            actual_answer="The answer is 4",
            scores={"bleu_score": 0.6, "rouge_l": 0.7},
            passed=True,
            error=None,
            latency_seconds=0.3,
            model="gpt-4o",
        ),
        EvalResult(
            test_case_id="tc-003",
            question="Explain quantum computing",
            expected_answer="Quantum computing uses qubits",
            actual_answer="I don't know",
            scores={"bleu_score": 0.1, "rouge_l": 0.15},
            passed=False,
            error=None,
            latency_seconds=0.8,
            model="gpt-4o",
        ),
    ]


# ===================================================================
# JSON Reporter
# ===================================================================


class TestJSONReporter:
    def test_generate_writes_valid_json(self, tmp_output_dir):
        results = _make_results()
        reporter = JSONReporter()
        out_path = tmp_output_dir / "report.json"

        returned_path = reporter.generate(results, out_path)

        assert returned_path == out_path
        assert out_path.exists()

        data = json.loads(out_path.read_text(encoding="utf-8"))
        assert "summary" in data
        assert "results" in data
        assert data["summary"]["total"] == 3
        assert data["summary"]["passed"] == 2

    def test_generate_creates_parent_dirs(self, tmp_output_dir):
        results = _make_results()
        reporter = JSONReporter()
        out_path = tmp_output_dir / "nested" / "dir" / "report.json"

        reporter.generate(results, out_path)
        assert out_path.exists()


# ===================================================================
# CSV Reporter
# ===================================================================


class TestCSVReporter:
    def test_to_csv_writes_correct_columns(self, tmp_output_dir):
        results = _make_results()
        reporter = JSONReporter()
        out_path = tmp_output_dir / "report.csv"

        returned_path = reporter.to_csv(results, out_path)

        assert returned_path == out_path
        assert out_path.exists()

        with open(out_path, encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)

        assert len(rows) == 3
        assert "test_case_id" in rows[0]
        assert "question" in rows[0]
        assert "passed" in rows[0]
        assert "score_bleu_score" in rows[0]
        assert "score_rouge_l" in rows[0]

    def test_csv_values(self, tmp_output_dir):
        results = _make_results()
        reporter = JSONReporter()
        out_path = tmp_output_dir / "report.csv"

        reporter.to_csv(results, out_path)

        with open(out_path, encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)

        assert rows[0]["test_case_id"] == "tc-001"
        assert rows[0]["model"] == "gpt-4o"


# ===================================================================
# HTML Reporter
# ===================================================================


class TestHTMLReporter:
    def test_generate_creates_html(self, tmp_output_dir):
        """HTML reporter generates a file with expected content sections."""
        # Import here to isolate template loading issues
        from src.reporters.html_reporter import HTMLReporter

        results = _make_results()
        reporter = HTMLReporter()
        out_path = tmp_output_dir / "report.html"

        returned_path = reporter.generate(results, out_path)

        assert returned_path == out_path
        assert out_path.exists()

        html = out_path.read_text(encoding="utf-8")
        assert "<html" in html.lower() or "<!doctype" in html.lower()


# ===================================================================
# DataFrame conversion
# ===================================================================


class TestDataFrame:
    def test_to_dataframe(self):
        results = _make_results()
        reporter = JSONReporter()
        df = reporter.to_dataframe(results)

        assert len(df) == 3
        assert "test_case_id" in df.columns
        assert "score_bleu_score" in df.columns
        assert df.iloc[0]["test_case_id"] == "tc-001"
