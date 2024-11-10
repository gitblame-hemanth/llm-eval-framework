"""Tests for CLI commands via click.testing.CliRunner."""

from __future__ import annotations

from click.testing import CliRunner

from src.cli import cli


class TestCLI:
    def setup_method(self):
        self.runner = CliRunner()

    def test_cli_help(self):
        result = self.runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "LLM Eval Framework" in result.output

    def test_run_help(self):
        result = self.runner.invoke(cli, ["run", "--help"])
        assert result.exit_code == 0
        assert "--suite" in result.output
        assert "--model" in result.output

    def test_compare_help(self):
        result = self.runner.invoke(cli, ["compare", "--help"])
        assert result.exit_code == 0
        assert "--models" in result.output
        assert "--suite" in result.output

    def test_report_help(self):
        result = self.runner.invoke(cli, ["report", "--help"])
        assert result.exit_code == 0
        assert "--input" in result.output
        assert "--format" in result.output

    def test_ci_help(self):
        result = self.runner.invoke(cli, ["ci", "--help"])
        assert result.exit_code == 0
        assert "--threshold" in result.output
        assert "--baseline" in result.output
