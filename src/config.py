"""Configuration models and YAML suite loader with schema validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# Core configuration dataclasses
# ---------------------------------------------------------------------------


@dataclass
class EvalConfig:
    """Runtime configuration for an evaluation run."""

    model_name: str
    provider: Literal["openai", "anthropic", "bedrock"]
    api_key: str = ""
    temperature: float = 0.0
    max_tokens: int = 1024
    rate_limit_rpm: int = 60
    timeout: float = 30.0


# ---------------------------------------------------------------------------
# Pydantic models — used for YAML validation
# ---------------------------------------------------------------------------


class TestCase(BaseModel):
    """A single evaluation test case."""

    id: str = Field(..., min_length=1, description="Unique test case identifier")
    question: str = Field(..., min_length=1)
    expected_answer: str = Field(..., min_length=1)
    context: str | None = Field(
        default=None, description="Optional retrieval context for RAG tests"
    )
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("tags", mode="before")
    @classmethod
    def _coerce_tags(cls, v: Any) -> list[str]:
        if v is None:
            return []
        return v


class SuiteConfig(BaseModel):
    """Represents a loaded test-suite YAML file."""

    name: str = Field(..., min_length=1)
    description: str = ""
    version: str = "1.0"
    tags: list[str] = Field(default_factory=list)
    test_cases: list[TestCase] = Field(..., min_length=1)

    @field_validator("tags", mode="before")
    @classmethod
    def _coerce_tags(cls, v: Any) -> list[str]:
        if v is None:
            return []
        return v


# ---------------------------------------------------------------------------
# Schema-based validation
# ---------------------------------------------------------------------------

_SCHEMA_PATH = Path(__file__).resolve().parent.parent / "test_suites" / "schema.yaml"


def _load_json_schema() -> dict[str, Any]:
    """Load the JSON Schema definition from test_suites/schema.yaml."""
    if not _SCHEMA_PATH.exists():
        raise FileNotFoundError(f"Suite schema not found at {_SCHEMA_PATH}")
    with open(_SCHEMA_PATH, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _validate_raw_yaml(data: dict[str, Any]) -> list[str]:
    """Validate raw YAML data against the JSON Schema.

    Returns a list of validation error messages (empty == valid).
    Uses a lightweight inline validator so we avoid a hard dep on jsonschema.
    """
    errors: list[str] = []
    schema = _load_json_schema()

    required_top = schema.get("required", [])
    for key in required_top:
        if key not in data:
            errors.append(f"Missing required top-level key: '{key}'")

    tc_items = schema.get("properties", {}).get("test_cases", {}).get("items", {})
    tc_required = tc_items.get("required", [])

    if "test_cases" in data:
        if not isinstance(data["test_cases"], list):
            errors.append("'test_cases' must be a list")
        else:
            for idx, tc in enumerate(data["test_cases"]):
                if not isinstance(tc, dict):
                    errors.append(f"test_cases[{idx}]: must be a mapping")
                    continue
                for key in tc_required:
                    if key not in tc:
                        errors.append(f"test_cases[{idx}]: missing required key '{key}'")

    return errors


# ---------------------------------------------------------------------------
# Public loader
# ---------------------------------------------------------------------------


def load_suite(path: str | Path) -> SuiteConfig:
    """Load and validate a test-suite YAML file, returning a SuiteConfig.

    Parameters
    ----------
    path:
        Filesystem path to a YAML test suite file.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If the YAML fails schema or Pydantic validation.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Suite file not found: {path}")

    with open(path, encoding="utf-8") as fh:
        raw: dict[str, Any] = yaml.safe_load(fh)

    if not isinstance(raw, dict):
        raise ValueError(f"Expected a YAML mapping at top level, got {type(raw).__name__}")

    # Stage 1: JSON-Schema-style structural check
    schema_errors = _validate_raw_yaml(raw)
    if schema_errors:
        msg = "Suite YAML validation failed:\n" + "\n".join(f"  - {e}" for e in schema_errors)
        raise ValueError(msg)

    # Stage 2: Pydantic model validation (types, constraints, coercion)
    try:
        suite = SuiteConfig(**raw)
    except Exception as exc:
        raise ValueError(f"Suite validation error: {exc}") from exc

    return suite
