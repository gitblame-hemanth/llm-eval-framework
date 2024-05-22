"""Custom metric plugin system with registry, decorator, and dynamic loading."""

from __future__ import annotations

import builtins
import importlib.util
import logging
import sys
import threading
from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------


class CustomMetric(ABC):
    """Base class for all custom metrics."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique metric name."""
        ...

    @abstractmethod
    def compute(self, response: str, reference: str, **kwargs: Any) -> float:
        """Compute the metric value.

        Args:
            response: The generated response text.
            reference: The reference / ground-truth text.
            **kwargs: Additional metric-specific arguments.

        Returns:
            A float score (interpretation depends on the metric).
        """
        ...


# ---------------------------------------------------------------------------
# Singleton registry
# ---------------------------------------------------------------------------


class MetricRegistry:
    """Thread-safe singleton registry for evaluation metrics."""

    _instance: MetricRegistry | None = None
    _lock = threading.Lock()

    def __new__(cls) -> MetricRegistry:
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._metrics: dict[str, Callable[..., float]] = {}
                cls._instance._initialized = False
            return cls._instance

    # ---- public API ----

    def register(self, name: str, fn: Callable[..., float]) -> None:
        """Register a metric function under *name*.

        Args:
            name: Unique identifier for the metric.
            fn: Callable that returns a float score.

        Raises:
            ValueError: If a metric with the same name is already registered
                and points to a different function.
        """
        if name in self._metrics and self._metrics[name] is not fn:
            logger.warning("Overwriting existing metric '%s'", name)
        self._metrics[name] = fn

    def get(self, name: str) -> Callable[..., float]:
        """Retrieve a registered metric by name.

        Raises:
            KeyError: If no metric with that name exists.
        """
        if name not in self._metrics:
            raise KeyError(f"Metric '{name}' not found. Available: {list(self._metrics.keys())}")
        return self._metrics[name]

    def list(self) -> builtins.list[str]:
        """Return a sorted list of all registered metric names."""
        return sorted(self._metrics.keys())

    def has(self, name: str) -> bool:
        """Check whether a metric is registered."""
        return name in self._metrics

    def unregister(self, name: str) -> None:
        """Remove a metric from the registry."""
        self._metrics.pop(name, None)

    def clear(self) -> None:
        """Remove all registered metrics (useful for testing)."""
        self._metrics.clear()
        self._initialized = False

    @classmethod
    def reset(cls) -> None:
        """Destroy the singleton instance (useful for testing)."""
        with cls._lock:
            cls._instance = None


# ---------------------------------------------------------------------------
# Decorator
# ---------------------------------------------------------------------------


def register_metric(name: str) -> Callable:
    """Decorator to register a function or ``CustomMetric`` subclass.

    Usage::

        @register_metric("my_metric")
        def my_metric(response: str, reference: str, **kwargs) -> float:
            ...

        @register_metric("my_class_metric")
        class MyMetric(CustomMetric):
            name = "my_class_metric"
            def compute(self, response, reference, **kwargs):
                return 1.0
    """

    def decorator(fn_or_cls: Any) -> Any:
        registry = MetricRegistry()
        if isinstance(fn_or_cls, type) and issubclass(fn_or_cls, CustomMetric):
            instance = fn_or_cls()
            registry.register(name, instance.compute)
        elif callable(fn_or_cls):
            registry.register(name, fn_or_cls)
        else:
            raise TypeError(
                f"@register_metric expects a callable or CustomMetric subclass, "
                f"got {type(fn_or_cls)}"
            )
        return fn_or_cls

    return decorator


# ---------------------------------------------------------------------------
# Dynamic loading
# ---------------------------------------------------------------------------


def load_custom_metrics(path: str) -> list[str]:
    """Load custom metrics from a Python file.

    The file is executed as a module. Any functions/classes decorated with
    ``@register_metric`` will be auto-registered.

    Args:
        path: Filesystem path to a ``.py`` file containing metric definitions.

    Returns:
        List of newly registered metric names.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ImportError: If the module fails to load.
    """
    filepath = Path(path).resolve()
    if not filepath.exists():
        raise FileNotFoundError(f"Metric file not found: {filepath}")
    if not filepath.suffix == ".py":
        raise ValueError(f"Expected a .py file, got: {filepath}")

    registry = MetricRegistry()
    before = set(registry.list())

    module_name = f"_custom_metrics_{filepath.stem}_{id(filepath)}"

    spec = importlib.util.spec_from_file_location(module_name, str(filepath))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create module spec from {filepath}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module

    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise

    after = set(registry.list())
    new_metrics = sorted(after - before)
    if new_metrics:
        logger.info("Loaded custom metrics from %s: %s", filepath, new_metrics)
    return new_metrics


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------


def get_metric(name: str) -> Callable[..., float]:
    """Retrieve a registered metric by name.

    Ensures built-in metrics are loaded before lookup.
    """
    _ensure_builtins()
    return MetricRegistry().get(name)


def list_metrics() -> list[str]:
    """Return all registered metric names (including built-ins)."""
    _ensure_builtins()
    return MetricRegistry().list()


# ---------------------------------------------------------------------------
# Auto-register built-in metrics
# ---------------------------------------------------------------------------


def _ensure_builtins() -> None:
    """Register built-in metrics if not already done."""
    registry = MetricRegistry()
    if registry._initialized:
        return
    registry._initialized = True

    try:
        from metrics.builtin import (
            bleu_score,
            cosine_similarity,
            exact_match,
            f1_token_overlap,
            rouge_l,
            semantic_similarity,
        )

        _builtin_map = {
            "bleu_score": bleu_score,
            "rouge_l": rouge_l,
            "cosine_similarity": cosine_similarity,
            "semantic_similarity": semantic_similarity,
            "exact_match": exact_match,
            "f1_token_overlap": f1_token_overlap,
        }

        for name, fn in _builtin_map.items():
            if not registry.has(name):
                registry.register(name, fn)
    except ImportError:
        logger.debug("Could not auto-register built-in metrics (import failed)")
