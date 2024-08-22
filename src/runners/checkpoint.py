"""Checkpoint/resume support for long-running evaluation suites."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.runners.async_runner import EvalResult

logger = logging.getLogger(__name__)


@dataclass
class CheckpointState:
    """Serializable checkpoint of an in-progress evaluation run."""

    suite_id: str
    completed: list[EvalResult]
    remaining_ids: list[str]
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "suite_id": self.suite_id,
            "completed": [r.to_dict() for r in self.completed],
            "remaining_ids": self.remaining_ids,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CheckpointState:
        return cls(
            suite_id=data["suite_id"],
            completed=[EvalResult.from_dict(r) for r in data.get("completed", [])],
            remaining_ids=data.get("remaining_ids", []),
            timestamp=data.get("timestamp", 0.0),
        )


class CheckpointManager:
    """Persist and restore evaluation progress to disk as JSON checkpoints."""

    def __init__(
        self,
        checkpoint_dir: str | Path = "checkpoints/",
        auto_save_interval: int = 5,
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.auto_save_interval = auto_save_interval
        self._counter: int = 0

    def _path_for(self, suite_id: str) -> Path:
        safe_name = suite_id.replace("/", "_").replace("\\", "_")
        return self.checkpoint_dir / f"{safe_name}.checkpoint.json"

    def save(
        self,
        suite_id: str,
        completed_results: list[EvalResult],
        remaining_ids: list[str],
    ) -> Path:
        """Persist current progress to a JSON checkpoint file.

        Returns the path to the written checkpoint file.
        """
        state = CheckpointState(
            suite_id=suite_id,
            completed=completed_results,
            remaining_ids=remaining_ids,
        )
        path = self._path_for(suite_id)
        path.write_text(json.dumps(state.to_dict(), indent=2, default=str), encoding="utf-8")
        logger.info(
            "Checkpoint saved: %d completed, %d remaining -> %s",
            len(completed_results),
            len(remaining_ids),
            path,
        )
        return path

    def load(self, suite_id: str) -> CheckpointState | None:
        """Load checkpoint state for a suite, or return None if none exists."""
        path = self._path_for(suite_id)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            state = CheckpointState.from_dict(data)
            logger.info(
                "Checkpoint loaded: %d completed, %d remaining from %s",
                len(state.completed),
                len(state.remaining_ids),
                path,
            )
            return state
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            logger.warning("Corrupt checkpoint file %s: %s", path, exc)
            return None

    def clear(self, suite_id: str) -> None:
        """Remove checkpoint file for a suite."""
        path = self._path_for(suite_id)
        if path.exists():
            path.unlink()
            logger.info("Checkpoint cleared: %s", path)

    def maybe_auto_save(
        self,
        suite_id: str,
        completed_results: list[EvalResult],
        remaining_ids: list[str],
    ) -> bool:
        """Auto-save if enough results have accumulated since last save.

        Returns True if a save was performed.
        """
        self._counter += 1
        if self._counter >= self.auto_save_interval:
            self.save(suite_id, completed_results, remaining_ids)
            self._counter = 0
            return True
        return False

    def create_callback(
        self,
        suite_id: str,
        all_test_case_ids: list[str],
    ):
        """Return a callback suitable for AsyncRunner.set_checkpoint_callback().

        The callback auto-saves every N results.
        """

        def _callback(result: EvalResult, all_results: list[EvalResult]) -> None:
            completed_ids = {r.test_case_id for r in all_results}
            remaining = [tid for tid in all_test_case_ids if tid not in completed_ids]
            self.maybe_auto_save(suite_id, all_results, remaining)

        return _callback
