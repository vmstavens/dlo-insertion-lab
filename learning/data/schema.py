"""Versioned manifest contract for raw and processed trajectory datasets."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DatasetManifest:
    schema_version: int
    dataset_id: str
    kind: str
    environment: str
    observation_layout: tuple[str, ...]
    observation_dim: int
    action_dim: int
    action_semantics: str
    control_dt: float
    episode_count: int
    git_commit: str | None = None
    source_dataset_ids: tuple[str, ...] = ()

    def validate(self) -> None:
        if self.schema_version != 1:
            raise ValueError(
                f"Unsupported dataset schema_version {self.schema_version}; expected 1."
            )
        if self.kind not in {"raw", "processed"}:
            raise ValueError("Dataset kind must be 'raw' or 'processed'.")
        if not self.dataset_id or not self.environment:
            raise ValueError("dataset_id and environment must be non-empty.")
        if self.observation_dim != len(self.observation_layout):
            raise ValueError(
                "observation_dim does not match the observation_layout length."
            )
        if self.observation_dim <= 0 or self.action_dim <= 0:
            raise ValueError("Observation and action dimensions must be positive.")
        if self.control_dt <= 0:
            raise ValueError("control_dt must be positive.")
        if self.episode_count < 0:
            raise ValueError("episode_count cannot be negative.")

    @classmethod
    def load(cls, path: str | Path) -> DatasetManifest:
        payload: dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
        payload["observation_layout"] = tuple(payload["observation_layout"])
        payload["source_dataset_ids"] = tuple(payload.get("source_dataset_ids", ()))
        manifest = cls(**payload)
        manifest.validate()
        return manifest

    def save(self, path: str | Path) -> None:
        self.validate()
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    def assert_compatible(
        self,
        *,
        environment: str,
        observation_dim: int,
        action_dim: int,
        control_dt: float,
    ) -> None:
        expected = {
            "environment": environment,
            "observation_dim": observation_dim,
            "action_dim": action_dim,
            "control_dt": control_dt,
        }
        actual = {name: getattr(self, name) for name in expected}
        mismatches = {}
        for name, value in expected.items():
            observed = actual[name]
            matches = (
                math.isclose(observed, value, rel_tol=1e-9, abs_tol=1e-12)
                if name == "control_dt"
                else observed == value
            )
            if not matches:
                mismatches[name] = (observed, value)
        if mismatches:
            raise ValueError(f"Dataset/environment contract mismatch: {mismatches}")
