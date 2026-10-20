"""Dataset preparation workflow boundary."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from learning.config.runtime import WorkflowNotImplementedError


def prepare_dataset(config: Mapping[str, Any], output_dir: Path) -> None:
    del config, output_dir
    raise WorkflowNotImplementedError(
        "Dataset transformation is scaffolded but not implemented yet. "
        "Run with runtime.dry_run=true to validate composition."
    )
