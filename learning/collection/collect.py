"""Demonstration collection workflow boundary."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from learning.config.runtime import WorkflowNotImplementedError


def collect_demonstrations(config: Mapping[str, Any], output_dir: Path) -> None:
    del config, output_dir
    raise WorkflowNotImplementedError(
        "Demonstration collection is scaffolded but has no controller/recorder "
        "implementation yet. Run with runtime.dry_run=true to validate composition."
    )
