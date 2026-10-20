"""mjxsim DRLR2 training boundary."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from learning.config.runtime import WorkflowNotImplementedError


def train_drlr2(config: Mapping[str, Any], output_dir: Path) -> None:
    del config, output_dir
    raise WorkflowNotImplementedError(
        "DRLR2 training is scaffolded but the DP loader, expert-memory adapter, "
        "RL model factory, and trainer integration are not implemented yet."
    )
