"""mjxsim Diffusion Policy training boundary."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from learning.config.runtime import WorkflowNotImplementedError


def train_diffusion_policy(config: Mapping[str, Any], output_dir: Path) -> None:
    del config, output_dir
    raise WorkflowNotImplementedError(
        "Diffusion Policy training is scaffolded but the sequence dataset and "
        "mjxsim model factory are not implemented yet."
    )
