"""Provider-neutral notification data passed by workflow runners."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

WorkflowStatus = Literal["succeeded", "failed", "interrupted"]


@dataclass(frozen=True)
class WorkflowNotification:
    """A deliberately small, non-sensitive workflow completion event."""

    status: WorkflowStatus
    stage: str
    task: str
    duration_seconds: float
    output_dir: Path
    hostname: str
    error_type: str | None = None
