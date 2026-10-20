"""Best-effort dispatch to configured notification providers."""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from typing import Any

from learning.notifications.model import WorkflowNotification
from learning.notifications.ntfy import publish_ntfy

_STATUS_FLAGS = {
    "succeeded": "on_success",
    "failed": "on_failure",
    "interrupted": "on_interrupted",
}


def send_workflow_notification(
    config: Mapping[str, Any] | None,
    event: WorkflowNotification,
) -> None:
    """Send an enabled notification without changing workflow outcome."""
    if not config or not config.get("enabled", False):
        return

    provider = config.get("provider", "ntfy")
    try:
        if not config.get(_STATUS_FLAGS[event.status], True):
            return
        minimum_duration = float(config.get("minimum_success_duration_seconds", 60.0))
        if minimum_duration < 0:
            raise ValueError(
                "notifications.minimum_success_duration_seconds must be non-negative"
            )
        if event.status == "succeeded" and event.duration_seconds < minimum_duration:
            return
        if provider == "ntfy":
            publish_ntfy(config, event)
        else:
            raise ValueError(f"unsupported notification provider {provider!r}")
    except Exception as error:
        warnings.warn(
            f"Could not send {provider} workflow notification: {error}",
            RuntimeWarning,
            stacklevel=2,
        )
