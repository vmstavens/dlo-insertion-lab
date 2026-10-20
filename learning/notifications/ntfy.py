"""ntfy publisher implemented with the Python standard library."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from typing import Any
from urllib.request import Request, urlopen

from learning.notifications.model import WorkflowNotification

_STATUS_STYLE = {
    "succeeded": (3, ["white_check_mark"]),
    "failed": (4, ["x"]),
    "interrupted": (3, ["warning"]),
}


def _required_environment(name: object, setting: str) -> str:
    if not isinstance(name, str) or not name:
        raise ValueError(f"notifications.{setting} must name an environment variable")
    value = os.environ.get(name)
    if not value:
        raise ValueError(
            f"notifications are enabled but environment variable {name!r} is unset"
        )
    return value


def publish_ntfy(
    config: Mapping[str, Any],
    event: WorkflowNotification,
) -> None:
    """Publish one event to ntfy's JSON HTTP endpoint."""
    server = config.get("server", "https://ntfy.sh")
    if not isinstance(server, str) or not server.startswith(("http://", "https://")):
        raise ValueError("notifications.server must be an HTTP(S) URL")

    topic = _required_environment(config.get("topic_env"), "topic_env")
    token_env = config.get("token_env", "NTFY_TOKEN")
    if not isinstance(token_env, str) or not token_env:
        raise ValueError("notifications.token_env must name an environment variable")
    token = os.environ.get(token_env)

    priority, tags = _STATUS_STYLE[event.status]
    lines = [
        f"Task: {event.task}",
        f"Duration: {event.duration_seconds:.1f}s",
        f"Host: {event.hostname}",
        f"Run: {event.output_dir}",
    ]
    if event.error_type is not None:
        lines.append(f"Error: {event.error_type}")

    payload = {
        "topic": topic,
        "title": f"{event.stage} {event.status}",
        "message": "\n".join(lines),
        "priority": priority,
        "tags": tags,
    }
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    request = Request(
        f"{server.rstrip('/')}/",
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    timeout = float(config.get("timeout_seconds", 5.0))
    if timeout <= 0:
        raise ValueError("notifications.timeout_seconds must be positive")
    with urlopen(request, timeout=timeout) as response:  # noqa: S310
        response.read()
