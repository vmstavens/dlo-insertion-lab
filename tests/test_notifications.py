from __future__ import annotations

import json
from pathlib import Path
from urllib.request import Request

import pytest

from learning.config import runtime
from learning.notifications import dispatch, ntfy
from learning.notifications.model import WorkflowNotification, WorkflowStatus


def make_event(
    *,
    status: WorkflowStatus = "succeeded",
    duration_seconds: float = 120.0,
    error_type: str | None = None,
) -> WorkflowNotification:
    return WorkflowNotification(
        status=status,
        stage="train_drlr2",
        task="grasped_end_dlo_insert",
        duration_seconds=duration_seconds,
        output_dir=Path("runs/example"),
        hostname="worker-01",
        error_type=error_type,
    )


def test_ntfy_publishes_json_using_environment_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class Response:
        def __enter__(self) -> Response:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            return b"{}"

    def fake_urlopen(request: Request, timeout: float) -> Response:
        captured["request"] = request
        captured["timeout"] = timeout
        return Response()

    monkeypatch.setenv("TEST_NTFY_TOPIC", "private-topic")
    monkeypatch.setenv("TEST_NTFY_TOKEN", "secret-token")
    monkeypatch.setattr(ntfy, "urlopen", fake_urlopen)

    ntfy.publish_ntfy(
        {
            "server": "https://ntfy.example.test",
            "topic_env": "TEST_NTFY_TOPIC",
            "token_env": "TEST_NTFY_TOKEN",
            "timeout_seconds": 2,
        },
        make_event(status="failed", error_type="RuntimeError"),
    )

    request = captured["request"]
    assert isinstance(request, Request)
    assert request.full_url == "https://ntfy.example.test/"
    assert request.get_header("Authorization") == "Bearer secret-token"
    assert captured["timeout"] == 2
    payload = json.loads(request.data)
    assert payload == {
        "topic": "private-topic",
        "title": "train_drlr2 failed",
        "message": (
            "Task: grasped_end_dlo_insert\n"
            "Duration: 120.0s\n"
            "Host: worker-01\n"
            "Run: runs/example\n"
            "Error: RuntimeError"
        ),
        "priority": 4,
        "tags": ["x"],
    }


def test_dispatch_skips_short_success_but_not_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    published: list[WorkflowNotification] = []
    monkeypatch.setattr(
        dispatch,
        "publish_ntfy",
        lambda config, event: published.append(event),
    )
    config = {
        "enabled": True,
        "provider": "ntfy",
        "minimum_success_duration_seconds": 60,
    }

    dispatch.send_workflow_notification(
        config,
        make_event(duration_seconds=59),
    )
    dispatch.send_workflow_notification(
        config,
        make_event(status="failed", duration_seconds=1),
    )

    assert [event.status for event in published] == ["failed"]


def test_dispatch_failure_is_best_effort(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(config: object, event: object) -> None:
        raise OSError("network unavailable")

    monkeypatch.setattr(dispatch, "publish_ntfy", fail)

    with pytest.warns(RuntimeWarning, match="network unavailable"):
        dispatch.send_workflow_notification(
            {"enabled": True, "minimum_success_duration_seconds": 0},
            make_event(),
        )


def test_invalid_dispatch_configuration_is_best_effort() -> None:
    with pytest.warns(RuntimeWarning, match="must be non-negative"):
        dispatch.send_workflow_notification(
            {
                "enabled": True,
                "minimum_success_duration_seconds": -1,
            },
            make_event(),
        )


@pytest.mark.parametrize(
    ("raised", "status"),
    [
        (None, "succeeded"),
        (RuntimeError("training failed"), "failed"),
        (KeyboardInterrupt(), "interrupted"),
    ],
)
def test_workflow_runner_reports_terminal_state_and_preserves_exceptions(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    raised: BaseException | None,
    status: str,
) -> None:
    events: list[WorkflowNotification] = []
    monkeypatch.setattr(
        runtime,
        "send_workflow_notification",
        lambda config, event: events.append(event),
    )

    def implementation(values: object, output_dir: Path) -> None:
        if raised is not None:
            raise raised

    values = {
        "stage": "train_drlr2",
        "task": {"name": "insert"},
        "runtime": {"notifications": {"enabled": True}},
    }

    if raised is None:
        runtime._execute_implementation(values, tmp_path, implementation)
    else:
        with pytest.raises(type(raised)):
            runtime._execute_implementation(values, tmp_path, implementation)

    assert len(events) == 1
    assert events[0].status == status
    assert events[0].error_type == (None if raised is None else type(raised).__name__)
