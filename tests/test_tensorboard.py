from unittest.mock import MagicMock

import pytest

from learning.tensorboard import tensorboard_session


def test_print_mode_quotes_paths_without_starting_process(
    tmp_path, monkeypatch, capsys
):
    launch = MagicMock()
    monkeypatch.setattr("learning.tensorboard.subprocess.Popen", launch)
    with tensorboard_session({"mode": "print"}, tmp_path / "space here"):
        pass
    assert "'" in capsys.readouterr().out
    launch.assert_not_called()


def test_dry_run_never_serves_or_opens_browser(tmp_path, monkeypatch):
    launch = MagicMock()
    browser = MagicMock()
    monkeypatch.setattr("learning.tensorboard.subprocess.Popen", launch)
    monkeypatch.setattr("learning.tensorboard.webbrowser.open", browser)
    with tensorboard_session(
        {"mode": "serve", "open_browser": True}, tmp_path, dry_run=True
    ):
        pass
    launch.assert_not_called()
    browser.assert_not_called()


def test_missing_tensorboard_does_not_stop_workflow(tmp_path, monkeypatch):
    monkeypatch.setattr("learning.tensorboard.shutil.which", lambda name: None)
    with pytest.warns(UserWarning, match="Install"):
        with tensorboard_session({"mode": "serve"}, tmp_path):
            reached = True
    assert reached


def test_owned_server_is_stopped_on_training_failure(tmp_path, monkeypatch):
    process = MagicMock()
    process.poll.return_value = None
    monkeypatch.setattr(
        "learning.tensorboard.shutil.which", lambda name: "/bin/tensorboard"
    )
    monkeypatch.setattr("learning.tensorboard.socket.socket", MagicMock())
    monkeypatch.setattr(
        "learning.tensorboard.subprocess.Popen", lambda *a, **k: process
    )
    monkeypatch.setattr("learning.tensorboard.urllib.request.urlopen", MagicMock())
    with pytest.raises(RuntimeError, match="training failed"):
        with tensorboard_session({"mode": "serve"}, tmp_path):
            raise RuntimeError("training failed")
    process.terminate.assert_called_once()
    process.wait.assert_called_once()
