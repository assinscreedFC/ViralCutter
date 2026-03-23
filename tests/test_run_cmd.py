"""Tests pour scripts/run_cmd.py."""
import subprocess
from unittest.mock import patch, MagicMock

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.run_cmd import run as run_cmd


@patch("scripts.run_cmd.subprocess.run")
def test_auto_timeout_ffmpeg(mock_run):
    mock_run.return_value = MagicMock(spec=subprocess.CompletedProcess)
    run_cmd(["ffmpeg", "-i", "input.mp4", "output.mp4"])
    _, kwargs = mock_run.call_args
    assert kwargs["timeout"] == 600


@patch("scripts.run_cmd.subprocess.run")
def test_auto_timeout_ffprobe(mock_run):
    mock_run.return_value = MagicMock(spec=subprocess.CompletedProcess)
    run_cmd(["ffprobe", "-v", "quiet", "input.mp4"])
    _, kwargs = mock_run.call_args
    assert kwargs["timeout"] == 600


@patch("scripts.run_cmd.subprocess.run")
def test_auto_timeout_other(mock_run):
    mock_run.return_value = MagicMock(spec=subprocess.CompletedProcess)
    run_cmd(["yt-dlp", "https://example.com/video"])
    _, kwargs = mock_run.call_args
    assert kwargs["timeout"] == 120


@patch("scripts.run_cmd.subprocess.run")
def test_explicit_timeout(mock_run):
    mock_run.return_value = MagicMock(spec=subprocess.CompletedProcess)
    run_cmd(["ffmpeg", "-i", "input.mp4"], timeout=30)
    _, kwargs = mock_run.call_args
    assert kwargs["timeout"] == 30


@patch("scripts.run_cmd.subprocess.run")
def test_success_returns_result(mock_run):
    expected = MagicMock(spec=subprocess.CompletedProcess)
    expected.returncode = 0
    mock_run.return_value = expected
    result = run_cmd(["echo", "hello"])
    assert result is expected


@patch("scripts.run_cmd.subprocess.run")
def test_check_false(mock_run):
    mock_run.return_value = MagicMock(spec=subprocess.CompletedProcess)
    run_cmd(["false"], check=False)
    _, kwargs = mock_run.call_args
    assert kwargs["check"] is False


@patch("scripts.run_cmd.subprocess.run")
def test_timeout_expired_raises(mock_run):
    mock_run.side_effect = subprocess.TimeoutExpired(cmd=["sleep", "999"], timeout=120)
    with pytest.raises(subprocess.TimeoutExpired):
        run_cmd(["sleep", "999"])


@patch("scripts.run_cmd.subprocess.run")
def test_called_process_error_raises(mock_run):
    error = subprocess.CalledProcessError(
        returncode=1,
        cmd=["ffmpeg", "-i", "bad.mp4"],
        stderr=b"No such file or directory",
    )
    mock_run.side_effect = error
    with pytest.raises(subprocess.CalledProcessError):
        run_cmd(["ffmpeg", "-i", "bad.mp4"])
