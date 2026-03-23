"""Shared pytest fixtures for ViralCutter test suite.

Centralises path setup and common fixtures to eliminate duplication
across test files.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Path bootstrap — must happen at module level, before any project import
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_segments() -> dict:
    """Two realistic Segment-compatible dicts wrapped in a segments envelope."""
    seg1 = {
        "start_time": "00:01:30",
        "end_time": "00:02:15",
        "duration": 45.0,
        "title": "Test Segment 1",
        "description": "First test",
        "tiktok_caption": "#test",
        "zoom_cues": [],
        "power_words": ["amazing"],
        "score": 75.0,
    }
    seg2 = {
        "start_time": "00:03:00",
        "end_time": "00:04:00",
        "duration": 60.0,
        "title": "Test Segment 2",
        "description": "Second test",
        "tiktok_caption": "#viral #fyp",
        "zoom_cues": [{"time": 10.0, "scale": 1.3}],
        "power_words": ["incredible", "must-watch"],
        "score": 88.0,
    }
    return {"segments": [seg1, seg2]}


@pytest.fixture
def mock_subprocess():
    """Patch subprocess.run globally and return the MagicMock."""
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(
            spec=subprocess.CompletedProcess,
            returncode=0,
            stdout="",
            stderr="",
        )
        yield mock_run


@pytest.fixture
def tmp_project(tmp_path: Path) -> Path:
    """Temporary project directory with the standard ViralCutter layout.

    Structure::

        <tmp_path>/
            input.mp4   (empty placeholder)
            cuts/       (empty directory)
            subs/       (empty directory)
    """
    (tmp_path / "input.mp4").touch()
    (tmp_path / "cuts").mkdir()
    (tmp_path / "subs").mkdir()
    return tmp_path
