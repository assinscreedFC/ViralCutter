"""Centralized subprocess runner with timeout and logging."""
from __future__ import annotations

import logging
import subprocess
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 120  # seconds
FFMPEG_TIMEOUT = 600   # 10 minutes for video processing


def run(
    cmd: list[str],
    *,
    timeout: Optional[int] = None,
    check: bool = True,
    capture_output: bool = True,
    text: bool = False,
    cwd: Optional[str] = None,
) -> subprocess.CompletedProcess:
    """Run a subprocess command with timeout and logging.

    Auto-detects ffmpeg commands and applies longer timeout.
    Logs command on error for easier debugging.
    """
    if timeout is None:
        timeout = FFMPEG_TIMEOUT if (cmd and cmd[0] in ("ffmpeg", "ffprobe")) else DEFAULT_TIMEOUT

    logger.debug(f"Running: {' '.join(cmd[:6])}{'...' if len(cmd) > 6 else ''}")

    try:
        result = subprocess.run(
            cmd,
            timeout=timeout,
            check=check,
            capture_output=capture_output,
            text=text,
            cwd=cwd,
        )
        return result
    except subprocess.TimeoutExpired:
        logger.error(f"Command timed out after {timeout}s: {' '.join(cmd[:4])}")
        raise
    except subprocess.CalledProcessError as e:
        stderr_text = ""
        if e.stderr:
            stderr_text = e.stderr.decode("utf-8", errors="replace") if isinstance(e.stderr, bytes) else e.stderr
        logger.error(f"Command failed (rc={e.returncode}): {' '.join(cmd[:4])}")
        if stderr_text:
            logger.error(f"stderr: {stderr_text[:500]}")
        raise
