"""Shared FFmpeg/FFprobe utilities used across the pipeline."""
from __future__ import annotations

import functools
import logging

from scripts.run_cmd import run as run_cmd

logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=64)
def get_video_duration(video_path: str) -> float:
    """Get video duration in seconds via ffprobe (cached).

    Returns 0.0 on any error.
    """
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]
    try:
        result = run_cmd(cmd, text=True, timeout=30)
        return float(result.stdout.strip())
    except Exception as e:
        logger.warning("Could not get duration for %s: %s", video_path, e)
        return 0.0


# Re-export get_best_encoder from edit_video (single source of truth).
# Lazy import to avoid circular dependencies at module load time.
def get_best_encoder() -> tuple[str, str]:
    """Return (encoder, preset) — delegates to edit_video.get_best_encoder()."""
    from scripts.edit_video import get_best_encoder as _get_best_encoder
    return _get_best_encoder()
