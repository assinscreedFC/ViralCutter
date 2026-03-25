"""Apply speed ramping to video clips — speed up dead moments, slow down highlights."""
from __future__ import annotations

import logging
import os
import subprocess
import tempfile

from scripts.core.run_cmd import run as run_cmd

logger = logging.getLogger(__name__)

# atempo filter range in ffmpeg
_ATEMPO_MIN = 0.5
_ATEMPO_MAX = 2.0


def _build_atempo_chain(factor: float) -> str:
    """Build chained atempo filters for factors outside 0.5-2.0 range."""
    if factor <= 0:
        raise ValueError(f"atempo factor must be positive, got {factor}")
    filters: list[str] = []
    remaining = factor
    while remaining > _ATEMPO_MAX + 0.001:
        filters.append(f"atempo={_ATEMPO_MAX}")
        remaining /= _ATEMPO_MAX
    while remaining < _ATEMPO_MIN - 0.001:
        filters.append(f"atempo={_ATEMPO_MIN}")
        remaining /= _ATEMPO_MIN
    filters.append(f"atempo={remaining:.4f}")
    return ",".join(filters)


def _build_segments(
    duration: float,
    silences: list[dict],
    highlights: list[dict] | None,
    speed_up_factor: float,
    slow_down_factor: float,
) -> list[dict]:
    """Build sorted, non-overlapping segments with assigned speed factors.

    Returns list of {"start": float, "end": float, "speed": float}.
    """
    tagged: list[dict] = []

    for s in silences:
        tagged.append({"start": s["start"], "end": s["end"], "speed": speed_up_factor})

    for h in (highlights or []):
        ts = h["timestamp"]
        dur = h.get("duration", 1.0)
        tagged.append({"start": max(0, ts - dur / 2), "end": min(duration, ts + dur / 2), "speed": slow_down_factor})

    # Sort and merge: later entries (highlights) override earlier (silences) on overlap
    tagged.sort(key=lambda x: x["start"])

    # Fill gaps with 1.0x segments
    segments: list[dict] = []
    cursor = 0.0
    for t in tagged:
        if t["start"] < cursor:
            t["start"] = cursor  # trim overlap
        if t["start"] >= t["end"]:
            continue
        if t["start"] > cursor + 0.01:
            segments.append({"start": cursor, "end": t["start"], "speed": 1.0})
        segments.append(t)
        cursor = t["end"]

    if cursor < duration - 0.01:
        segments.append({"start": cursor, "end": duration, "speed": 1.0})

    return segments


def apply_speed_ramp(
    input_path: str,
    output_path: str,
    silences: list[dict],
    speed_up_factor: float = 1.5,
    highlights: list[dict] | None = None,
    slow_down_factor: float = 0.8,
) -> bool:
    """Apply speed ramping: speed up silences, slow down highlights.

    Returns True if the video was modified, False otherwise.
    """
    from scripts.audio.remove_silence import get_video_duration

    duration = get_video_duration(input_path)
    if duration <= 0:
        logger.warning("Could not get video duration for %s", input_path)
        return False

    segments = _build_segments(duration, silences, highlights, speed_up_factor, slow_down_factor)
    if not segments:
        return False

    # Skip if all segments are 1.0x
    if all(abs(s["speed"] - 1.0) < 0.01 for s in segments):
        logger.info("No speed changes needed for %s", input_path)
        return False

    n = len(segments)
    filters: list[str] = []
    streams: list[str] = []

    for i, seg in enumerate(segments):
        spd = seg["speed"]
        filters.append(
            f"[0:v]trim=start={seg['start']:.4f}:end={seg['end']:.4f},"
            f"setpts=PTS-STARTPTS,setpts=PTS/{spd:.4f}[v{i}]"
        )
        atempo = _build_atempo_chain(spd)
        filters.append(
            f"[0:a]atrim=start={seg['start']:.4f}:end={seg['end']:.4f},"
            f"asetpts=PTS-STARTPTS,{atempo}[a{i}]"
        )
        streams.append(f"[v{i}][a{i}]")

    concat_input = "".join(streams)
    filters.append(f"{concat_input}concat=n={n}:v=1:a=1[outv][outa]")
    filter_complex = ";".join(filters)

    cmd = [
        "ffmpeg", "-y",
        "-loglevel", "error", "-hide_banner",
        "-i", input_path,
        "-filter_complex", filter_complex,
        "-map", "[outv]", "-map", "[outa]",
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
        "-c:a", "aac", "-b:a", "128k",
        output_path,
    ]

    try:
        run_cmd(cmd, text=True)
        logger.info("Speed ramp applied: %s", output_path)
        return True
    except subprocess.CalledProcessError as e:
        logger.error("ffmpeg speed ramp failed: %s", e.stderr[:500])
        return False
