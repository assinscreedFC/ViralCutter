"""Overlay effects for video post-production: progress bar, transitions, emoji."""
from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import tempfile
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from PIL import Image, ImageDraw, ImageFont
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False
    logger.debug("Pillow not available — emoji overlay will be disabled")


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _get_duration(video_path: str) -> float:
    """Get video duration in seconds via ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "json",
        video_path,
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=True, timeout=30,
        )
        data = json.loads(result.stdout)
        return float(data["format"]["duration"])
    except (subprocess.CalledProcessError, KeyError, ValueError, json.JSONDecodeError) as e:
        logger.error("Failed to get duration for %s: %s", video_path, e)
        return 0.0


# ---------------------------------------------------------------------------
# B3 — Progress bar overlay
# ---------------------------------------------------------------------------

def add_progress_bar(
    input_path: str,
    output_path: str,
    bar_color: str = "white",
    bar_height: int = 6,
    bar_position: str = "top",
) -> bool:
    """Draw an animated progress bar that fills across the video duration.

    Args:
        bar_position: "top" (y=0) or "bottom" (y=ih-height).
        bar_color: Any ffmpeg color name or hex (e.g. "white", "0xFF0000").
    """
    _VALID_COLOR = re.compile(r'^[a-zA-Z0-9#@.]+$')
    if not _VALID_COLOR.match(bar_color):
        logger.error("Invalid bar_color: %s", bar_color)
        return False
    if bar_position not in ("top", "bottom"):
        logger.error("Invalid bar_position: %s", bar_position)
        return False

    duration = _get_duration(input_path)
    if duration <= 0:
        logger.error("Cannot add progress bar: invalid duration")
        return False

    y_expr = "0" if bar_position == "top" else f"ih-{bar_height}"
    drawbox = (
        f"drawbox=x=0:y={y_expr}:"
        f"w='iw*t/{duration}':h={bar_height}:"
        f"color={bar_color}@0.8:t=fill"
    )

    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-vf", drawbox,
        "-c:a", "copy",
        output_path,
    ]
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=600)
        logger.info("Progress bar added → %s", output_path)
        return True
    except subprocess.CalledProcessError as e:
        logger.error("add_progress_bar failed: %s", e.stderr[:500] if e.stderr else e)
        return False


# ---------------------------------------------------------------------------
# B9 — Transitions between clips
# ---------------------------------------------------------------------------

_SUPPORTED_TRANSITIONS = {"fade", "wipeleft", "wiperight", "slideup", "slidedown", "circlecrop"}


def add_transition(
    video_a: str,
    video_b: str,
    output: str,
    transition_type: str = "fade",
    duration: float = 0.5,
) -> bool:
    """Join two clips with an xfade transition.

    Args:
        transition_type: One of fade, wipeleft, wiperight, slideup, slidedown, circlecrop.
        duration: Transition duration in seconds.
    """
    if transition_type not in _SUPPORTED_TRANSITIONS:
        logger.error("Unsupported transition '%s'. Use one of %s", transition_type, _SUPPORTED_TRANSITIONS)
        return False

    dur_a = _get_duration(video_a)
    if dur_a <= 0:
        logger.error("Cannot compute xfade offset: invalid duration for %s", video_a)
        return False

    offset = max(0.0, dur_a - duration)

    cmd = [
        "ffmpeg", "-y",
        "-i", video_a,
        "-i", video_b,
        "-filter_complex",
        f"xfade=transition={transition_type}:duration={duration}:offset={offset}",
        "-c:a", "copy",
        output,
    ]
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=600)
        logger.info("Transition '%s' applied → %s", transition_type, output)
        return True
    except subprocess.CalledProcessError as e:
        logger.error("add_transition failed: %s", e.stderr[:500] if e.stderr else e)
        return False


# ---------------------------------------------------------------------------
# B16 — Emoji / sticker overlay
# ---------------------------------------------------------------------------

_POSITION_MAP = {
    "top-left":     ("10", "10"),
    "top-right":    ("main_w-overlay_w-10", "10"),
    "bottom-left":  ("10", "main_h-overlay_h-10"),
    "bottom-right": ("main_w-overlay_w-10", "main_h-overlay_h-10"),
    "center":       ("(main_w-overlay_w)/2", "(main_h-overlay_h)/2"),
}

# Common emoji shortnames → unicode codepoints
_EMOJI_MAP: dict[str, str] = {
    "fire": "\U0001F525",
    "heart": "\u2764\uFE0F",
    "thumbsup": "\U0001F44D",
    "laugh": "\U0001F602",
    "star": "\u2B50",
    "100": "\U0001F4AF",
    "clap": "\U0001F44F",
    "rocket": "\U0001F680",
    "eyes": "\U0001F440",
    "crown": "\U0001F451",
}


def _render_emoji_png(emoji_text: str, size: int = 128) -> Optional[str]:
    """Render emoji text to a temporary PNG file via Pillow. Returns path or None."""
    if not _HAS_PIL:
        logger.error("Pillow is required for emoji overlay")
        return None

    resolved = _EMOJI_MAP.get(emoji_text, emoji_text)

    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Try system emoji fonts, fall back to default
    font = None
    font_candidates = [
        "C:/Windows/Fonts/seguiemj.ttf",  # Windows Segoe UI Emoji
        "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf",
        "/System/Library/Fonts/Apple Color Emoji.ttc",
    ]
    font_size = int(size * 0.75)
    for fpath in font_candidates:
        if os.path.isfile(fpath):
            try:
                font = ImageFont.truetype(fpath, font_size)
                break
            except (OSError, IOError):
                continue
    if font is None:
        font = ImageFont.load_default()

    draw.text((size // 8, size // 8), resolved, font=font, embedded_color=True)

    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    img.save(tmp.name, "PNG")
    tmp.close()
    return tmp.name


def add_emoji_overlay(
    input_path: str,
    output_path: str,
    emojis: list[dict],
) -> bool:
    """Overlay emoji/sticker images at timed positions.

    Args:
        emojis: list of dicts with keys:
            emoji (str): emoji shortname or unicode char
            timestamp (float): start time in seconds
            duration (float): display duration in seconds
            position (str): top-left|top-right|bottom-left|bottom-right|center
    """
    if not emojis:
        logger.warning("No emojis provided, skipping")
        return False

    png_paths: list[str] = []
    inputs = ["-i", input_path]
    overlay_chain = "[0:v]"
    filters: list[str] = []

    try:
        for idx, entry in enumerate(emojis):
            emoji_name = entry.get("emoji", "fire")
            timestamp = float(entry.get("timestamp", 0))
            dur = float(entry.get("duration", 1.5))
            pos_key = entry.get("position", "center")

            png_path = _render_emoji_png(emoji_name)
            if png_path is None:
                return False
            png_paths.append(png_path)
            inputs.extend(["-i", png_path])

            x_expr, y_expr = _POSITION_MAP.get(pos_key, _POSITION_MAP["center"])
            end_t = timestamp + dur
            tag_in = overlay_chain
            tag_out = f"[ov{idx}]"

            filters.append(
                f"{tag_in}[{idx + 1}:v]overlay={x_expr}:{y_expr}"
                f":enable='between(t,{timestamp},{end_t})'{tag_out}"
            )
            overlay_chain = tag_out

        # Last filter outputs without tag
        last = filters[-1]
        last_tag = f"[ov{len(emojis) - 1}]"
        filters[-1] = last.replace(last_tag, "")

        filter_complex = ";".join(filters)
        cmd = [
            "ffmpeg", "-y",
            *inputs,
            "-filter_complex", filter_complex,
            "-c:a", "copy",
            output_path,
        ]
        subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=600)
        logger.info("Emoji overlay applied (%d emojis) → %s", len(emojis), output_path)
        return True

    except subprocess.CalledProcessError as e:
        logger.error("add_emoji_overlay failed: %s", e.stderr[:500] if e.stderr else e)
        return False
    finally:
        for p in png_paths:
            try:
                os.unlink(p)
            except OSError:
                pass
