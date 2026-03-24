"""Overlay effects for video post-production: progress bar, transitions, emoji."""
from __future__ import annotations

import json
import logging
import os
import re
import tempfile

from scripts.run_cmd import run as run_cmd
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

# Moved to scripts.ffmpeg_utils.
from scripts.ffmpeg_utils import get_video_duration, get_best_encoder, build_quality_params, _build_preset_flags


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

    duration = get_video_duration(input_path)
    logger.info("Progress bar: duration=%.2f for %s", duration, input_path)  # FIX: log diagnostique
    if duration <= 0:
        logger.error("Cannot add progress bar: invalid duration (%.2f)", duration)
        return False

    y_expr = "0" if bar_position == "top" else f"ih-{bar_height}"
    drawbox = (
        f"drawbox=x=0:y={y_expr}:"
        f"w=iw*t/{duration}:h={bar_height}:"
        f"color={bar_color}@0.8:thickness=fill"  # FIX: 'thickness' au lieu de 't' pour eviter ambiguite avec variable temps
    )

    encoder_name, encoder_preset = get_best_encoder()
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-vf", drawbox,
        "-c:v", encoder_name, *_build_preset_flags(encoder_name, encoder_preset),
        *build_quality_params(encoder_name),
        "-pix_fmt", "yuv420p",
        "-c:a", "copy",
        output_path,
    ]
    logger.debug("Progress bar cmd: %s", " ".join(cmd))  # FIX: log commande avant execution
    try:
        run_cmd(cmd, text=True)
        logger.info("Progress bar added → %s", output_path)
        return True
    except Exception as e:
        logger.error("add_progress_bar failed: %s", e)
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

    dur_a = get_video_duration(video_a)
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
        run_cmd(cmd, text=True)
        logger.info("Transition '%s' applied → %s", transition_type, output)
        return True
    except Exception as e:
        logger.error("add_transition failed: %s", e)
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
        encoder_name, encoder_preset = get_best_encoder()
        cmd = [
            "ffmpeg", "-y",
            *inputs,
            "-filter_complex", filter_complex,
            "-c:v", encoder_name, *_build_preset_flags(encoder_name, encoder_preset),
            *build_quality_params(encoder_name),
            "-pix_fmt", "yuv420p",
            "-c:a", "copy",
            output_path,
        ]
        run_cmd(cmd, text=True)
        logger.info("Emoji overlay applied (%d emojis) → %s", len(emojis), output_path)
        return True

    except Exception as e:
        logger.error("add_emoji_overlay failed: %s", e)
        return False
    finally:
        for p in png_paths:
            try:
                os.unlink(p)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Combined single-pass post-production
# ---------------------------------------------------------------------------

_VALID_COLOR = re.compile(r'^[a-zA-Z0-9#@.]+$')


def apply_post_production(
    input_path: str,
    output_path: str,
    *,
    # Color grading params
    lut_name: str | None = None,
    lut_intensity: float = 0.5,
    lut_dir: str | None = None,
    # Progress bar params
    progress_bar: bool = False,
    bar_color: str = "white",
    bar_height: int = 6,
    bar_position: str = "bottom",
    # Emoji params
    emojis: list[dict] | None = None,
) -> bool:
    """Apply color grading + progress bar + emoji overlay in ONE FFmpeg pass.

    Only enabled effects are included in the filter chain.  If no effects
    are enabled the function returns True immediately (no re-encode).

    Returns True on success, False on failure.
    """
    from scripts.color_grading import build_lut_filter

    # ---- Determine which effects are active --------------------------------
    has_lut = lut_name is not None
    has_bar = progress_bar
    has_emoji = bool(emojis)

    if not (has_lut or has_bar or has_emoji):
        logger.debug("apply_post_production: no effects enabled, skipping")
        return True

    # ---- Validate inputs early ---------------------------------------------
    if has_bar:
        if not _VALID_COLOR.match(bar_color):
            logger.error("Invalid bar_color: %s", bar_color)
            return False
        if bar_position not in ("top", "bottom"):
            logger.error("Invalid bar_position: %s", bar_position)
            return False

    duration = get_video_duration(input_path)
    if duration <= 0:
        logger.error("apply_post_production: invalid duration (%.2f) for %s", duration, input_path)
        return False

    # ---- Build filter chain ------------------------------------------------
    # We use filter_complex (not -vf) because emoji overlays need multiple
    # inputs.  Each stage consumes the previous tag and produces the next.
    #
    # Convention: current_tag tracks the *output* of the last filter.
    # When a stage is the very last one it must NOT emit a tag (FFmpeg
    # requirement: the final output is untagged).

    filters: list[str] = []
    current_tag = "[0:v]"
    extra_inputs: list[str] = []   # -i flags for emoji PNGs
    png_paths: list[str] = []      # for cleanup
    # emoji input index offset: 0 is the video, emoji PNGs start at 1
    emoji_input_offset = 1

    try:
        # -- LUT color grading -----------------------------------------------
        if has_lut:
            lut_filter = build_lut_filter(
                lut_name=lut_name,
                intensity=lut_intensity,
                lut_dir=lut_dir,
            )
            if lut_filter is None:
                logger.warning("LUT filter could not be built, skipping color grading")
                has_lut = False
            else:
                # The build_lut_filter uses internal tags __lut_a/__lut_b/__lut_g.
                # We need to feed current_tag into split and capture output.
                out_tag = "[__pp_lut]"
                # Replace the bare "split" with "{current_tag}split" and append output tag
                lut_adapted = lut_filter.replace(
                    "split[__lut_a][__lut_b]",
                    f"{current_tag}split[__lut_a][__lut_b]",
                )
                # The blend output needs a tag for the next stage
                lut_adapted += out_tag
                filters.append(lut_adapted)
                current_tag = out_tag

        # -- Progress bar (drawbox) ------------------------------------------
        if has_bar:
            y_expr = "0" if bar_position == "top" else f"ih-{bar_height}"
            out_tag = "[__pp_bar]"
            drawbox = (
                f"{current_tag}drawbox=x=0:y={y_expr}:"
                f"w=iw*t/{duration}:h={bar_height}:"
                f"color={bar_color}@0.8:thickness=fill"
                f"{out_tag}"
            )
            filters.append(drawbox)
            current_tag = out_tag

        # -- Emoji overlays --------------------------------------------------
        if has_emoji:
            for idx, entry in enumerate(emojis):
                emoji_name = entry.get("emoji", "fire")
                timestamp = float(entry.get("timestamp", 0))
                dur = float(entry.get("duration", 1.5))
                pos_key = entry.get("position", "center")

                png_path = _render_emoji_png(emoji_name)
                if png_path is None:
                    logger.warning("Could not render emoji '%s', skipping it", emoji_name)
                    continue
                png_paths.append(png_path)
                extra_inputs.extend(["-i", png_path])

                x_expr, y_expr_e = _POSITION_MAP.get(pos_key, _POSITION_MAP["center"])
                end_t = timestamp + dur
                input_idx = emoji_input_offset + len(png_paths) - 1
                out_tag = f"[__pp_emo{idx}]"

                filters.append(
                    f"{current_tag}[{input_idx}:v]overlay={x_expr}:{y_expr_e}"
                    f":enable='between(t,{timestamp},{end_t})'{out_tag}"
                )
                current_tag = out_tag

        # If after skipping we have no filters at all, bail out
        if not filters:
            logger.debug("apply_post_production: all requested effects were skipped")
            return False

        filter_complex = ";".join(filters)
        logger.debug("apply_post_production filter_complex: %s", filter_complex)

        # ---- Build and run FFmpeg command -----------------------------------
        # Use -map to select the last named output pad (more robust than
        # stripping the output tag via string manipulation).
        encoder_name, encoder_preset = get_best_encoder()
        cmd = [
            "ffmpeg", "-y",
            "-i", input_path,
            *extra_inputs,
            "-filter_complex", filter_complex,
            "-map", current_tag,
            "-map", "0:a?",
            "-c:v", encoder_name, *_build_preset_flags(encoder_name, encoder_preset),
            *build_quality_params(encoder_name),
            "-pix_fmt", "yuv420p",
            "-c:a", "copy",
            output_path,
        ]
        logger.debug("apply_post_production cmd: %s", " ".join(cmd))
        run_cmd(cmd, text=True)
        logger.info(
            "Post-production applied (lut=%s, bar=%s, emoji=%d) -> %s",
            bool(has_lut), has_bar, len(png_paths), output_path,
        )
        return True

    except Exception as e:
        logger.error("apply_post_production failed: %s", e)
        return False
    finally:
        for p in png_paths:
            try:
                os.unlink(p)
            except OSError:
                pass
