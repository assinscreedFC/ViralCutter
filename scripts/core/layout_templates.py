from __future__ import annotations

import logging
import os

from scripts.core.run_cmd import run as run_cmd

logger = logging.getLogger(__name__)

PIP_POSITIONS: dict[str, tuple[str, str]] = {
    "top-left": ("20", "20"),
    "top-right": ("W-w-20", "20"),
    "bottom-left": ("20", "H-h-20"),
    "bottom-right": ("W-w-20", "H-h-20"),
}


def apply_pip_layout(
    main_video: str,
    pip_video: str,
    output_path: str,
    pip_position: str = "bottom-right",
    pip_size: float = 0.25,
) -> bool:
    """Overlay pip_video on main_video at the given corner position."""
    if pip_position not in PIP_POSITIONS:
        logger.error("Invalid pip_position %r, must be one of %s", pip_position, list(PIP_POSITIONS))
        return False

    pip_size = max(0.1, min(0.5, pip_size))
    x_expr, y_expr = PIP_POSITIONS[pip_position]

    filter_complex = (
        f"[1:v]scale=iw*{pip_size}:ih*{pip_size}[pip];"
        f"[0:v][pip]overlay={x_expr}:{y_expr}"
    )

    cmd = [
        "ffmpeg", "-y",
        "-i", main_video,
        "-i", pip_video,
        "-filter_complex", filter_complex,
        "-c:a", "copy",
        output_path,
    ]

    try:
        run_cmd(cmd, text=True)
        logger.info("PiP layout saved to %s", output_path)
        return True
    except Exception as exc:
        logger.error("PiP layout failed: %s", exc)
        return False


def apply_lower_third(
    input_path: str,
    output_path: str,
    text: str,
    display_start: float = 0.0,
    display_duration: float = 3.0,
    font_size: int = 36,
    font_color: str = "white",
    bg_color: str = "black@0.6",
) -> bool:
    """Draw a temporary lower-third text banner on the video."""
    if not os.path.isfile(input_path):
        logger.error("Input file not found: %s", input_path)
        return False

    escaped_text = (
        text.replace("\\", "\\\\")
            .replace("'", r"\'")
            .replace(":", r"\:")
            .replace("%", r"\%")
            .replace("[", r"\[")
            .replace("]", r"\]")
    )
    end_time = display_start + display_duration

    drawtext = (
        f"drawtext=text='{escaped_text}'"
        f":fontsize={font_size}"
        f":fontcolor={font_color}"
        f":x=(w-text_w)/2"
        f":y=h-h/6"
        f":box=1:boxcolor={bg_color}:boxborderw=10"
        f":enable='between(t,{display_start},{end_time})'"
    )

    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-vf", drawtext,
        "-c:a", "copy",
        output_path,
    ]

    try:
        run_cmd(cmd, text=True)
        logger.info("Lower third saved to %s", output_path)
        return True
    except Exception as exc:
        logger.error("Lower third failed: %s", exc)
        return False
