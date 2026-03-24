from __future__ import annotations

import logging
import os
from scripts.run_cmd import run as run_cmd
from scripts.ffmpeg_utils import get_best_encoder, build_quality_params, _build_preset_flags

logger = logging.getLogger(__name__)

LUT_PRESETS = {
    "cinematic": "cinematic.cube",
    "vintage": "vintage.cube",
    "warm": "warm.cube",
    "cool": "cool.cube",
    "high_contrast": "contrast.cube",
}

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_LUT_DIR = os.path.join(PROJECT_ROOT, "luts")


def build_lut_filter(
    lut_name: str = "cinematic",
    intensity: float = 0.7,
    lut_dir: str | None = None,
) -> str | None:
    """Build the LUT filter string for use in a combined filter_complex.

    Returns the filter string (e.g. ``split[a][b];[b]lut3d=...``) or None
    if the LUT file cannot be resolved.  Does NOT run FFmpeg.
    """
    if lut_dir is None:
        lut_dir = DEFAULT_LUT_DIR

    lut_filename = LUT_PRESETS.get(lut_name, lut_name)
    lut_path = os.path.realpath(os.path.join(lut_dir, lut_filename))
    if not lut_path.startswith(os.path.realpath(lut_dir)):
        logger.error("LUT path traversal attempt: %s", lut_name)
        return None

    if not os.path.isfile(lut_path):
        logger.warning("LUT file not found: %s", lut_path)
        return None

    intensity = max(0.0, min(1.0, intensity))

    # Escape backslashes and single quotes for ffmpeg filter string
    safe_lut = lut_path.replace("\\", "/").replace("'", "'\\''")

    return (
        f"split[__lut_a][__lut_b];"
        f"[__lut_b]lut3d='{safe_lut}'[__lut_g];"
        f"[__lut_a][__lut_g]blend=all_mode=normal:all_opacity={intensity}"
    )


def apply_lut(
    input_path: str,
    output_path: str,
    lut_name: str = "cinematic",
    intensity: float = 0.7,
    lut_dir: str | None = None,
) -> bool:
    """Apply a 3D LUT color grading to a video file.

    Args:
        input_path: Path to the source video.
        output_path: Path for the graded output video.
        lut_name: Key from LUT_PRESETS or a direct .cube filename.
        intensity: Blend opacity between original and graded (0.0-1.0).
        lut_dir: Directory containing .cube files. Defaults to ``luts/`` at project root.

    Returns:
        True on success, False on failure.
    """
    if lut_dir is None:
        lut_dir = DEFAULT_LUT_DIR

    lut_filename = LUT_PRESETS.get(lut_name, lut_name)
    lut_path = os.path.realpath(os.path.join(lut_dir, lut_filename))
    if not lut_path.startswith(os.path.realpath(lut_dir)):
        logger.error("LUT path traversal attempt: %s", lut_name)
        return False

    if not os.path.isfile(lut_path):
        logger.warning("LUT file not found: %s", lut_path)
        return False

    if not os.path.isfile(input_path):
        logger.warning("Input video not found: %s", input_path)
        return False

    intensity = max(0.0, min(1.0, intensity))

    # Escape backslashes and single quotes for ffmpeg filter string
    safe_lut = lut_path.replace("\\", "/").replace("'", "'\\''")

    vf = (
        f"split[a][b];"
        f"[b]lut3d='{safe_lut}'[g];"
        f"[a][g]blend=all_mode=normal:all_opacity={intensity}"
    )

    encoder_name, encoder_preset = get_best_encoder()
    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-vf", vf,
        "-c:v", encoder_name, *_build_preset_flags(encoder_name, encoder_preset),
        *build_quality_params(encoder_name),
        "-pix_fmt", "yuv420p",
        "-c:a", "copy",
        output_path,
    ]

    logger.info("Applying LUT '%s' (intensity=%.2f) to %s", lut_name, intensity, input_path)

    try:
        run_cmd(cmd, text=True)
        logger.info("Color grading saved to %s", output_path)
        return True
    except Exception as exc:
        logger.error("ffmpeg color grading failed: %s", getattr(exc, 'stderr', exc))
        return False
