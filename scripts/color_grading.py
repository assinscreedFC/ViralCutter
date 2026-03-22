from __future__ import annotations

import logging
import os
import subprocess

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

    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-vf", vf,
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
        "-c:a", "copy",
        output_path,
    ]

    logger.info("Applying LUT '%s' (intensity=%.2f) to %s", lut_name, intensity, input_path)

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info("Color grading saved to %s", output_path)
        return True
    except subprocess.CalledProcessError as exc:
        logger.error("ffmpeg color grading failed: %s", exc.stderr)
        return False
