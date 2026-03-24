"""Shared FFmpeg/FFprobe utilities used across the pipeline."""
from __future__ import annotations

import logging
import os
import subprocess

from scripts.run_cmd import run as run_cmd

logger = logging.getLogger(__name__)


# Global cache for encoder
CACHED_ENCODER = None


def get_best_encoder() -> tuple[str, str]:
    """Return (encoder_name, preset) — auto-detects best available H.264 encoder."""
    global CACHED_ENCODER
    if CACHED_ENCODER:
        return CACHED_ENCODER

    try:
        # Check available encoders
        result = run_cmd(['ffmpeg', '-hide_banner', '-encoders'], check=False, text=True)
        output = result.stdout

        # Priority: NVENC (NVIDIA) > AMF (AMD) > QSV (Intel) > CPU
        if "h264_nvenc" in output:
            logger.info("Encoder Detected: NVIDIA (h264_nvenc)")
            CACHED_ENCODER = ("h264_nvenc", "p1")
            return CACHED_ENCODER

        if "h264_amf" in output:
            logger.info("Encoder Detected: AMD (h264_amf)")
            CACHED_ENCODER = ("h264_amf", "balanced")
            return CACHED_ENCODER

        if "h264_qsv" in output:
            logger.info("Encoder Detected: Intel QSV (h264_qsv)")
            CACHED_ENCODER = ("h264_qsv", "faster")
            return CACHED_ENCODER

        # Mac OS (VideoToolbox)
        if "h264_videotoolbox" in output:
            logger.info("Encoder Detected: MacOS (h264_videotoolbox)")
            CACHED_ENCODER = ("h264_videotoolbox", "default")
            return CACHED_ENCODER

    except Exception as e:
        logger.error(f"Error checking encoders: {e}")

    logger.info("Encoder Detected: CPU (libx264)")
    CACHED_ENCODER = ("libx264", "fast")
    return CACHED_ENCODER


def build_quality_params(encoder_name: str) -> list[str]:
    """Return encoder-specific quality flags for FFmpeg."""
    if "nvenc" in encoder_name:
        return ["-rc:v", "vbr", "-cq", "19", "-maxrate", "15M", "-bufsize", "30M"]
    if "amf" in encoder_name:
        return ["-rc", "vbr_peak", "-qp_i", "19", "-qp_p", "21", "-maxrate", "15M", "-bufsize", "30M"]
    if "qsv" in encoder_name:
        return ["-global_quality", "19", "-maxrate", "15M", "-bufsize", "30M"]
    if "videotoolbox" in encoder_name:
        return ["-q:v", "65", "-maxrate", "15M", "-bufsize", "30M"]
    # CPU (libx264)
    return ["-crf", "18", "-maxrate", "15M", "-bufsize", "30M"]


def _build_preset_flags(encoder_name: str, encoder_preset: str) -> list[str]:
    """Return the correct preset flag(s) for each encoder family.
    AMF uses -quality, VideoToolbox has no preset, others use -preset."""
    if "amf" in encoder_name:
        return ["-quality", encoder_preset]
    if "videotoolbox" in encoder_name:
        return []  # VideoToolbox does not support -preset
    return ["-preset", encoder_preset]


def create_ffmpeg_pipe(
    output_path: str,
    fps: float,
    width: int = 1080,
    height: int = 1920,
) -> subprocess.Popen:
    """Create an FFmpeg subprocess that receives raw BGR24 frames via stdin pipe.

    Returns a Popen with stdin=PIPE ready for frame.tobytes() writes.
    """
    encoder_name, encoder_preset = get_best_encoder()
    cmd = [
        'ffmpeg', '-y', '-loglevel', 'error', '-hide_banner', '-stats',
        '-f', 'rawvideo', '-vcodec', 'rawvideo',
        '-s', f'{width}x{height}', '-pix_fmt', 'bgr24', '-r', str(fps),
        '-i', '-',
        '-c:v', encoder_name,
        *_build_preset_flags(encoder_name, encoder_preset),
        '-pix_fmt', 'yuv420p',
        *build_quality_params(encoder_name),
        output_path,
    ]
    logger.debug("FFmpeg pipe cmd: %s", " ".join(cmd))
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

# FIX: Cache manuel avec verification mtime au lieu de lru_cache.
# lru_cache sur le path seul peut retourner une duree perimee si le fichier
# est remplace entre deux appels (meme chemin, contenu different).
_duration_cache: dict[str, tuple[float, float]] = {}  # path -> (mtime, duration)


def get_video_duration(video_path: str) -> float:
    """Get video duration in seconds via ffprobe (cached par mtime).

    Returns 0.0 on any error.
    """
    # FIX: verifier le mtime pour invalider le cache si le fichier a change
    try:
        current_mtime = os.path.getmtime(video_path)
    except OSError:
        current_mtime = 0.0

    cached = _duration_cache.get(video_path)
    if cached and cached[0] == current_mtime and current_mtime > 0:
        logger.debug("get_video_duration cache hit: %s -> %.2f", video_path, cached[1])
        return cached[1]

    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]
    try:
        result = run_cmd(cmd, text=True, timeout=30)
        duration = float(result.stdout.strip())
        logger.debug("get_video_duration: %s -> %.2f", video_path, duration)  # FIX: log diagnostique
        _duration_cache[video_path] = (current_mtime, duration)
        return duration
    except Exception as e:
        logger.warning("Could not get duration for %s: %s (returning 0.0)", video_path, e)  # FIX: log plus explicite
        return 0.0
