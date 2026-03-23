"""Blur detection: detect blurry frames in video clips using Laplacian variance."""
from __future__ import annotations

import logging

import cv2
import numpy as np

from scripts.frame_utils import downscale_for_analysis

logger = logging.getLogger(__name__)


def detect_blur_frames(
    video_path: str,
    sample_interval: float = 1.0,
    blur_threshold: float = 100.0,
) -> dict:
    """Sample frames and detect blur via Laplacian variance.

    A frame with Laplacian variance below blur_threshold is considered blurry.

    Args:
        video_path: Path to video file.
        sample_interval: Seconds between sampled frames.
        blur_threshold: Laplacian variance below this = blurry.

    Returns:
        dict with keys: blur_ratio, avg_sharpness, min_sharpness, blurry_timestamps.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.warning(f"Could not open video: {video_path}")
        return {
            "blur_ratio": 0.0, "avg_sharpness": 0.0,
            "min_sharpness": 0.0, "blurry_timestamps": [],
        }

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_step = max(1, int(fps * sample_interval))

    sharpness_values = []
    blurry_timestamps = []
    frame_count = 0

    while frame_count < total_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_step == 0:
            small, _ = downscale_for_analysis(frame, max_width=360)
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            sharpness_values.append(laplacian_var)

            if laplacian_var < blur_threshold:
                timestamp = frame_count / fps
                blurry_timestamps.append(round(timestamp, 2))
        frame_count += 1

    cap.release()

    if not sharpness_values:
        return {
            "blur_ratio": 0.0, "avg_sharpness": 0.0,
            "min_sharpness": 0.0, "blurry_timestamps": [],
        }

    total_samples = len(sharpness_values)
    blur_count = len(blurry_timestamps)

    return {
        "blur_ratio": round(blur_count / total_samples, 3) if total_samples > 0 else 0.0,
        "avg_sharpness": round(float(np.mean(sharpness_values)), 1),
        "min_sharpness": round(float(np.min(sharpness_values)), 1),
        "blurry_timestamps": blurry_timestamps,
    }
