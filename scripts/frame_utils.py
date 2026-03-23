"""Frame utilities for analysis — downscale before CPU-heavy operations."""
from __future__ import annotations

import cv2
import numpy as np


def downscale_for_analysis(
    frame: np.ndarray, max_width: int = 480
) -> tuple[np.ndarray, float]:
    """Downscale a frame for analysis (detection, blur, scene diff).

    Returns the resized frame and the scale factor (original / resized).
    If the frame is already smaller than max_width, returns it unchanged.

    Args:
        frame: BGR or grayscale numpy array.
        max_width: Target width in pixels.

    Returns:
        (resized_frame, scale) where scale >= 1.0.
    """
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame, 1.0
    scale = w / max_width
    new_w = max_width
    new_h = int(h / scale)
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale
