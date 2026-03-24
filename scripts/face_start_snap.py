"""Face-start snap: ensure the first frame of a clip shows a face."""
from __future__ import annotations

import logging
import os

import cv2

logger = logging.getLogger(__name__)

# Haar cascade path (same as used in clip_validator.py)
_CASCADE_PATH = os.path.join(
    os.path.dirname(cv2.__file__), "data", "haarcascade_frontalface_default.xml"
)


def snap_to_first_face(
    video_path: str,
    start_time: float,
    max_search_forward: float = 3.0,
    sample_interval: float = 0.1,
) -> float:
    """Scan forward from *start_time* to find the first frame with a face.

    Uses Haar Cascade (lightweight, no GPU needed) to detect faces.
    Samples one frame every *sample_interval* seconds for up to
    *max_search_forward* seconds.

    Returns:
        Adjusted start_time (unchanged if a face is already present at
        *start_time* or if no face is found within the search window).
    """
    if not os.path.isfile(video_path):
        return start_time

    cascade = cv2.CascadeClassifier(_CASCADE_PATH)
    if cascade.empty():
        logger.warning("Haar cascade not found at %s — skipping face snap", _CASCADE_PATH)
        return start_time

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return start_time

    start_ms = start_time * 1000.0
    max_ms = (start_time + max_search_forward) * 1000.0
    interval_ms = sample_interval * 1000.0

    try:
        current_ms = start_ms
        while current_ms <= max_ms:
            cap.set(cv2.CAP_PROP_POS_MSEC, current_ms)
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,       # finer scale search (was 1.15)
                minNeighbors=6,        # higher confidence (was 4)
                minSize=(120, 120),    # larger min face (was 80,80)
            )

            if len(faces) > 0:
                # Filter: aspect ratio ~square and minimum relative size
                h_frame = gray.shape[0]
                min_face_h = int(h_frame * 0.08)  # face >= 8% of frame height
                valid = [
                    (x, y, w, h) for (x, y, w, h) in faces
                    if 0.7 < (w / h if h > 0 else 0) < 1.4 and h >= min_face_h
                ]
                if valid:
                    found_time = current_ms / 1000.0
                    if found_time != start_time:
                        logger.debug(
                            "Face snap: found face at %.2fs (searched from %.2fs)",
                            found_time, start_time,
                        )
                    return found_time

            current_ms += interval_ms
    finally:
        cap.release()

    # No face found — keep original start
    return start_time
