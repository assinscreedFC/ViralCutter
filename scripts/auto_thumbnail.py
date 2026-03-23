"""Extract the best frame from a video clip for use as a thumbnail."""
from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from scripts.frame_utils import downscale_for_analysis

logger = logging.getLogger(__name__)

HAARCASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
_face_cascade = cv2.CascadeClassifier(HAARCASCADE_PATH)


def _score_sharpness(gray: np.ndarray) -> float:
    """Laplacian variance — higher = sharper."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _score_faces(gray: np.ndarray) -> tuple[float, list]:
    """Return (score, faces). Score is capped at 1.0 for >=1 face."""
    cascade = _face_cascade
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    faces = list(faces) if len(faces) > 0 else []
    return (min(len(faces), 1.0), faces)


def _score_composition(gray: np.ndarray, faces: list) -> float:
    """Rule of thirds: check if faces or high-contrast areas sit near 1/3 lines."""
    h, w = gray.shape
    third_xs = [w / 3, 2 * w / 3]
    third_ys = [h / 3, 2 * h / 3]

    def _proximity(cx: float, cy: float) -> float:
        dx = min(abs(cx - tx) for tx in third_xs) / (w / 3)
        dy = min(abs(cy - ty) for ty in third_ys) / (h / 3)
        return max(0.0, 1.0 - (dx + dy) / 2)

    if faces:
        scores = [_proximity(x + fw / 2, y + fh / 2) for x, y, fw, fh in faces]
        return max(scores)

    # Fallback: find high-contrast point via Sobel magnitude
    sobel = cv2.magnitude(
        cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3),
        cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3),
    )
    _, _, _, max_loc = cv2.minMaxLoc(sobel)
    return _proximity(float(max_loc[0]), float(max_loc[1]))


def extract_best_frame(video_path: str, sample_count: int = 20) -> tuple[np.ndarray, float]:
    """Sample N evenly-spaced frames and return the best (frame, timestamp_seconds)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    sample_count = min(sample_count, total_frames)
    indices = np.linspace(0, total_frames - 1, sample_count, dtype=int)

    best_score, best_frame, best_ts = -1.0, None, 0.0
    sharpness_values: list[float] = []
    candidates: list[tuple[np.ndarray, float, float, float, list, np.ndarray]] = []

    try:
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if not ok:
                continue
            small, _scale = downscale_for_analysis(frame, max_width=480)
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            sharpness = _score_sharpness(gray)
            face_score, faces = _score_faces(gray)
            ts = float(idx) / fps
            sharpness_values.append(sharpness)
            candidates.append((frame, ts, face_score, sharpness, faces, gray))
    finally:
        cap.release()

    if not candidates:
        raise RuntimeError(f"No readable frames in {video_path}")

    max_sharp = max(sharpness_values) or 1.0
    for frame, ts, face_score, sharpness, faces, gray in candidates:
        norm_sharp = sharpness / max_sharp
        comp_score = _score_composition(gray, faces)
        score = 0.4 * norm_sharp + 0.35 * face_score + 0.25 * comp_score
        if score > best_score:
            best_score, best_frame, best_ts = score, frame, ts

    logger.info("Best thumbnail at %.2fs (score=%.3f) from %s", best_ts, best_score, video_path)
    return best_frame, best_ts


def save_thumbnail(frame: np.ndarray, output_path: str, quality: int = 95) -> str:
    """Save frame as JPEG with given quality. Returns output_path."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    logger.info("Thumbnail saved: %s", output_path)
    return output_path
