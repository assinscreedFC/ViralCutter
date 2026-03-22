"""Scene detection: detect scene boundaries to avoid mid-scene cuts."""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Try importing scenedetect, fallback to OpenCV-based detection
SCENEDETECT_AVAILABLE = False
try:
    from scenedetect import open_video, SceneManager
    from scenedetect.detectors import ContentDetector
    SCENEDETECT_AVAILABLE = True
except ImportError:
    pass


def detect_scenes_scenedetect(video_path: str, threshold: float = 27.0) -> list[dict]:
    """Detect scenes using PySceneDetect ContentDetector.

    Args:
        video_path: Path to video file.
        threshold: Content change threshold (higher = less sensitive).

    Returns:
        List of {"start": float, "end": float, "scene_idx": int} dicts.
    """
    if not SCENEDETECT_AVAILABLE:
        logger.warning("scenedetect not installed, falling back to OpenCV-based detection")
        return detect_scenes_opencv(video_path, threshold=threshold)

    try:
        video = open_video(video_path)
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector(threshold=threshold))
        scene_manager.detect_scenes(video)
        scene_list = scene_manager.get_scene_list()

        scenes = []
        for i, (start, end) in enumerate(scene_list):
            scenes.append({
                "start": start.get_seconds(),
                "end": end.get_seconds(),
                "scene_idx": i,
            })
        return scenes
    except Exception as e:
        logger.error(f"PySceneDetect failed: {e}")
        return detect_scenes_opencv(video_path, threshold=threshold)


def detect_scenes_opencv(video_path: str, threshold: float = 30.0, sample_interval: float = 0.2) -> list[dict]:
    """Fallback scene detection using OpenCV frame differencing.

    Args:
        video_path: Path to video file.
        threshold: Mean absolute diff threshold to detect scene change.
        sample_interval: Seconds between sampled frames.

    Returns:
        List of {"start": float, "end": float, "scene_idx": int} dicts.
    """
    import cv2
    import numpy as np

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    frame_step = max(1, int(fps * sample_interval))

    boundaries = [0.0]
    prev_gray = None
    frame_idx = 0

    while frame_idx < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_gray is not None:
            diff = cv2.absdiff(prev_gray, gray)
            if np.mean(diff) > threshold:
                timestamp = frame_idx / fps
                boundaries.append(round(timestamp, 3))

        prev_gray = gray
        frame_idx += frame_step

    cap.release()

    boundaries.append(round(duration, 3))

    # Convert boundaries to scene intervals
    scenes = []
    for i in range(len(boundaries) - 1):
        scenes.append({
            "start": boundaries[i],
            "end": boundaries[i + 1],
            "scene_idx": i,
        })

    return scenes


def detect_scenes(video_path: str, threshold: float = 27.0) -> list[dict]:
    """Detect scenes using best available method.

    Args:
        video_path: Path to video file.
        threshold: Detection threshold.

    Returns:
        List of {"start": float, "end": float, "scene_idx": int} dicts.
    """
    if SCENEDETECT_AVAILABLE:
        return detect_scenes_scenedetect(video_path, threshold)
    return detect_scenes_opencv(video_path, threshold)


def validate_cut_boundaries(
    start_sec: float,
    end_sec: float,
    scenes: list[dict],
    min_distance_from_cut: float = 0.5,
) -> dict:
    """Check if cut boundaries fall near scene changes and suggest adjustments.

    Args:
        start_sec: Cut start time.
        end_sec: Cut end time.
        scenes: Scene list from detect_scenes().
        min_distance_from_cut: Minimum distance from a scene boundary.

    Returns:
        dict with keys: cuts_mid_scene, nearest_scene_boundary_start,
        nearest_scene_boundary_end, suggested_start, suggested_end.
    """
    if not scenes:
        return {
            "cuts_mid_scene": False,
            "nearest_scene_boundary_start": None,
            "nearest_scene_boundary_end": None,
            "suggested_start": start_sec,
            "suggested_end": end_sec,
        }

    # Collect all scene boundaries
    boundaries = set()
    for s in scenes:
        boundaries.add(s["start"])
        boundaries.add(s["end"])
    boundaries = sorted(boundaries)

    # Find nearest boundary to start
    nearest_start = min(boundaries, key=lambda b: abs(b - start_sec))
    near_start_dist = abs(nearest_start - start_sec)

    # Find nearest boundary to end
    nearest_end = min(boundaries, key=lambda b: abs(b - end_sec))
    near_end_dist = abs(nearest_end - end_sec)

    # Check if cuts are mid-scene (far from any boundary)
    cuts_mid = near_start_dist > min_distance_from_cut or near_end_dist > min_distance_from_cut

    # Suggest snapping to boundaries if close enough (within 2s)
    suggested_start = start_sec
    if near_start_dist <= 2.0 and near_start_dist > 0.1:
        suggested_start = nearest_start

    suggested_end = end_sec
    if near_end_dist <= 2.0 and near_end_dist > 0.1:
        suggested_end = nearest_end

    # Don't create too-short segments
    if suggested_end - suggested_start < 3.0:
        suggested_start = start_sec
        suggested_end = end_sec

    return {
        "cuts_mid_scene": cuts_mid,
        "nearest_scene_boundary_start": round(nearest_start, 3),
        "nearest_scene_boundary_end": round(nearest_end, 3),
        "suggested_start": round(suggested_start, 3),
        "suggested_end": round(suggested_end, 3),
    }
