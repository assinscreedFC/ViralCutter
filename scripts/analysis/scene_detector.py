"""Scene detection: detect scene boundaries to avoid mid-scene cuts."""
from __future__ import annotations

import logging

from scripts.core.ffmpeg_utils import get_video_duration
from scripts.core.frame_utils import downscale_for_analysis

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


def detect_scenes_opencv(video_path: str, threshold: float = 30.0, sample_interval: float = 0.5) -> list[dict]:
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
    frame_count = 0

    while frame_count < total_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_step == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray, _ = downscale_for_analysis(gray, max_width=240)
            if prev_gray is not None:
                diff = cv2.absdiff(prev_gray, gray)
                if np.mean(diff) > threshold:
                    timestamp = frame_count / fps
                    boundaries.append(round(timestamp, 3))
            prev_gray = gray
        frame_count += 1

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


def _detect_scenes_ffmpeg_range(
    video_path: str, start_sec: float, duration_sec: float, scene_threshold: float = 0.4,
) -> list[float]:
    """Detect scene change timestamps in a time range using ffmpeg.

    Uses ffmpeg's native scene detection (C-optimized, hardware-accelerated
    decoding). Much faster than Python/OpenCV for 4K videos.

    Returns list of scene-change timestamps (absolute, not relative to range).
    """
    import re
    import subprocess

    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "info",
        "-ss", f"{start_sec:.3f}",
        "-t", f"{duration_sec:.3f}",
        "-i", video_path,
        "-vf", f"select='gt(scene,{scene_threshold})',showinfo",
        "-an", "-f", "null",
    ]
    # Use /dev/null on Unix, NUL on Windows
    import sys
    cmd.append("NUL" if sys.platform == "win32" else "/dev/null")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        stderr = result.stderr
    except (subprocess.TimeoutExpired, Exception) as e:
        logger.warning(f"ffmpeg scene detection failed for range {start_sec:.0f}-{start_sec+duration_sec:.0f}: {e}")
        return []

    # Parse scene timestamps from showinfo output:
    # [Parsed_showinfo_...] n:  5 pts: 163840 pts_time:12.8
    timestamps = []
    for match in re.finditer(r'pts_time:([\d.]+)', stderr):
        relative_ts = float(match.group(1))
        absolute_ts = round(start_sec + relative_ts, 3)
        timestamps.append(absolute_ts)

    return timestamps


def detect_scenes_for_segments(
    video_path: str,
    segments: list[dict],
    threshold: float = 27.0,
    margin_sec: float = 10.0,
) -> list[dict]:
    """Detect scenes only around segment boundaries using ffmpeg.

    Uses ffmpeg's native scene detection with fast seeking per range.
    For a 55min 4K video with 9 segments, scans ~5min of footage in ~3min
    instead of ~55min for the full video.

    Quality: identical scene boundaries detected for the segment regions.

    Args:
        video_path: Path to video file.
        segments: List of segment dicts with start_time/end_time or duration.
        threshold: OpenCV-style threshold (converted to ffmpeg scene threshold).
        margin_sec: Seconds to scan before/after each segment boundary.

    Returns:
        List of {"start": float, "end": float, "scene_idx": int} dicts.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    # Get video duration via cached ffprobe
    video_duration = get_video_duration(video_path)

    # Collect time ranges to scan (±margin around each segment boundary)
    ranges: list[tuple[float, float]] = []
    for seg in segments:
        st = seg.get("start_time", 0)
        try:
            start = float(st)
        except (ValueError, TypeError):
            start = 0.0

        dur = seg.get("duration", 0)
        try:
            duration = float(dur)
        except (ValueError, TypeError):
            duration = 0.0

        end = start + duration
        ranges.append((max(0, start - margin_sec), start + margin_sec))
        ranges.append((max(0, end - margin_sec), end + margin_sec))

    # Merge overlapping ranges
    ranges.sort()
    merged: list[tuple[float, float]] = []
    for r_start, r_end in ranges:
        if merged and r_start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], r_end))
        else:
            merged.append((r_start, r_end))

    scan_seconds = sum(r[1] - r[0] for r in merged)
    logger.info(f"Scene detection (ffmpeg): scanning {scan_seconds:.0f}s in "
                f"{len(merged)} ranges for {len(segments)} segments")

    # Convert OpenCV-style threshold to ffmpeg scene threshold
    # OpenCV absdiff mean ~27-30 ≈ ffmpeg scene ~0.4
    scene_thresh = max(0.3, min(0.6, threshold / 70.0))

    # Run ffmpeg scene detection per range in parallel
    all_boundaries: set[float] = set()
    max_workers = min(4, len(merged))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _detect_scenes_ffmpeg_range,
                video_path, r_start, r_end - r_start, scene_thresh,
            ): (r_start, r_end)
            for r_start, r_end in merged
        }
        for future in as_completed(futures):
            r_start, r_end = futures[future]
            try:
                timestamps = future.result()
                all_boundaries.update(timestamps)
                if timestamps:
                    logger.debug(f"  Range {r_start:.0f}-{r_end:.0f}: {len(timestamps)} changes")
            except Exception as e:
                logger.warning(f"  Range {r_start:.0f}-{r_end:.0f} failed: {e}")

    # Build scene intervals from boundaries
    sorted_bounds = sorted(all_boundaries)
    if not sorted_bounds:
        return [{"start": 0.0, "end": video_duration, "scene_idx": 0}]

    all_bounds = [0.0] + sorted_bounds + [round(video_duration, 3)]
    scenes = []
    for i in range(len(all_bounds) - 1):
        scenes.append({
            "start": all_bounds[i],
            "end": all_bounds[i + 1],
            "scene_idx": i,
        })

    logger.info(f"Scene detection complete: {len(scenes)} scenes from {len(all_boundaries)} boundaries")
    return scenes


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
