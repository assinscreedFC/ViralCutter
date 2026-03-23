"""
B-roll auto-insert — Fetch stock footage from Pexels and insert at static moments.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
from urllib.parse import quote

import cv2
import numpy as np
import requests

logger = logging.getLogger(__name__)


def _has_audio_stream(video_path: str) -> bool:
    """Return True if the video file contains at least one audio stream."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "a:0",
        "-show_entries", "stream=codec_type",
        "-of", "json",
        video_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        data = json.loads(result.stdout)
        return bool(data.get("streams"))
    except (subprocess.SubprocessError, json.JSONDecodeError, OSError):
        return False


def fetch_broll_from_pexels(
    query: str, api_key: str, output_dir: str, count: int = 3
) -> list[str]:
    """Download B-roll clips from Pexels Videos API.

    Returns list of downloaded file paths.
    """
    count = min(count, 10)
    os.makedirs(output_dir, exist_ok=True)
    url = f"https://api.pexels.com/videos/search?query={quote(query)}&per_page={count}"
    headers = {"Authorization": api_key}

    try:
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as e:
        logger.error("Pexels API request failed: %s", e)
        return []

    data = resp.json()
    videos = data.get("videos", [])
    if not videos:
        logger.warning("No Pexels results for query '%s'", query)
        return []

    downloaded: list[str] = []
    for video in videos:
        video_files = video.get("video_files", [])
        if not video_files:
            continue

        # Pick highest quality file
        best = max(video_files, key=lambda f: (f.get("height", 0) or 0))
        download_url = best.get("link")
        if not download_url:
            continue

        filename = f"broll_{video['id']}.mp4"
        filepath = os.path.join(output_dir, filename)

        try:
            with requests.get(download_url, stream=True, timeout=120) as r:
                r.raise_for_status()
                with open(filepath, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            downloaded.append(filepath)
            logger.info("Downloaded B-roll: %s", filepath)
        except requests.RequestException as e:
            logger.error("Failed to download %s: %s", download_url, e)

    return downloaded


def detect_static_moments(
    video_path: str,
    motion_threshold: float = 2.0,
    sample_interval: float = 1.0,
) -> list[dict]:
    """Detect low-motion intervals using frame differencing.

    Returns list of {"start": float, "end": float, "motion_score": float}.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error("Cannot open video: %s", video_path)
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_step = max(1, int(fps * sample_interval))

    prev_gray = None
    static_frames: list[dict] = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_step == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev_gray is not None:
                diff = cv2.absdiff(prev_gray, gray)
                score = float(np.mean(diff))
                timestamp = frame_idx / fps

                if score < motion_threshold:
                    static_frames.append({
                        "timestamp": timestamp,
                        "motion_score": round(score, 3),
                    })
            prev_gray = gray

        frame_idx += 1

    cap.release()

    # Merge consecutive static frames into intervals
    if not static_frames:
        return []

    intervals: list[dict] = []
    current_start = static_frames[0]["timestamp"]
    current_end = current_start + sample_interval
    scores = [static_frames[0]["motion_score"]]

    for sf in static_frames[1:]:
        if sf["timestamp"] <= current_end + sample_interval * 1.5:
            current_end = sf["timestamp"] + sample_interval
            scores.append(sf["motion_score"])
        else:
            intervals.append({
                "start": round(current_start, 2),
                "end": round(current_end, 2),
                "motion_score": round(sum(scores) / len(scores), 3),
            })
            current_start = sf["timestamp"]
            current_end = current_start + sample_interval
            scores = [sf["motion_score"]]

    intervals.append({
        "start": round(current_start, 2),
        "end": round(current_end, 2),
        "motion_score": round(sum(scores) / len(scores), 3),
    })

    return intervals


def insert_broll(
    main_video: str,
    broll_video: str,
    output_path: str,
    insert_start: float,
    insert_duration: float,
) -> bool:
    """Insert a B-roll clip into the main video at insert_start for insert_duration seconds.

    Uses ffmpeg trim + concat. Returns True on success.
    """
    try:
        # Check if both inputs have audio streams
        main_has_audio = _has_audio_stream(main_video)
        broll_has_audio = _has_audio_stream(broll_video)
        use_audio = main_has_audio and broll_has_audio

        if use_audio:
            filter_complex = (
                f"[0:v]trim=0:{insert_start},setpts=PTS-STARTPTS[before_v];"
                f"[0:a]atrim=0:{insert_start},asetpts=PTS-STARTPTS[before_a];"
                f"[1:v]trim=0:{insert_duration},setpts=PTS-STARTPTS[broll_v];"
                f"[1:a]atrim=0:{insert_duration},asetpts=PTS-STARTPTS[broll_a];"
                f"[0:v]trim=start={insert_start + insert_duration},"
                f"setpts=PTS-STARTPTS[after_v];"
                f"[0:a]atrim=start={insert_start + insert_duration},"
                f"asetpts=PTS-STARTPTS[after_a];"
                f"[before_v][before_a][broll_v][broll_a][after_v][after_a]"
                f"concat=n=3:v=1:a=1[outv][outa]"
            )
            map_args = ["-map", "[outv]", "-map", "[outa]"]
        else:
            filter_complex = (
                f"[0:v]trim=0:{insert_start},setpts=PTS-STARTPTS[before_v];"
                f"[1:v]trim=0:{insert_duration},setpts=PTS-STARTPTS[broll_v];"
                f"[0:v]trim=start={insert_start + insert_duration},"
                f"setpts=PTS-STARTPTS[after_v];"
                f"[before_v][broll_v][after_v]concat=n=3:v=1:a=0[outv]"
            )
            map_args = ["-map", "[outv]"]

        cmd = [
            "ffmpeg", "-y",
            "-i", main_video,
            "-i", broll_video,
            "-filter_complex", filter_complex,
            *map_args,
            "-c:v", "libx264", "-preset", "ultrafast",
            "-c:a", "aac",
            output_path,
        ]

        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300
        )
        if result.returncode != 0:
            logger.error("ffmpeg failed: %s", result.stderr[-500:] if result.stderr else "unknown")
            return False

        logger.info("B-roll inserted -> %s", output_path)
        return True

    except subprocess.TimeoutExpired:
        logger.error("ffmpeg timed out for %s", output_path)
        return False
    except (OSError, subprocess.SubprocessError) as e:
        logger.error("insert_broll error: %s", e, exc_info=True)
        return False
