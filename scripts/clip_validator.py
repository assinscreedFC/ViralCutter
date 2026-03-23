"""Clip quality validation: silence boundaries, speech ratio, visual variety, speaker activity."""
from __future__ import annotations

import logging
import os
import re
from scripts.frame_utils import downscale_for_analysis
from scripts.run_cmd import run as run_cmd

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


# Moved to scripts.ffmpeg_utils — import kept for backward compatibility.
from scripts.ffmpeg_utils import get_video_duration  # noqa: F401


def detect_silences(video_path: str, noise_db: float = -30, min_duration: float = 0.3) -> list[dict]:
    """Run ffmpeg silencedetect and return list of {start, end} dicts."""
    cmd = [
        "ffmpeg", "-i", video_path,
        "-af", f"silencedetect=noise={noise_db}dB:d={min_duration}",
        "-f", "null", "-",
    ]
    result = run_cmd(cmd, check=False, text=True)
    if result.returncode not in (0, 1):
        logger.warning(f"ffmpeg silencedetect failed (rc={result.returncode}): {result.stderr[-300:]}")
        return []
    stderr = result.stderr

    silences = []
    starts = re.findall(r"silence_start:\s*([\d.]+)", stderr)
    ends = re.findall(r"silence_end:\s*([\d.]+)", stderr)

    duration = get_video_duration(video_path)
    for i, start_str in enumerate(starts):
        start = float(start_str)
        end = float(ends[i]) if i < len(ends) else duration
        silences.append({"start": start, "end": end})

    return silences


def validate_clip_boundaries(
    video_path: str,
    noise_db: float = -30,
    min_silence: float = 0.3,
    boundary_check_duration: float = 1.5,
) -> dict:
    """Validate clip start/end for silence and compute speech ratio.

    Args:
        video_path: Path to the clip video file.
        noise_db: Silence detection threshold in dB.
        min_silence: Minimum silence duration to detect.
        boundary_check_duration: Duration at start/end to check for silence.

    Returns:
        dict with keys: starts_on_silence, ends_on_silence, speech_ratio,
        total_silence_sec, quality_pass.
    """
    if not os.path.exists(video_path):
        return {"starts_on_silence": False, "ends_on_silence": False,
                "speech_ratio": 1.0, "total_silence_sec": 0, "quality_pass": True}

    duration = get_video_duration(video_path)
    if duration <= 0:
        return {"starts_on_silence": False, "ends_on_silence": False,
                "speech_ratio": 1.0, "total_silence_sec": 0, "quality_pass": True}

    silences = detect_silences(video_path, noise_db, min_silence)

    total_silence = sum(max(0.0, min(s["end"], duration) - max(0.0, s["start"])) for s in silences)
    speech_ratio = max(0.0, 1.0 - (total_silence / duration)) if duration > 0 else 1.0

    # Check if clip starts on silence
    starts_on_silence = False
    for s in silences:
        if s["start"] <= 0.1 and s["end"] >= boundary_check_duration:
            starts_on_silence = True
            break

    # Check if clip ends on silence
    ends_on_silence = False
    for s in silences:
        if s["start"] <= duration - boundary_check_duration and s["end"] >= duration - 0.1:
            ends_on_silence = True
            break

    return {
        "starts_on_silence": starts_on_silence,
        "ends_on_silence": ends_on_silence,
        "speech_ratio": round(speech_ratio, 3),
        "total_silence_sec": round(total_silence, 2),
        "quality_pass": speech_ratio >= 0.5 and not starts_on_silence,
    }


def score_visual_variety(
    video_path: str,
    sample_interval: float = 1.0,
    diff_threshold: float = 30.0,
) -> dict:
    """Score visual variety of a clip via frame differencing and face presence.

    Args:
        video_path: Path to video file.
        sample_interval: Seconds between sampled frames.
        diff_threshold: Mean absolute diff threshold to count as scene change.

    Returns:
        dict with keys: scene_change_count, face_presence_ratio, visual_variety_score.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"scene_change_count": 0, "face_presence_ratio": 0.0, "visual_variety_score": 0.0}

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    frame_step = max(1, int(fps * sample_interval))

    prev_gray = None
    scene_changes = 0
    face_count = 0
    total_samples = 0

    frame_idx = 0
    try:
        while True:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break

            small, _ = downscale_for_analysis(frame, max_width=360)
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            total_samples += 1

            # Scene change detection via frame differencing
            if prev_gray is not None:
                diff = cv2.absdiff(prev_gray, gray)
                if np.mean(diff) > diff_threshold:
                    scene_changes += 1

            # Face detection (lightweight Haar)
            faces = _FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=3, minSize=(60, 60))
            if len(faces) > 0:
                face_count += 1

            prev_gray = gray
            frame_idx += frame_step
            if frame_idx >= total_frames:
                break
    finally:
        cap.release()

    face_ratio = face_count / total_samples if total_samples > 0 else 0.0

    # Composite score: more scene changes + face presence = higher variety
    # Normalize scene changes: 0-5+ mapped to 0-50
    scene_score = min(50, scene_changes * 10)
    face_score = face_ratio * 50
    variety_score = min(100, scene_score + face_score)

    return {
        "scene_change_count": scene_changes,
        "face_presence_ratio": round(face_ratio, 3),
        "visual_variety_score": round(variety_score, 1),
    }


def analyze_speaker_activity(
    transcript_words: list[dict],
    start_sec: float = 0.0,
    end_sec: float | None = None,
) -> dict:
    """Analyze speaker activity from WhisperX word timestamps.

    Args:
        transcript_words: List of {"word", "start", "end"} dicts.
        start_sec: Start of analysis window.
        end_sec: End of analysis window.

    Returns:
        dict with keys: speaking_time_ratio, total_speaking_sec, gap_count.
    """
    if not transcript_words:
        return {"speaking_time_ratio": 0.0, "total_speaking_sec": 0.0, "gap_count": 0}

    # Filter words in range
    words = transcript_words
    if end_sec is not None:
        words = [w for w in words if w.get("start", 0) >= start_sec and w.get("end", 0) <= end_sec]

    if not words:
        return {"speaking_time_ratio": 0.0, "total_speaking_sec": 0.0, "gap_count": 0}

    # Calculate total speaking time (merge overlapping word intervals)
    sorted_words = sorted(words, key=lambda w: w["start"])
    merged = []
    for w in sorted_words:
        if merged and w["start"] <= merged[-1]["end"] + 0.05:
            merged[-1]["end"] = max(merged[-1]["end"], w["end"])
        else:
            merged.append({"start": w["start"], "end": w["end"]})

    total_speaking = sum(m["end"] - m["start"] for m in merged)
    total_duration = (end_sec or sorted_words[-1]["end"]) - start_sec
    ratio = total_speaking / total_duration if total_duration > 0 else 0.0

    # Count gaps between speech segments (>0.5s = notable gap)
    gap_count = 0
    for i in range(1, len(merged)):
        if merged[i]["start"] - merged[i - 1]["end"] > 0.5:
            gap_count += 1

    return {
        "speaking_time_ratio": round(min(1.0, ratio), 3),
        "total_speaking_sec": round(total_speaking, 2),
        "gap_count": gap_count,
    }


def validate_all(
    video_path: str,
    transcript_words: list[dict] | None = None,
    noise_db: float = -30,
) -> dict:
    """Run all validations on a clip and return combined results."""
    results = {}

    # Silence validation
    boundary = validate_clip_boundaries(video_path, noise_db=noise_db)
    results.update(boundary)

    # Visual variety
    variety = score_visual_variety(video_path)
    results.update(variety)

    # Speaker activity
    if transcript_words:
        duration = get_video_duration(video_path)
        activity = analyze_speaker_activity(transcript_words, 0, duration)
        results.update(activity)

    return results
