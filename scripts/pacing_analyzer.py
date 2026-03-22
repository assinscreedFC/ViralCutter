"""Pacing/energy analysis: measure speech pace and audio energy for a clip."""
from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def analyze_pacing(
    video_path: str,
    transcript_words: list[dict],
    start_sec: float = 0.0,
    end_sec: float | None = None,
    energy_threshold: float = 0.02,
) -> dict:
    """Analyze speech pacing and audio energy for a clip.

    Args:
        video_path: Path to the clip video file.
        transcript_words: List of {"word", "start", "end"} from WhisperX.
        start_sec: Start of analysis window (seconds).
        end_sec: End of analysis window. If None, uses all words.
        energy_threshold: RMS threshold for "energetic" audio.

    Returns:
        dict with keys: words_per_sec, avg_rms_energy, energy_variance, pacing_score (0-100).
    """
    # Filter words in range
    if end_sec is not None:
        words_in_range = [
            w for w in transcript_words
            if w.get("start", 0) >= start_sec and w.get("end", 0) <= end_sec
        ]
        duration = end_sec - start_sec
    else:
        words_in_range = [w for w in transcript_words if w.get("start", 0) >= start_sec]
        if words_in_range:
            duration = max(w.get("end", 0) for w in words_in_range) - start_sec
        else:
            duration = 0

    words_per_sec = len(words_in_range) / duration if duration > 0 else 0.0

    # Audio energy via hook_scorer's existing function
    avg_rms_energy = 0.0
    energy_variance = 0.0
    try:
        from scripts.hook_scorer import compute_audio_energy_rms
        # Sample energy at 3-second windows across the clip
        if duration > 0:
            window = 3.0
            energies = []
            t = start_sec
            while t < (end_sec if end_sec is not None else duration):
                e = compute_audio_energy_rms(video_path, t, min(window, (end_sec if end_sec is not None else duration) - t))
                energies.append(e)
                t += window
            if energies:
                avg_rms_energy = float(np.mean(energies))
                energy_variance = float(np.var(energies))
    except Exception as e:
        logger.warning(f"Audio energy analysis failed: {e}")

    # Compute pacing score (0-100)
    # words_per_sec: ideal 2.5-4.0 for viral content
    if words_per_sec >= 3.0:
        wps_score = 40
    elif words_per_sec >= 2.0:
        wps_score = 30
    elif words_per_sec >= 1.0:
        wps_score = 15
    else:
        wps_score = 0

    # Energy score: higher energy = more engaging
    if avg_rms_energy > energy_threshold * 2:
        energy_score = 35
    elif avg_rms_energy > energy_threshold:
        energy_score = 20
    else:
        energy_score = 0

    # Energy variance bonus: dynamic clips are more engaging
    if energy_variance > 0.0005:
        variance_score = 25
    elif energy_variance > 0.0001:
        variance_score = 15
    else:
        variance_score = 5

    pacing_score = min(100, wps_score + energy_score + variance_score)

    return {
        "words_per_sec": round(words_per_sec, 2),
        "avg_rms_energy": round(avg_rms_energy, 5),
        "energy_variance": round(energy_variance, 7),
        "pacing_score": pacing_score,
    }
