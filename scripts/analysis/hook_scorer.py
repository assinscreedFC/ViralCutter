"""Hook detection: score the first N seconds of a clip for stop-scroll potential."""
from __future__ import annotations

import io
import logging
from scripts.core.run_cmd import run as run_cmd

import numpy as np

logger = logging.getLogger(__name__)


def compute_audio_energy_rms(video_path: str, start: float = 0.0, duration: float = 3.0) -> float:
    """Extract audio segment and compute RMS energy via librosa.

    Args:
        video_path: Path to video file.
        start: Start time in seconds.
        duration: Duration to analyze in seconds.

    Returns:
        RMS energy value (float). Higher = louder/more energetic.
    """
    cmd = [
        "ffmpeg",
        "-ss", f"{start:.3f}",
        "-t", f"{duration:.3f}",
        "-i", video_path,
        "-vn", "-ar", "16000", "-ac", "1",
        "-f", "wav", "pipe:1",
    ]

    try:
        result = run_cmd(cmd)
    except Exception:
        logger.warning(f"Could not extract audio from {video_path}")
        return 0.0

    try:
        import librosa
        audio_data = np.frombuffer(result.stdout, dtype=np.int16).astype(np.float32) / 32768.0
        if len(audio_data) < 1600:  # Less than 0.1s at 16kHz
            return 0.0
        rms = librosa.feature.rms(y=audio_data, frame_length=2048, hop_length=512)[0]
        return float(np.mean(rms))
    except Exception as e:
        logger.warning(f"librosa RMS computation failed: {e}")
        return 0.0


def score_hook(
    video_path: str,
    transcript_words: list[dict],
    duration_to_analyze: float = 3.0,
    energy_threshold: float = 0.02,
) -> dict:
    """Score the first N seconds of a clip for hook strength.

    Scoring dimensions (total 0-100):
    - Audio energy above threshold: +20pts
    - Words per second > 2.0: +20pts
    - Starts with question: +20pts
    - Starts with number/stat: +15pts
    - No silence in first 1s: +25pts

    Args:
        video_path: Path to the clip video.
        transcript_words: List of {"word", "start", "end"} from WhisperX.
        duration_to_analyze: Duration of hook to analyze (seconds).
        energy_threshold: RMS energy threshold for "energetic" audio.

    Returns:
        dict with keys: hook_score, audio_energy, words_per_sec,
        starts_with_question, starts_with_number.
    """
    score = 0

    # 1. Audio energy
    audio_energy = compute_audio_energy_rms(video_path, 0.0, duration_to_analyze)
    if audio_energy > energy_threshold:
        score += 20

    # 2. Words per second in hook window
    hook_words = [
        w for w in transcript_words
        if w.get("start", 0) < duration_to_analyze
    ]
    words_per_sec = len(hook_words) / duration_to_analyze if duration_to_analyze > 0 else 0
    if words_per_sec > 2.0:
        score += 20

    # 3. Starts with question
    starts_with_question = False
    if hook_words:
        first_few = " ".join(w["word"] for w in hook_words[:5]).strip()
        question_starters = {
            "qui", "que", "quoi", "comment", "pourquoi", "quand", "ou", "est-ce",
            "what", "why", "how", "when", "where", "who", "which", "do", "does",
            "did", "is", "are", "was", "were", "can", "could", "would", "should",
            "have", "has", "will",
        }
        first_word = hook_words[0]["word"].strip().lower().rstrip(".,!?")
        if first_word in question_starters or first_few.rstrip().endswith("?"):
            starts_with_question = True
            score += 20

    # 4. Starts with number/stat
    starts_with_number = False
    if hook_words:
        first_word = hook_words[0]["word"].strip()
        if any(c.isdigit() for c in first_word):
            starts_with_number = True
            score += 15

    # 5. No silence in first 1s (check if first word starts before 1s)
    if hook_words and hook_words[0].get("start", 99) < 1.0:
        score += 25

    return {
        "hook_score": min(100, score),
        "audio_energy": round(audio_energy, 5),
        "words_per_sec": round(words_per_sec, 2),
        "starts_with_question": starts_with_question,
        "starts_with_number": starts_with_number,
    }
