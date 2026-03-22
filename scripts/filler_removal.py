"""Filler word removal: detect and remove filler words from video clips."""
from __future__ import annotations

import json
import logging
import os

logger = logging.getLogger(__name__)

FILLER_WORDS = {
    "fr": {"euh", "genre", "en fait", "du coup", "bah", "ben", "hein", "voilà", "bon"},
    "en": {"um", "uh", "like", "you know", "basically", "i mean", "right", "so", "well", "actually"},
    "es": {"este", "pues", "bueno", "o sea", "eh"},
    "de": {"ähm", "äh", "halt", "also", "sozusagen", "quasi"},
    "pt": {"tipo", "né", "assim", "então"},
    "tr": {"yani", "şey", "hani", "işte"},
}


def detect_fillers(
    transcript_words: list[dict],
    language: str = "auto",
    custom_fillers: set[str] | None = None,
) -> list[dict]:
    """Detect filler words in transcript.

    Args:
        transcript_words: List of {"word", "start", "end"} from WhisperX.
        language: Language code (fr, en, etc.) or "auto" for all.
        custom_fillers: Additional filler words to detect.

    Returns:
        List of {"word", "start", "end"} for detected fillers.
    """
    if language == "auto":
        fillers = set()
        for lang_fillers in FILLER_WORDS.values():
            fillers.update(lang_fillers)
    else:
        fillers = set(FILLER_WORDS.get(language, set()))

    if custom_fillers:
        fillers.update(custom_fillers)

    detected = []
    for w in transcript_words:
        word_lower = w.get("word", "").strip().lower().rstrip(".,!?")
        if word_lower in fillers:
            start = w.get("start", 0)
            end = w.get("end", 0)
            if end - start >= 0.1:  # Skip very short detections
                detected.append({"word": word_lower, "start": start, "end": end})

    return detected


def remove_fillers_from_video(
    input_path: str,
    output_path: str,
    fillers: list[dict],
    min_filler_duration: float = 0.15,
) -> bool:
    """Remove filler word segments from video.

    Reuses the same trim+concat pattern as remove_silence.py.

    Args:
        input_path: Input video path.
        output_path: Output video path.
        fillers: List of {"word", "start", "end"} from detect_fillers.
        min_filler_duration: Minimum filler duration to remove (seconds).

    Returns:
        True if video was modified, False otherwise.
    """
    # Filter by minimum duration
    to_remove = [f for f in fillers if (f["end"] - f["start"]) >= min_filler_duration]
    if not to_remove:
        return False

    # Sort by start time
    to_remove.sort(key=lambda x: x["start"])

    # Get video duration
    from scripts.remove_silence import get_video_duration, compute_keep_intervals, remove_silence_from_video

    duration = get_video_duration(input_path)
    if duration <= 0:
        return False

    # Convert fillers to silence-like format for reuse
    silence_intervals = [{"start": f["start"], "end": f["end"]} for f in to_remove]
    keep_intervals = compute_keep_intervals(duration, silence_intervals, max_silence=0.0)

    if len(keep_intervals) <= 1:
        return False

    result = remove_silence_from_video(input_path, output_path, keep_intervals)
    if result:
        logger.info(f"Removed {len(to_remove)} fillers from {os.path.basename(input_path)}")
    return result


def update_subtitle_json(
    json_path: str,
    fillers: list[dict],
    output_path: str,
) -> None:
    """Recalculate subtitle timestamps after filler removal.

    Args:
        json_path: Input subtitle JSON path.
        fillers: List of removed fillers with start/end.
        output_path: Output JSON path.
    """
    if not os.path.exists(json_path):
        return

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    fillers_sorted = sorted(fillers, key=lambda x: x["start"])

    def adjust_time(t: float) -> float:
        """Subtract total filler duration before time t."""
        offset = 0.0
        for f in fillers_sorted:
            if f["end"] <= t:
                offset += f["end"] - f["start"]
            elif f["start"] < t:
                offset += t - f["start"]
            else:
                break
        return t - offset

    # Adjust word-level timestamps
    segments = data.get("segments", [])
    for seg in segments:
        if "start" in seg:
            seg["start"] = round(adjust_time(seg["start"]), 3)
        if "end" in seg:
            seg["end"] = round(adjust_time(seg["end"]), 3)
        for w in seg.get("words", []):
            if "start" in w:
                w["start"] = round(adjust_time(w["start"]), 3)
            if "end" in w:
                w["end"] = round(adjust_time(w["end"]), 3)

    # Remove filler words from segments
    for seg in segments:
        seg["words"] = [
            w for w in seg.get("words", [])
            if w.get("word", "").strip().lower().rstrip(".,!?") not in
            {f["word"] for f in fillers}
        ]
        # Rebuild segment text
        seg["text"] = " ".join(w.get("word", "") for w in seg.get("words", []))

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
