"""Smart trim: snap cut boundaries to sentence boundaries using WhisperX word timestamps."""
from __future__ import annotations

import json
import logging
import os

logger = logging.getLogger(__name__)


def load_whisperx_words(json_path: str) -> list[dict]:
    """Load all words from WhisperX JSON, flattened and sorted by start time."""
    if not os.path.exists(json_path):
        logger.warning(f"WhisperX JSON not found: {json_path}")
        return []

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        logger.warning(f"Could not read WhisperX JSON: {json_path}")
        return []

    words = []
    for segment in data.get("segments", []):
        for word in segment.get("words", []):
            if "start" in word and "end" in word and "word" in word:
                words.append({
                    "word": word["word"],
                    "start": float(word["start"]),
                    "end": float(word["end"]),
                })

    words.sort(key=lambda w: w["start"])
    return words


def _is_sentence_end(text: str, terminators: str = ".!?") -> bool:
    """Check if word text ends with a sentence terminator."""
    stripped = text.rstrip()
    return len(stripped) > 0 and stripped[-1] in terminators


def snap_to_sentence_boundary(
    start_sec: float,
    end_sec: float,
    transcript_words: list[dict],
    pad_start: float = 0.3,
    pad_end: float = 0.5,
    sentence_terminators: str = ".!?",
    max_shift: float = 3.0,
) -> tuple[float, float]:
    """Adjust start/end to nearest sentence boundary using word timestamps.

    Args:
        start_sec: Original start time in seconds.
        end_sec: Original end time in seconds.
        transcript_words: Flattened list of {"word", "start", "end"} from WhisperX.
        pad_start: Padding before the first word (seconds).
        pad_end: Padding after the last word (seconds).
        sentence_terminators: Characters that indicate end of sentence.
        max_shift: Maximum allowed shift from original timestamp (seconds).

    Returns:
        (adjusted_start, adjusted_end) tuple.
    """
    if not transcript_words:
        return start_sec, end_sec

    # Find words in the segment range (with some margin for search)
    search_margin = max_shift + 1.0
    words_near_start = [
        w for w in transcript_words
        if start_sec - search_margin <= w["start"] <= start_sec + search_margin
    ]
    words_near_end = [
        w for w in transcript_words
        if end_sec - search_margin <= w["end"] <= end_sec + search_margin
    ]

    # --- Adjust START ---
    # Walk backwards from start_sec to find the previous sentence terminator
    adjusted_start = start_sec
    words_before_start = [w for w in transcript_words if w["end"] <= start_sec + 0.5]
    words_before_start.sort(key=lambda w: w["end"], reverse=True)

    for w in words_before_start:
        if abs(w["end"] - start_sec) > max_shift:
            break
        if _is_sentence_end(w["word"], sentence_terminators):
            # Start after this sentence-ending word
            # Find the next word after this one
            next_words = [
                nw for nw in transcript_words if nw["start"] > w["end"]
            ]
            if next_words:
                adjusted_start = max(0, next_words[0]["start"] - pad_start)
            else:
                adjusted_start = max(0, w["end"])
            break

    # --- Adjust END ---
    # Walk forward from end_sec to find the next sentence terminator
    adjusted_end = end_sec
    words_after_end = [w for w in transcript_words if w["start"] >= end_sec - 0.5]
    words_after_end.sort(key=lambda w: w["start"])

    for w in words_after_end:
        if abs(w["start"] - end_sec) > max_shift:
            break
        if _is_sentence_end(w["word"], sentence_terminators):
            adjusted_end = w["end"] + pad_end
            break

    # Fallback: if no sentence boundary found, snap to nearest word boundary
    if adjusted_start == start_sec and words_near_start:
        # Snap to nearest word start
        closest = min(words_near_start, key=lambda w: abs(w["start"] - start_sec))
        adjusted_start = max(0, closest["start"] - pad_start)

    if adjusted_end == end_sec and words_near_end:
        # Snap to nearest word end
        closest = min(words_near_end, key=lambda w: abs(w["end"] - end_sec))
        adjusted_end = closest["end"] + pad_end

    # Ensure minimum duration
    if adjusted_end - adjusted_start < 3.0:
        return start_sec, end_sec

    return round(adjusted_start, 3), round(adjusted_end, 3)
