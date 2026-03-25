"""Split long viral segments into multi-part series using LLM-guided narrative cuts."""
from __future__ import annotations

import json
import logging
import os
import re
import unicodedata

logger = logging.getLogger(__name__)


def _extract_transcript_range(
    transcript_segments: list[dict], start_time: float, end_time: float
) -> str:
    """Extract transcript text with time tags for a specific time range."""
    full_text = ""
    last_tag_time = start_time - 5  # force first tag early

    for seg in transcript_segments:
        seg_start = seg.get("start", seg.get("end", 0))
        seg_end = seg.get("end", seg_start)

        # Skip segments outside range (with small margin)
        if seg_end < start_time - 1 or seg_start > end_time + 1:
            continue

        text = seg.get("text", "").strip()
        if not text:
            continue

        full_text += text + " "

        if seg_end - last_tag_time >= 4:
            full_text += f"({int(seg_end)}s) "
            last_tag_time = seg_end

    return full_text.strip()


def _normalize(text: str) -> str:
    """Lowercase, remove punctuation and extra spaces for fuzzy matching."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _align_part_boundaries(
    parts: list[dict],
    word_data: list[dict],
    parent_start: float,
    parent_end: float,
) -> list[dict]:
    """Align LLM-proposed part boundaries to exact word timestamps."""
    # Build flat word list within parent range
    words = []
    for seg in word_data:
        for w in seg.get("words", []):
            w_start = w.get("start", 0)
            w_end = w.get("end", w_start)
            if w_end >= parent_start - 1 and w_start <= parent_end + 1:
                words.append(w)

    if not words:
        return parts

    aligned = []
    for pi, part in enumerate(parts):
        start_ref = part.get("start_time_ref", 0)
        end_ref = part.get("end_time_ref", 0)
        start_text = _normalize(part.get("start_text", ""))
        end_text = _normalize(part.get("end_text", ""))

        # Find start boundary
        found_start = _find_text_in_words(words, start_text, start_ref, window=15)
        if found_start is not None:
            part_start = found_start
        elif pi == 0:
            part_start = parent_start
        else:
            # Use previous part's end
            part_start = aligned[-1]["end_time"] if aligned else parent_start

        # Find end boundary
        found_end = _find_text_in_words(words, end_text, end_ref, window=15, find_end=True)
        if found_end is not None:
            part_end = found_end
        elif pi == len(parts) - 1:
            part_end = parent_end
        else:
            part_end = float(end_ref) if end_ref else part_start + 55

        # Clamp to parent boundaries
        part_start = max(part_start, parent_start)
        part_end = min(part_end, parent_end)

        if part_end <= part_start:
            part_end = min(part_start + 55, parent_end)

        part["start_time"] = round(part_start, 3)
        part["end_time"] = round(part_end, 3)
        part["duration"] = round(part_end - part_start, 3)
        aligned.append(part)

    # Fix gaps/overlaps between consecutive parts
    for i in range(1, len(aligned)):
        if aligned[i]["start_time"] != aligned[i - 1]["end_time"]:
            aligned[i]["start_time"] = aligned[i - 1]["end_time"]
            aligned[i]["duration"] = round(
                aligned[i]["end_time"] - aligned[i]["start_time"], 3
            )

    return aligned


def _find_text_in_words(
    words: list[dict],
    target_text: str,
    time_ref: float,
    window: float = 15,
    find_end: bool = False,
) -> float | None:
    """Find target text in word list near time_ref. Return timestamp or None."""
    if not target_text or len(target_text) < 3:
        return None

    target_words = target_text.split()
    if len(target_words) < 2:
        return None

    best_match = None
    best_score = 0

    for i in range(len(words)):
        w_time = words[i].get("start", 0)
        if abs(w_time - time_ref) > window:
            continue

        # Build sliding window
        window_size = min(len(target_words) + 2, len(words) - i)
        if window_size < 2:
            continue

        candidate = " ".join(
            _normalize(words[i + j].get("word", ""))
            for j in range(window_size)
        )

        # Simple substring match score
        if target_text in candidate:
            score = len(target_text)
            if score > best_score:
                best_score = score
                if find_end:
                    end_idx = min(i + window_size - 1, len(words) - 1)
                    best_match = words[end_idx].get("end", words[end_idx].get("start", 0))
                else:
                    best_match = words[i].get("start", 0)

    return best_match


def _fallback_split(
    segment: dict,
    transcript_segments: list[dict],
    word_data: list[dict],
    target_duration: int,
    min_duration: int,
) -> list[dict]:
    """Deterministic fallback: split at sentence boundaries nearest to equidistant points."""
    start = segment["start_time"]
    end = segment["end_time"]
    duration = end - start

    n_parts = max(2, round(duration / target_duration))

    # Collect sentence-end timestamps within range
    sentence_ends = []
    words = []
    for seg in word_data:
        for w in seg.get("words", []):
            w_start = w.get("start", 0)
            w_end = w.get("end", w_start)
            if w_end < start or w_start > end:
                continue
            words.append(w)
            word_text = w.get("word", "")
            if word_text and word_text[-1] in ".!?":
                sentence_ends.append(w_end)

    # If no sentence boundaries found, split at longest pauses
    if len(sentence_ends) < n_parts - 1:
        pauses = []
        for i in range(1, len(words)):
            gap = words[i].get("start", 0) - words[i - 1].get("end", 0)
            if gap > 0.3:
                pauses.append((gap, words[i].get("start", 0)))
        pauses.sort(reverse=True)
        sentence_ends = [t for _, t in pauses[: n_parts * 2]]
        sentence_ends.sort()

    # Find best split points near equidistant ideal points
    ideal_points = [start + (k + 1) * duration / n_parts for k in range(n_parts - 1)]
    split_points = []

    for ideal in ideal_points:
        best = None
        best_dist = float("inf")
        for se in sentence_ends:
            dist = abs(se - ideal)
            if dist < best_dist and se not in split_points:
                # Ensure minimum part duration
                prev = split_points[-1] if split_points else start
                if se - prev >= min_duration:
                    best_dist = dist
                    best = se
        if best is not None:
            split_points.append(best)

    # Build parts from split points
    boundaries = [start] + split_points + [end]
    total_parts = len(boundaries) - 1
    base_title = segment.get("title", "Segment")

    parts = []
    for k in range(total_parts):
        part_start = boundaries[k]
        part_end = boundaries[k + 1]
        parts.append({
            "title": f"{base_title} — Part {k + 1}/{total_parts}",
            "start_time": round(part_start, 3),
            "end_time": round(part_end, 3),
            "duration": round(part_end - part_start, 3),
            "part_number": k + 1,
            "total_parts": total_parts,
            "hook": segment.get("hook", "") if k == 0 else "",
            "cliffhanger": None,
            "reasoning": segment.get("reasoning", ""),
            "score": segment.get("score", 0),
            "tiktok_caption": segment.get("tiktok_caption", ""),
            "caption_variants": segment.get("caption_variants", []),
        })

    return parts


def split_long_segments(
    segments_data: dict,
    transcript_json_path: str,
    transcript_segments: list[dict],
    target_part_duration: int = 55,
    min_part_duration: int = 30,
    max_normal_duration: int = 90,
    ai_mode: str = "gemini",
    api_key: str | None = None,
    model_name: str | None = None,
) -> dict:
    """Split segments longer than max_normal_duration into LLM-guided multi-part series."""
    from scripts.analysis.create_viral_segments import _call_ai, clean_json_response_simple

    segments = segments_data.get("segments", [])
    if not segments:
        return segments_data

    # Load word-level transcript
    word_data = []
    if os.path.exists(transcript_json_path):
        try:
            with open(transcript_json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            word_data = data.get("segments", [])
        except Exception as e:
            logger.warning(f"Could not load word-level transcript: {e}")

    # Load split prompt template
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    prompt_path = os.path.join(base_dir, "prompts", "split_parts.txt")

    if not os.path.exists(prompt_path):
        logger.warning("prompts/split_parts.txt not found, skipping parts split.")
        return segments_data

    with open(prompt_path, "r", encoding="utf-8") as f:
        prompt_template = f.read()

    result_segments = []
    group_counter = 0

    for idx, segment in enumerate(segments):
        duration = segment.get("duration", 0)
        if not duration:
            st = segment.get("start_time", 0)
            et = segment.get("end_time", 0)
            duration = et - st

        if duration <= max_normal_duration:
            # Short segment — keep as-is with metadata
            segment["part_number"] = 1
            segment["total_parts"] = 1
            segment["group_id"] = f"s{group_counter}"
            group_counter += 1
            result_segments.append(segment)
            continue

        # Long segment — LLM split
        logger.info(
            f"[PARTS] Splitting '{segment.get('title', '')}' "
            f"({duration:.0f}s) into parts..."
        )

        group_id = f"g{group_counter}"
        group_counter += 1

        # Extract transcript excerpt for this segment
        excerpt = _extract_transcript_range(
            transcript_segments,
            segment.get("start_time", 0),
            segment.get("end_time", 0),
        )

        if not excerpt:
            logger.warning(f"[PARTS] No transcript for segment {idx}, using fallback split.")
            parts = _fallback_split(
                segment, transcript_segments, word_data,
                target_part_duration, min_part_duration,
            )
            for p in parts:
                p["group_id"] = group_id
            result_segments.extend(parts)
            continue

        # Format prompt
        prompt = prompt_template.format(
            target_part_duration=target_part_duration,
            min_part_duration=min_part_duration,
            title=segment.get("title", ""),
            score=segment.get("score", 0),
            duration=int(duration),
            start_time=int(segment.get("start_time", 0)),
            end_time=int(segment.get("end_time", 0)),
            transcript_excerpt=excerpt,
        )

        # Call LLM
        response = _call_ai(prompt, ai_mode, api_key, model_name)

        if not response:
            logger.warning(f"[PARTS] LLM returned empty for segment {idx}, using fallback.")
            parts = _fallback_split(
                segment, transcript_segments, word_data,
                target_part_duration, min_part_duration,
            )
            for p in parts:
                p["group_id"] = group_id
            result_segments.extend(parts)
            continue

        # Parse LLM response
        try:
            # Try clean_json_response_simple first, then manual parse
            parsed = None
            try:
                parsed = clean_json_response_simple(response)
            except Exception:
                pass

            if not parsed or "parts" not in parsed:
                # Try extracting JSON manually
                cleaned = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
                json_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
                if json_match:
                    parsed = json.loads(json_match.group())

            if not parsed or "parts" not in parsed:
                raise ValueError("No 'parts' key in LLM response")

            raw_parts = parsed["parts"]
            if not isinstance(raw_parts, list) or len(raw_parts) < 2:
                raise ValueError(f"Expected >=2 parts, got {len(raw_parts) if isinstance(raw_parts, list) else 0}")

            # Align to word-level timestamps
            aligned_parts = _align_part_boundaries(
                raw_parts, word_data,
                segment.get("start_time", 0),
                segment.get("end_time", 0),
            )

            # Build final part segments
            total_parts = len(aligned_parts)
            base_title = segment.get("title", "Segment")

            for part in aligned_parts:
                pn = part.get("part_number", 1)
                result_segments.append({
                    "title": part.get("title", f"{base_title} — Part {pn}/{total_parts}"),
                    "start_time": part["start_time"],
                    "end_time": part["end_time"],
                    "duration": part["duration"],
                    "group_id": group_id,
                    "part_number": pn,
                    "total_parts": total_parts,
                    "hook": part.get("hook", segment.get("hook", "") if pn == 1 else ""),
                    "cliffhanger": part.get("cliffhanger"),
                    "reasoning": segment.get("reasoning", ""),
                    "score": segment.get("score", 0),
                    "viral_score": segment.get("viral_score", 0),
                    "tiktok_caption": segment.get("tiktok_caption", ""),
                    "caption_variants": segment.get("caption_variants", []),
                })

            logger.info(f"[PARTS] Split into {total_parts} parts successfully.")

        except Exception as e:
            logger.warning(f"[PARTS] LLM parse failed for segment {idx}: {e}. Using fallback.")
            parts = _fallback_split(
                segment, transcript_segments, word_data,
                target_part_duration, min_part_duration,
            )
            for p in parts:
                p["group_id"] = group_id
            result_segments.extend(parts)

    # Return updated segments_data
    result = dict(segments_data)
    result["segments"] = result_segments
    return result
