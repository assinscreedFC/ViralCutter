"""Remove silent portions from cut segments (jump cuts)."""
from __future__ import annotations

import glob
import json
import logging
import os
import re
import tempfile

from scripts.core.run_cmd import run as run_cmd

logger = logging.getLogger(__name__)


# Moved to scripts.ffmpeg_utils — import kept for backward compatibility.
from scripts.core.ffmpeg_utils import get_video_duration  # noqa: F401


def detect_silences(video_path: str, noise_db: float = -30, min_duration: float = 0.5) -> list[dict]:
    """Run ffmpeg silencedetect and return list of {start, end} dicts."""
    cmd = [
        "ffmpeg", "-i", video_path,
        "-af", f"silencedetect=noise={noise_db}dB:d={min_duration}",
        "-f", "null", "-",
    ]
    result = run_cmd(cmd, check=False, text=True)
    stderr = result.stderr

    silences = []
    starts = re.findall(r"silence_start:\s*([\d.]+)", stderr)
    ends = re.findall(r"silence_end:\s*([\d.]+)", stderr)

    for i, start_str in enumerate(starts):
        start = float(start_str)
        end = float(ends[i]) if i < len(ends) else get_video_duration(video_path)
        silences.append({"start": start, "end": end})

    return silences


def compute_keep_intervals(
    duration: float,
    silences: list[dict],
    max_silence: float = 0.3,
) -> list[dict]:
    """Compute intervals to keep from detected silences.

    If max_silence > 0, trims each silence to max_silence (centered)
    instead of removing entirely.
    Returns list of {start, end, new_start} dicts.
    """
    if not silences:
        return [{"start": 0, "end": duration, "new_start": 0}]

    # Build "remove" intervals (silence minus the kept portion)
    remove = []
    for s in silences:
        s_dur = s["end"] - s["start"]
        if s_dur <= max_silence:
            continue  # Silence is short enough to keep entirely
        # Keep max_silence/2 at the start and end of the silence
        half = max_silence / 2
        remove.append({"start": s["start"] + half, "end": s["end"] - half})

    if not remove:
        return [{"start": 0, "end": duration, "new_start": 0}]

    # Build keep intervals from the gaps between removed portions
    keep = []
    prev_end = 0.0
    for r in remove:
        if r["start"] > prev_end:
            keep.append({"start": prev_end, "end": r["start"]})
        prev_end = r["end"]
    if prev_end < duration:
        keep.append({"start": prev_end, "end": duration})

    # Calculate new_start (cumulative offset after removal)
    cumulative = 0.0
    for k in keep:
        k["new_start"] = cumulative
        cumulative += k["end"] - k["start"]

    return keep


def remove_silence_from_video(input_path: str, output_path: str, keep_intervals: list[dict]) -> bool:
    """Reassemble video from keep intervals using ffmpeg trim+concat."""
    if len(keep_intervals) <= 1 and keep_intervals[0]["start"] == 0:
        # Nothing to remove
        return False

    n = len(keep_intervals)

    # For many intervals, use concat demuxer with temp files
    if n > 50:
        return _remove_silence_concat_demuxer(input_path, output_path, keep_intervals)

    # Build filter_complex string
    filters = []
    streams = []
    for i, k in enumerate(keep_intervals):
        filters.append(f"[0:v]trim=start={k['start']:.4f}:end={k['end']:.4f},setpts=PTS-STARTPTS[v{i}]")
        filters.append(f"[0:a]atrim=start={k['start']:.4f}:end={k['end']:.4f},asetpts=PTS-STARTPTS[a{i}]")
        streams.append(f"[v{i}][a{i}]")

    concat_input = "".join(streams)
    filters.append(f"{concat_input}concat=n={n}:v=1:a=1[outv][outa]")

    filter_complex = ";".join(filters)

    cmd = [
        "ffmpeg", "-y",
        "-loglevel", "error", "-hide_banner",
        "-i", input_path,
        "-filter_complex", filter_complex,
        "-map", "[outv]", "-map", "[outa]",
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
        "-c:a", "aac", "-b:a", "128k",
        output_path,
    ]

    try:
        run_cmd(cmd, text=True)
        return True
    except Exception as e:
        logger.error(f"ffmpeg silence removal failed: {e}")
        return False


def _remove_silence_concat_demuxer(input_path: str, output_path: str, keep_intervals: list[dict]) -> bool:
    """Fallback for many intervals: cut individual clips then concat via demuxer."""
    tmp_dir = tempfile.mkdtemp(prefix="silence_")
    clip_paths = []

    try:
        for i, k in enumerate(keep_intervals):
            clip_path = os.path.join(tmp_dir, f"clip_{i:04d}.mp4")
            dur = k["end"] - k["start"]
            cmd = [
                "ffmpeg", "-y",
                "-loglevel", "error", "-hide_banner",
                "-ss", f"{k['start']:.4f}",
                "-i", input_path,
                "-t", f"{dur:.4f}",
                "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
                "-c:a", "aac", "-b:a", "128k",
                clip_path,
            ]
            run_cmd(cmd, text=True)
            clip_paths.append(clip_path)

        # Write concat list
        list_path = os.path.join(tmp_dir, "concat.txt")
        with open(list_path, "w") as f:
            for cp in clip_paths:
                f.write(f"file '{cp}'\n")

        cmd = [
            "ffmpeg", "-y",
            "-loglevel", "error", "-hide_banner",
            "-f", "concat", "-safe", "0",
            "-i", list_path,
            "-c", "copy",
            output_path,
        ]
        run_cmd(cmd, text=True)
        return True

    except Exception as e:
        logger.error(f"ffmpeg concat demuxer failed: {e}")
        return False
    finally:
        # Cleanup temp files
        for cp in clip_paths:
            if os.path.exists(cp):
                os.remove(cp)
        list_path = os.path.join(tmp_dir, "concat.txt")
        if os.path.exists(list_path):
            os.remove(list_path)
        if os.path.exists(tmp_dir):
            os.rmdir(tmp_dir)


def adjust_subtitles(json_path: str, keep_intervals: list[dict]) -> None:
    """Remap word/segment timestamps in subtitle JSON after silence removal."""
    if not os.path.exists(json_path):
        return

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        logger.warning(f"Could not read subtitle JSON: {json_path}")
        return

    def remap_time(t: float) -> float | None:
        """Map original time to new time after silence removal. Returns None if in removed section."""
        for k in keep_intervals:
            if k["start"] <= t <= k["end"]:
                return k["new_start"] + (t - k["start"])
        # Time is in a removed section — clamp to nearest kept boundary
        for i, k in enumerate(keep_intervals):
            if t < k["start"]:
                return k["new_start"]
        # After all intervals
        if keep_intervals:
            last = keep_intervals[-1]
            return last["new_start"] + (last["end"] - last["start"])
        return t

    new_segments = []
    for segment in data.get("segments", []):
        new_start = remap_time(segment.get("start", 0))
        new_end = remap_time(segment.get("end", 0))

        if new_start is None or new_end is None or new_end <= new_start:
            continue

        new_seg = segment.copy()
        new_seg["start"] = round(new_start, 3)
        new_seg["end"] = round(new_end, 3)

        if "words" in segment:
            new_words = []
            for word in segment["words"]:
                w_start = remap_time(word.get("start", 0))
                w_end = remap_time(word.get("end", 0))
                if w_start is not None and w_end is not None and w_end > w_start:
                    new_word = word.copy()
                    new_word["start"] = round(w_start, 3)
                    new_word["end"] = round(w_end, 3)
                    new_words.append(new_word)
            new_seg["words"] = new_words

        new_segments.append(new_seg)

    data["segments"] = new_segments

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    logger.info(f"Subtitles adjusted: {json_path}")


def process_project(
    project_folder: str,
    noise_db: float = -30,
    min_silence_duration: float = 0.5,
    max_silence_keep: float = 0.3,
) -> None:
    """Process all cut segments in a project: detect and remove silences."""
    cuts_folder = os.path.join(project_folder, "cuts")
    subs_folder = os.path.join(project_folder, "subs")

    if not os.path.exists(cuts_folder):
        logger.warning(f"Cuts folder not found: {cuts_folder}")
        return

    video_files = sorted(glob.glob(os.path.join(cuts_folder, "*_original_scale.mp4")))
    if not video_files:
        logger.warning("No cut segments found for silence removal.")
        return

    logger.info(f"Silence removal: processing {len(video_files)} segments (threshold={noise_db}dB, min_dur={min_silence_duration}s, max_keep={max_silence_keep}s)")

    for video_path in video_files:
        filename = os.path.basename(video_path)
        logger.info(f"Analyzing silence: {filename}")

        # 1. Detect silences
        silences = detect_silences(video_path, noise_db, min_silence_duration)

        if not silences:
            logger.info(f"  No silence detected, skipping.")
            continue

        total_silence = sum(s["end"] - s["start"] for s in silences)
        logger.info(f"  Found {len(silences)} silence(s), total {total_silence:.1f}s")

        # 2. Get video duration
        duration = get_video_duration(video_path)
        if duration <= 0:
            logger.warning(f"  Could not get duration, skipping.")
            continue

        # Skip if entire segment is silence
        if total_silence >= duration * 0.95:
            logger.warning(f"  Segment is almost entirely silent, skipping.")
            continue

        # 3. Compute keep intervals
        keep_intervals = compute_keep_intervals(duration, silences, max_silence_keep)

        if len(keep_intervals) == 1 and keep_intervals[0]["start"] == 0 and abs(keep_intervals[0]["end"] - duration) < 0.01:
            logger.info(f"  Nothing to remove after applying max_silence_keep, skipping.")
            continue

        new_duration = sum(k["end"] - k["start"] for k in keep_intervals)
        removed = duration - new_duration
        logger.info(f"  Removing {removed:.1f}s of silence ({duration:.1f}s -> {new_duration:.1f}s)")

        # 4. Remove silence from video
        tmp_output = video_path + ".tmp.mp4"
        success = remove_silence_from_video(video_path, tmp_output, keep_intervals)

        if success and os.path.exists(tmp_output) and os.path.getsize(tmp_output) > 0:
            os.replace(tmp_output, video_path)
            logger.info(f"  Video updated: {filename}")
        else:
            if os.path.exists(tmp_output):
                os.remove(tmp_output)
            logger.error(f"  Failed to remove silence from {filename}")
            continue

        # 5. Adjust subtitles
        base_name = filename.replace("_original_scale.mp4", "")
        json_filename = f"{base_name}_processed.json"
        json_path = os.path.join(subs_folder, json_filename)
        adjust_subtitles(json_path, keep_intervals)

    logger.info("Silence removal complete.")
