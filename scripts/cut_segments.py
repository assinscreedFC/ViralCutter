from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from scripts import cut_json
import os
from scripts.run_cmd import run as run_cmd
import json

logger = logging.getLogger(__name__)

def cut(
    segments: dict | None,
    project_folder: str = "tmp",
    skip_video: bool = False,
    smart_trim: bool = False,
    trim_pad_start: float = 0.3,
    trim_pad_end: float = 0.5,
    scene_detection: bool = False,
) -> None:

    def generate_segments(response, project_folder, skip_video, smart_trim=smart_trim,
                          trim_pad_start=trim_pad_start, trim_pad_end=trim_pad_end,
                          scene_detection=scene_detection):
        # Procurar input_video.mp4 no project_folder ou tmp
        input_file = os.path.join(project_folder, "input.mp4")
        if not os.path.exists(input_file):
            # Tenta fallback legado
            input_file_legacy = os.path.join(project_folder, "input_video.mp4")
            if os.path.exists(input_file_legacy):
                input_file = input_file_legacy
            else:
                logger.error(f"Input file not found in {project_folder}")
                return

        # Pasta de saida para os cortes
        cuts_folder = os.path.join(project_folder, "cuts")
        os.makedirs(cuts_folder, exist_ok=True)
        
        # Pasta de saida para legendas json cortadas
        subs_folder = os.path.join(project_folder, "subs")
        os.makedirs(subs_folder, exist_ok=True)

        # Input JSON (Transkription original)
        input_json_path = os.path.join(project_folder, "input.json")

        segments = response.get("segments", [])

        # Load whisperx words once before the loop (N+1 fix)
        words = []
        if smart_trim:
            try:
                from scripts.smart_trim import load_whisperx_words
                words = load_whisperx_words(input_json_path)
            except Exception as e:
                logger.warning(f"Failed to preload whisperx words: {e}")

        # Preload scenes once before the loop (avoid N+1 re-analysis)
        cached_scenes = None
        if scene_detection:
            try:
                from scripts.scene_detector import detect_scenes
                logger.info("Pre-computing scene detection (one-time)...")
                cached_scenes = detect_scenes(input_file)
                logger.info(f"  Found {len(cached_scenes)} scenes.")
            except Exception as e:
                logger.warning(f"Scene detection preload failed: {e}")

        def _cut_single_segment(i, segment, total):
            """Process a single segment: ffmpeg cut + JSON transcript cut."""
            try:
                start_time = segment.get("start_time", "00:00:00")
                duration = segment.get("duration", 0)

                # Heurística para duration:
                if isinstance(duration, (int, float)):
                    if duration < 1000:
                        duration_seconds = float(duration)
                    else:
                        duration_seconds = duration / 1000.0
                    duration_str = f"{duration_seconds:.3f}"
                else:
                    try:
                        duration_seconds = float(duration)
                        duration_str = f"{duration_seconds:.3f}"
                    except ValueError:
                        duration_seconds = 0
                        duration_str = duration

                # Refazendo a logica original exata para seguranca e capturando o float:
                if isinstance(start_time, int):
                    start_time_seconds = start_time / 1000.0
                    start_time_str = f"{start_time_seconds:.3f}"
                elif isinstance(start_time, float):
                    start_time_seconds = start_time
                    start_time_str = f"{start_time_seconds:.3f}"
                else:
                    try:
                        start_time_seconds = float(start_time)
                        start_time_str = f"{start_time_seconds:.3f}"
                    except ValueError:
                        h, m, s = str(start_time).split(':')
                        start_time_seconds = int(h) * 3600 + int(m) * 60 + float(s)
                        start_time_str = str(start_time)

                # --- Smart Trim: snap to sentence boundaries ---
                if smart_trim:
                    try:
                        from scripts.smart_trim import snap_to_sentence_boundary
                        if words:
                            end_time_sec = start_time_seconds + duration_seconds
                            adj_start, adj_end = snap_to_sentence_boundary(
                                start_time_seconds, end_time_sec, words,
                                pad_start=trim_pad_start, pad_end=trim_pad_end,
                            )
                            logger.info(f"  [Seg {i}] Smart trim: {start_time_seconds:.2f}-{end_time_sec:.2f} -> {adj_start:.2f}-{adj_end:.2f}")
                            start_time_seconds = adj_start
                            start_time_str = f"{adj_start:.3f}"
                            duration_seconds = adj_end - adj_start
                            duration_str = f"{duration_seconds:.3f}"
                    except Exception as e:
                        logger.warning(f"[Seg {i}] Smart trim failed, using original timestamps: {e}")

                # --- Scene Detection: avoid cutting mid-scene ---
                if scene_detection and cached_scenes:
                    try:
                        from scripts.scene_detector import validate_cut_boundaries
                        scenes = cached_scenes
                        if scenes:
                            end_time_sec = start_time_seconds + duration_seconds
                            result = validate_cut_boundaries(start_time_seconds, end_time_sec, scenes)
                            if result["cuts_mid_scene"]:
                                new_start = result["suggested_start"]
                                new_end = result["suggested_end"]
                                logger.info(f"  [Seg {i}] Scene snap: {start_time_seconds:.2f}-{end_time_sec:.2f} -> {new_start:.2f}-{new_end:.2f}")
                                start_time_seconds = new_start
                                start_time_str = f"{new_start:.3f}"
                                duration_seconds = new_end - new_start
                                duration_str = f"{duration_seconds:.3f}"
                    except Exception as e:
                        logger.warning(f"[Seg {i}] Scene detection failed, using original timestamps: {e}")

                # Título para nome de arquivo
                title = segment.get("title", f"Segment_{i}")
                safe_title = "".join([c for c in title if c.isalnum() or c in " _-"]).strip()
                safe_title = safe_title.replace(" ", "_")[:60]

                part_num = segment.get("part_number", 1)
                total_parts = segment.get("total_parts", 1)
                if total_parts > 1:
                    base_name = f"{i:03d}_{safe_title}_Part{part_num}of{total_parts}"
                else:
                    base_name = f"{i:03d}_{safe_title}"

                output_filename = f"{base_name}_original_scale.mp4"
                output_path = os.path.join(cuts_folder, output_filename)

                logger.info(f"[Seg {i}] Processing segment {i+1}/{total}")
                logger.info(f"[Seg {i}] Start time: {start_time}, Duration: {duration}")

                # VIDEO GENERATION
                if not skip_video:
                    command = [
                        "ffmpeg",
                        "-y",
                        "-loglevel", "error", "-hide_banner",
                        "-ss", start_time_str,
                        "-i", input_file,
                        "-t", duration_str,
                        "-c", "copy",
                        "-avoid_negative_ts", "make_zero",
                        output_path
                    ]

                    try:
                        run_cmd(command, text=True)
                        if os.path.exists(output_path):
                            file_size = os.path.getsize(output_path)
                            logger.info(f"[Seg {i}] Generated segment: {output_filename}, Size: {file_size} bytes")
                    except Exception as e:
                        logger.error(f"[Seg {i}] Error executing ffmpeg: {e}")
                else:
                    logger.info(f"[Seg {i}] Skipping video generation for {output_filename} (using existing). check json...")

                # --- JSON CUTTING (ALWAYS RUN) ---
                end_time_seconds = start_time_seconds + float(duration_seconds)

                json_output_filename = f"{base_name}_processed.json"
                json_output_path = os.path.join(subs_folder, json_output_filename)

                cut_json.cut_json_transcript(input_json_path, json_output_path, start_time_seconds, end_time_seconds)
                # --------------------

                logger.info(f"[Seg {i}] Done.")
                return i, True

            except Exception as e:
                logger.error(f"[Seg {i}] Segment failed: {e}", exc_info=True)
                return i, False

        # --- Parallel segment cutting ---
        total = len(segments)
        max_workers = min(4, total) if total > 0 else 1

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_cut_single_segment, i, seg, total): i
                for i, seg in enumerate(segments)
            }
            failed = []
            for future in as_completed(futures):
                seg_idx = futures[future]
                try:
                    idx, success = future.result()
                    if not success:
                        failed.append(idx)
                except Exception as e:
                    logger.error(f"[Seg {seg_idx}] Unexpected error: {e}", exc_info=True)
                    failed.append(seg_idx)

            if failed:
                logger.warning(f"Failed segments: {sorted(failed)}")
            else:
                logger.info(f"All {total} segments cut successfully.")

    # Reading the JSON file if segments not provided (legacy behavior)
    if segments is None:
        json_path = os.path.join(project_folder, 'viral_segments.txt')
        with open(json_path, 'r', encoding='utf-8') as file:
            response = json.load(file)
    else:
        response = segments

    generate_segments(response, project_folder, skip_video,
                      smart_trim=smart_trim, trim_pad_start=trim_pad_start,
                      trim_pad_end=trim_pad_end, scene_detection=scene_detection)
