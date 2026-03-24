"""Pipeline stages — each modifies PipelineContext in-place."""
from __future__ import annotations

import json
import logging
import os
import sys
import time

from i18n.i18n import I18nAuto
from scripts import (
    download_video,
    transcribe_video,
    create_viral_segments,
    cut_segments,
    edit_video,
    adjust_subtitles,
    burn_subtitles,
    add_music,
    save_json,
    translate_json,
)
from scripts.models import Segment
from scripts.pipeline.config import ProcessingConfig
from scripts.pipeline.context import PipelineContext
from scripts.pipeline.errors import PipelineError

logger = logging.getLogger(__name__)
i18n = I18nAuto()


# ---------------------------------------------------------------------------
# Helpers (kept from original main_improved.py)
# ---------------------------------------------------------------------------

COLORS = {
    "red": "0000FF", "yellow": "00FFFF", "green": "00FF00",
    "white": "FFFFFF", "black": "000000", "grey": "808080",
}


def get_subtitle_config(config_path=None) -> dict:
    """Return the subtitle configuration dictionary."""
    base_color_transparency = "00"
    outline_transparency = "FF"
    highlight_color_transparency = "00"
    shadow_color_transparency = "00"

    config = {
        "font": "Montserrat-Regular",
        "base_size": 30,
        "base_color": f"&H{base_color_transparency}{COLORS['white']}&",
        "highlight_size": 35,
        "words_per_block": 3,
        "gap_limit": 0.5,
        "mode": 'highlight',
        "highlight_color": f"&H{highlight_color_transparency}{COLORS['green']}&",
        "vertical_position": 210,
        "alignment": 2,
        "bold": 0,
        "italic": 0,
        "underline": 0,
        "strikeout": 0,
        "border_style": 2,
        "outline_thickness": 1.5,
        "outline_color": f"&H{outline_transparency}{COLORS['grey']}&",
        "shadow_size": 2,
        "shadow_color": f"&H{shadow_color_transparency}{COLORS['black']}&",
        "remove_punctuation": True,
        "animation": "pop",
    }

    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                loaded_config = json.load(f)
                config.update(loaded_config)
                config["animation"] = config.get("animation", "pop")
                logger.info(i18n("Loaded subtitle config from {}").format(config_path))
        except Exception as e:
            logger.error(i18n("Error loading subtitle config: {}. Using defaults.").format(e))

    return config


# ---------------------------------------------------------------------------
# Stage 1: Download
# ---------------------------------------------------------------------------

def stage_download(ctx: PipelineContext) -> None:
    """Download video or reuse existing input."""
    cfg = ctx.cfg
    logger.debug(f"Checking input_video state. input_video={ctx.input_video}")

    if not ctx.input_video:
        if not ctx.url:
            raise PipelineError(i18n("Error: No URL provided and no existing video selected."))

        logger.info(i18n("Starting download..."))
        download_subs = not cfg.input.skip_youtube_subs
        download_result = download_video.download(ctx.url, download_subs=download_subs, quality=cfg.input.video_quality)

        if isinstance(download_result, tuple):
            ctx.input_video, ctx.project_folder = download_result
        else:
            ctx.input_video = download_result
            ctx.project_folder = os.path.dirname(ctx.input_video)

        logger.debug(f"Download finished. input_video={ctx.input_video}, project_folder={ctx.project_folder}")
    else:
        logger.debug("Using existing video logic.")
        ctx.project_folder = os.path.dirname(ctx.input_video)

    logger.info(f"Project Folder: {ctx.project_folder}")


# ---------------------------------------------------------------------------
# Stage 2: Transcribe
# ---------------------------------------------------------------------------

def stage_transcribe(ctx: PipelineContext) -> None:
    """Run Whisper transcription (skipped for Workflow 3)."""
    if ctx.workflow_choice == "3":
        logger.info(i18n("Workflow 3: Skipping Transcribe."))
        return

    cfg = ctx.cfg
    logger.info(i18n("Transcribing with model {}...").format(cfg.model))
    transcribe_video.transcribe(ctx.input_video, cfg.model, project_folder=ctx.project_folder)


# ---------------------------------------------------------------------------
# Stage 3: Viral Segments
# ---------------------------------------------------------------------------

def stage_viral_segments(ctx: PipelineContext) -> None:
    """Create or load viral segments, align, generate captions, split parts."""
    cfg = ctx.cfg
    if ctx.workflow_choice == "3":
        return

    # Load or create segments
    if not ctx.viral_segments:
        viral_segments_file_late = os.path.join(ctx.project_folder, "viral_segments.txt")
        if os.path.exists(viral_segments_file_late):
            logger.info(i18n("Found existing viral segments file at {}").format(viral_segments_file_late))
            if cfg.skip_prompts:
                logger.info(i18n("Skipping prompts enabled. Loading existing segments."))
            else:
                logger.info(i18n("Loading existing viral segments found at {}").format(viral_segments_file_late))
            try:
                with open(viral_segments_file_late, 'r', encoding='utf-8') as f:
                    ctx.viral_segments = json.load(f)
            except Exception as e:
                logger.error(i18n("Error loading existing JSON: {}. Proceeding to create new segments.").format(e))

        if not ctx.viral_segments:
            logger.info(i18n("Creating viral segments using {}...").format(ctx.ai_backend.upper()))
            raw_ct = cfg.ai.content_type or []
            content_type_arg = [ct for ct in raw_ct if ct != "auto"] or None
            ctx.viral_segments = create_viral_segments.create(
                ctx.num_segments,
                ctx.viral_mode,
                ctx.themes,
                cfg.segment.min_duration,
                cfg.segment.max_duration,
                ai_mode=ctx.ai_backend,
                api_key=ctx.api_key,
                project_folder=ctx.project_folder,
                chunk_size_arg=cfg.ai.chunk_size,
                model_name_arg=cfg.ai.ai_model_name,
                content_type=content_type_arg,
                enable_scoring=cfg.ai.enable_scoring,
                min_score=cfg.ai.min_score,
                enable_validation=cfg.ai.enable_validation,
                enable_parts=cfg.segment.enable_parts,
                ab_variants=cfg.post_production.ab_variants,
                num_variants=cfg.post_production.num_variants,
            )

        if not ctx.viral_segments or not ctx.viral_segments.get("segments"):
            logger.error(i18n("Error: No viral segments were generated."))
            logger.error(i18n("Possible reasons: API error, Model not found, or empty response."))
            raise PipelineError(i18n("No viral segments were generated."))

        save_json.save_viral_segments(ctx.viral_segments, project_folder=ctx.project_folder)
        ctx.viral_segments["segments"] = [
            Segment.from_dict(s).to_dict() for s in ctx.viral_segments.get("segments", [])
        ]

    # Fix raw segments (missing timestamps)
    if ctx.viral_segments and "segments" in ctx.viral_segments:
        segs = ctx.viral_segments.get("segments", [])
        if segs and segs[0].get("duration", 0) == 0:
            logger.info(i18n("Detected raw AI segments without timestamps (Duration 0). Running alignment..."))
            try:
                transcript = create_viral_segments.load_transcript(ctx.project_folder)
                ctx.viral_segments = create_viral_segments.process_segments(
                    segs, transcript, cfg.segment.min_duration, cfg.segment.max_duration, output_count=None
                )
                save_json.save_viral_segments(ctx.viral_segments, project_folder=ctx.project_folder)
                ctx.viral_segments["segments"] = [
                    Segment.from_dict(s).to_dict() for s in ctx.viral_segments.get("segments", [])
                ]
                logger.info(i18n("Segments aligned and saved."))
            except Exception as e:
                logger.error(i18n("Failed to align raw segments: {}").format(e))

    # Generate TikTok captions
    _segs_for_caption = ctx.viral_segments.get("segments", []) if ctx.viral_segments else []
    _needs_captions = any(not s.get("tiktok_caption") for s in _segs_for_caption)
    if ctx.viral_segments and ctx.ai_backend in ("pleiade", "gemini", "g4f") and _needs_captions:
        logger.info(i18n("Generating TikTok captions..."))
        try:
            transcript_for_captions = create_viral_segments.load_transcript(ctx.project_folder)
            transcript_text = create_viral_segments.preprocess_transcript_for_ai(transcript_for_captions)
            ctx.viral_segments["segments"] = create_viral_segments.generate_tiktok_captions(
                ctx.viral_segments["segments"],
                transcript_text,
                ai_mode=ctx.ai_backend,
                api_key=ctx.api_key,
                model_name=cfg.ai.ai_model_name,
                content_type=ctx.viral_segments.get("content_type")
            )
            if cfg.ai.enable_validation:
                logger.info(i18n("Validating TikTok captions..."))
                ctx.viral_segments["segments"] = create_viral_segments.validate_captions(
                    ctx.viral_segments["segments"],
                    transcript_text,
                    ai_mode=ctx.ai_backend,
                    api_key=ctx.api_key,
                    model_name=cfg.ai.ai_model_name,
                )
            save_json.save_viral_segments(ctx.viral_segments, project_folder=ctx.project_folder)
        except Exception as e:
            logger.warning(f"TikTok caption generation failed: {e}")

    # Split long segments into parts
    if cfg.segment.enable_parts and ctx.viral_segments and "segments" in ctx.viral_segments:
        from scripts.split_parts import split_long_segments
        logger.info(i18n("Splitting long segments into parts (AI-guided)..."))
        transcript_segments_for_split = create_viral_segments.load_transcript(ctx.project_folder)
        ctx.viral_segments = split_long_segments(
            ctx.viral_segments,
            transcript_json_path=os.path.join(ctx.project_folder, "input.json"),
            transcript_segments=transcript_segments_for_split,
            target_part_duration=cfg.segment.target_part_duration,
            min_part_duration=max(cfg.segment.min_duration, 30),
            max_normal_duration=cfg.segment.max_duration,
            ai_mode=ctx.ai_backend,
            api_key=ctx.api_key,
            model_name=cfg.ai.ai_model_name,
        )
        save_json.save_viral_segments(ctx.viral_segments, project_folder=ctx.project_folder)
        ctx.viral_segments["segments"] = [
            Segment.from_dict(s).to_dict() for s in ctx.viral_segments.get("segments", [])
        ]
        logger.info(i18n("{} segments after splitting into parts.").format(
            len(ctx.viral_segments.get("segments", []))))


# ---------------------------------------------------------------------------
# Stage 4: Cut Segments
# ---------------------------------------------------------------------------

def stage_cut(ctx: PipelineContext) -> None:
    """Cut video segments and optionally remove silence / fillers / speed ramp."""
    cfg = ctx.cfg

    if ctx.workflow_choice == "3":
        logger.info(i18n("Workflow 3 (Subtitles Only): Skipping Cut and Edit."))
        return

    cuts_folder = os.path.join(ctx.project_folder, "cuts")
    skip_cutting = False

    if os.path.exists(cuts_folder) and os.listdir(cuts_folder):
        logger.info(i18n("Existing cuts found in: {}").format(cuts_folder))
        if cfg.skip_prompts:
            cut_again_resp = 'no'
        else:
            cut_again_resp = input(i18n("Cuts already exist. Cut again? (yes/no) [default: no]: ")).strip().lower()
        if cut_again_resp not in ['y', 'yes']:
            skip_cutting = True

    if skip_cutting:
        logger.info(i18n("Skipping Video Rendering (using existing cuts), but updating Subtitle JSONs..."))
    else:
        logger.info(i18n("Cutting segments..."))

    cut_segments.cut(
        ctx.viral_segments,
        project_folder=ctx.project_folder,
        skip_video=skip_cutting,
        smart_trim=cfg.quality.smart_trim,
        trim_pad_start=cfg.quality.trim_pad_start,
        trim_pad_end=cfg.quality.trim_pad_end,
        scene_detection=cfg.quality.scene_detection,
    )

    # Remove silence (jump cuts)
    if cfg.audio.remove_silence:
        from scripts import remove_silence
        logger.info(i18n("Removing silences (jump cuts)..."))
        remove_silence.process_project(
            project_folder=ctx.project_folder,
            noise_db=cfg.audio.silence_threshold,
            min_silence_duration=cfg.audio.silence_min_duration,
            max_silence_keep=cfg.audio.silence_max_keep,
        )


# ---------------------------------------------------------------------------
# Stage 5: Filler Removal + Speed Ramp
# ---------------------------------------------------------------------------

def _process_filler_single(video_path: str, json_path: str) -> None:
    """Remove fillers from a single video (thread-safe)."""
    from scripts.filler_removal import detect_fillers, remove_fillers_from_video, update_subtitle_json
    from scripts.smart_trim import load_whisperx_words

    filename = os.path.basename(video_path)
    words = load_whisperx_words(json_path) if os.path.exists(json_path) else []
    if words:
        fillers = detect_fillers(words)
        if fillers:
            logger.info(f"Found {len(fillers)} fillers in {filename}")
            temp_out = video_path + ".tmp.mp4"
            if remove_fillers_from_video(video_path, temp_out, fillers):
                os.replace(temp_out, video_path)
                update_subtitle_json(json_path, fillers, json_path)
            elif os.path.exists(temp_out):
                os.remove(temp_out)


def _process_speed_ramp_single(
    video_path: str, speed_up_factor: float, zoom_cues: list,
) -> None:
    """Apply speed ramp to a single video (thread-safe)."""
    from scripts.speed_ramp import apply_speed_ramp
    from scripts.remove_silence import detect_silences

    silences = detect_silences(video_path, noise_db=-35, min_duration=0.8)
    highlights = None
    if zoom_cues:
        highlights = [{"timestamp": z.get("timestamp", 0), "duration": z.get("duration", 1.5)} for z in zoom_cues]
    if silences:
        temp_out = video_path + ".tmp.mp4"
        if apply_speed_ramp(video_path, temp_out, silences, speed_up_factor=speed_up_factor, highlights=highlights):
            os.replace(temp_out, video_path)
            logger.info(f"Speed ramp applied to {os.path.basename(video_path)}")
        elif os.path.exists(temp_out):
            os.remove(temp_out)


def stage_filler_speed(ctx: PipelineContext) -> None:
    """Remove filler words and apply speed ramp (parallelized per segment)."""
    cfg = ctx.cfg
    if ctx.workflow_choice == "3":
        return

    import glob as glob_mod
    from concurrent.futures import ThreadPoolExecutor, as_completed

    cuts_folder = os.path.join(ctx.project_folder, "cuts")
    subs_folder = os.path.join(ctx.project_folder, "subs")
    video_files = sorted(glob_mod.glob(os.path.join(cuts_folder, "*_original_scale.mp4")))
    max_workers = min(4, len(video_files))

    # Filler word removal (parallel)
    if cfg.post_production.remove_fillers and video_files:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for video_path in video_files:
                filename = os.path.basename(video_path)
                base_name = filename.replace("_original_scale.mp4", "")
                json_path = os.path.join(subs_folder, f"{base_name}_processed.json")
                futures.append(executor.submit(_process_filler_single, video_path, json_path))
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Filler removal failed: {e}")

    # Speed ramp (parallel, after fillers complete)
    if cfg.post_production.speed_ramp and video_files:
        segments_path = os.path.join(ctx.project_folder, "viral_segments.txt")
        all_zoom_cues: list[list] = []
        if os.path.exists(segments_path):
            with open(segments_path, "r", encoding="utf-8") as f:
                vs = json.load(f)
            all_zoom_cues = [s.get("zoom_cues", []) for s in vs.get("segments", [])]

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for idx, video_path in enumerate(video_files):
                zoom_cues = all_zoom_cues[idx] if idx < len(all_zoom_cues) else []
                futures.append(executor.submit(
                    _process_speed_ramp_single, video_path, cfg.post_production.speed_up_factor, zoom_cues,
                ))
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Speed ramp failed: {e}")


# ---------------------------------------------------------------------------
# Stage 6: Clip Quality Validation
# ---------------------------------------------------------------------------

def _analyze_single_segment(
    video_path: str, json_path: str, cfg: ProcessingConfig, engagement_model: str | None = None,
) -> dict:
    """Analyze a single segment's quality metrics (thread-safe, no shared state)."""
    from scripts.smart_trim import load_whisperx_words
    filename = os.path.basename(video_path)
    result: dict = {}

    words = load_whisperx_words(json_path) if os.path.exists(json_path) else []

    if cfg.quality.validate_clips:
        from scripts.clip_validator import validate_clip_boundaries
        boundary = validate_clip_boundaries(video_path, noise_db=cfg.audio.silence_threshold)
        result["speech_ratio"] = boundary["speech_ratio"]
        result["starts_on_silence"] = boundary["starts_on_silence"]
        result["ends_on_silence"] = boundary["ends_on_silence"]
        if boundary["starts_on_silence"]:
            logger.warning(f"  {filename}: starts on silence!")
        logger.info(f"  {filename}: speech_ratio={boundary['speech_ratio']}")

    if cfg.quality.hook_detection:
        from scripts.hook_scorer import score_hook
        hook = score_hook(video_path, words)
        result["hook_score"] = hook["hook_score"]
        result["hook_audio_energy"] = hook["audio_energy"]
        logger.info(f"  {filename}: hook_score={hook['hook_score']}")

    if cfg.quality.blur_detection:
        from scripts.blur_detector import detect_blur_frames
        blur = detect_blur_frames(video_path)
        result["blur_ratio"] = blur["blur_ratio"]
        result["avg_sharpness"] = blur["avg_sharpness"]
        if blur["blur_ratio"] > cfg.quality.max_blur_ratio:
            logger.warning(f"  {filename}: high blur ratio {blur['blur_ratio']:.2f}")
        logger.info(f"  {filename}: blur_ratio={blur['blur_ratio']}, sharpness={blur['avg_sharpness']}")

    if cfg.quality.pacing_analysis:
        from scripts.pacing_analyzer import analyze_pacing
        pacing = analyze_pacing(video_path, words)
        result["pacing_score"] = pacing["pacing_score"]
        result["words_per_sec"] = pacing["words_per_sec"]
        result["avg_rms_energy"] = pacing["avg_rms_energy"]
        logger.info(f"  {filename}: pacing_score={pacing['pacing_score']}, wps={pacing['words_per_sec']}")

    if cfg.quality.validate_clips:
        from scripts.clip_validator import score_visual_variety, analyze_speaker_activity
        variety = score_visual_variety(video_path)
        result["visual_variety_score"] = variety["visual_variety_score"]
        result["scene_change_count"] = variety["scene_change_count"]
        logger.info(f"  {filename}: visual_variety={variety['visual_variety_score']}")

        if words:
            speaker = analyze_speaker_activity(words, 0, words[-1].get("end", 0))
            result["speaking_time_ratio"] = speaker["speaking_time_ratio"]
            logger.info(f"  {filename}: speaking_ratio={speaker['speaking_time_ratio']}")

    # Composite and engagement depend on prior results — compute inline
    if cfg.quality.composite_scoring:
        from scripts.composite_scorer import compute_composite_score
        composite = compute_composite_score(
            hook_score=result.get("hook_score", 50.0),
            speech_ratio=result.get("speech_ratio", 0.8),
            pacing_score=result.get("pacing_score", 50.0),
            blur_ratio=result.get("blur_ratio", 0.0),
            visual_variety_score=result.get("visual_variety_score", 50.0),
        )
        result["composite_quality_score"] = composite
        logger.info(f"  {filename}: composite_score={composite}")

    if cfg.advanced_ai.engagement_prediction:
        from scripts.engagement_predictor import predict_from_metadata
        engagement = predict_from_metadata(result, model_path=engagement_model)
        result["engagement_score"] = engagement
        logger.info(f"  {filename}: engagement_score={engagement}")

    return result


def stage_quality(ctx: PipelineContext) -> None:
    """Validate clip quality: silence, hooks, blur, pacing, composite, engagement."""
    cfg = ctx.cfg
    if ctx.workflow_choice == "3":
        return

    if not any([cfg.quality.validate_clips, cfg.quality.hook_detection, cfg.quality.blur_detection, cfg.quality.pacing_analysis, cfg.quality.composite_scoring]):
        return

    import glob as glob_mod
    from concurrent.futures import ThreadPoolExecutor, as_completed
    cuts_folder = os.path.join(ctx.project_folder, "cuts")
    subs_folder = os.path.join(ctx.project_folder, "subs")
    video_files = sorted(glob_mod.glob(os.path.join(cuts_folder, "*_original_scale.mp4")))

    if not video_files or not ctx.viral_segments or "segments" not in ctx.viral_segments:
        return

    segments = ctx.viral_segments["segments"]
    valid_pairs = []
    for idx, video_path in enumerate(video_files):
        if idx >= len(segments):
            logger.warning(f"Skipping validation for {os.path.basename(video_path)}: no matching segment data")
            continue
        filename = os.path.basename(video_path)
        base_name = filename.replace("_original_scale.mp4", "")
        json_path = os.path.join(subs_folder, f"{base_name}_processed.json")
        valid_pairs.append((idx, video_path, json_path))

    max_workers = min(4, len(valid_pairs))
    logger.info(i18n("Validating clip quality...") + f" ({len(valid_pairs)} segments, {max_workers} workers)")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _analyze_single_segment, video_path, json_path, cfg,
                cfg.advanced_ai.engagement_model,
            ): idx
            for idx, video_path, json_path in valid_pairs
        }
        for future in as_completed(futures):
            idx = futures[future]
            try:
                result = future.result()
                segments[idx].update(result)
            except Exception as e:
                logger.error(f"Quality analysis failed for segment {idx}: {e}")

    logger.info(i18n("Clip quality validation complete."))

    # Save updated metadata
    segments_path = os.path.join(ctx.project_folder, "viral_segments.txt")
    with open(segments_path, "w", encoding="utf-8") as f:
        json.dump(ctx.viral_segments, f, indent=2, ensure_ascii=False)
    logger.info(i18n("Clip quality validation complete."))


# ---------------------------------------------------------------------------
# Stage 7: Face Edit (Crop 9:16)
# ---------------------------------------------------------------------------

def stage_face_edit(ctx: PipelineContext) -> None:
    """Face detection & cropping, or file renaming for Workflow 3."""
    cfg = ctx.cfg

    if ctx.workflow_choice != "3":
        logger.info(i18n("Editing video with {} (Mode: {})...").format(ctx.face_model, ctx.face_mode))

        try:
            dead_zone_val = int(cfg.face.face_dead_zone)
        except (ValueError, TypeError):
            dead_zone_val = 40

        edit_video.edit(
            project_folder=ctx.project_folder,
            face_model=ctx.face_model,
            face_mode=ctx.face_mode,
            detection_period=ctx.detection_intervals,
            filter_threshold=cfg.face.face_filter_threshold,
            two_face_threshold=cfg.face.face_two_threshold,
            confidence_threshold=cfg.face.face_confidence_threshold,
            dead_zone=dead_zone_val,
            focus_active_speaker=cfg.face.focus_active_speaker,
            active_speaker_mar=cfg.face.active_speaker_mar,
            active_speaker_score_diff=cfg.face.active_speaker_score_diff,
            include_motion=cfg.face.include_motion,
            active_speaker_motion_deadzone=cfg.face.active_speaker_motion_threshold,
            active_speaker_motion_sensitivity=cfg.face.active_speaker_motion_sensitivity,
            active_speaker_decay=cfg.face.active_speaker_decay,
            segments_data=ctx.viral_segments.get("segments", []) if ctx.viral_segments else None,
            no_face_mode=cfg.face.no_face_mode,
            zoom_out_factor=cfg.face.zoom_out_factor,
            # NEW: 1-face visual enhancement params
            vertical_offset=cfg.face.vertical_offset,
            single_face_zoom=cfg.face.single_face_zoom,
            ema_alpha=cfg.face.ema_alpha,
            detection_resolution=cfg.face.detection_resolution,
        )
    else:
        logger.info(i18n("Workflow 3: Skipping Face Crop."))
        if ctx.viral_segments and "segments" in ctx.viral_segments:
            segments_data = ctx.viral_segments.get("segments", [])
            final_folder = os.path.join(ctx.project_folder, "final")
            subs_folder = os.path.join(ctx.project_folder, "subs")

            logger.info(i18n("Renaming existing files with titles..."))
            for idx, segment in enumerate(segments_data):
                title = segment.get("title", f"Segment_{idx}")
                safe_title = "".join([c for c in title if c.isalnum() or c in " _-"]).strip()
                safe_title = safe_title.replace(" ", "_")[:60]
                new_base_name = f"{idx:03d}_{safe_title}"

                # MP4
                old_mp4_name = f"final-output{idx:03d}_processed.mp4"
                old_mp4_path = os.path.join(final_folder, old_mp4_name)
                new_mp4_path = os.path.join(final_folder, f"{new_base_name}.mp4")
                if os.path.exists(old_mp4_path) and not os.path.exists(new_mp4_path):
                    os.rename(old_mp4_path, new_mp4_path)
                    logger.info(f"Renamed (Workflow 3): {old_mp4_name} -> {new_base_name}.mp4")

                # JSON Sub
                old_json_name = f"final-output{idx:03d}_processed.json"
                old_json_path = os.path.join(subs_folder, old_json_name)
                new_json_path = os.path.join(subs_folder, f"{new_base_name}_processed.json")
                if os.path.exists(old_json_path) and not os.path.exists(new_json_path):
                    os.rename(old_json_path, new_json_path)
                    logger.info(f"Renamed (Workflow 3): {old_json_name} -> {new_base_name}_processed.json")

                # Timeline
                old_tl_name = f"temp_video_no_audio_{idx}_timeline.json"
                old_tl_path = os.path.join(final_folder, old_tl_name)
                new_tl_path = os.path.join(final_folder, f"{new_base_name}_timeline.json")
                if os.path.exists(old_tl_path) and not os.path.exists(new_tl_path):
                    os.rename(old_tl_path, new_tl_path)
                    logger.info(f"Renamed (Workflow 3): {old_tl_name} -> {new_base_name}_timeline.json")


# ---------------------------------------------------------------------------
# Stage 8: Subtitles
# ---------------------------------------------------------------------------

def stage_subtitles(ctx: PipelineContext) -> None:
    """Process, translate, and burn subtitles; add music and distraction."""
    cfg = ctx.cfg

    # Translation
    if cfg.translate_target and cfg.translate_target.lower() != "none":
        logger.info(i18n("Translating subtitles to: {}").format(cfg.translate_target))
        import asyncio
        try:
            asyncio.run(translate_json.translate_project_subs(ctx.project_folder, cfg.translate_target))
        except Exception as e:
            logger.error(i18n("Translation failed: {}").format(e))

    ctx.sub_config = get_subtitle_config(cfg.subtitle.subtitle_config)

    # Split-screen auto-adjust
    if cfg.distraction.add_distraction:
        _ratio = cfg.distraction.distraction_ratio
        ctx.sub_config['vertical_position'] = int(620 * (1 - _ratio))

    # Generate .ass files
    try:
        adjust_subtitles.adjust(project_folder=ctx.project_folder, **ctx.sub_config)
    except FileNotFoundError as fnf_error:
        raise PipelineError(
            i18n("[ERROR] Subtitle processing failed: {}").format(str(fnf_error))
            + "\n" + i18n("Tip: If you are using Workflow 3 (Subtitles Only), ensure the 'subs' folder exists and contains valid JSON files.")
        ) from fnf_error
    except Exception as e:
        logger.error(i18n("[ERROR] Unexpected error during subtitle processing: {}").format(str(e)))
        raise e

    # Add music
    if cfg.audio.add_music:
        logger.info(i18n("Adding background music..."))
        try:
            _segs = ctx.viral_segments.get("segments", []) if ctx.viral_segments else []
            add_music.add_music_to_project(
                project_folder=ctx.project_folder,
                music_dir=cfg.audio.music_dir,
                music_file=cfg.audio.music_file,
                music_volume=cfg.audio.music_volume,
                segments=_segs,
            )
        except Exception as e:
            logger.warning(f"Music addition failed: {e}")

    # Split-screen distraction
    if cfg.distraction.add_distraction:
        logger.info("Adding split-screen distraction video...")
        try:
            from scripts.add_distraction_video import add_distraction_to_project
            from scripts.add_distraction_video import DEFAULT_DISTRACTION_DIR as _DIST_DIR
            add_distraction_to_project(
                project_folder=ctx.project_folder,
                distraction_dir=cfg.distraction.distraction_dir or _DIST_DIR,
                distraction_file=cfg.distraction.distraction_file,
                no_fetch=cfg.distraction.distraction_no_fetch,
                main_crop_y=0 if cfg.face.face_mode == '2' else None,
                distraction_ratio=cfg.distraction.distraction_ratio,
            )
        except Exception as e:
            logger.warning(f"Split-screen distraction failed: {e}")

    # Burn subtitles
    try:
        if cfg.distraction.add_distraction:
            split_screen_folder = os.path.join(ctx.project_folder, 'split_screen')
            split_suffix = "_music_split" if cfg.audio.add_music else "_split"
            burn_subtitles.burn(
                project_folder=ctx.project_folder,
                source_folder=split_screen_folder,
                name_suffix_strip=split_suffix,
            )
        else:
            burn_subtitles.burn(project_folder=ctx.project_folder)
    except Exception as e:
        logger.error(i18n("[ERROR] Unexpected error during subtitle burning: {}").format(str(e)))
        raise e


# ---------------------------------------------------------------------------
# Stage 8b: A/B Caption Variants
# ---------------------------------------------------------------------------

def stage_ab_variants(ctx: PipelineContext) -> None:
    """Generate A/B caption variants if enabled."""
    cfg = ctx.cfg
    if not cfg.post_production.ab_variants:
        return

    from scripts.ab_variants import generate_variants
    sub_config = ctx.sub_config if hasattr(ctx, "sub_config") else None
    variants = generate_variants(
        project_folder=ctx.project_folder,
        num_variants=cfg.post_production.num_variants,
        sub_config=sub_config,
    )
    if variants:
        logger.info(f"Generated {len(variants)} A/B variant videos in burned_sub/")


# ---------------------------------------------------------------------------
# Stage 9: Post-production (thumbnails, overlays, color grading, dubbing)
# ---------------------------------------------------------------------------

def stage_post_production(ctx: PipelineContext) -> None:
    """Apply post-production effects: thumbnails, color grading, progress bar, emoji, dubbing."""
    cfg = ctx.cfg
    if ctx.workflow_choice == "3":
        return

    import glob as glob_mod

    # Auto thumbnails
    if cfg.post_production.auto_thumbnail:
        from scripts.auto_thumbnail import extract_best_frame, save_thumbnail

        burned_folder = os.path.join(ctx.project_folder, "burned_sub")
        if not os.path.isdir(burned_folder):
            burned_folder = os.path.join(ctx.project_folder, "cuts")
        video_files = sorted(glob_mod.glob(os.path.join(burned_folder, "*.mp4")))

        for video_path in video_files:
            frame, ts = extract_best_frame(video_path)
            if frame is not None:
                thumb_path = video_path.rsplit(".", 1)[0] + "_thumbnail.jpg"
                save_thumbnail(frame, thumb_path)
                logger.info(f"Thumbnail saved: {os.path.basename(thumb_path)}")

    # Post-production overlays
    final_folder = os.path.join(ctx.project_folder, "burned_sub")
    if not os.path.isdir(final_folder):
        final_folder = os.path.join(ctx.project_folder, "cuts")
    video_files = sorted(glob_mod.glob(os.path.join(final_folder, "*.mp4")))

    has_any_effect = cfg.post_production.color_grade or cfg.post_production.progress_bar or cfg.post_production.emoji_overlay

    for idx, video_path in enumerate(video_files):
        # Resolve emoji cues for this segment
        emoji_cues: list[dict] = []
        if cfg.post_production.emoji_overlay and ctx.viral_segments and "segments" in ctx.viral_segments:
            if idx < len(ctx.viral_segments["segments"]):
                emoji_cues = ctx.viral_segments["segments"][idx].get("emoji_cues", [])

        if not has_any_effect:
            continue

        temp_out = video_path + ".tmp.mp4"
        try:
            from scripts.overlay_effects import apply_post_production
            ok = apply_post_production(
                video_path,
                temp_out,
                lut_name=cfg.post_production.color_grade if cfg.post_production.color_grade else None,
                lut_intensity=cfg.post_production.grade_intensity,
                progress_bar=bool(cfg.post_production.progress_bar),
                bar_color=cfg.post_production.bar_color,
                bar_position=cfg.post_production.bar_position,
                emojis=emoji_cues or None,
            )
            if ok and os.path.isfile(temp_out):
                os.replace(temp_out, video_path)
                logger.info(f"Post-production applied (single pass): {os.path.basename(video_path)}")
        finally:
            if os.path.exists(temp_out):
                os.remove(temp_out)

    # AI Dubbing
    if cfg.advanced_ai.dubbing:
        from scripts.ai_dubbing import dub_segment

        dub_folder = os.path.join(ctx.project_folder, "burned_sub")
        if not os.path.isdir(dub_folder):
            dub_folder = os.path.join(ctx.project_folder, "cuts")
        video_files = sorted(glob_mod.glob(os.path.join(dub_folder, "*.mp4")))

        segments_path = os.path.join(ctx.project_folder, "viral_segments.txt")
        segment_texts = []
        if os.path.exists(segments_path):
            with open(segments_path, "r", encoding="utf-8") as f:
                vs = json.load(f)
            segment_texts = [s.get("description", s.get("title", "")) for s in vs.get("segments", [])]

        for idx, video_path in enumerate(video_files):
            text = segment_texts[idx] if idx < len(segment_texts) else ""
            if text:
                dubbed_path = video_path.rsplit(".", 1)[0] + f"_dubbed_{cfg.advanced_ai.dubbing_language}.mp4"
                if dub_segment(video_path, text, cfg.advanced_ai.dubbing_language, dubbed_path, original_volume=cfg.advanced_ai.dubbing_original_volume):
                    logger.info(f"Dubbed: {os.path.basename(dubbed_path)}")

    # NEW: Rename non-retained segments with _DRAFT suffix
    if ctx.viral_segments and "segments" in ctx.viral_segments:
        burned_folder = os.path.join(ctx.project_folder, "burned_sub")
        if os.path.isdir(burned_folder):
            draft_videos = sorted(glob_mod.glob(os.path.join(burned_folder, "*.mp4")))
            segments_list = ctx.viral_segments["segments"]
            for idx, vf in enumerate(draft_videos):
                if idx < len(segments_list):
                    seg = segments_list[idx]
                    if not seg.get("retained", True):
                        base, ext = os.path.splitext(vf)
                        draft_path = base + "_DRAFT" + ext
                        os.rename(vf, draft_path)
                        logger.info(f"Tagged as DRAFT: {os.path.basename(draft_path)}")


# ---------------------------------------------------------------------------
# Stage 10: Save Config
# ---------------------------------------------------------------------------

def stage_save_config(ctx: PipelineContext) -> None:
    """Save processing configuration to project folder."""
    cfg = ctx.cfg
    try:
        used_ai_model = cfg.ai.ai_model_name
        if not used_ai_model and ctx.ai_backend != "manual":
            if ctx.ai_backend == "gemini":
                used_ai_model = ctx.api_config.get("gemini", {}).get("model", "default")
            elif ctx.ai_backend == "g4f":
                used_ai_model = ctx.api_config.get("g4f", {}).get("model", "default")

        current_sub_config = ctx.sub_config if ctx.sub_config else get_subtitle_config(cfg.subtitle.subtitle_config)

        final_config = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "workflow": ctx.workflow_choice,
            "ai_config": {
                "backend": ctx.ai_backend,
                "model_name": used_ai_model,
                "viral_mode": ctx.viral_mode,
                "themes": ctx.themes,
                "num_segments": ctx.num_segments,
                "chunk_size": cfg.ai.chunk_size
            },
            "face_config": {
                "model": ctx.face_model,
                "mode": ctx.face_mode,
                "detect_interval": cfg.face.face_detect_interval,
                "filter_threshold": cfg.face.face_filter_threshold,
                "two_face_threshold": cfg.face.face_two_threshold,
                "confidence_threshold": cfg.face.face_confidence_threshold,
                "dead_zone": cfg.face.face_dead_zone,
                "focus_active_speaker": cfg.face.focus_active_speaker,
                "active_speaker_mar": cfg.face.active_speaker_mar,
                "active_speaker_score_diff": cfg.face.active_speaker_score_diff,
                "include_motion": cfg.face.include_motion,
                # NEW: 1-face visual params
                "vertical_offset": cfg.face.vertical_offset,
                "single_face_zoom": cfg.face.single_face_zoom,
                "ema_alpha": cfg.face.ema_alpha,
                "detection_resolution": cfg.face.detection_resolution,
            },
            "video_config": {
                "min_duration": cfg.segment.min_duration,
                "max_duration": cfg.segment.max_duration,
                "whisper_model": cfg.model
            },
            "subtitle_config": current_sub_config
        }

        config_save_path = os.path.join(ctx.project_folder, "process_config.json")
        with open(config_save_path, "w", encoding="utf-8") as f:
            json.dump(final_config, f, indent=4, ensure_ascii=False)
        logger.info(i18n("Configuration saved to: {}").format(config_save_path))

    except Exception as e:
        logger.error(i18n("Error saving configuration JSON: {}").format(e))

    logger.info(i18n("Process completed! Check your results in: {}").format(ctx.project_folder))
