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
from scripts.pipeline.context import PipelineContext

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
    }

    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                loaded_config = json.load(f)
                config.update(loaded_config)
                logger.info(i18n("Loaded subtitle config from {}").format(config_path))
        except Exception as e:
            logger.error(i18n("Error loading subtitle config: {}. Using defaults.").format(e))

    return config


# ---------------------------------------------------------------------------
# Stage 1: Download
# ---------------------------------------------------------------------------

def stage_download(ctx: PipelineContext) -> None:
    """Download video or reuse existing input."""
    args = ctx.args
    logger.debug(f"Checking input_video state. input_video={ctx.input_video}")

    if not ctx.input_video:
        if not ctx.url:
            logger.error(i18n("Error: No URL provided and no existing video selected."))
            sys.exit(1)

        logger.info(i18n("Starting download..."))
        download_subs = not args.skip_youtube_subs
        download_result = download_video.download(ctx.url, download_subs=download_subs, quality=args.video_quality)

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

    logger.info(i18n("Transcribing with model {}...").format(ctx.args.model))
    transcribe_video.transcribe(ctx.input_video, ctx.args.model, project_folder=ctx.project_folder)


# ---------------------------------------------------------------------------
# Stage 3: Viral Segments
# ---------------------------------------------------------------------------

def stage_viral_segments(ctx: PipelineContext) -> None:
    """Create or load viral segments, align, generate captions, split parts."""
    args = ctx.args
    if ctx.workflow_choice == "3":
        return

    # Load or create segments
    if not ctx.viral_segments:
        viral_segments_file_late = os.path.join(ctx.project_folder, "viral_segments.txt")
        if os.path.exists(viral_segments_file_late):
            logger.info(i18n("Found existing viral segments file at {}").format(viral_segments_file_late))
            if args.skip_prompts:
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
            raw_ct = args.content_type or []
            content_type_arg = [ct for ct in raw_ct if ct != "auto"] or None
            ctx.viral_segments = create_viral_segments.create(
                ctx.num_segments,
                ctx.viral_mode,
                ctx.themes,
                args.min_duration,
                args.max_duration,
                ai_mode=ctx.ai_backend,
                api_key=ctx.api_key,
                project_folder=ctx.project_folder,
                chunk_size_arg=args.chunk_size,
                model_name_arg=args.ai_model_name,
                content_type=content_type_arg,
                enable_scoring=args.enable_scoring,
                min_score=args.min_score,
                enable_validation=args.enable_validation,
                enable_parts=args.enable_parts
            )

        if not ctx.viral_segments or not ctx.viral_segments.get("segments"):
            logger.error(i18n("Error: No viral segments were generated."))
            logger.error(i18n("Possible reasons: API error, Model not found, or empty response."))
            logger.error(i18n("Stopping execution."))
            sys.exit(1)

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
                    segs, transcript, args.min_duration, args.max_duration, output_count=None
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
                model_name=args.ai_model_name,
                content_type=ctx.viral_segments.get("content_type")
            )
            if args.enable_validation:
                logger.info(i18n("Validating TikTok captions..."))
                ctx.viral_segments["segments"] = create_viral_segments.validate_captions(
                    ctx.viral_segments["segments"],
                    transcript_text,
                    ai_mode=ctx.ai_backend,
                    api_key=ctx.api_key,
                    model_name=args.ai_model_name,
                )
            save_json.save_viral_segments(ctx.viral_segments, project_folder=ctx.project_folder)
        except Exception as e:
            logger.warning(f"TikTok caption generation failed: {e}")

    # Split long segments into parts
    if args.enable_parts and ctx.viral_segments and "segments" in ctx.viral_segments:
        from scripts.split_parts import split_long_segments
        logger.info(i18n("Splitting long segments into parts (AI-guided)..."))
        transcript_segments_for_split = create_viral_segments.load_transcript(ctx.project_folder)
        ctx.viral_segments = split_long_segments(
            ctx.viral_segments,
            transcript_json_path=os.path.join(ctx.project_folder, "input.json"),
            transcript_segments=transcript_segments_for_split,
            target_part_duration=args.target_part_duration,
            min_part_duration=max(args.min_duration, 30),
            max_normal_duration=args.max_duration,
            ai_mode=ctx.ai_backend,
            api_key=ctx.api_key,
            model_name=args.ai_model_name,
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
    args = ctx.args

    if ctx.workflow_choice == "3":
        logger.info(i18n("Workflow 3 (Subtitles Only): Skipping Cut and Edit."))
        return

    cuts_folder = os.path.join(ctx.project_folder, "cuts")
    skip_cutting = False

    if os.path.exists(cuts_folder) and os.listdir(cuts_folder):
        logger.info(i18n("Existing cuts found in: {}").format(cuts_folder))
        if args.skip_prompts:
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
        smart_trim=args.smart_trim,
        trim_pad_start=args.trim_pad_start,
        trim_pad_end=args.trim_pad_end,
        scene_detection=args.scene_detection,
    )

    # Remove silence (jump cuts)
    if args.remove_silence:
        from scripts import remove_silence
        logger.info(i18n("Removing silences (jump cuts)..."))
        remove_silence.process_project(
            project_folder=ctx.project_folder,
            noise_db=args.silence_threshold,
            min_silence_duration=args.silence_min_duration,
            max_silence_keep=args.silence_max_keep,
        )


# ---------------------------------------------------------------------------
# Stage 5: Filler Removal + Speed Ramp
# ---------------------------------------------------------------------------

def stage_filler_speed(ctx: PipelineContext) -> None:
    """Remove filler words and apply speed ramp."""
    args = ctx.args
    if ctx.workflow_choice == "3":
        return

    # Filler word removal
    if args.remove_fillers:
        import glob as glob_mod
        from scripts.filler_removal import detect_fillers, remove_fillers_from_video, update_subtitle_json
        from scripts.smart_trim import load_whisperx_words

        cuts_folder = os.path.join(ctx.project_folder, "cuts")
        subs_folder = os.path.join(ctx.project_folder, "subs")
        video_files = sorted(glob_mod.glob(os.path.join(cuts_folder, "*_original_scale.mp4")))

        for video_path in video_files:
            filename = os.path.basename(video_path)
            base_name = filename.replace("_original_scale.mp4", "")
            json_path = os.path.join(subs_folder, f"{base_name}_processed.json")
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

    # Speed ramp
    if args.speed_ramp:
        import glob as glob_mod
        from scripts.speed_ramp import apply_speed_ramp
        from scripts.remove_silence import detect_silences

        cuts_folder = os.path.join(ctx.project_folder, "cuts")
        video_files = sorted(glob_mod.glob(os.path.join(cuts_folder, "*_original_scale.mp4")))

        segments_path = os.path.join(ctx.project_folder, "viral_segments.txt")
        all_zoom_cues = []
        if os.path.exists(segments_path):
            with open(segments_path, "r", encoding="utf-8") as f:
                vs = json.load(f)
            all_zoom_cues = [s.get("zoom_cues", []) for s in vs.get("segments", [])]

        for idx, video_path in enumerate(video_files):
            silences = detect_silences(video_path, noise_db=-35, min_duration=0.8)
            highlights = None
            if idx < len(all_zoom_cues) and all_zoom_cues[idx]:
                highlights = [{"timestamp": z.get("timestamp", 0), "duration": z.get("duration", 1.5)} for z in all_zoom_cues[idx]]
            if silences:
                temp_out = video_path + ".tmp.mp4"
                if apply_speed_ramp(video_path, temp_out, silences, speed_up_factor=args.speed_up_factor, highlights=highlights):
                    os.replace(temp_out, video_path)
                    logger.info(f"Speed ramp applied to {os.path.basename(video_path)}")
                elif os.path.exists(temp_out):
                    os.remove(temp_out)


# ---------------------------------------------------------------------------
# Stage 6: Clip Quality Validation
# ---------------------------------------------------------------------------

def stage_quality(ctx: PipelineContext) -> None:
    """Validate clip quality: silence, hooks, blur, pacing, composite, engagement."""
    args = ctx.args
    if ctx.workflow_choice == "3":
        return

    if not any([args.validate_clips, args.hook_detection, args.blur_detection, args.pacing_analysis, args.composite_scoring]):
        return

    import glob as glob_mod
    from scripts.smart_trim import load_whisperx_words
    cuts_folder = os.path.join(ctx.project_folder, "cuts")
    subs_folder = os.path.join(ctx.project_folder, "subs")
    video_files = sorted(glob_mod.glob(os.path.join(cuts_folder, "*_original_scale.mp4")))

    if not video_files or not ctx.viral_segments or "segments" not in ctx.viral_segments:
        return

    logger.info(i18n("Validating clip quality..."))

    for idx, video_path in enumerate(video_files):
        if idx >= len(ctx.viral_segments["segments"]):
            logger.warning(f"Skipping validation for {os.path.basename(video_path)}: no matching segment data")
            continue
        seg = ctx.viral_segments["segments"][idx]
        filename = os.path.basename(video_path)

        if args.validate_clips:
            from scripts.clip_validator import validate_clip_boundaries
            boundary = validate_clip_boundaries(video_path, noise_db=args.silence_threshold)
            seg["speech_ratio"] = boundary["speech_ratio"]
            seg["starts_on_silence"] = boundary["starts_on_silence"]
            seg["ends_on_silence"] = boundary["ends_on_silence"]
            if boundary["starts_on_silence"]:
                logger.warning(f"  {filename}: starts on silence!")
            logger.info(f"  {filename}: speech_ratio={boundary['speech_ratio']}")

        if args.hook_detection:
            from scripts.hook_scorer import score_hook
            base_name = filename.replace("_original_scale.mp4", "")
            json_path = os.path.join(subs_folder, f"{base_name}_processed.json")
            words = load_whisperx_words(json_path) if os.path.exists(json_path) else []
            hook = score_hook(video_path, words)
            seg["hook_score"] = hook["hook_score"]
            seg["hook_audio_energy"] = hook["audio_energy"]
            logger.info(f"  {filename}: hook_score={hook['hook_score']}")

        if args.blur_detection:
            from scripts.blur_detector import detect_blur_frames
            blur = detect_blur_frames(video_path)
            seg["blur_ratio"] = blur["blur_ratio"]
            seg["avg_sharpness"] = blur["avg_sharpness"]
            if blur["blur_ratio"] > args.max_blur_ratio:
                logger.warning(f"  {filename}: high blur ratio {blur['blur_ratio']:.2f}")
            logger.info(f"  {filename}: blur_ratio={blur['blur_ratio']}, sharpness={blur['avg_sharpness']}")

        if args.pacing_analysis:
            from scripts.pacing_analyzer import analyze_pacing
            base_name = filename.replace("_original_scale.mp4", "")
            json_path = os.path.join(subs_folder, f"{base_name}_processed.json")
            words = load_whisperx_words(json_path) if os.path.exists(json_path) else []
            pacing = analyze_pacing(video_path, words)
            seg["pacing_score"] = pacing["pacing_score"]
            seg["words_per_sec"] = pacing["words_per_sec"]
            seg["avg_rms_energy"] = pacing["avg_rms_energy"]
            logger.info(f"  {filename}: pacing_score={pacing['pacing_score']}, wps={pacing['words_per_sec']}")

        if args.validate_clips:
            from scripts.clip_validator import score_visual_variety, analyze_speaker_activity
            variety = score_visual_variety(video_path)
            seg["visual_variety_score"] = variety["visual_variety_score"]
            seg["scene_change_count"] = variety["scene_change_count"]
            logger.info(f"  {filename}: visual_variety={variety['visual_variety_score']}")

            base_name = filename.replace("_original_scale.mp4", "")
            json_path = os.path.join(subs_folder, f"{base_name}_processed.json")
            words = load_whisperx_words(json_path) if os.path.exists(json_path) else []
            if words:
                speaker = analyze_speaker_activity(words, 0, words[-1].get("end", 0))
                seg["speaking_time_ratio"] = speaker["speaking_time_ratio"]
                logger.info(f"  {filename}: speaking_ratio={speaker['speaking_time_ratio']}")

        if args.composite_scoring:
            from scripts.composite_scorer import compute_composite_score
            composite = compute_composite_score(
                hook_score=seg.get("hook_score", 50.0),
                speech_ratio=seg.get("speech_ratio", 0.8),
                pacing_score=seg.get("pacing_score", 50.0),
                blur_ratio=seg.get("blur_ratio", 0.0),
                visual_variety_score=seg.get("visual_variety_score", 50.0),
            )
            seg["composite_quality_score"] = composite
            logger.info(f"  {filename}: composite_score={composite}")

        if args.engagement_prediction:
            from scripts.engagement_predictor import predict_from_metadata
            engagement = predict_from_metadata(seg, model_path=args.engagement_model)
            seg["engagement_score"] = engagement
            logger.info(f"  {filename}: engagement_score={engagement}")

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
    args = ctx.args

    if ctx.workflow_choice != "3":
        logger.info(i18n("Editing video with {} (Mode: {})...").format(ctx.face_model, ctx.face_mode))

        try:
            dead_zone_val = int(args.face_dead_zone)
        except (ValueError, TypeError):
            dead_zone_val = 40

        edit_video.edit(
            project_folder=ctx.project_folder,
            face_model=ctx.face_model,
            face_mode=ctx.face_mode,
            detection_period=ctx.detection_intervals,
            filter_threshold=args.face_filter_threshold,
            two_face_threshold=args.face_two_threshold,
            confidence_threshold=args.face_confidence_threshold,
            dead_zone=dead_zone_val,
            focus_active_speaker=args.focus_active_speaker,
            active_speaker_mar=args.active_speaker_mar,
            active_speaker_score_diff=args.active_speaker_score_diff,
            include_motion=args.include_motion,
            active_speaker_motion_deadzone=args.active_speaker_motion_threshold,
            active_speaker_motion_sensitivity=args.active_speaker_motion_sensitivity,
            active_speaker_decay=args.active_speaker_decay,
            segments_data=ctx.viral_segments.get("segments", []) if ctx.viral_segments else None,
            no_face_mode=args.no_face_mode,
            zoom_out_factor=args.zoom_out_factor
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
    args = ctx.args

    # Translation
    if args.translate_target and args.translate_target.lower() != "none":
        logger.info(i18n("Translating subtitles to: {}").format(args.translate_target))
        import asyncio
        try:
            asyncio.run(translate_json.translate_project_subs(ctx.project_folder, args.translate_target))
        except Exception as e:
            logger.error(i18n("Translation failed: {}").format(e))

    ctx.sub_config = get_subtitle_config(args.subtitle_config)

    # Split-screen auto-adjust
    if getattr(args, 'add_distraction', False):
        _ratio = getattr(args, 'distraction_ratio', 0.5)
        ctx.sub_config['vertical_position'] = int(620 * (1 - _ratio))

    # Generate .ass files
    try:
        adjust_subtitles.adjust(project_folder=ctx.project_folder, **ctx.sub_config)
    except FileNotFoundError as fnf_error:
        logger.error(i18n("[ERROR] Subtitle processing failed: {}").format(str(fnf_error)))
        logger.info(i18n("Tip: If you are using Workflow 3 (Subtitles Only), ensure the 'subs' folder exists and contains valid JSON files."))
        sys.exit(1)
    except Exception as e:
        logger.error(i18n("[ERROR] Unexpected error during subtitle processing: {}").format(str(e)))
        raise e

    # Add music
    if args.add_music:
        logger.info(i18n("Adding background music..."))
        try:
            _segs = ctx.viral_segments.get("segments", []) if ctx.viral_segments else []
            add_music.add_music_to_project(
                project_folder=ctx.project_folder,
                music_dir=args.music_dir,
                music_file=args.music_file,
                music_volume=args.music_volume,
                segments=_segs,
            )
        except Exception as e:
            logger.warning(f"Music addition failed: {e}")

    # Split-screen distraction
    if args.add_distraction:
        logger.info("Adding split-screen distraction video...")
        try:
            from scripts.add_distraction_video import add_distraction_to_project
            from scripts.add_distraction_video import DEFAULT_DISTRACTION_DIR as _DIST_DIR
            add_distraction_to_project(
                project_folder=ctx.project_folder,
                distraction_dir=args.distraction_dir or _DIST_DIR,
                distraction_file=args.distraction_file,
                no_fetch=args.distraction_no_fetch,
                main_crop_y=0 if getattr(args, 'face_mode', '1') == '2' else None,
                distraction_ratio=getattr(args, 'distraction_ratio', 0.35),
            )
        except Exception as e:
            logger.warning(f"Split-screen distraction failed: {e}")

    # Burn subtitles
    try:
        if args.add_distraction:
            split_screen_folder = os.path.join(ctx.project_folder, 'split_screen')
            split_suffix = "_music_split" if args.add_music else "_split"
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
# Stage 9: Post-production (thumbnails, overlays, color grading, dubbing)
# ---------------------------------------------------------------------------

def stage_post_production(ctx: PipelineContext) -> None:
    """Apply post-production effects: thumbnails, color grading, progress bar, emoji, dubbing."""
    args = ctx.args
    if ctx.workflow_choice == "3":
        return

    import glob as glob_mod

    # Auto thumbnails
    if args.auto_thumbnail:
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

    for video_path in video_files:
        temp_out = video_path + ".tmp.mp4"

        if args.color_grade:
            from scripts.color_grading import apply_lut
            if apply_lut(video_path, temp_out, lut_name=args.color_grade, intensity=args.grade_intensity):
                os.replace(temp_out, video_path)
                logger.info(f"Color grading applied: {os.path.basename(video_path)}")

        if args.progress_bar:
            from scripts.overlay_effects import add_progress_bar
            if add_progress_bar(video_path, temp_out, bar_color=args.bar_color, bar_position=args.bar_position):
                os.replace(temp_out, video_path)
                logger.info(f"Progress bar added: {os.path.basename(video_path)}")

        if args.emoji_overlay and ctx.viral_segments and "segments" in ctx.viral_segments:
            from scripts.overlay_effects import add_emoji_overlay
            idx = video_files.index(video_path)
            if idx < len(ctx.viral_segments["segments"]):
                emoji_cues = ctx.viral_segments["segments"][idx].get("emoji_cues", [])
                if emoji_cues:
                    if add_emoji_overlay(video_path, temp_out, emoji_cues):
                        os.replace(temp_out, video_path)
                        logger.info(f"Emoji overlay added: {os.path.basename(video_path)}")

        if os.path.exists(temp_out):
            os.remove(temp_out)

    # AI Dubbing
    if args.dubbing:
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
                dubbed_path = video_path.rsplit(".", 1)[0] + f"_dubbed_{args.dubbing_language}.mp4"
                if dub_segment(video_path, text, args.dubbing_language, dubbed_path, original_volume=args.dubbing_original_volume):
                    logger.info(f"Dubbed: {os.path.basename(dubbed_path)}")


# ---------------------------------------------------------------------------
# Stage 10: Save Config
# ---------------------------------------------------------------------------

def stage_save_config(ctx: PipelineContext) -> None:
    """Save processing configuration to project folder."""
    args = ctx.args
    try:
        used_ai_model = args.ai_model_name
        if not used_ai_model and ctx.ai_backend != "manual":
            if ctx.ai_backend == "gemini":
                used_ai_model = ctx.api_config.get("gemini", {}).get("model", "default")
            elif ctx.ai_backend == "g4f":
                used_ai_model = ctx.api_config.get("g4f", {}).get("model", "default")

        current_sub_config = ctx.sub_config if ctx.sub_config else get_subtitle_config(args.subtitle_config)

        final_config = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "workflow": ctx.workflow_choice,
            "ai_config": {
                "backend": ctx.ai_backend,
                "model_name": used_ai_model,
                "viral_mode": ctx.viral_mode,
                "themes": ctx.themes,
                "num_segments": ctx.num_segments,
                "chunk_size": args.chunk_size
            },
            "face_config": {
                "model": ctx.face_model,
                "mode": ctx.face_mode,
                "detect_interval": args.face_detect_interval,
                "filter_threshold": args.face_filter_threshold,
                "two_face_threshold": args.face_two_threshold,
                "confidence_threshold": args.face_confidence_threshold,
                "dead_zone": args.face_dead_zone,
                "focus_active_speaker": args.focus_active_speaker,
                "active_speaker_mar": args.active_speaker_mar,
                "active_speaker_score_diff": args.active_speaker_score_diff,
                "include_motion": args.include_motion
            },
            "video_config": {
                "min_duration": args.min_duration,
                "max_duration": args.max_duration,
                "whisper_model": args.model
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
