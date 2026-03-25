"""Process management and pipeline argument building for WebUI."""
from __future__ import annotations

import datetime
import json
import logging
import multiprocessing
import os
import queue
import shutil
import sys
import time

import gradio as gr
import psutil

from webui.presets import WORKING_DIR, VIRALS_DIR, convert_color_to_ass

# Ensure WORKING_DIR is on sys.path so that project-level imports work
if WORKING_DIR not in sys.path:
    sys.path.insert(0, WORKING_DIR)

from i18n.i18n import I18nAuto

i18n = I18nAuto()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Import webui-local modules (they live in the same package directory)
# ---------------------------------------------------------------------------
_WEBUI_DIR = os.path.dirname(os.path.abspath(__file__))
if _WEBUI_DIR not in sys.path:
    sys.path.insert(0, _WEBUI_DIR)
import library  # type: ignore[import-untyped]

# ---------------------------------------------------------------------------
# Global variables
# ---------------------------------------------------------------------------
current_worker: multiprocessing.Process | None = None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_captions_for_project(proj_name: str):
    """Genere les captions TikTok pour un projet existant (appel LLM a la demande)."""
    if not proj_name:
        return "Aucun projet selectionne.", None

    try:
        from scripts.analysis import create_viral_segments
        from scripts.transcription import save_json

        # Charger api_config
        api_config_path = os.path.join(WORKING_DIR, "api_config.json")
        with open(api_config_path, "r", encoding="utf-8") as f:
            api_cfg = json.load(f)

        ai_mode = api_cfg.get("selected_api", "gemini")
        api_key = api_cfg.get(ai_mode, {}).get("api_key") or os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY")
        model_name = api_cfg.get(ai_mode, {}).get("model")

        # Fallback : si pas de cle valide, utiliser le backend de webui/settings.json
        if (not api_key or api_key in ("", "SUA_KEY_AQUI")) and ai_mode not in ("pleiade", "g4f"):
            settings_path = os.path.join(os.path.dirname(__file__), "settings.json")
            if os.path.exists(settings_path):
                try:
                    with open(settings_path, "r", encoding="utf-8") as _sf:
                        _settings = json.load(_sf)
                    _fallback_mode = _settings.get("ai_backend", "")
                    if _fallback_mode and _fallback_mode != "manual":
                        ai_mode = _fallback_mode
                        api_key = api_cfg.get(ai_mode, {}).get("api_key") or ""
                        model_name = _settings.get("ai_model_name") or api_cfg.get(ai_mode, {}).get("model")
                except Exception:
                    logger.debug("Failed to load settings.json fallback", exc_info=True)

        # Trouver le dossier projet
        project_folder = os.path.join(VIRALS_DIR, proj_name) if not os.path.isabs(proj_name) else proj_name

        # Priorite : backend du process_config.json du projet (plus fiable)
        proc_cfg_path = os.path.join(project_folder, "process_config.json")
        if os.path.exists(proc_cfg_path):
            try:
                with open(proc_cfg_path, "r", encoding="utf-8") as f:
                    proc_cfg = json.load(f)
                proj_backend = proc_cfg.get("ai_config", {}).get("backend", "")
                if proj_backend and proj_backend != "manual":
                    ai_mode = proj_backend
                    model_name = proc_cfg.get("ai_config", {}).get("model_name") or api_cfg.get(ai_mode, {}).get("model")
                    api_key = api_cfg.get(ai_mode, {}).get("api_key") or os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY")
            except Exception:
                logger.debug("Failed to load process_config.json for AI backend", exc_info=True)

        if ai_mode not in ("gemini", "g4f", "pleiade"):
            return f"Backend '{ai_mode}' ne supporte pas la generation automatique.", None

        segments_file = os.path.join(project_folder, "viral_segments.txt")
        if not os.path.exists(segments_file):
            return "viral_segments.txt introuvable pour ce projet.", None

        with open(segments_file, "r", encoding="utf-8") as f:
            viral_segments = json.load(f)

        segments = viral_segments.get("segments", [])
        if not segments:
            return "Aucun segment trouve.", None

        # Charger la transcription
        transcript = create_viral_segments.load_transcript(project_folder)
        transcript_text = create_viral_segments.preprocess_transcript_for_ai(transcript)

        # Generer les captions
        updated = create_viral_segments.generate_tiktok_captions(
            segments, transcript_text, ai_mode=ai_mode, api_key=api_key, model_name=model_name,
            content_type=viral_segments.get("content_type")
        )
        # Validation des captions
        updated = create_viral_segments.validate_captions(
            updated, transcript_text, ai_mode=ai_mode, api_key=api_key, model_name=model_name
        )
        viral_segments["segments"] = updated
        save_json.save_viral_segments(viral_segments, project_folder=project_folder)

        count = sum(1 for s in updated if s.get("tiktok_caption"))
        gallery = library.generate_project_gallery(proj_name)
        return f"{count}/{len(updated)} captions generees et validees.", gallery

    except Exception as e:
        return f"Erreur : {e}", None


def kill_process():
    global current_worker
    if current_worker and current_worker.is_alive():
        try:
            parent = psutil.Process(current_worker.pid)
            for child in parent.children(recursive=True):
                child.kill()
            parent.kill()
            current_worker.join(timeout=5)
            current_worker = None
            return i18n("Process terminated.")
        except psutil.NoSuchProcess:
            current_worker = None
            return i18n("Process already terminated.")
        except Exception as e:
            return i18n("Error terminating process: {}").format(e)
    return i18n("No process running.")


def run_viral_cutter(input_source, project_name, url, video_file, segments, viral, themes, min_duration, max_duration, model, ai_backend, api_key, ai_model_name, chunk_size, workflow, face_model, face_mode, face_detect_interval, no_face_mode,
                     face_filter_thresh, face_two_thresh, face_conf_thresh, face_dead_zone, zoom_out_factor,
                     vertical_offset, single_face_zoom, ema_alpha, detection_resolution,  # NEW: 1-face visual params
                     focus_active_speaker, active_speaker_mar, active_speaker_score_diff, include_motion, active_speaker_motion_threshold, active_speaker_motion_sensitivity, active_speaker_decay,
                     content_type, enable_scoring, min_score, enable_validation,
                     use_custom_subs, font_name, font_size, font_color, highlight_color, outline_color, outline_thickness, shadow_color, shadow_size, is_bold, is_italic, is_uppercase, vertical_pos, alignment,
                     h_size, w_block, gap, mode, under, strike, border_s, remove_punc, animation,
                     video_quality, use_youtube_subs, translate_target,
                     add_music, music_dir, music_file, music_volume,
                     add_distraction, distraction_dir, distraction_file, distraction_no_fetch, distraction_ratio,
                     smart_trim, trim_pad_start, trim_pad_end, scene_detection,
                     validate_clips, hook_detection, min_hook_score, blur_detection, max_blur_ratio,
                     pacing_analysis, composite_scoring,
                     remove_fillers, auto_thumbnail, auto_zoom, speed_ramp, speed_up_factor,
                     progress_bar, bar_color, bar_position, ab_variants, num_variants,
                     layout_template, auto_broll, transitions, output_resolution,
                     emoji_overlay, color_grade, grade_intensity,
                     engagement_prediction, dubbing, dubbing_language, dubbing_original_volume,
                     remove_silence, silence_threshold, silence_min_duration, silence_max_keep,
                     enable_parts, target_part_duration,
                     post_youtube, post_tiktok, youtube_privacy, post_interval_minutes, post_first_time):

    global current_worker
    min_score = min_score if min_score is not None else 70
    yield "", gr.update(value=i18n("Running..."), interactive=False), gr.update(visible=True), None

    # -----------------------------------------------------------------
    # Build args dict (replaces cmd.extend)
    # -----------------------------------------------------------------
    args_dict: dict = {}
    project_folder_path = None

    # Input source handling
    if input_source == "Existing Project":
        if not project_name:
            yield i18n("Error: No project selected."), gr.update(value=i18n("Start Processing"), interactive=True), gr.update(visible=False), None
            return
        full_project_path = os.path.join(VIRALS_DIR, project_name)
        args_dict["project_path"] = full_project_path
        project_folder_path = full_project_path
    elif input_source == "Upload Video":
        if not video_file:
            yield i18n("Error: No video file uploaded."), gr.update(value=i18n("Start Processing"), interactive=True), gr.update(visible=False), None
            return

        # Determine project name from filename
        original_filename = os.path.basename(video_file)
        name_no_ext = os.path.splitext(original_filename)[0]
        # Sanitize: Allow alphanumeric, space, dash, underscore
        safe_name = "".join([c for c in name_no_ext if c.isalnum() or c in " _-"]).strip()
        if not safe_name:
            safe_name = "Untitled_Upload"

        # Always append timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        project_name_upload = f"{safe_name}_{timestamp}"
        project_path = os.path.join(VIRALS_DIR, project_name_upload)

        os.makedirs(project_path, exist_ok=True)

        target_path = os.path.join(project_path, "input.mp4")
        shutil.copy(video_file, target_path)

        args_dict["project_path"] = project_path
        args_dict["skip_youtube_subs"] = True
    else:
        if url:
            args_dict["url"] = url
        if video_quality:
            args_dict["video_quality"] = video_quality
        if not use_youtube_subs:
            args_dict["skip_youtube_subs"] = True

    # Translation
    if translate_target and translate_target != "None" and translate_target.strip():
        args_dict["translate_target"] = translate_target.strip()

    # Basic params
    args_dict["segments"] = int(segments)
    args_dict["viral"] = bool(viral)
    if themes:
        args_dict["themes"] = themes
    args_dict["min_duration"] = int(min_duration)
    args_dict["max_duration"] = int(max_duration)
    args_dict["model"] = model
    args_dict["ai_backend"] = ai_backend
    if ai_model_name:
        args_dict["ai_model_name"] = str(ai_model_name)
    if chunk_size:
        args_dict["chunk_size"] = int(chunk_size)
    args_dict["workflow"] = {"Full": "1", "Cut Only": "2", "Subtitles Only": "3"}.get(str(workflow), "1")
    args_dict["skip_prompts"] = True

    # Face params
    args_dict["face_model"] = face_model
    args_dict["face_mode"] = face_mode
    args_dict["face_detect_interval"] = str(face_detect_interval)
    args_dict["no_face_mode"] = no_face_mode
    if face_filter_thresh is not None:
        args_dict["face_filter_threshold"] = float(face_filter_thresh)
    if face_two_thresh is not None:
        args_dict["face_two_threshold"] = float(face_two_thresh)
    if face_conf_thresh is not None:
        args_dict["face_confidence_threshold"] = float(face_conf_thresh)
    if face_dead_zone is not None:
        args_dict["face_dead_zone"] = str(int(face_dead_zone))
    if zoom_out_factor is not None:
        args_dict["zoom_out_factor"] = float(zoom_out_factor)
    # NEW: 1-face visual params
    if vertical_offset is not None:
        args_dict["vertical_offset"] = float(vertical_offset)
    if single_face_zoom is not None:
        args_dict["single_face_zoom"] = float(single_face_zoom)
    if ema_alpha is not None:
        args_dict["ema_alpha"] = float(ema_alpha)
    if detection_resolution is not None:
        args_dict["detection_resolution"] = int(detection_resolution)

    # Active speaker
    if focus_active_speaker:
        args_dict["focus_active_speaker"] = True
        if active_speaker_mar is not None:
            args_dict["active_speaker_mar"] = float(active_speaker_mar)
        if active_speaker_score_diff is not None:
            args_dict["active_speaker_score_diff"] = float(active_speaker_score_diff)
        if include_motion:
            args_dict["include_motion"] = True
        if active_speaker_motion_threshold is not None:
            args_dict["active_speaker_motion_threshold"] = float(active_speaker_motion_threshold)
        if active_speaker_motion_sensitivity is not None:
            args_dict["active_speaker_motion_sensitivity"] = float(active_speaker_motion_sensitivity)
        if active_speaker_decay is not None:
            args_dict["active_speaker_decay"] = float(active_speaker_decay)

    # Content type
    if content_type:
        types = content_type if isinstance(content_type, list) else [content_type]
        filtered = [t for t in types if t and t != "auto"]
        if filtered:
            args_dict["content_type"] = filtered

    # Scoring / validation
    if enable_scoring:
        args_dict["enable_scoring"] = True
        args_dict["min_score"] = int(min_score)
    if enable_validation:
        args_dict["enable_validation"] = True

    # Parts
    if enable_parts:
        args_dict["enable_parts"] = True
        if target_part_duration is not None:
            args_dict["target_part_duration"] = int(target_part_duration)

    # Music
    if add_music:
        args_dict["add_music"] = True
        if music_dir:
            args_dict["music_dir"] = music_dir
        if music_file:
            args_dict["music_file"] = music_file
        if music_volume is not None:
            args_dict["music_volume"] = float(music_volume)

    # Split-screen distraction
    if add_distraction:
        args_dict["add_distraction"] = True
        if distraction_dir:
            args_dict["distraction_dir"] = distraction_dir
        if distraction_file:
            args_dict["distraction_file"] = distraction_file
        if distraction_no_fetch:
            args_dict["distraction_no_fetch"] = True
        if distraction_ratio is not None:
            args_dict["distraction_ratio"] = float(distraction_ratio)

    # Video quality features
    if smart_trim:
        args_dict["smart_trim"] = True
        if trim_pad_start is not None:
            args_dict["trim_pad_start"] = float(trim_pad_start)
        if trim_pad_end is not None:
            args_dict["trim_pad_end"] = float(trim_pad_end)
    if scene_detection:
        args_dict["scene_detection"] = True
    if validate_clips:
        args_dict["validate_clips"] = True
    if hook_detection:
        args_dict["hook_detection"] = True
        if min_hook_score is not None:
            args_dict["min_hook_score"] = int(min_hook_score)
    if blur_detection:
        args_dict["blur_detection"] = True
        if max_blur_ratio is not None:
            args_dict["max_blur_ratio"] = float(max_blur_ratio)
    if pacing_analysis:
        args_dict["pacing_analysis"] = True
    if composite_scoring:
        args_dict["composite_scoring"] = True

    # Phase 3 features
    if remove_fillers:
        args_dict["remove_fillers"] = True
    if auto_thumbnail:
        args_dict["auto_thumbnail"] = True
    if auto_zoom:
        args_dict["auto_zoom"] = True
    if speed_ramp:
        args_dict["speed_ramp"] = True
        if speed_up_factor is not None:
            args_dict["speed_up_factor"] = float(speed_up_factor)

    # Phase 4 post-production
    if progress_bar:
        args_dict["progress_bar"] = True
        args_dict["bar_color"] = bar_color or "white"
        args_dict["bar_position"] = bar_position or "top"
    if ab_variants:
        args_dict["ab_variants"] = True
        if num_variants is not None:
            args_dict["num_variants"] = int(num_variants)
    if layout_template:
        args_dict["layout"] = str(layout_template)
    if auto_broll:
        args_dict["auto_broll"] = True
    if transitions:
        args_dict["transitions"] = str(transitions)
    args_dict["output_resolution"] = output_resolution or "1080p"
    if emoji_overlay:
        args_dict["emoji_overlay"] = True
    if color_grade:
        args_dict["color_grade"] = str(color_grade)
        if grade_intensity is not None:
            args_dict["grade_intensity"] = float(grade_intensity)

    # Phase 5 advanced AI
    if engagement_prediction:
        args_dict["engagement_prediction"] = True
    if dubbing:
        args_dict["dubbing"] = True
        args_dict["dubbing_language"] = dubbing_language or "en"
        if dubbing_original_volume is not None:
            args_dict["dubbing_original_volume"] = float(dubbing_original_volume)

    # Jump Cuts (Silence Removal)
    if remove_silence:
        args_dict["remove_silence"] = True
        if silence_threshold is not None:
            args_dict["silence_threshold"] = float(silence_threshold)
        if silence_min_duration is not None:
            args_dict["silence_min_duration"] = float(silence_min_duration)
        if silence_max_keep is not None:
            args_dict["silence_max_keep"] = float(silence_max_keep)

    # Subtitle config
    if use_custom_subs:
        # En mode distraction, placer les subs exactement sur la ligne de coupure.
        # PlayResY=640. Formule : int(620 * (1 - ratio))
        #   ratio=0.50 -> 310 (split a y~960px), ratio=0.35 -> 403, ratio=0.30 -> 434
        if add_distraction:
            _ratio = distraction_ratio if distraction_ratio is not None else 0.5
            effective_vertical_pos = int(620 * (1 - _ratio))
        else:
            effective_vertical_pos = vertical_pos
        subtitle_config = {
            "font": font_name, "base_size": int(font_size), "base_color": convert_color_to_ass(font_color), "highlight_color": convert_color_to_ass(highlight_color),
            "outline_color": convert_color_to_ass(outline_color), "outline_thickness": outline_thickness, "shadow_color": convert_color_to_ass(shadow_color),
            "shadow_size": shadow_size, "vertical_position": effective_vertical_pos, "alignment": alignment, "bold": 1 if is_bold else 0, "italic": 1 if is_italic else 0,
            "underline": 1 if under else 0, "strikeout": 1 if strike else 0, "border_style": border_s, "words_per_block": int(w_block), "gap_limit": gap,
            "mode": mode, "highlight_size": int(h_size), "remove_punctuation": remove_punc,
            "animation": animation
        }
        subtitle_config["uppercase"] = 1 if is_uppercase else 0

        subtitle_config_path = os.path.join(WORKING_DIR, "temp_subtitle_config.json")
        try:
            with open(subtitle_config_path, "w", encoding="utf-8") as f:
                json.dump(subtitle_config, f, indent=4)
            args_dict["subtitle_config"] = subtitle_config_path
        except Exception:
            logger.debug("Failed to write subtitle config", exc_info=True)

    # -----------------------------------------------------------------
    # Environment vars (API keys — not in args_dict)
    # -----------------------------------------------------------------
    env_vars: dict[str, str] = {}
    if api_key:
        env_vars["GEMINI_API_KEY"] = api_key

    # -----------------------------------------------------------------
    # Spawn worker process
    # -----------------------------------------------------------------
    from webui.pipeline_worker import pipeline_worker

    progress_q: multiprocessing.Queue = multiprocessing.Queue()
    worker = multiprocessing.Process(
        target=pipeline_worker,
        args=(args_dict, progress_q, env_vars, WORKING_DIR),
        daemon=True,
    )
    worker.start()
    current_worker = worker

    logs = ""
    formatter = logging.Formatter('[%(levelname)s] %(name)s: %(message)s')
    last_yield_time = time.time()

    try:
        while True:
            try:
                record = progress_q.get(timeout=0.2)
            except queue.Empty:
                if not worker.is_alive():
                    break
                now = time.time()
                if now - last_yield_time > 0.5:
                    yield logs, gr.update(visible=True, interactive=False), gr.update(visible=True), None
                    last_yield_time = now
                continue

            if isinstance(record, dict):
                if record["type"] == "done":
                    project_folder_path = record.get("project_folder", "")
                    break
                elif record["type"] == "error":
                    logs += f"\n[PIPELINE ERROR]\n{record['text']}\n"
                    break
            elif isinstance(record, logging.LogRecord):
                line = formatter.format(record)
                logs += line + "\n"
                if "Project Folder:" in line:
                    parts = line.split("Project Folder:")
                    if len(parts) > 1:
                        project_folder_path = parts[1].strip()

            now = time.time()
            if now - last_yield_time > 0.2:
                yield logs, gr.update(visible=True, interactive=False), gr.update(visible=True), None
                last_yield_time = now

        # Final yield to ensure all logs are shown
        yield logs, gr.update(visible=True, interactive=False), gr.update(visible=True), None
    except Exception as e:
        logs += f"\nError running process: {str(e)}\n"
        yield logs, gr.update(visible=True, interactive=False), gr.update(visible=True), None
    finally:
        if worker.is_alive():
            worker.join(timeout=5)
        current_worker = None

    # Brief wait to ensure filesystem flush
    time.sleep(0.1)

    # -- Auto Post ---------------------------------------------------------
    if (post_youtube or post_tiktok) and project_folder_path and os.path.exists(project_folder_path):
        try:
            import sys as _sys
            if WORKING_DIR not in _sys.path:
                _sys.path.insert(0, WORKING_DIR)
            from scripts.export.post_social import post_all_segments
            logs += "\n[Auto Post] Starting...\n"
            yield logs, gr.update(interactive=False), gr.update(visible=True), None
            post_results = post_all_segments(
                project_folder=project_folder_path,
                post_youtube=post_youtube,
                post_tiktok=post_tiktok,
                youtube_privacy=youtube_privacy,
                interval_minutes=float(post_interval_minutes),
                first_post_time=post_first_time,
            )
            for r in post_results:
                if r.get("status") == "success":
                    scheduled = f" (scheduled {r['scheduled_at']})" if r.get("scheduled_at") else ""
                    logs += f"[Auto Post] {r['platform'].upper()} uploaded{scheduled} — {r.get('url') or r.get('title')}\n"
                else:
                    logs += f"[Auto Post] {r['platform'].upper()} FAILED — {r.get('error')}\n"
        except Exception as _e:
            logs += f"[Auto Post] Error: {_e}\n"
    # ----------------------------------------------------------------------

    html_output = ""
    if project_folder_path and os.path.exists(project_folder_path):
        html_output = library.generate_project_gallery(project_folder_path, is_full_path=True)
    else:
        html_output = f"<h3>{i18n('Error: Project folder could not be determined from logs.')}</h3>"
    yield logs, gr.update(value=i18n("Start Processing"), interactive=True), gr.update(visible=False), html_output
