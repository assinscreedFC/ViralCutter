"""Process management and CLI command building for WebUI."""
from __future__ import annotations

import datetime
import json
import logging
import os
import shutil
import subprocess
import sys
import time

import gradio as gr
import psutil

from webui.presets import MAIN_SCRIPT_PATH, WORKING_DIR, VIRALS_DIR, convert_color_to_ass

# Ensure WORKING_DIR is on sys.path so that project-level imports work
if WORKING_DIR not in sys.path:
    sys.path.insert(0, WORKING_DIR)

from i18n.i18n import I18nAuto

i18n = I18nAuto()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Import webui-local modules (they live in the same package directory)
# ---------------------------------------------------------------------------
import library  # type: ignore[import-untyped]

# ---------------------------------------------------------------------------
# Global variables
# ---------------------------------------------------------------------------
current_process = None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_captions_for_project(proj_name: str):
    """Génère les captions TikTok pour un projet existant (appel LLM à la demande)."""
    if not proj_name:
        return "Aucun projet sélectionné.", None

    try:
        from scripts import create_viral_segments
        from scripts import save_json

        # Charger api_config
        api_config_path = os.path.join(WORKING_DIR, "api_config.json")
        with open(api_config_path, "r", encoding="utf-8") as f:
            api_cfg = json.load(f)

        ai_mode = api_cfg.get("selected_api", "gemini")
        api_key = api_cfg.get(ai_mode, {}).get("api_key") or os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY")
        model_name = api_cfg.get(ai_mode, {}).get("model")

        # Fallback : si pas de clé valide, utiliser le backend de webui/settings.json
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

        # Priorité : backend du process_config.json du projet (plus fiable)
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
            return f"Backend '{ai_mode}' ne supporte pas la génération automatique.", None

        segments_file = os.path.join(project_folder, "viral_segments.txt")
        if not os.path.exists(segments_file):
            return "viral_segments.txt introuvable pour ce projet.", None

        with open(segments_file, "r", encoding="utf-8") as f:
            viral_segments = json.load(f)

        segments = viral_segments.get("segments", [])
        if not segments:
            return "Aucun segment trouvé.", None

        # Charger la transcription
        transcript = create_viral_segments.load_transcript(project_folder)
        transcript_text = create_viral_segments.preprocess_transcript_for_ai(transcript)

        # Générer les captions
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
        return f"{count}/{len(updated)} captions générées et validées.", gallery

    except Exception as e:
        return f"Erreur : {e}", None


def kill_process():
    global current_process
    if current_process:
        try:
            parent = psutil.Process(current_process.pid)
            for child in parent.children(recursive=True):
                child.kill()
            parent.kill()
            current_process = None
            return i18n("Process terminated.")
        except Exception as e:
            return i18n("Error terminating process: {}").format(e)
    return i18n("No process running.")


def run_viral_cutter(input_source, project_name, url, video_file, segments, viral, themes, min_duration, max_duration, model, ai_backend, api_key, ai_model_name, chunk_size, workflow, face_model, face_mode, face_detect_interval, no_face_mode,
                     face_filter_thresh, face_two_thresh, face_conf_thresh, face_dead_zone, zoom_out_factor, focus_active_speaker, active_speaker_mar, active_speaker_score_diff, include_motion, active_speaker_motion_threshold, active_speaker_motion_sensitivity, active_speaker_decay,
                     content_type, enable_scoring, min_score, enable_validation,
                     use_custom_subs, font_name, font_size, font_color, highlight_color, outline_color, outline_thickness, shadow_color, shadow_size, is_bold, is_italic, is_uppercase, vertical_pos, alignment,
                     h_size, w_block, gap, mode, under, strike, border_s, remove_punc, video_quality, use_youtube_subs, translate_target,
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

    global current_process
    min_score = min_score if min_score is not None else 70
    yield "", gr.update(value=i18n("Running..."), interactive=False), gr.update(visible=True), None

    cmd = [sys.executable, MAIN_SCRIPT_PATH]

    # Input Source Logic
    if input_source == "Existing Project":
        if not project_name:
             yield i18n("Error: No project selected."), gr.update(value=i18n("Start Processing"), interactive=True), gr.update(visible=False), None
             return
        full_project_path = os.path.join(VIRALS_DIR, project_name)
        cmd.extend(["--project-path", full_project_path])
    elif input_source == "Upload Video":
        if not video_file:
             yield i18n("Error: No video file uploaded."), gr.update(value=i18n("Start Processing"), interactive=True), gr.update(visible=False), None
             return

        # Determine project name from filename
        original_filename = os.path.basename(video_file)
        name_no_ext = os.path.splitext(original_filename)[0]
        # Sanitize: Allow alphanumeric, space, dash, underscore
        safe_name = "".join([c for c in name_no_ext if c.isalnum() or c in " _-"]).strip()
        if not safe_name: safe_name = "Untitled_Upload"

        # Always append timestamp as requested
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        project_name_upload = f"{safe_name}_{timestamp}"
        project_path = os.path.join(VIRALS_DIR, project_name_upload)

        os.makedirs(project_path, exist_ok=True)

        target_path = os.path.join(project_path, "input.mp4")
        shutil.copy(video_file, target_path)

        cmd.extend(["--project-path", project_path])
        # Skip YouTube subs as it is a local upload
        cmd.append("--skip-youtube-subs")

    else:
        if url: cmd.extend(["--url", url])
        # Pass Video Quality
        if video_quality: cmd.extend(["--video-quality", video_quality])
        # Pass Subtitle Option (if False, we skip)
        if not use_youtube_subs: cmd.append("--skip-youtube-subs")

    # Translation
    if translate_target and translate_target != "None":
            cmd.extend(["--translate-target", translate_target])


    cmd.extend(["--segments", str(int(segments))])
    if viral: cmd.append("--viral")
    if themes: cmd.extend(["--themes", themes])
    cmd.extend(["--min-duration", str(int(min_duration))])
    cmd.extend(["--max-duration", str(int(max_duration))])
    cmd.extend(["--model", model])
    cmd.extend(["--ai-backend", ai_backend])
    # api_key is passed via environment variable to avoid exposure in process list

    # New AI Params
    if ai_model_name: cmd.extend(["--ai-model-name", str(ai_model_name)])
    if chunk_size: cmd.extend(["--chunk-size", str(int(chunk_size))])

    workflow_map = {"Full": "1", "Cut Only": "2", "Subtitles Only": "3"}
    cmd.extend(["--workflow", workflow_map.get(workflow, "1")])
    cmd.extend(["--face-model", face_model])
    cmd.extend(["--face-mode", face_mode])
    if face_detect_interval: cmd.extend(["--face-detect-interval", str(face_detect_interval)])
    if no_face_mode: cmd.extend(["--no-face-mode", no_face_mode])

    # New Face Params
    if face_filter_thresh is not None: cmd.extend(["--face-filter-threshold", str(face_filter_thresh)])
    if face_two_thresh is not None: cmd.extend(["--face-two-threshold", str(face_two_thresh)])
    if face_conf_thresh is not None: cmd.extend(["--face-confidence-threshold", str(face_conf_thresh)])
    if face_dead_zone is not None: cmd.extend(["--face-dead-zone", str(face_dead_zone)])



    cmd.append("--skip-prompts")

    if focus_active_speaker:
        cmd.append("--focus-active-speaker")
        if active_speaker_mar is not None: cmd.extend(["--active-speaker-mar", str(active_speaker_mar)])
        if active_speaker_score_diff is not None: cmd.extend(["--active-speaker-score-diff", str(active_speaker_score_diff)])
        if include_motion: cmd.append("--include-motion")
        if active_speaker_motion_threshold is not None: cmd.extend(["--active-speaker-motion-threshold", str(active_speaker_motion_threshold)])
        if active_speaker_motion_sensitivity is not None: cmd.extend(["--active-speaker-motion-sensitivity", str(active_speaker_motion_sensitivity)])
        if active_speaker_decay is not None: cmd.extend(["--active-speaker-decay", str(active_speaker_decay)])

    # Zoom out factor
    if zoom_out_factor is not None: cmd.extend(["--zoom-out-factor", str(zoom_out_factor)])

    # Advanced AI settings (content_type est une liste, vide = auto-détection)
    types = content_type if isinstance(content_type, list) else ([content_type] if content_type else [])
    for ct in [t for t in types if t and t != "auto"]:
        cmd.extend(["--content-type", ct])
    if enable_scoring:
        cmd.append("--enable-scoring")
        if min_score is not None: cmd.extend(["--min-score", str(int(min_score))])
    if enable_validation:
        cmd.append("--enable-validation")

    # Parts mode
    if enable_parts:
        cmd.append("--enable-parts")
        if target_part_duration is not None:
            cmd.extend(["--target-part-duration", str(int(target_part_duration))])

    # Music
    if add_music:
        cmd.append("--add-music")
        if music_dir: cmd.extend(["--music-dir", music_dir])
        if music_file: cmd.extend(["--music-file", music_file])
        if music_volume is not None: cmd.extend(["--music-volume", str(music_volume)])

    # Split-screen distraction
    if add_distraction:
        cmd.append("--add-distraction")
        if distraction_dir: cmd.extend(["--distraction-dir", distraction_dir])
        if distraction_file: cmd.extend(["--distraction-file", distraction_file])
        if distraction_no_fetch: cmd.append("--distraction-no-fetch")
        if distraction_ratio is not None: cmd.extend(["--distraction-ratio", str(distraction_ratio)])

    # Video Quality
    if smart_trim:
        cmd.append("--smart-trim")
        if trim_pad_start is not None: cmd.extend(["--trim-pad-start", str(trim_pad_start)])
        if trim_pad_end is not None: cmd.extend(["--trim-pad-end", str(trim_pad_end)])
    if scene_detection:
        cmd.append("--scene-detection")
    if validate_clips:
        cmd.append("--validate-clips")
    if hook_detection:
        cmd.append("--hook-detection")
        if min_hook_score is not None: cmd.extend(["--min-hook-score", str(int(min_hook_score))])
    if blur_detection:
        cmd.append("--blur-detection")
        if max_blur_ratio is not None: cmd.extend(["--max-blur-ratio", str(max_blur_ratio)])
    if pacing_analysis:
        cmd.append("--pacing-analysis")
    if composite_scoring:
        cmd.append("--composite-scoring")

    # Phase 3 features
    if remove_fillers:
        cmd.append("--remove-fillers")
    if auto_thumbnail:
        cmd.append("--auto-thumbnail")
    if auto_zoom:
        cmd.append("--auto-zoom")
    if speed_ramp:
        cmd.append("--speed-ramp")
        if speed_up_factor is not None: cmd.extend(["--speed-up-factor", str(speed_up_factor)])

    # Phase 4 post-production
    if progress_bar:
        cmd.append("--progress-bar")
        if bar_color: cmd.extend(["--bar-color", str(bar_color)])
        if bar_position: cmd.extend(["--bar-position", str(bar_position)])
    if ab_variants:
        cmd.append("--ab-variants")
        if num_variants is not None: cmd.extend(["--num-variants", str(int(num_variants))])
    if layout_template:
        cmd.extend(["--layout", str(layout_template)])
    if auto_broll:
        cmd.append("--auto-broll")
    if transitions:
        cmd.extend(["--transitions", str(transitions)])
    if output_resolution and output_resolution != "1080p":
        cmd.extend(["--output-resolution", str(output_resolution)])
    if emoji_overlay:
        cmd.append("--emoji-overlay")
    if color_grade:
        cmd.extend(["--color-grade", str(color_grade)])
        if grade_intensity is not None: cmd.extend(["--grade-intensity", str(grade_intensity)])

    # Phase 5 advanced AI
    if engagement_prediction:
        cmd.append("--engagement-prediction")
    if dubbing:
        cmd.append("--dubbing")
        if dubbing_language: cmd.extend(["--dubbing-language", str(dubbing_language)])
        if dubbing_original_volume is not None: cmd.extend(["--dubbing-original-volume", str(dubbing_original_volume)])

    # Jump Cuts (Silence Removal)
    if remove_silence:
        cmd.append("--remove-silence")
        if silence_threshold is not None: cmd.extend(["--silence-threshold", str(silence_threshold)])
        if silence_min_duration is not None: cmd.extend(["--silence-min-duration", str(silence_min_duration)])
        if silence_max_keep is not None: cmd.extend(["--silence-max-keep", str(silence_max_keep)])

    cmd.append("--skip-prompts") # Always skip prompts in WebUI to prevent freezing

    if use_custom_subs:
        # En mode distraction, placer les subs exactement sur la ligne de coupure.
        # PlayResY=640. Formule : int(620 * (1 - ratio))
        #   ratio=0.50 → 310 (split à y≈960px), ratio=0.35 → 403, ratio=0.30 → 434
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
            "mode": mode, "highlight_size": int(h_size), "remove_punctuation": remove_punc
        }
        # Uppercase is handled in main script or logic?
        # Actually subtitle_config doesn't seem to natively support "uppercase" in get_subtitle_config default, but app.py was using it.
        # I should probably add it back if I want to support it, but user said "PROHIBITED to remove existing ones".
        # I'll re-add 'uppercase': 1 if is_uppercase else 0 to the dict if the backend supports it, otherwise it's just ignored.
        # But wait, main_improved.py doesn't have 'uppercase' in get_subtitle_config.
        # I'll keep it in the dict just in case logic uses it elsewhere or if I missed it.
        # Actually, standard ASS doesn't support uppercase flag directly in Style, it needs to be text transform.
        # But I'll leave it in the dict.
        subtitle_config["uppercase"] = 1 if is_uppercase else 0

        subtitle_config_path = os.path.join(WORKING_DIR, "temp_subtitle_config.json")
        try:
            with open(subtitle_config_path, "w", encoding="utf-8") as f:
                json.dump(subtitle_config, f, indent=4)
            cmd.extend(["--subtitle-config", subtitle_config_path])
        except Exception:
            logger.debug("Failed to write subtitle config", exc_info=True)

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    if api_key:
        env["GEMINI_API_KEY"] = api_key
    try:
        current_process = subprocess.Popen(cmd, cwd=WORKING_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, universal_newlines=True, env=env)
        logs = ""
        project_folder_path = None
        if input_source == "Existing Project" and project_name:
             # If using existing project, we already know the path, but let's see if logs confirm it
             project_folder_path = os.path.join(VIRALS_DIR, project_name)

        last_update_time = time.time()

        while True:
            line = current_process.stdout.readline()
            if not line and current_process.poll() is not None:
                break

            if line:
                logs += line
                if "Project Folder:" in line:
                    parts = line.split("Project Folder:")
                    if len(parts) > 1: project_folder_path = parts[1].strip()

                # Throttle updates to avoid browser freeze (0.2s interval)
                current_time = time.time()
                if current_time - last_update_time > 0.2:
                    yield logs, gr.update(visible=True, interactive=False), gr.update(visible=True), None
                    last_update_time = current_time

        # Final yield to ensure all logs are shown
        yield logs, gr.update(visible=True, interactive=False), gr.update(visible=True), None
    except Exception as e:
        logs += f"\nError running process: {str(e)}\n"
        yield logs, gr.update(visible=True, interactive=False), gr.update(visible=True), None
    finally:
        if current_process:
            if current_process.stdout:
                try:
                    current_process.stdout.close()
                except Exception:
                    logger.debug("Failed to close process stdout", exc_info=True)
            if current_process.poll() is None:
                # If we are here, it means we finished reading or errored out, but process is still running.
                # If it was a normal break from loop, process should be done or close to done.
                # If we are stopping, current_process.terminate() might be needed outside?
                # But here we just wait.
                try:
                    current_process.wait()
                except Exception:
                    logger.debug("Failed to wait for process termination", exc_info=True)
            current_process = None

    # Wait to ensure filesystem flush
    time.sleep(1.0)

    # ── Auto Post ─────────────────────────────────────────────────────────────
    if (post_youtube or post_tiktok) and project_folder_path and os.path.exists(project_folder_path):
        try:
            import sys as _sys
            if WORKING_DIR not in _sys.path:
                _sys.path.insert(0, WORKING_DIR)
            from scripts.post_social import post_all_segments
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
    # ──────────────────────────────────────────────────────────────────────────

    html_output = ""
    if project_folder_path and os.path.exists(project_folder_path):
        html_output = library.generate_project_gallery(project_folder_path, is_full_path=True)
    else:
        html_output = f"<h3>{i18n('Error: Project folder could not be determined from logs.')}</h3>"
    yield logs, gr.update(value=i18n("Start Processing"), interactive=True), gr.update(visible=False), html_output
