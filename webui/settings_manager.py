"""Settings persistence for WebUI."""
from __future__ import annotations

import json
import logging
import os
import sys

import gradio as gr

# Allow importing project-level modules (i18n)
_WORKING_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _WORKING_DIR not in sys.path:
    sys.path.insert(0, _WORKING_DIR)

from i18n.i18n import I18nAuto

i18n = I18nAuto()
logger = logging.getLogger(__name__)

SETTINGS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "settings.json")

SETTINGS_KEYS = [
    "segments", "viral", "themes", "min_duration", "max_duration",
    "model", "ai_backend", "ai_model_name", "chunk_size",
    "workflow", "face_model", "face_mode", "face_detect_interval", "no_face_mode",
    "face_filter_thresh", "face_two_thresh", "face_conf_thresh", "face_dead_zone", "zoom_out_factor",
    "vertical_offset", "single_face_zoom", "ema_alpha", "detection_resolution",  # NEW: 1-face visual params
    "focus_active_speaker",
    "active_speaker_mar", "active_speaker_score_diff", "include_motion",
    "active_speaker_motion_threshold", "active_speaker_motion_sensitivity", "active_speaker_decay",
    "content_type", "enable_scoring", "min_score", "enable_validation",
    "use_custom_subs", "subtitle_preset",
    "font_name", "font_size", "font_color", "highlight_color",
    "outline_color", "outline_thickness", "shadow_color", "shadow_size",
    "bold", "italic", "uppercase", "vertical_pos", "alignment",
    "highlight_size", "words_per_block", "gap", "mode",
    "underline", "strikeout", "border_style", "remove_punc",
    "video_quality", "use_youtube_subs", "translate_target",
    "add_music", "music_dir", "music_file", "music_volume",
    "add_distraction", "distraction_dir", "distraction_file", "distraction_no_fetch", "distraction_ratio",
    "smart_trim", "trim_pad_start", "trim_pad_end", "scene_detection",
    "validate_clips", "hook_detection", "min_hook_score", "blur_detection", "max_blur_ratio",
    "pacing_analysis", "composite_scoring",
    "remove_fillers", "auto_thumbnail", "auto_zoom", "speed_ramp", "speed_up_factor",
    "progress_bar", "bar_color", "bar_position", "ab_variants", "num_variants",
    "layout_template", "auto_broll", "transitions", "output_resolution",
    "emoji_overlay", "color_grade", "grade_intensity",
    "engagement_prediction", "dubbing", "dubbing_language", "dubbing_original_volume",
    "remove_silence", "silence_threshold", "silence_min_duration", "silence_max_keep",
    "enable_parts", "target_part_duration",
    "post_youtube", "post_tiktok", "youtube_privacy", "post_interval_minutes", "post_first_time",
]


def save_settings(*values):
    data = dict(zip(SETTINGS_KEYS, values))
    try:
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return i18n("Settings saved.")
    except Exception as e:
        return i18n("Error saving settings: {}").format(e)


def load_settings():
    if not os.path.exists(SETTINGS_FILE):
        return [gr.skip() for _ in SETTINGS_KEYS]
    try:
        with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        results = []
        for k in SETTINGS_KEYS:
            if k in data and data[k] is not None:
                results.append(gr.update(value=data[k]))
            else:
                results.append(gr.skip())
        return results
    except Exception:
        logger.debug("Failed to load settings file", exc_info=True)
        return [gr.skip() for _ in SETTINGS_KEYS]
