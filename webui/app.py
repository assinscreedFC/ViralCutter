import os
import sys
import logging
import warnings
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

import gradio as gr
import subprocess
import json
import time
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import uvicorn

# Ensure project root is on sys.path so `webui.*` package imports work
# even when app.py is launched directly (e.g. `python webui/app.py`).
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Imports from extracted modules
# ---------------------------------------------------------------------------
from webui.presets import (
    MAIN_SCRIPT_PATH, WORKING_DIR, VIRALS_DIR, MODELS_DIR,
    FACE_PRESETS, EXPERIMENTAL_PRESETS, GEMINI_MODELS, G4F_MODELS,
    get_local_models, apply_face_preset, apply_experimental_preset,
    convert_color_to_ass,
)
from webui.settings_manager import SETTINGS_KEYS, save_settings, load_settings
from webui.process_runner import run_viral_cutter, kill_process, generate_captions_for_project

# Ensure WORKING_DIR is on sys.path for project-level imports (i18n, scripts)
if WORKING_DIR not in sys.path:
    sys.path.append(WORKING_DIR)

# Ensure webui dir is on sys.path for sibling modules (library, subtitle_handler)
_WEBUI_DIR = os.path.dirname(os.path.abspath(__file__))
if _WEBUI_DIR not in sys.path:
    sys.path.insert(0, _WEBUI_DIR)

from i18n.i18n import I18nAuto
i18n = I18nAuto()

import library # Module for Library Logic
import subtitle_handler as subs # Module for Subtitles
import subtitle_editor as editor # Module for Editor Logic

css = """
/* Global Dark Theme Overrides */
body, .gradio-container {
    background-color: #0b0b0b !important;
    color: #ffffff !important;
}

/* Force dark background for specific inputs that might be white */
input[type="password"], textarea, select {
    background-color: #1f1f1f !important;
    color: #ffffff !important;
    border: 1px solid #333 !important;
}

/* Hide Footer */
footer {visibility: hidden}

/* Container Width */
.gradio-container {
    max-width: 98% !important;
    width: 98% !important;
    margin: 0 auto !important;
}
"""

import header

with gr.Blocks(title=i18n("ViralCutter WebUI"), theme=gr.themes.Default(primary_hue="orange", neutral_hue="slate"), css=css) as demo:
    gr.Markdown(header.badges)
    gr.Markdown(header.description)
    with gr.Tabs():
        with gr.Tab(i18n("Create New")):
             with gr.Row():
                with gr.Column(scale=1):
                    input_source = gr.Radio([(i18n("YouTube URL"), "YouTube URL"), (i18n("Existing Project"), "Existing Project"), (i18n("Upload Video"), "Upload Video")], label=i18n("Input Source"), value="YouTube URL")

                    url_input = gr.Textbox(label=i18n("YouTube URL"), placeholder="https://www.youtube.com/watch?v=...", visible=True)
                    video_upload = gr.File(label=i18n("Upload Video"), file_count="single", file_types=["video"], visible=False)

                    with gr.Row():
                        video_quality_input = gr.Dropdown(choices=["best", "1080p", "720p", "480p"], label=i18n("Video Quality"), value="best")
                        translate_input = gr.Dropdown(choices=["None", "pt", "en", "es", "fr", "de", "it", "ru", "ja", "ko", "zh-CN"], label=i18n("Translate Subtitles To"), value="None")
                        use_youtube_subs_input = gr.Checkbox(label=i18n("Use YouTube Subs"), value=True, info=i18n("Download and use official subtitles if available. (Recommended, it speeds up the process)"))

                    project_selector = gr.Dropdown(choices=[], label=i18n("Select Project"), visible=False)

                    def on_source_change(source):
                        if source == "YouTube URL":
                            return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(value="Full")
                        elif source == "Upload Video":
                             return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), gr.update(value="Full")
                        else:
                            # Load projects
                            projs = library.get_existing_projects()
                            return gr.update(visible=False), gr.update(choices=projs, visible=True), gr.update(visible=False), gr.update(value="Subtitles Only")


                    with gr.Row():
                        segments_input = gr.Number(label=i18n("Segments"), value=3, precision=0)
                        viral_input = gr.Checkbox(label=i18n("Viral Mode"), value=True)
                    themes_input = gr.Textbox(label=i18n("Themes"), placeholder=i18n("funny, sad..."), visible=False)
                    viral_input.change(lambda x: gr.update(visible=not x), viral_input, themes_input)
                    with gr.Row():
                        min_dur_input = gr.Number(label=i18n("Min Duration (s)"), value=15)
                        max_dur_input = gr.Number(label=i18n("Max Duration (s)"), value=90)
                with gr.Column(scale=1):
                    with gr.Row():
                        ai_backend_input = gr.Dropdown(choices=[(i18n("Gemini"), "gemini"), (i18n("G4F"), "g4f"), ("Pléiade", "pleiade"), (i18n("Local (GGUF)"), "local"), (i18n("Manual"), "manual")], label=i18n("AI Backend"), value="gemini", scale=2)
                        api_key_input = gr.Textbox(label=i18n("Gemini API Key"), type="password", scale=3)

                    # New Dynamic Inputs
                    with gr.Row():
                        ai_model_input = gr.Dropdown(choices=GEMINI_MODELS, label=i18n("AI Model"), value=GEMINI_MODELS[1], allow_custom_value=True, visible=True, scale=5)
                        refresh_models_btn = gr.Button("🔄", size="sm", visible=False, scale=0, min_width=50) # Only local
                        chunk_size_input = gr.Number(label=i18n("Chunk Size"), value=70000, precision=0, scale=2)

                    # Update listeners with logic to hide/show API key
                    def update_ai_ui(backend):
                        show_api = (backend == "gemini")
                        show_refresh = (backend == "local")
                        # Pléiade utilise .env, pas besoin d'API key dans l'UI

                        # Definições padrão para evitar que fiquem vazios
                        new_choices = []
                        new_val = ""
                        new_chunk = 70000

                        if backend == "gemini":
                            new_choices = GEMINI_MODELS
                            new_val = GEMINI_MODELS[1]
                            new_chunk = 70000
                        elif backend == "g4f":
                            new_choices = G4F_MODELS
                            new_val = G4F_MODELS[5]
                            new_chunk = 70000
                        elif backend == "pleiade":
                            new_choices = ["athene-v2:latest", "Qwen3:30b", "deepseek-R1:70b"]
                            new_val = os.getenv("PLEIADE_CHAT_MODEL", "athene-v2:latest")
                            new_chunk = 12000
                        elif backend == "local":
                            models = get_local_models()
                            new_choices = models if models else [i18n("No models found")]
                            new_val = new_choices[0]
                            new_chunk = 30000
                        else: # Manual
                             pass

                        return (
                            gr.update(visible=show_api), # API Key Visibility (Fixes hole 1)
                            gr.update(choices=new_choices, value=new_val, visible=(backend != "manual")), # Model Dropdown
                            gr.update(visible=show_refresh), # Refresh Button
                            gr.update(value=new_chunk) # Chunk Size
                        )

                    def refresh_local_models():
                        models = get_local_models()
                        val = models[0] if models else i18n("No models found")
                        return gr.update(choices=models, value=val)

                    refresh_models_btn.click(refresh_local_models, outputs=ai_model_input)
                    ai_backend_input.change(update_ai_ui, inputs=ai_backend_input, outputs=[api_key_input, ai_model_input, refresh_models_btn, chunk_size_input])

                    with gr.Accordion(i18n("Advanced AI Settings"), open=False):
                        content_type_input = gr.Dropdown(
                            choices=[
                                ("Anime", "anime"), ("Comedy", "comedy"), ("Commentary", "commentary"),
                                ("Cooking", "cooking"), ("Education", "education"),
                                ("Gaming", "gaming"), ("Manga", "manga"), ("Motivation", "motivation"),
                                ("Music", "music"), ("News", "news"),
                                ("Podcast", "podcast"), ("Sport", "sport"),
                                ("Talk Show", "talkshow"), ("Vlog", "vlog"),
                            ],
                            label=i18n("Content Type"),
                            value=[],
                            multiselect=True,
                            info=i18n("Leave empty for auto-detection. Select one or more types (e.g. Gaming + Comedy)")
                        )
                        with gr.Row():
                            enable_scoring_input = gr.Checkbox(
                                label=i18n("Enable Scoring Pass"),
                                value=True,
                                info=i18n("Score segments (0-100) and filter weak ones")
                            )
                            with gr.Row(visible=True) as min_score_row:
                                min_score_input = gr.Slider(
                                    label=i18n("Min Score"),
                                    minimum=0, maximum=100, value=70, step=5,
                                    info=i18n("Discard segments below this score")
                                )
                        enable_scoring_input.change(
                            lambda x: (gr.update(visible=x), gr.update(value=70)),
                            inputs=enable_scoring_input,
                            outputs=[min_score_row, min_score_input]
                        )
                        enable_validation_input = gr.Checkbox(
                            label=i18n("Enable Validation Pass"),
                            value=True,
                            info=i18n("LLM reviews each segment: hook strength, standalone test, narrative arc, viral value. Rejects weak segments.")
                        )
                        with gr.Row():
                            enable_parts_input = gr.Checkbox(
                                label=i18n("Enable Parts Mode"),
                                value=False,
                                info=i18n("Allow AI to select long passages. They are auto-split into a multi-part series (Part 1, Part 2...).")
                            )
                            with gr.Row(visible=False) as parts_duration_row:
                                target_part_duration_input = gr.Slider(
                                    label=i18n("Target part duration (s)"),
                                    minimum=20, maximum=90, value=55, step=5,
                                    info=i18n("Target duration for each part after splitting")
                                )
                        enable_parts_input.change(
                            lambda x: gr.update(visible=x),
                            inputs=enable_parts_input,
                            outputs=parts_duration_row
                        )

                    model_input = gr.Dropdown(["tiny", "small", "medium", "large", "large-v1", "large-v2", "large-v3", "turbo", "large-v3-turbo", "distil-large-v2", "distil-medium.en", "distil-small.en", "distil-large-v3"], label=i18n("Whisper Model"), value="large-v3-turbo")
                    with gr.Row():
                        workflow_input = gr.Dropdown(choices=[(i18n("Full"), "Full"), (i18n("Cut Only"), "Cut Only"), (i18n("Subtitles Only"), "Subtitles Only")], label=i18n("Workflow"), value="Full")
                        face_model_input = gr.Dropdown(["insightface", "mediapipe"], label=i18n("Face Model"), value="insightface")
                    with gr.Row():
                        face_mode_input = gr.Dropdown(choices=[(i18n("Auto"), "auto"), ("1", "1"), ("2", "2")], label=i18n("Face Mode"), value="auto")
                        face_detect_interval_input = gr.Textbox(label=i18n("Face Det. Interval"), value="0.17,1.0")
                        no_face_mode_input = gr.Dropdown(choices=[(i18n("Padding (9:16)"), "padding"), (i18n("Zoom (Center)"), "zoom"), (i18n("Saliency (Anime/Manga)"), "saliency"), (i18n("Motion (Stickman/Action)"), "motion")], label=i18n("No Face Fallback"), value="zoom", info=i18n("'Saliency' detects visually interesting regions (static panels). 'Motion' follows movement (stickman animations)."))


                    # Update listeners now that all components are defined
                    input_source.change(on_source_change, inputs=input_source, outputs=[url_input, project_selector, video_upload, workflow_input])

             with gr.Accordion(i18n("Advanced Face Settings"), open=False):
                 face_preset_input = gr.Dropdown(choices=[(i18n(k), k) for k in FACE_PRESETS.keys()], label=i18n("Configuration Presets"), value="Default (Balanced)", interactive=True)
                 with gr.Row():
                      face_filter_thresh_input = gr.Slider(label=i18n("Ignore Small Faces (0.0 - 1.0)"), minimum=0.0, maximum=1.0, value=0.35, step=0.05, info=i18n("Relative size to ignore background."))
                      face_two_thresh_input = gr.Slider(label=i18n("Threshold for 2 Faces (0.0 - 1.0)"), minimum=0.0, maximum=1.0, value=0.60, step=0.05, info=i18n("Size of 2nd face to activate split mode."))
                      face_conf_thresh_input = gr.Slider(label=i18n("Minimum Confidence (0.0 - 1.0)"), minimum=0.0, maximum=1.0, value=0.40, step=0.05, info=i18n("Ignore detections with low confidence."))
                      face_dead_zone_input = gr.Slider(label=i18n("Dead Zone (Stabilization)"), minimum=0, maximum=200, value=150, step=5, info=i18n("Movement pixels to ignore."))
                 zoom_out_factor_input = gr.Slider(label=i18n("Zoom Out Factor (2-face mode)"), minimum=1.0, maximum=4.0, value=2.2, step=0.1, info=i18n("Higher = more zoom out when 2 faces detected. Falls back to center-crop if crop quality < 45%."))

                 face_preset_input.change(apply_face_preset, inputs=face_preset_input, outputs=[face_filter_thresh_input, face_two_thresh_input, face_conf_thresh_input, face_dead_zone_input])

                 with gr.Accordion(i18n("Experimental: Active Speaker & Motion"), open=False):
                        experimental_preset_input = gr.Dropdown(choices=[(i18n(k), k) for k in EXPERIMENTAL_PRESETS.keys()], label=i18n("Configuration Presets"), value="Default (Off)", interactive=True)
                        focus_active_speaker_input = gr.Checkbox(label=i18n("Experimental: Focus on Speaker"), value=False, info=i18n("Tries to focus only on the speaking person instead of split screen."))
                        with gr.Row():
                            active_speaker_mar_input = gr.Slider(label=i18n("MAR Threshold (Mouth Open)"), minimum=0.01, maximum=0.20, value=0.03, step=0.005, info=i18n("Mouth open sensitivity."))
                            active_speaker_score_diff_input = gr.Slider(label=i18n("Score Difference"), minimum=0.5, maximum=10.0, value=1.5, step=0.5, info=i18n("Minimum difference to focus on 1 face."))

                        with gr.Row():
                            include_motion_input = gr.Checkbox(label=i18n("Consider Motion"), value=False, info=i18n("Increases score with motion (gestures)."))

                        with gr.Row():
                            active_speaker_motion_threshold_input = gr.Slider(label=i18n("Motion Dead Zone"), minimum=0.0, maximum=20.0, value=3.0, step=0.5, info=i18n("Pixels ignored."))
                            active_speaker_motion_sensitivity_input = gr.Slider(label=i18n("Motion Sensitivity"), minimum=0.01, maximum=0.5, value=0.05, step=0.01, info=i18n("Points per pixel."))
                            active_speaker_decay_input = gr.Slider(label=i18n("Switch Speed"), minimum=0.5, maximum=5.0, value=2.0, step=0.5, info=i18n("Speed to lose focus."))

                        experimental_preset_input.change(apply_experimental_preset, inputs=experimental_preset_input, outputs=[focus_active_speaker_input, active_speaker_mar_input, active_speaker_score_diff_input, include_motion_input, active_speaker_motion_threshold_input, active_speaker_motion_sensitivity_input, active_speaker_decay_input])
             with gr.Accordion(i18n("Subtitle Settings (alpha)"), open=False):
                preset_input = gr.Dropdown(choices=[(i18n("Manual"), "Manual")] + [(i18n(k), k) for k in subs.SUBTITLE_PRESETS.keys()], label=i18n("Quick Presets"), value="Hormozi (Classic)")
                use_custom_subs = gr.Checkbox(label=i18n("Enable Subtitle Customization (Includes Preset)"), value=True)

                # Previews (Always Visible)
                preview_html = gr.HTML(value=f"<div style='text-align:center; padding:10px; color:#666;'>{i18n('Select options or preset to preview')}</div>")

                with gr.Row():
                    preview_vid_btn = gr.Button(i18n("🎬 Render Animated Preview (Slow)"), size="sm")
                preview_vid = gr.Video(label=i18n("Animated Preview"), height=300, autoplay=True, interactive=False)

                with gr.Accordion(i18n("Advanced Settings"), open=False):
                    gr.Markdown(f"### {i18n('Appearance')}")
                    with gr.Row():
                        font_name_input = gr.Textbox(label=i18n("Font Name"), value="Montserrat-Regular")
                        font_size_input = gr.Slider(label=i18n("Font Size (Base)"), minimum=1, maximum=80, value=12)
                        highlight_size_input = gr.Slider(label=i18n("Highlight Size"), minimum=1, maximum=80, value=14)

                    with gr.Row():
                        font_color_input = gr.ColorPicker(label=i18n("Base Color"), value="#FFFFFF")
                        highlight_color_input = gr.ColorPicker(label=i18n("Highlight Color"), value="#00FF00")
                        outline_color_input = gr.ColorPicker(label=i18n("Outline Color"), value="#000000")
                        shadow_color_input = gr.ColorPicker(label=i18n("Shadow Color"), value="#000000")

                    gr.Markdown(f"### {i18n('Styling & Effects')}")
                    with gr.Row():
                        outline_thickness_input = gr.Slider(label=i18n("Outline Thickness"), minimum=0, maximum=10, value=1.5)
                        shadow_size_input = gr.Slider(label=i18n("Shadow Size"), minimum=0, maximum=10, value=2)
                        border_style_input = gr.Dropdown(choices=[(i18n("Outline"), 1), (i18n("Opaque Box"), 3)], label=i18n("Border Style"), value=1)

                    with gr.Row():
                        bold_input = gr.Checkbox(label=i18n("Bold"))
                        italic_input = gr.Checkbox(label=i18n("Italic"))
                        uppercase_input = gr.Checkbox(label=i18n("Uppercase"))
                        remove_punc_input = gr.Checkbox(label=i18n("Remove Punctuation"), value=True)
                        underline_input = gr.Checkbox(label=i18n("Underline"))
                        strikeout_input = gr.Checkbox(label=i18n("Strikeout"))

                    gr.Markdown(f"### {i18n('Positioning & Layout')}")
                    with gr.Row():
                        vertical_pos_input = gr.Slider(label=i18n("V-Pos (Margin V)"), minimum=0, maximum=500, value=210)
                        alignment_input = gr.Dropdown(choices=[(i18n("Left"), 1), (i18n("Center"), 2), (i18n("Right"), 3)], label=i18n("Alignment"), value=2)
                        gap_limit_input = gr.Slider(label=i18n("Gap Limit"), minimum=0.0, maximum=5.0, value=0.5, step=0.1)
                        mode_input = gr.Dropdown(choices=[(i18n("Highlight"), "highlight"), (i18n("Word by Word"), "word_by_word"), (i18n("No Highlight"), "no_highlight")], label=i18n("Mode"), value="highlight")
                        words_per_block_input = gr.Slider(label=i18n("Words per Block"), minimum=1, maximum=20, value=3, step=1)

                manual_inputs = [
                    font_name_input, font_size_input, font_color_input, highlight_color_input,
                    outline_color_input, outline_thickness_input, shadow_color_input, shadow_size_input,
                    bold_input, italic_input, uppercase_input,
                    highlight_size_input, words_per_block_input, gap_limit_input, mode_input,
                    underline_input, strikeout_input, border_style_input,
                    vertical_pos_input, alignment_input,
                    remove_punc_input
                ]

                # Update manual inputs when preset changes
                preset_input.change(subs.apply_preset, inputs=[preset_input], outputs=manual_inputs)

                # Auto-update PREVIEW HTML on any change; also switch preset to Manual when user edits directly
                for inp in manual_inputs:
                    inp.change(subs.generate_preview_html, inputs=manual_inputs, outputs=preview_html)
                    inp.change(lambda: "Manual", inputs=[], outputs=[preset_input])

                # Render video button
                preview_vid_btn.click(
                    subs.render_preview_video,
                    inputs=manual_inputs,
                    outputs=preview_vid
                )

                # Initial load
                demo.load(subs.generate_preview_html, inputs=manual_inputs, outputs=preview_html)
                demo.load(subs.apply_preset, inputs=[preset_input], outputs=manual_inputs) # Apply default preset on load

             with gr.Accordion(i18n("Music Settings"), open=False):
                 add_music_input = gr.Checkbox(label=i18n("Add Background Music"), value=False, info=i18n("Mix royalty-free music from the music/ folder into the clips"))
                 with gr.Row(visible=False) as music_options_row:
                     music_dir_input = gr.Textbox(label=i18n("Music Folder"), placeholder="music/", value="", info=i18n("Folder with .mp3/.wav files (uses music/ by default)"))
                     music_file_input = gr.Textbox(label=i18n("Specific Music File"), placeholder="path/to/song.mp3", value="", info=i18n("Force a specific file instead of auto-selection"))
                     music_volume_input = gr.Slider(label=i18n("Music Volume"), minimum=0.01, maximum=0.5, value=0.12, step=0.01, info=i18n("Volume ratio of background music (default 12%)"))
                 add_music_input.change(lambda x: gr.update(visible=x), inputs=add_music_input, outputs=music_options_row)

             with gr.Accordion(i18n("Split-Screen (TikTok Format)"), open=False):
                 add_distraction_input = gr.Checkbox(
                     label=i18n("Add Distraction Video (bottom half)"), value=False,
                     info=i18n("Stack a satisfying/gameplay video below the main clip (1080×1920 output). Subtitle position is automatically adjusted to appear at the split line.")
                 )
                 with gr.Row(visible=False) as distraction_options_row:
                     with gr.Column():
                         distraction_ratio_input = gr.Slider(
                             label=i18n("Distraction Height (%)"),
                             minimum=0.20, maximum=0.50, value=0.35, step=0.05,
                             info=i18n("Share of screen height for the bottom distraction video (default 35%)")
                         )
                         distraction_no_fetch_input = gr.Checkbox(
                             label=i18n("Disable auto-fetch (use cache only)"), value=False,
                             info=i18n("Never download distraction videos, use whatever is already in the cache")
                         )
                     with gr.Column():
                         distraction_dir_input = gr.Textbox(
                             label=i18n("Distraction Folder"), placeholder="distraction/", value="",
                             info=i18n("Folder with distraction videos (auto-downloaded to distraction/ by default)")
                         )
                         distraction_file_input = gr.Textbox(
                             label=i18n("Specific Distraction File"), placeholder="path/to/video.mp4", value="",
                             info=i18n("Force a specific video instead of random selection")
                         )
                 add_distraction_input.change(lambda x: gr.update(visible=x), inputs=add_distraction_input, outputs=distraction_options_row)

             with gr.Accordion(i18n("Video Quality"), open=False):
                 smart_trim_input = gr.Checkbox(
                     label=i18n("Smart Trim (sentence boundaries)"), value=False,
                     info=i18n("Snap cuts to sentence boundaries using word timestamps + padding")
                 )
                 with gr.Row(visible=False) as trim_options_row:
                     trim_pad_start_input = gr.Slider(
                         label=i18n("Start Padding (s)"), minimum=0.0, maximum=1.0, value=0.3, step=0.05
                     )
                     trim_pad_end_input = gr.Slider(
                         label=i18n("End Padding (s)"), minimum=0.0, maximum=1.5, value=0.5, step=0.05
                     )
                 smart_trim_input.change(lambda x: gr.update(visible=x), inputs=smart_trim_input, outputs=trim_options_row)

                 scene_detection_input = gr.Checkbox(
                     label=i18n("Scene Detection"), value=False,
                     info=i18n("Detect scene changes to avoid cutting mid-transition")
                 )
                 validate_clips_input = gr.Checkbox(
                     label=i18n("Validate Clip Boundaries"), value=False,
                     info=i18n("Check clips for silence at start/end and compute speech ratio")
                 )
                 hook_detection_input = gr.Checkbox(
                     label=i18n("Hook Detection"), value=False,
                     info=i18n("Score first 3 seconds of each clip for stop-scroll potential")
                 )
                 with gr.Row(visible=False) as hook_options_row:
                     min_hook_score_input = gr.Slider(
                         label=i18n("Min Hook Score"), minimum=0, maximum=100, value=40, step=5
                     )
                 hook_detection_input.change(lambda x: gr.update(visible=x), inputs=hook_detection_input, outputs=hook_options_row)

                 blur_detection_input = gr.Checkbox(
                     label=i18n("Blur Detection"), value=False,
                     info=i18n("Detect blurry frames and flag low-quality clips")
                 )
                 with gr.Row(visible=False) as blur_options_row:
                     max_blur_ratio_input = gr.Slider(
                         label=i18n("Max Blur Ratio"), minimum=0.0, maximum=1.0, value=0.3, step=0.05,
                         info=i18n("Maximum ratio of blurry frames allowed (0-1)")
                     )
                 blur_detection_input.change(lambda x: gr.update(visible=x), inputs=blur_detection_input, outputs=blur_options_row)

                 pacing_analysis_input = gr.Checkbox(
                     label=i18n("Pacing & Energy Analysis"), value=False,
                     info=i18n("Analyze speech pace and audio energy to score clip dynamism")
                 )
                 composite_scoring_input = gr.Checkbox(
                     label=i18n("Composite Quality Score"), value=False,
                     info=i18n("Aggregate all quality signals into a single 0-100 score")
                 )

             with gr.Accordion(i18n("Content Enhancement"), open=False):
                 remove_fillers_input = gr.Checkbox(
                     label=i18n("Remove Filler Words"), value=False,
                     info=i18n("Detect and remove filler words (um, uh, like, euh, genre...)")
                 )
                 auto_thumbnail_input = gr.Checkbox(
                     label=i18n("Auto Thumbnail"), value=False,
                     info=i18n("Generate best-frame thumbnails for each clip")
                 )
                 auto_zoom_input = gr.Checkbox(
                     label=i18n("Auto Zoom (Dynamic)"), value=False,
                     info=i18n("Apply dynamic zoom effects on punchlines and key moments")
                 )
                 speed_ramp_input = gr.Checkbox(
                     label=i18n("Speed Ramp"), value=False,
                     info=i18n("Speed up dead moments, slow down highlights")
                 )
                 with gr.Row(visible=False) as speed_options_row:
                     speed_up_factor_input = gr.Slider(
                         label=i18n("Speed Up Factor"), minimum=1.1, maximum=3.0, value=1.5, step=0.1,
                         info=i18n("How much to speed up dead moments (1.5 = 50% faster)")
                     )
                 speed_ramp_input.change(lambda x: gr.update(visible=x), inputs=speed_ramp_input, outputs=speed_options_row)

             with gr.Accordion(i18n("Overlays & Effects"), open=False):
                 progress_bar_input = gr.Checkbox(
                     label=i18n("Progress Bar"), value=False,
                     info=i18n("Add animated progress bar to keep viewers watching")
                 )
                 with gr.Row(visible=False) as bar_options_row:
                     bar_color_input = gr.Dropdown(
                         choices=["white", "red", "yellow", "green", "blue", "purple"],
                         label=i18n("Bar Color"), value="white"
                     )
                     bar_position_input = gr.Dropdown(
                         choices=["top", "bottom"],
                         label=i18n("Bar Position"), value="top"
                     )
                 progress_bar_input.change(lambda x: gr.update(visible=x), inputs=progress_bar_input, outputs=bar_options_row)

                 emoji_overlay_input = gr.Checkbox(
                     label=i18n("Emoji Overlay"), value=False,
                     info=i18n("Add emoji reactions at key moments (requires LLM emoji_cues)")
                 )
                 color_grade_input = gr.Dropdown(
                     choices=[None, "cinematic", "vintage", "warm", "cool", "high_contrast"],
                     label=i18n("Color Grading"), value=None,
                     info=i18n("Apply color grading LUT preset")
                 )
                 with gr.Row(visible=False) as grade_options_row:
                     grade_intensity_input = gr.Slider(
                         label=i18n("Grading Intensity"), minimum=0.0, maximum=1.0, value=0.7, step=0.05
                     )
                 color_grade_input.change(lambda x: gr.update(visible=x is not None), inputs=color_grade_input, outputs=grade_options_row)

                 transitions_input = gr.Dropdown(
                     choices=[None, "fade", "wipeleft", "wiperight", "slideup", "slidedown"],
                     label=i18n("Transitions"), value=None,
                     info=i18n("Transition effect between multi-part clips")
                 )
                 output_resolution_input = gr.Dropdown(
                     choices=["720p", "1080p", "4k"],
                     label=i18n("Output Resolution"), value="1080p"
                 )
                 ab_variants_input = gr.Checkbox(
                     label=i18n("A/B Caption Variants"), value=False,
                     info=i18n("Generate multiple caption variants for A/B testing")
                 )
                 with gr.Row(visible=False) as ab_options_row:
                     num_variants_input = gr.Slider(
                         label=i18n("Number of Variants"), minimum=2, maximum=5, value=3, step=1
                     )
                 ab_variants_input.change(lambda x: gr.update(visible=x), inputs=ab_variants_input, outputs=ab_options_row)

                 layout_template_input = gr.Dropdown(
                     choices=[None, "pip", "lower-third"],
                     label=i18n("Layout Template"), value=None,
                     info=i18n("Apply visual layout (PiP, lower third)")
                 )
                 auto_broll_input = gr.Checkbox(
                     label=i18n("Auto B-roll (Pexels)"), value=False,
                     info=i18n("Auto-insert stock footage at static moments (requires Pexels API key)")
                 )

             with gr.Accordion(i18n("Advanced AI"), open=False):
                 engagement_prediction_input = gr.Checkbox(
                     label=i18n("Engagement Prediction"), value=False,
                     info=i18n("Predict engagement score using ML model (requires trained model)")
                 )
                 dubbing_input = gr.Checkbox(
                     label=i18n("AI Dubbing"), value=False,
                     info=i18n("Translate and voice-over clips in target language (edge-tts)")
                 )
                 with gr.Row(visible=False) as dubbing_options_row:
                     dubbing_language_input = gr.Dropdown(
                         choices=["en", "fr", "es", "de", "pt", "tr", "ja", "zh"],
                         label=i18n("Dubbing Language"), value="en"
                     )
                     dubbing_original_volume_input = gr.Slider(
                         label=i18n("Original Audio Volume"), minimum=0.0, maximum=1.0, value=0.2, step=0.05,
                         info=i18n("Volume of original audio during dubbing (0 = muted)")
                     )
                 dubbing_input.change(lambda x: gr.update(visible=x), inputs=dubbing_input, outputs=dubbing_options_row)

             with gr.Accordion(i18n("Jump Cuts (Silence Removal)"), open=False):
                 remove_silence_input = gr.Checkbox(
                     label=i18n("Remove Silences"), value=False,
                     info=i18n("Automatically detect and remove silent portions for tighter pacing")
                 )
                 with gr.Row(visible=False) as silence_options_row:
                     silence_threshold_input = gr.Slider(
                         label=i18n("Silence Threshold (dB)"),
                         minimum=-60, maximum=-10, value=-30, step=1,
                         info=i18n("Audio level below which is considered silence (default -30 dB)")
                     )
                     silence_min_duration_input = gr.Slider(
                         label=i18n("Min Silence to Cut (s)"),
                         minimum=0.1, maximum=5.0, value=0.5, step=0.1,
                         info=i18n("Only remove silences longer than this (default 0.5s)")
                     )
                     silence_max_keep_input = gr.Slider(
                         label=i18n("Max Silence to Keep (s)"),
                         minimum=0.0, maximum=2.0, value=0.3, step=0.05,
                         info=i18n("Keep this much silence for natural pacing (0 = remove all)")
                     )
                 remove_silence_input.change(lambda x: gr.update(visible=x), inputs=remove_silence_input, outputs=silence_options_row)

             with gr.Accordion(i18n("Auto Post"), open=False):
                 with gr.Row():
                     post_youtube_input = gr.Checkbox(
                         label=i18n("Post to YouTube Shorts"), value=False,
                         info=i18n("Requires YOUTUBE_CLIENT_ID and YOUTUBE_CLIENT_SECRET in .env")
                     )
                     post_tiktok_input = gr.Checkbox(
                         label=i18n("Post to TikTok"), value=False,
                         info=i18n("Requires TIKTOK_COOKIES_PATH in .env")
                     )
                 youtube_privacy_input = gr.Dropdown(
                     choices=["public", "private", "unlisted"],
                     label=i18n("YouTube Privacy"), value="public", visible=False,
                 )
                 post_first_time_input = gr.Textbox(
                     label=i18n("First post time (optional)"),
                     placeholder="HH:MM  or  YYYY-MM-DD HH:MM",
                     value="",
                     info=i18n("Leave empty to post immediately after pipeline. Use HH:MM for today or YYYY-MM-DD HH:MM for a specific date.")
                 )
                 post_interval_input = gr.Number(
                     label=i18n("Interval between posts (minutes)"),
                     value=30, minimum=0, maximum=1440,
                     info=i18n("Wait this many minutes between each video posted")
                 )
                 post_youtube_input.change(
                     lambda x: gr.update(visible=x),
                     inputs=post_youtube_input, outputs=youtube_privacy_input
                 )

             with gr.Row():
                 start_btn = gr.Button(i18n("Start Processing"), variant="primary")
                 stop_btn = gr.Button(i18n("Stop"), variant="stop", visible=False)
                 save_settings_btn = gr.Button(i18n("Save Settings"), variant="secondary", size="sm", scale=0, min_width=130)
             save_settings_status = gr.Markdown("")
             stop_btn.click(kill_process, outputs=[])
             logs_output = gr.Textbox(label=i18n("Logs"), lines=10, autoscroll=True, elem_id="logs_output")

             # Force scroll to bottom via JS
             logs_output.change(fn=None, inputs=[], outputs=[], js="""
                function() {
                    var ta = document.querySelector('#logs_output textarea');
                    if(ta) {
                        // Setup scroll listener once to track user intent
                        if (!ta._scrollerSetup) {
                            ta._isSticky = true; // Default to sticky
                            ta.addEventListener('scroll', function() {
                                var diff = ta.scrollHeight - ta.scrollTop - ta.clientHeight;
                                // If near bottom (<50px), enable sticky. Else disable.
                                if (diff <= 50) {
                                     ta._isSticky = true;
                                } else {
                                     ta._isSticky = false;
                                }
                            });
                            ta._scrollerSetup = true;
                        }

                        // Apply scroll only if sticky
                        if(ta._isSticky === undefined || ta._isSticky === true) {
                            ta.scrollTop = ta.scrollHeight;
                        }
                    }
                }
             """)
             results_html = gr.HTML(label=i18n("Results"))

             # MUST pass all all new inputs to the run function
             start_btn.click(run_viral_cutter, inputs=[
                 input_source, project_selector, url_input, video_upload, segments_input, viral_input, themes_input, min_dur_input, max_dur_input,
                 model_input, ai_backend_input, api_key_input, ai_model_input, chunk_size_input,
                 workflow_input, face_model_input, face_mode_input, face_detect_interval_input, no_face_mode_input,
                 face_filter_thresh_input, face_two_thresh_input, face_conf_thresh_input, face_dead_zone_input, zoom_out_factor_input, focus_active_speaker_input,
                 active_speaker_mar_input, active_speaker_score_diff_input, include_motion_input, active_speaker_motion_threshold_input, active_speaker_motion_sensitivity_input, active_speaker_decay_input,
                 content_type_input, enable_scoring_input, min_score_input, enable_validation_input,
                 use_custom_subs,
                 # Expanded Manual Inputs mapping
                 font_name_input, font_size_input, font_color_input, highlight_color_input,
                 outline_color_input, outline_thickness_input, shadow_color_input, shadow_size_input,
                 bold_input, italic_input, uppercase_input, vertical_pos_input, alignment_input,
                 # New Inputs
                 highlight_size_input, words_per_block_input, gap_limit_input, mode_input,
                 underline_input, strikeout_input, border_style_input, remove_punc_input,
                 video_quality_input, use_youtube_subs_input, translate_input,
                 # Music
                 add_music_input, music_dir_input, music_file_input, music_volume_input,
                 # Split-screen distraction
                 add_distraction_input, distraction_dir_input, distraction_file_input, distraction_no_fetch_input, distraction_ratio_input,
                 # Video Quality
                 smart_trim_input, trim_pad_start_input, trim_pad_end_input, scene_detection_input,
                 validate_clips_input, hook_detection_input, min_hook_score_input, blur_detection_input, max_blur_ratio_input,
                 pacing_analysis_input, composite_scoring_input,
                 # Content Enhancement (Phase 3)
                 remove_fillers_input, auto_thumbnail_input, auto_zoom_input, speed_ramp_input, speed_up_factor_input,
                 # Overlays & Effects (Phase 4)
                 progress_bar_input, bar_color_input, bar_position_input, ab_variants_input, num_variants_input,
                 layout_template_input, auto_broll_input, transitions_input, output_resolution_input,
                 emoji_overlay_input, color_grade_input, grade_intensity_input,
                 # Advanced AI (Phase 5)
                 engagement_prediction_input, dubbing_input, dubbing_language_input, dubbing_original_volume_input,
                 # Jump Cuts (Silence Removal)
                 remove_silence_input, silence_threshold_input, silence_min_duration_input, silence_max_keep_input,
                 # Parts Mode
                 enable_parts_input, target_part_duration_input,
                 # Auto Post
                 post_youtube_input, post_tiktok_input, youtube_privacy_input, post_interval_input, post_first_time_input,
             ], outputs=[logs_output, start_btn, stop_btn, results_html])

             # Saveable inputs (same order as SETTINGS_KEYS)
             _saveable_inputs = [
                 segments_input, viral_input, themes_input, min_dur_input, max_dur_input,
                 model_input, ai_backend_input, api_key_input, ai_model_input, chunk_size_input,
                 workflow_input, face_model_input, face_mode_input, face_detect_interval_input, no_face_mode_input,
                 face_filter_thresh_input, face_two_thresh_input, face_conf_thresh_input, face_dead_zone_input, zoom_out_factor_input,
                 focus_active_speaker_input,
                 active_speaker_mar_input, active_speaker_score_diff_input, include_motion_input,
                 active_speaker_motion_threshold_input, active_speaker_motion_sensitivity_input, active_speaker_decay_input,
                 content_type_input, enable_scoring_input, min_score_input, enable_validation_input,
                 use_custom_subs, preset_input,
                 font_name_input, font_size_input, font_color_input, highlight_color_input,
                 outline_color_input, outline_thickness_input, shadow_color_input, shadow_size_input,
                 bold_input, italic_input, uppercase_input, vertical_pos_input, alignment_input,
                 highlight_size_input, words_per_block_input, gap_limit_input, mode_input,
                 underline_input, strikeout_input, border_style_input, remove_punc_input,
                 video_quality_input, use_youtube_subs_input, translate_input,
                 add_music_input, music_dir_input, music_file_input, music_volume_input,
                 add_distraction_input, distraction_dir_input, distraction_file_input, distraction_no_fetch_input, distraction_ratio_input,
                 smart_trim_input, trim_pad_start_input, trim_pad_end_input, scene_detection_input,
                 validate_clips_input, hook_detection_input, min_hook_score_input, blur_detection_input, max_blur_ratio_input,
                 pacing_analysis_input, composite_scoring_input,
                 # Content Enhancement (Phase 3)
                 remove_fillers_input, auto_thumbnail_input, auto_zoom_input, speed_ramp_input, speed_up_factor_input,
                 # Overlays & Effects (Phase 4)
                 progress_bar_input, bar_color_input, bar_position_input, ab_variants_input, num_variants_input,
                 layout_template_input, auto_broll_input, transitions_input, output_resolution_input,
                 emoji_overlay_input, color_grade_input, grade_intensity_input,
                 engagement_prediction_input, dubbing_input, dubbing_language_input, dubbing_original_volume_input,
                 remove_silence_input, silence_threshold_input, silence_min_duration_input, silence_max_keep_input,
                 # Parts Mode
                 enable_parts_input, target_part_duration_input,
                 # Auto Post
                 post_youtube_input, post_tiktok_input, youtube_privacy_input, post_interval_input, post_first_time_input,
             ]
             save_settings_btn.click(save_settings, inputs=_saveable_inputs, outputs=[save_settings_status])
             demo.load(load_settings, inputs=[], outputs=_saveable_inputs)


        with gr.Tab(i18n("Subtitle Editor")):
            gr.Markdown(f"### {i18n('Edit Subtitles (Smart Mode)')}")

            with gr.Group():
                editor_project_dropdown = gr.Dropdown(choices=library.get_existing_projects(), label=i18n("Select Project"), value=None)
                editor_refresh_btn = gr.Button(i18n("Refresh"), size="sm")

            with gr.Group():
                editor_file_dropdown = gr.Dropdown(choices=[], label=i18n("Select Subtitle File"), interactive=True)
                editor_load_btn = gr.Button(i18n("Load Subtitles"), variant="secondary")

            # Hidden state to store full path of currently loaded JSON
            current_json_path = gr.State()

            # The Dataframe Editor
            # Headers: Start, End, Text
            subtitle_dataframe = gr.Dataframe(
                headers=["Start", "End", "Text"],
                datatype=["str", "str", "str"],
                col_count=(3, "fixed"),
                interactive=True,
                label=i18n("Subtitle Segments"),
                wrap=True
            )

            with gr.Row():
                editor_save_btn = gr.Button(i18n("💾 Save Changes"), variant="primary")
                editor_render_single_btn = gr.Button(i18n("⚡ Render This Segment (Very-Fast)"), variant="secondary")
                editor_render_all_btn = gr.Button(i18n("🎬 Render All (Fast)"), variant="stop")

            editor_status = gr.Textbox(label=i18n("Status"), interactive=False)

            # --- Callbacks for Editor ---
            editor_refresh_btn.click(library.refresh_projects, outputs=editor_project_dropdown)

            def update_file_list(proj_name):
                if not proj_name: return gr.update(choices=[])
                proj_path = os.path.join(VIRALS_DIR, proj_name)
                files = editor.list_editable_files(proj_path)
                return gr.update(choices=files, value=files[0] if files else None)

            editor_project_dropdown.change(update_file_list, inputs=editor_project_dropdown, outputs=editor_file_dropdown)

            def load_subs(proj_name, file_name):
                if not proj_name or not file_name:
                    return [], None, i18n("Please select project and file.")

                full_path = os.path.join(VIRALS_DIR, proj_name, 'subs', file_name)
                data = editor.load_transcription_for_editor(full_path)
                return data, full_path, i18n("Loaded {} segments.").format(len(data))

            editor_load_btn.click(load_subs, inputs=[editor_project_dropdown, editor_file_dropdown], outputs=[subtitle_dataframe, current_json_path, editor_status])

            def save_subs(json_path, df):
                if not json_path: return i18n("No file loaded.")
                data_list = df.values.tolist() if hasattr(df, 'values') else df
                msg = editor.save_editor_changes(json_path, data_list)
                return msg

            editor_save_btn.click(save_subs, inputs=[current_json_path, subtitle_dataframe], outputs=editor_status)

            def render_single(json_path, use_custom, font_name, font_size, font_color, highlight_color,
                              outline_color, outline_thickness, shadow_color, shadow_size,
                              is_bold, is_italic, is_uppercase,
                              h_size, w_block, gap, mode, under, strike, border_s,
                              vertical_pos, alignment, remove_punc):

                if not json_path: return i18n("No file loaded.")

                subtitle_config_path = os.path.join(WORKING_DIR, "temp_subtitle_config.json")

                # Save config if custom subs enabled
                if use_custom:
                    subtitle_config = {
                        "font": font_name, "base_size": int(font_size),
                        "base_color": convert_color_to_ass(font_color),
                        "highlight_color": convert_color_to_ass(highlight_color),
                        "outline_color": convert_color_to_ass(outline_color),
                        "outline_thickness": outline_thickness,
                        "shadow_color": convert_color_to_ass(shadow_color),
                        "shadow_size": shadow_size, "vertical_position": vertical_pos,
                        "alignment": alignment, "bold": 1 if is_bold else 0,
                        "italic": 1 if is_italic else 0,
                        "underline": 1 if under else 0, "strikeout": 1 if strike else 0,
                        "border_style": border_s, "words_per_block": int(w_block),
                        "gap_limit": gap, "mode": mode, "highlight_size": int(h_size),
                        "uppercase": 1 if is_uppercase else 0,
                        "remove_punctuation": remove_punc
                    }
                    try:
                        with open(subtitle_config_path, "w", encoding="utf-8") as f:
                            json.dump(subtitle_config, f, indent=4)
                    except Exception:
                        logger.debug("Failed to write subtitle config for render", exc_info=True)
                else:
                    # Remove temp config if it exists to ensure defaults are used
                    try:
                        if os.path.exists(subtitle_config_path):
                            os.remove(subtitle_config_path)
                    except Exception:
                        logger.debug("Failed to remove temp subtitle config", exc_info=True)

                # We expect user to SAVE first, but we could auto-save.
                # For now assume saved.
                msg = editor.render_specific_video(json_path)
                return msg

            editor_render_single_btn.click(
                render_single,
                inputs=[current_json_path, use_custom_subs] + manual_inputs,
                outputs=editor_status
            )

            def render_all(proj_name, use_custom, font_name, font_size, font_color, highlight_color,
                           outline_color, outline_thickness, shadow_color, shadow_size,
                           is_bold, is_italic, is_uppercase,
                           h_size, w_block, gap, mode, under, strike, border_s,
                           vertical_pos, alignment, remove_punc):
                if not proj_name: return i18n("No project selected.")

                # Save config
                if use_custom:
                    subtitle_config = {
                        "font": font_name, "base_size": int(font_size),
                        "base_color": convert_color_to_ass(font_color),
                        "highlight_color": convert_color_to_ass(highlight_color),
                        "outline_color": convert_color_to_ass(outline_color),
                        "outline_thickness": outline_thickness,
                        "shadow_color": convert_color_to_ass(shadow_color),
                        "shadow_size": shadow_size, "vertical_position": vertical_pos,
                        "alignment": alignment, "bold": 1 if is_bold else 0,
                        "italic": 1 if is_italic else 0,
                        "underline": 1 if under else 0, "strikeout": 1 if strike else 0,
                        "border_style": border_s, "words_per_block": int(w_block),
                        "gap_limit": gap, "mode": mode, "highlight_size": int(h_size),
                        "uppercase": 1 if is_uppercase else 0,
                        "remove_punctuation": remove_punc
                    }
                    subtitle_config_path = os.path.join(WORKING_DIR, "temp_subtitle_config.json")
                    try:
                        with open(subtitle_config_path, "w", encoding="utf-8") as f:
                            json.dump(subtitle_config, f, indent=4)
                    except Exception:
                        logger.debug("Failed to write subtitle config for render_all", exc_info=True)

                proj_path = os.path.join(VIRALS_DIR, proj_name)

                # IMPORTANT: Pass the config file path to the command
                subtitle_config_path = os.path.join(WORKING_DIR, "temp_subtitle_config.json")
                cmd = [sys.executable, MAIN_SCRIPT_PATH, "--project-path", proj_path, "--workflow", "3", "--skip-prompts"]

                if use_custom and os.path.exists(subtitle_config_path):
                     cmd.extend(["--subtitle-config", subtitle_config_path])

                try:
                    subprocess.Popen(cmd, cwd=WORKING_DIR)
                    return i18n("Render All started in background... Check terminal/logs.")
                except Exception as e:
                    return i18n("Error starting render: {}").format(e)

            editor_render_all_btn.click(
                render_all,
                inputs=[editor_project_dropdown, use_custom_subs] + manual_inputs,
                outputs=editor_status
            )


        with gr.Tab(i18n("Library")):
            gr.Markdown(f"### {i18n('Existing Projects')}")
            with gr.Row():
                project_dropdown = gr.Dropdown(choices=library.get_existing_projects(), label=i18n("Select Project"), value=None)
                refresh_btn = gr.Button(i18n("Refresh List"))
            with gr.Row():
                gen_captions_btn = gr.Button("Generate TikTok Captions", variant="secondary")
                captions_status = gr.Textbox(label="", interactive=False, show_label=False, scale=3)

            with gr.Accordion(i18n("Auto Post"), open=False):
                with gr.Row():
                    lib_post_youtube = gr.Checkbox(
                        label=i18n("Post to YouTube Shorts"), value=False,
                        info=i18n("Requires YOUTUBE_CLIENT_ID + YOUTUBE_CLIENT_SECRET in .env")
                    )
                    lib_post_tiktok = gr.Checkbox(
                        label=i18n("Post to TikTok"), value=False,
                        info=i18n("Requires TIKTOK_COOKIES_PATH in .env")
                    )
                lib_youtube_privacy = gr.Dropdown(
                    choices=["public", "private", "unlisted"],
                    label=i18n("YouTube Privacy"), value="public", visible=False,
                )
                lib_post_youtube.change(
                    lambda x: gr.update(visible=x),
                    inputs=lib_post_youtube, outputs=lib_youtube_privacy
                )
                lib_first_time = gr.Textbox(
                    label=i18n("First post time (optional)"),
                    placeholder="HH:MM  or  YYYY-MM-DD HH:MM",
                    value="",
                    info=i18n("Leave empty to post immediately. Use HH:MM for today or YYYY-MM-DD HH:MM for a specific date.")
                )
                lib_interval = gr.Number(
                    label=i18n("Interval between posts (minutes)"),
                    value=30, minimum=0, maximum=1440,
                    info=i18n("Wait this many minutes between each video posted")
                )
                lib_post_btn = gr.Button(i18n("Post Selected Project"), variant="primary")
                lib_post_status = gr.Textbox(label=i18n("Post Status"), interactive=False, lines=4)

            project_gallery_html = gr.HTML()
            refresh_btn.click(library.refresh_projects, outputs=project_dropdown)
            def on_select_project(proj_name): return library.generate_project_gallery(proj_name)
            project_dropdown.change(on_select_project, project_dropdown, project_gallery_html)
            gen_captions_btn.click(
                generate_captions_for_project,
                inputs=project_dropdown,
                outputs=[captions_status, project_gallery_html]
            )

            def post_library_project(proj_name, post_yt, post_tt, privacy, first_time, interval):
                if not proj_name:
                    return i18n("No project selected.")
                if not post_yt and not post_tt:
                    return i18n("No platform selected. Check YouTube or TikTok.")
                proj_path = os.path.join(VIRALS_DIR, proj_name)
                if not os.path.exists(proj_path):
                    return i18n("Project folder not found: {}").format(proj_path)
                try:
                    import sys as _sys
                    if WORKING_DIR not in _sys.path:
                        _sys.path.insert(0, WORKING_DIR)
                    from scripts.post_social import post_all_segments
                    results = post_all_segments(
                        project_folder=proj_path,
                        post_youtube=post_yt,
                        post_tiktok=post_tt,
                        youtube_privacy=privacy,
                        interval_minutes=float(interval),
                        first_post_time=first_time,
                    )
                except Exception as e:
                    return f"Error: {e}"
                if not results:
                    return i18n("No segments posted. Check viral_segments.txt and filepath fields.")
                lines = []
                for r in results:
                    if r.get("status") == "success":
                        scheduled = f" (scheduled {r['scheduled_at']})" if r.get("scheduled_at") else ""
                        lines.append(f"{r['platform'].upper()} uploaded{scheduled} — {r.get('url') or r.get('title')}")
                    else:
                        lines.append(f"{r['platform'].upper()} FAILED — {r.get('error')}")
                return "\n".join(lines)

            lib_post_btn.click(
                post_library_project,
                inputs=[project_dropdown, lib_post_youtube, lib_post_tiktok, lib_youtube_privacy, lib_first_time, lib_interval],
                outputs=lib_post_status,
            )

    gr.Markdown(f"""
        <hr>
        <div style='text-align: center; font-size: 0.9em; color: #777;'>
            <p>
                {i18n('100% local • open source • no subscription required')}
            </p>
        </div>
        """)
if __name__ == "__main__":
    import webbrowser
    import threading
    import time
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--colab", action="store_true", help="Run in Google Colab mode")
    args = parser.parse_args()

    if args.colab:
        logger.info("Running in Colab mode. Generating public link with Static Mounts...")
        library.set_url_mode("fastapi")

        # Broaden allowed paths for Colab
        allowed_dirs = [VIRALS_DIR, WORKING_DIR, os.getcwd(), "."]

        # Explicitly set static paths
        try:
            gr.set_static_paths(paths=allowed_dirs)
            logger.debug(f"Registered static paths: {allowed_dirs}")
        except AttributeError:
            logger.debug("gr.set_static_paths not available")

        logger.debug(f"Allowed paths for Gradio: {allowed_dirs}")

        # Launch with prevent_thread_lock to allow mounting
        app, local_url, share_url = demo.queue().launch(
            share=True,
            allowed_paths=allowed_dirs,
            prevent_thread_lock=True
        )

        # Mount the VIRALS directory explicitly
        app.mount("/virals", StaticFiles(directory=VIRALS_DIR), name="virals")
        logger.info(f"Mounted /virals to {VIRALS_DIR}")

        demo.block_thread()
    else:
        # Check environment
        is_windows = (os.name == 'nt')

        library.set_url_mode("fastapi")
        allowed_dirs = [VIRALS_DIR, WORKING_DIR, os.getcwd(), "."]
        try:
            gr.set_static_paths(paths=allowed_dirs)
        except AttributeError: pass

        from fastapi.responses import FileResponse
        from fastapi import BackgroundTasks

        # Helper to attach routes to any FastAPI app (whether created by Gradio or us)
        def attach_extra_routes(fastapi_app):
            fastapi_app.mount("/virals", StaticFiles(directory=VIRALS_DIR), name="virals")

            @fastapi_app.get("/export_xml_api")
            def export_xml_api(project: str, segment: int, background_tasks: BackgroundTasks, format: str = "premiere"):
                try:
                    project_path = os.path.join(VIRALS_DIR, project)
                    real_path = os.path.realpath(project_path)
                    if not real_path.startswith(os.path.realpath(VIRALS_DIR)):
                        from fastapi.responses import JSONResponse
                        return JSONResponse(status_code=400, content={"error": "Invalid project path"})
                    script_path = os.path.join(WORKING_DIR, "scripts", "export_xml.py")
                    cmd = [sys.executable, script_path, "--project", project_path, "--segment", str(segment), "--format", format]
                    subprocess.run(cmd, check=True)
                    proj_name = os.path.basename(project_path)
                    zip_filename = f"export_{proj_name}_seg{segment}.zip"
                    file_path = os.path.join(project_path, zip_filename)
                    if os.path.exists(file_path):
                        return FileResponse(file_path, filename=zip_filename, media_type='application/zip')
                    else:
                        return {"error": f"File generation failed. Expected: {file_path}"}
                except Exception as e:
                    return {"error": str(e)}

            logger.info(f"Mounted /virals to {VIRALS_DIR}")

        if is_windows:
            logger.info("Running in Windows environment (using Gradio launch for convenience).")
            # Windows: Use demo.launch() for convenience (auto-browser, etc)
            app, local_url, share_url = demo.queue().launch(
                share=False,
                allowed_paths=allowed_dirs,
                inbrowser=True,
                server_name="127.0.0.1",
                server_port=7860,
                prevent_thread_lock=True
            )
            attach_extra_routes(app)
            demo.block_thread()
        else:
            logger.info("Running in Linux/Container environment (using Uvicorn for stability).")
            # Linux/HF: Use Uvicorn for explicit loop control
            app = FastAPI()
            attach_extra_routes(app)
            # Disable SSR to prevent Node proxying issues on HF Spaces
            app = gr.mount_gradio_app(app, demo.queue(), path="/", allowed_paths=allowed_dirs, ssr_mode=False)
            uvicorn.run(app, host="0.0.0.0", port=7860)
