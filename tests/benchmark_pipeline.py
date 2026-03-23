"""Benchmark pipeline stages on a real video with saved WebUI settings.

Usage:
    python tests/benchmark_pipeline.py [project_folder]

If no project_folder given, uses the first folder found in virals/.
Loads settings from webui/settings.json (same as the WebUI).
Outputs timing per stage to console + benchmark_report.json.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time

# Ensure project root is on path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.chdir(ROOT)

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("benchmark")


def load_settings_as_args_dict(project_path: str) -> dict:
    """Load webui/settings.json and convert to args_dict like process_runner does."""
    settings_path = os.path.join(ROOT, "webui", "settings.json")
    with open(settings_path, "r", encoding="utf-8") as f:
        s = json.load(f)

    args_dict: dict = {}
    args_dict["project_path"] = project_path
    args_dict["skip_prompts"] = True

    # Basic params
    args_dict["segments"] = int(s.get("segments", 6))
    args_dict["viral"] = bool(s.get("viral", True))
    if s.get("themes"):
        args_dict["themes"] = s["themes"]
    args_dict["min_duration"] = int(s.get("min_duration", 15))
    args_dict["max_duration"] = int(s.get("max_duration", 90))
    args_dict["model"] = s.get("model", "large-v3-turbo")
    args_dict["ai_backend"] = s.get("ai_backend", "manual")
    if s.get("ai_model_name"):
        args_dict["ai_model_name"] = str(s["ai_model_name"])
    if s.get("chunk_size"):
        args_dict["chunk_size"] = int(s["chunk_size"])
    workflow_map = {"Full": "1", "Cut Only": "2", "Subtitles Only": "3"}
    args_dict["workflow"] = workflow_map.get(str(s.get("workflow", "Full")), str(s.get("workflow", "1")))

    # Face params
    args_dict["face_model"] = s.get("face_model", "insightface")
    args_dict["face_mode"] = str(s.get("face_mode", "auto"))
    args_dict["face_detect_interval"] = str(s.get("face_detect_interval", "0.17,1.0"))
    args_dict["no_face_mode"] = s.get("no_face_mode", "padding")
    if s.get("face_filter_thresh") is not None:
        args_dict["face_filter_threshold"] = float(s["face_filter_thresh"])
    if s.get("face_two_thresh") is not None:
        args_dict["face_two_threshold"] = float(s["face_two_thresh"])
    if s.get("face_conf_thresh") is not None:
        args_dict["face_confidence_threshold"] = float(s["face_conf_thresh"])
    if s.get("face_dead_zone") is not None:
        args_dict["face_dead_zone"] = str(int(s["face_dead_zone"]))
    if s.get("zoom_out_factor") is not None:
        args_dict["zoom_out_factor"] = float(s["zoom_out_factor"])

    # Active speaker
    if s.get("focus_active_speaker"):
        args_dict["focus_active_speaker"] = True
        for k in ("active_speaker_mar", "active_speaker_score_diff",
                   "active_speaker_motion_threshold", "active_speaker_motion_sensitivity",
                   "active_speaker_decay"):
            if s.get(k) is not None:
                args_dict[k] = float(s[k])
        if s.get("include_motion"):
            args_dict["include_motion"] = True

    # Content type
    ct = s.get("content_type")
    if ct and isinstance(ct, list):
        filtered = [t for t in ct if t and t != "auto"]
        if filtered:
            args_dict["content_type"] = filtered

    # Scoring / validation
    if s.get("enable_scoring"):
        args_dict["enable_scoring"] = True
        args_dict["min_score"] = int(s.get("min_score", 70))
    if s.get("enable_validation"):
        args_dict["enable_validation"] = True

    # Parts
    if s.get("enable_parts"):
        args_dict["enable_parts"] = True
        if s.get("target_part_duration") is not None:
            args_dict["target_part_duration"] = int(s["target_part_duration"])

    # Music
    if s.get("add_music"):
        args_dict["add_music"] = True
        if s.get("music_dir"):
            args_dict["music_dir"] = s["music_dir"]
        if s.get("music_file"):
            args_dict["music_file"] = s["music_file"]
        if s.get("music_volume") is not None:
            args_dict["music_volume"] = float(s["music_volume"])

    # Distraction
    if s.get("add_distraction"):
        args_dict["add_distraction"] = True
        if s.get("distraction_dir"):
            args_dict["distraction_dir"] = s["distraction_dir"]
        if s.get("distraction_file"):
            args_dict["distraction_file"] = s["distraction_file"]
        if s.get("distraction_no_fetch"):
            args_dict["distraction_no_fetch"] = True
        if s.get("distraction_ratio") is not None:
            args_dict["distraction_ratio"] = float(s["distraction_ratio"])

    # Quality features
    if s.get("smart_trim"):
        args_dict["smart_trim"] = True
        if s.get("trim_pad_start") is not None:
            args_dict["trim_pad_start"] = float(s["trim_pad_start"])
        if s.get("trim_pad_end") is not None:
            args_dict["trim_pad_end"] = float(s["trim_pad_end"])
    if s.get("scene_detection"):
        args_dict["scene_detection"] = True
    if s.get("validate_clips"):
        args_dict["validate_clips"] = True
    if s.get("hook_detection"):
        args_dict["hook_detection"] = True
        if s.get("min_hook_score") is not None:
            args_dict["min_hook_score"] = int(s["min_hook_score"])
    if s.get("blur_detection"):
        args_dict["blur_detection"] = True
        if s.get("max_blur_ratio") is not None:
            args_dict["max_blur_ratio"] = float(s["max_blur_ratio"])
    if s.get("pacing_analysis"):
        args_dict["pacing_analysis"] = True
    if s.get("composite_scoring"):
        args_dict["composite_scoring"] = True

    # Phase 3
    if s.get("remove_fillers"):
        args_dict["remove_fillers"] = True
    if s.get("auto_thumbnail"):
        args_dict["auto_thumbnail"] = True
    if s.get("auto_zoom"):
        args_dict["auto_zoom"] = True
    if s.get("speed_ramp"):
        args_dict["speed_ramp"] = True
        if s.get("speed_up_factor") is not None:
            args_dict["speed_up_factor"] = float(s["speed_up_factor"])

    # Phase 4 post-production
    if s.get("progress_bar"):
        args_dict["progress_bar"] = True
        args_dict["bar_color"] = s.get("bar_color", "white")
        args_dict["bar_position"] = s.get("bar_position", "top")
    if s.get("ab_variants"):
        args_dict["ab_variants"] = True
        if s.get("num_variants") is not None:
            args_dict["num_variants"] = int(s["num_variants"])
    if s.get("layout_template"):
        args_dict["layout"] = str(s["layout_template"])
    if s.get("auto_broll"):
        args_dict["auto_broll"] = True
    if s.get("transitions"):
        args_dict["transitions"] = str(s["transitions"])
    args_dict["output_resolution"] = s.get("output_resolution", "1080p")
    if s.get("emoji_overlay"):
        args_dict["emoji_overlay"] = True
    if s.get("color_grade"):
        args_dict["color_grade"] = str(s["color_grade"])
        if s.get("grade_intensity") is not None:
            args_dict["grade_intensity"] = float(s["grade_intensity"])

    # Phase 5
    if s.get("engagement_prediction"):
        args_dict["engagement_prediction"] = True
    if s.get("dubbing"):
        args_dict["dubbing"] = True
        args_dict["dubbing_language"] = s.get("dubbing_language", "en")
        if s.get("dubbing_original_volume") is not None:
            args_dict["dubbing_original_volume"] = float(s["dubbing_original_volume"])

    # Silence removal
    if s.get("remove_silence"):
        args_dict["remove_silence"] = True
        if s.get("silence_threshold") is not None:
            args_dict["silence_threshold"] = float(s["silence_threshold"])
        if s.get("silence_min_duration") is not None:
            args_dict["silence_min_duration"] = float(s["silence_min_duration"])
        if s.get("silence_max_keep") is not None:
            args_dict["silence_max_keep"] = float(s["silence_max_keep"])

    # Subtitle config
    if s.get("use_custom_subs"):
        from webui.presets import convert_color_to_ass
        subtitle_config = {
            "font": s.get("font_name", "Montserrat-Regular"),
            "base_size": int(s.get("font_size", 30)),
            "base_color": convert_color_to_ass(s.get("font_color", "#FFFFFF")),
            "highlight_color": convert_color_to_ass(s.get("highlight_color", "#00FF00")),
            "outline_color": convert_color_to_ass(s.get("outline_color", "#000000")),
            "outline_thickness": s.get("outline_thickness", 1.5),
            "shadow_color": convert_color_to_ass(s.get("shadow_color", "#000000")),
            "shadow_size": s.get("shadow_size", 2),
            "vertical_position": s.get("vertical_pos", 210),
            "alignment": s.get("alignment", 2),
            "bold": 1 if s.get("bold") else 0,
            "italic": 1 if s.get("italic") else 0,
            "underline": 1 if s.get("underline") else 0,
            "strikeout": 1 if s.get("strikeout") else 0,
            "border_style": s.get("border_style", 2),
            "words_per_block": int(s.get("words_per_block", 3)),
            "gap_limit": s.get("gap", 0.5),
            "mode": s.get("mode", "highlight"),
            "highlight_size": int(s.get("highlight_size", 35)),
            "remove_punctuation": s.get("remove_punc", True),
            "animation": s.get("animation", "pop"),
        }
        if s.get("uppercase"):
            subtitle_config["uppercase"] = 1
        config_path = os.path.join(ROOT, "temp_subtitle_config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(subtitle_config, f, indent=4)
        args_dict["subtitle_config"] = config_path

    return args_dict


def find_project_folder() -> str:
    """Find first project folder in virals/."""
    virals_dir = os.path.join(ROOT, "virals")
    if not os.path.isdir(virals_dir):
        raise FileNotFoundError(f"virals/ directory not found at {virals_dir}")
    subdirs = [os.path.join(virals_dir, d) for d in os.listdir(virals_dir)
               if os.path.isdir(os.path.join(virals_dir, d))]
    if not subdirs:
        raise FileNotFoundError("No project folders in virals/")
    return max(subdirs, key=os.path.getmtime)


def time_stage(name: str, func, *args, **kwargs):
    """Run a stage function and return (elapsed_seconds, result_or_error)."""
    logger.info(f"{'='*60}")
    logger.info(f"  STAGE: {name}")
    logger.info(f"{'='*60}")
    t0 = time.perf_counter()
    error = None
    try:
        func(*args, **kwargs)
    except Exception as e:
        error = str(e)
        logger.error(f"  STAGE {name} FAILED: {e}")
    elapsed = time.perf_counter() - t0
    logger.info(f"  => {name}: {elapsed:.1f}s {'(FAILED)' if error else '(OK)'}")
    return elapsed, error


def main():
    import warnings
    warnings.filterwarnings("ignore")

    # Determine project folder
    if len(sys.argv) > 1:
        project_path = sys.argv[1]
    else:
        project_path = find_project_folder()

    logger.info(f"Project: {project_path}")
    logger.info(f"Video: {os.path.join(project_path, 'input.mp4')}")

    # Check prerequisites
    for f in ("input.mp4", "input.json", "viral_segments.txt"):
        fp = os.path.join(project_path, f)
        if not os.path.exists(fp):
            logger.error(f"Missing: {fp}")
            sys.exit(1)

    # Ensure project_path is absolute
    project_path = os.path.abspath(project_path)

    # Clean output dirs for fresh benchmark (tolerate locked files)
    import shutil
    for subdir in ("cuts", "subs", "final", "burned_sub", "split_screen"):
        d = os.path.join(project_path, subdir)
        if os.path.isdir(d):
            try:
                shutil.rmtree(d)
                logger.info(f"Cleaned: {subdir}/")
            except PermissionError as e:
                logger.warning(f"Could not clean {subdir}/ (files locked): {e}")
        os.makedirs(d, exist_ok=True)

    # Build context from saved settings
    args_dict = load_settings_as_args_dict(project_path)
    logger.info(f"Settings loaded. Workflow={args_dict.get('workflow')}, "
                f"face_model={args_dict.get('face_model')}, "
                f"scene_detection={args_dict.get('scene_detection')}")

    from webui.pipeline_bridge import build_context_from_dict
    ctx = build_context_from_dict(args_dict)

    # stage_download sets project_folder — since we skip it, set manually
    if not ctx.project_folder:
        if ctx.input_video:
            ctx.project_folder = os.path.dirname(ctx.input_video)
        else:
            ctx.project_folder = project_path
    logger.info(f"project_folder={ctx.project_folder}")

    # Import stages
    from scripts.pipeline.stages import (
        stage_viral_segments,
        stage_cut,
        stage_filler_speed,
        stage_quality,
        stage_face_edit,
        stage_subtitles,
        stage_post_production,
        stage_save_config,
    )

    # Run benchmark
    report = {"project": project_path, "settings": args_dict.get("workflow"), "stages": {}}
    total_start = time.perf_counter()

    stages = [
        ("viral_segments", stage_viral_segments),
        ("cut", stage_cut),
        ("filler_speed", stage_filler_speed),
        ("quality", stage_quality),
        ("face_edit", stage_face_edit),
        ("subtitles", stage_subtitles),
        ("post_production", stage_post_production),
        ("save_config", stage_save_config),
    ]

    for name, func in stages:
        elapsed, error = time_stage(name, func, ctx)
        report["stages"][name] = {"elapsed_s": round(elapsed, 2), "error": error}

    total_elapsed = time.perf_counter() - total_start
    report["total_s"] = round(total_elapsed, 2)

    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info(f"  BENCHMARK SUMMARY")
    logger.info(f"{'='*60}")
    for name, data in report["stages"].items():
        status = "FAIL" if data["error"] else "OK"
        pct = (data["elapsed_s"] / total_elapsed * 100) if total_elapsed > 0 else 0
        logger.info(f"  {name:25s}  {data['elapsed_s']:8.1f}s  ({pct:5.1f}%)  [{status}]")
    logger.info(f"  {'─'*50}")
    logger.info(f"  {'TOTAL':25s}  {total_elapsed:8.1f}s")

    # Save report
    report_path = os.path.join(ROOT, "benchmark_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info(f"\nReport saved: {report_path}")


if __name__ == "__main__":
    main()
