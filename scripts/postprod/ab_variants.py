"""A/B caption variants: generate multiple hook variants for each clip."""
from __future__ import annotations

import copy
import glob
import json
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


def generate_variants(
    project_folder: str,
    num_variants: int = 3,
    sub_config: dict | None = None,
) -> list[str]:
    """Generate A/B caption variant videos for each clip.

    For each segment with caption_variants in viral_segments.txt:
      1. Read variant hook texts
      2. For each variant:
         a. Copy the subtitle JSON, replace first words (hook) with variant text
         b. Generate .ass from modified JSON
         c. Burn subtitles onto the base video from final/
         d. Save in burned_sub/ with _varA, _varB suffix

    Returns list of generated variant file paths.
    """
    segments_path = os.path.join(project_folder, "viral_segments.txt")
    if not os.path.isfile(segments_path):
        logger.warning("[AB] viral_segments.txt not found")
        return []

    with open(segments_path, "r", encoding="utf-8") as f:
        vs = json.load(f)

    segments = vs.get("segments", [])
    if not segments:
        return []

    subs_folder = os.path.join(project_folder, "subs")
    final_folder = os.path.join(project_folder, "final")
    burned_folder = os.path.join(project_folder, "burned_sub")
    os.makedirs(burned_folder, exist_ok=True)

    generated: list[str] = []
    variant_labels = [chr(65 + i) for i in range(num_variants)]  # A, B, C...

    for seg_idx, segment in enumerate(segments):
        caption_variants = segment.get("caption_variants", [])
        if not caption_variants:
            continue

        # Find the subtitle JSON for this segment
        sub_json_path = _find_subtitle_json(subs_folder, seg_idx, segment)
        if not sub_json_path:
            logger.warning("[AB] Subtitle JSON not found for segment %d", seg_idx)
            continue

        base_name = os.path.basename(sub_json_path).replace("_processed.json", "")

        # Find the base video in final/
        video_candidates = sorted(
            [f for f in os.listdir(final_folder) if f.startswith(f"{seg_idx:03d}_") and f.endswith(".mp4")]
        ) if os.path.isdir(final_folder) else []

        if not video_candidates:
            logger.warning("[AB] Base video not found for segment %d in final/", seg_idx)
            continue

        base_video = os.path.join(final_folder, video_candidates[0])

        # Load original subtitle JSON
        with open(sub_json_path, "r", encoding="utf-8") as f:
            sub_data = json.load(f)

        for var_idx, variant_text in enumerate(caption_variants[:num_variants]):
            label = variant_labels[var_idx] if var_idx < len(variant_labels) else str(var_idx)

            # Create modified subtitle JSON with variant hook
            variant_sub = _modify_hook_text(sub_data, variant_text)

            # Save variant subtitle JSON
            variant_json_path = os.path.join(subs_folder, f"{base_name}_var{label}_processed.json")
            with open(variant_json_path, "w", encoding="utf-8") as f:
                json.dump(variant_sub, f, ensure_ascii=False, indent=2)

            # Generate .ass from variant JSON
            variant_ass_path = variant_json_path.replace(".json", ".ass")
            try:
                from scripts.editing.adjust_subtitles import generate_ass_from_file

                config = sub_config or {}
                generate_ass_from_file(
                    input_path=variant_json_path,
                    output_path=variant_ass_path,
                    project_folder=project_folder,
                    base_color=config.get("base_color", "&HFFFFFF&"),
                    base_size=config.get("base_size", 20),
                    highlight_size=config.get("highlight_size", 24),
                    highlight_color=config.get("highlight_color", "&H00D7FF&"),
                    words_per_block=config.get("words_per_block", 4),
                    gap_limit=config.get("gap_limit", 0.25),
                    mode=config.get("mode", "highlight"),
                    vertical_position=config.get("vertical_position", 180),
                    alignment=config.get("alignment", 2),
                    font=config.get("font", "Montserrat-ExtraBold"),
                    outline_color=config.get("outline_color", "&H000000&"),
                    shadow_color=config.get("shadow_color", "&H000000&"),
                    bold=config.get("bold", True),
                    italic=config.get("italic", False),
                    underline=config.get("underline", False),
                    strikeout=config.get("strikeout", False),
                    border_style=config.get("border_style", 1),
                    outline_thickness=config.get("outline_thickness", 3),
                    shadow_size=config.get("shadow_size", 2),
                    uppercase=config.get("uppercase", True),
                    remove_punctuation=config.get("remove_punctuation", True),
                    animation=config.get("animation", "pop"),
                )
            except Exception as e:
                logger.error("[AB] Failed to generate ASS for variant %s of segment %d: %s", label, seg_idx, e)
                continue

            if not os.path.isfile(variant_ass_path):
                continue

            # Burn subtitles onto base video
            variant_output = os.path.join(burned_folder, f"{base_name}_var{label}_subtitled.mp4")
            try:
                from scripts.editing.burn_subtitles import burn_video_file
                ok, msg = burn_video_file(base_video, variant_ass_path, variant_output)
                if ok:
                    generated.append(variant_output)
                    logger.info("[AB] Generated variant %s for segment %d: %s", label, seg_idx, os.path.basename(variant_output))
                else:
                    logger.error("[AB] Burn failed for variant %s: %s", label, msg)
            except Exception as e:
                logger.error("[AB] Burn failed for variant %s of segment %d: %s", label, seg_idx, e)

            # Cleanup temp files
            for tmp_path in [variant_json_path, variant_ass_path]:
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    logger.info("[AB] Generated %d variant videos total", len(generated))
    return generated


def _find_subtitle_json(subs_folder: str, seg_idx: int, segment: dict) -> str | None:
    """Locate the processed subtitle JSON for a given segment index."""
    title = segment.get("title", f"Segment_{seg_idx}")
    safe_title = "".join([c for c in title if c.isalnum() or c in " _-"]).strip().replace(" ", "_")[:60]
    base_name = f"{seg_idx:03d}_{safe_title}"

    sub_json_path = os.path.join(subs_folder, f"{base_name}_processed.json")
    if os.path.isfile(sub_json_path):
        return sub_json_path

    # Fallback: match by index prefix
    candidates = sorted(glob.glob(os.path.join(subs_folder, f"{seg_idx:03d}_*_processed.json")))
    if candidates:
        return candidates[0]

    return None


def _modify_hook_text(sub_data: list | dict, variant_text: str) -> list | dict:
    """Replace the first few words in the subtitle data with the variant hook text.

    The subtitle JSON is a list of word entries with 'word', 'start', 'end' keys.
    We replace the text of the first N words (keeping timing intact) with the variant text.
    """
    if isinstance(sub_data, dict):
        words = sub_data.get("words", sub_data.get("segments", []))
    else:
        words = sub_data

    if not words:
        return sub_data

    # Deep copy to avoid mutating original
    result = copy.deepcopy(sub_data)

    if isinstance(result, dict):
        result_words = result.get("words", result.get("segments", []))
    else:
        result_words = result

    # Split variant text into words
    variant_words = variant_text.strip().split()

    # Replace first N words (up to the length of variant text)
    for i, new_word in enumerate(variant_words):
        if i < len(result_words):
            if isinstance(result_words[i], dict):
                result_words[i]["word"] = new_word

    return result
