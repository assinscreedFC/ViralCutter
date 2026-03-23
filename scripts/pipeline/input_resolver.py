"""Resolve video input source (URL, project path, or latest project)."""
from __future__ import annotations

import json
import logging
import os

from i18n.i18n import I18nAuto
from scripts.models import Segment
from scripts.pipeline.context import PipelineContext
from scripts.pipeline.errors import PipelineError

logger = logging.getLogger(__name__)
i18n = I18nAuto()


def resolve_input(args) -> PipelineContext:
    """Determine input source and return an initialized PipelineContext."""
    ctx = PipelineContext(args=args)
    ctx.workflow_choice = args.workflow

    # Burn-only mode → Workflow 3
    if args.burn_only:
        logger.info(i18n("Burn only mode activated. Switching to Workflow 3..."))
        ctx.workflow_choice = "3"

    url = args.url
    project_path_arg = args.project_path
    input_video = None

    # If project_path provided, skip URL
    if project_path_arg:
        if os.path.exists(project_path_arg):
            logger.info(i18n("Using provided project path: {}").format(project_path_arg))
            possible_input = os.path.join(project_path_arg, "input.mp4")
            if os.path.exists(possible_input):
                input_video = possible_input
            else:
                input_video = os.path.join(project_path_arg, "dummy_input.mp4")
        else:
            raise PipelineError(i18n("Error: Provided project path does not exist."))

    # If no URL and no project path, prompt or use latest
    if not url and not project_path_arg:
        if args.skip_prompts:
            logger.info(i18n("No URL provided and skipping prompts. Trying to load latest project..."))
        else:
            user_input = input(i18n("Enter the YouTube video URL (or press Enter to use latest project): ")).strip()
            if user_input:
                url = user_input

    if not url and not input_video:
        base_virals = "VIRALS"
        if os.path.exists(base_virals):
            subdirs = [os.path.join(base_virals, d) for d in os.listdir(base_virals) if os.path.isdir(os.path.join(base_virals, d))]
            if subdirs:
                latest_project = max(subdirs, key=os.path.getmtime)
                detected_video = os.path.join(latest_project, "input.mp4")
                if os.path.exists(detected_video):
                    input_video = detected_video
                    logger.info(i18n("Using latest project: {}").format(latest_project))
                else:
                    raise PipelineError(i18n("Latest project found but 'input.mp4' is missing."))
            else:
                raise PipelineError(i18n("No existing projects found in VIRALS folder."))
        else:
            raise PipelineError(i18n("VIRALS folder not found. Cannot load latest project."))

    # Early check for existing viral segments
    viral_segments = None
    if input_video:
        project_folder_anticipated = os.path.dirname(input_video)
        viral_segments_file = os.path.join(project_folder_anticipated, "viral_segments.txt")

        if os.path.exists(viral_segments_file):
            logger.info(i18n("Existing viral segments found: {}").format(viral_segments_file))
            if args.skip_prompts:
                use_existing_json = 'yes'
            else:
                use_existing_json = input(i18n("Use existing viral segments? (yes/no) [default: yes]: ")).strip().lower()

            if use_existing_json in ['', 'y', 'yes']:
                try:
                    with open(viral_segments_file, 'r', encoding='utf-8') as f:
                        viral_segments = json.load(f)
                    logger.info(i18n("Loaded existing viral segments. Skipping configuration prompts."))
                    if viral_segments and "segments" in viral_segments:
                        viral_segments["segments"] = [
                            Segment.from_dict(s).to_dict() for s in viral_segments.get("segments", [])
                        ]
                        logger.debug(f"Loaded {len(viral_segments['segments'])} segments from file.")
                    else:
                        logger.debug("Loaded JSON but 'segments' key is missing or empty.")
                except Exception as e:
                    logger.error(i18n("Error loading JSON: {}.").format(e))

    ctx.url = url
    ctx.input_video = input_video
    ctx.viral_segments = viral_segments
    return ctx
