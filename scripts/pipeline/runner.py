"""Pipeline orchestrator — runs stages in sequence."""
from __future__ import annotations

import logging
import sys

from scripts.pipeline.context import PipelineContext
from scripts.pipeline.stages import (
    stage_cut,
    stage_download,
    stage_face_edit,
    stage_filler_speed,
    stage_post_production,
    stage_quality,
    stage_save_config,
    stage_subtitles,
    stage_transcribe,
    stage_viral_segments,
)

logger = logging.getLogger(__name__)


def run_pipeline(ctx: PipelineContext) -> None:
    """Execute the full ViralCutter pipeline."""
    try:
        stage_download(ctx)
        stage_transcribe(ctx)
        stage_viral_segments(ctx)
        stage_cut(ctx)

        if ctx.workflow_choice == "2":
            stage_save_config(ctx)
            return

        stage_filler_speed(ctx)
        stage_quality(ctx)
        stage_face_edit(ctx)
        stage_subtitles(ctx)
        stage_post_production(ctx)
        stage_save_config(ctx)
    except Exception:
        logger.exception("Pipeline failed")
        sys.exit(1)
