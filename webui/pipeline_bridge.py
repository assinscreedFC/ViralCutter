"""Bridge between WebUI parameter dicts and the pipeline's PipelineContext."""
from __future__ import annotations

from argparse import Namespace

from scripts.pipeline.cli import build_parser
from scripts.pipeline.config import ProcessingConfig
from scripts.pipeline.config_prompts import resolve_config
from scripts.pipeline.context import PipelineContext
from scripts.pipeline.input_resolver import resolve_input


def gui_params_to_namespace(params: dict) -> Namespace:
    """Convert a flat dict of GUI values to an argparse.Namespace.

    Uses build_parser defaults as base, then overlays the provided params.
    Keys use underscores (matching argparse dest names).
    """
    defaults = vars(build_parser().parse_args([]))
    defaults.update({k: v for k, v in params.items() if v is not None})
    return Namespace(**defaults)


def build_context_from_dict(params: dict) -> PipelineContext:
    """Build a PipelineContext from a raw parameter dict.

    The GUI has already resolved input source and AI config,
    so skip_prompts is forced True.
    """
    ns = gui_params_to_namespace(params)
    ns.skip_prompts = True
    ctx = resolve_input(ns)
    ctx = resolve_config(ctx)
    # Build typed config from the (possibly modified) Namespace
    ctx.config = ProcessingConfig.from_namespace(ctx.args)
    return ctx
