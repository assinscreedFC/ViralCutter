"""Interactive configuration prompts for AI backend, model, and segments."""
from __future__ import annotations

import json
import logging
import os

from i18n.i18n import I18nAuto
from scripts.config import load_api_config
from scripts.pipeline.context import PipelineContext

logger = logging.getLogger(__name__)
i18n = I18nAuto()


def _interactive_input_int(prompt_text: str) -> int:
    """Prompt user for a positive integer via terminal."""
    while True:
        try:
            value = int(input(i18n(prompt_text)))
            if value > 0:
                return value
            logger.error(i18n("Error: Number must be greater than 0."))
        except ValueError:
            logger.error(i18n("Error: The value you entered is not an integer. Please try again."))


def resolve_config(ctx: PipelineContext) -> PipelineContext:
    """Resolve AI configuration: backend, model, api_key, segments, themes."""
    args = ctx.args
    loaded_cfg = load_api_config()

    # If segments already loaded, skip most config
    if not ctx.viral_segments:
        num_segments = args.segments
        if not num_segments:
            if args.skip_prompts:
                logger.info(i18n("No segments count provided and skip-prompts is ON. Using default 3."))
                num_segments = 3
            else:
                num_segments = _interactive_input_int("Enter the number of viral segments to create: ")
        ctx.num_segments = num_segments

        viral_mode = args.viral
        if not args.viral and not args.themes:
            if args.skip_prompts:
                logger.info(i18n("Viral mode not set, defaulting to True."))
                viral_mode = True
            else:
                response = input(i18n("Do you want viral mode? (yes/no): ")).lower()
                viral_mode = response in ['yes', 'y']
        ctx.viral_mode = viral_mode

        themes = args.themes if args.themes else ""
        if not viral_mode and not themes:
            if not args.skip_prompts:
                themes = input(i18n("Enter themes (comma-separated, leave blank if viral mode is True): "))
        ctx.themes = themes

        # Duration Config
        logger.info(i18n("Current duration settings: {}s - {}s").format(args.min_duration, args.max_duration))
        if not args.skip_prompts:
            change_dur = input(i18n("Change duration? (y/n) [default: n]: ")).strip().lower()
            if change_dur in ['y', 'yes']:
                try:
                    min_d = input(i18n("Minimum duration [{}]: ").format(args.min_duration)).strip()
                    if min_d:
                        args.min_duration = int(min_d)
                    max_d = input(i18n("Maximum duration [{}]: ").format(args.max_duration)).strip()
                    if max_d:
                        args.max_duration = int(max_d)
                except ValueError:
                    logger.warning(i18n("Invalid number. Using previous values."))

        # Load raw API config for model-name lookups
        _raw_cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'api_config.json')
        api_config = {}
        if os.path.exists(_raw_cfg_path):
            try:
                with open(_raw_cfg_path, 'r', encoding='utf-8') as f:
                    api_config = json.load(f)
            except (json.JSONDecodeError, OSError):
                pass
        ctx.api_config = api_config

        # AI Backend selection
        ai_backend = args.ai_backend
        if not ai_backend and loaded_cfg.get("backend"):
            ai_backend = loaded_cfg["backend"]
            logger.info(i18n("Using AI Backend from config: {}").format(ai_backend))

        if not ai_backend:
            if args.skip_prompts:
                logger.info(i18n("No AI backend selected, defaulting to Manual."))
                ai_backend = "manual"
            else:
                logger.info(i18n("Select AI Backend for Viral Analysis:"))
                logger.info(i18n("1. Gemini API (Best / Recommended)"))
                logger.info(i18n("2. G4F (Free / Experimental)"))
                logger.info(i18n("3. Local (GGUF via llama.cpp)"))
                logger.info(i18n("4. Manual (Copy/Paste Prompt)"))
                choice = input(i18n("Choose (1-4): ")).strip()

                if choice == "1":
                    ai_backend = "gemini"
                elif choice == "2":
                    ai_backend = "g4f"
                elif choice == "3":
                    ai_backend = "local"
                    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "models")
                    if not os.path.exists(models_dir):
                        os.makedirs(models_dir)
                    models = [f for f in os.listdir(models_dir) if f.endswith(".gguf")]

                    if not models:
                        logger.warning(i18n("No .gguf models found in 'models' directory."))
                        logger.info(i18n("Please place a module file in: {}").format(models_dir))
                        logger.info(i18n("Falling back to Manual..."))
                        ai_backend = "manual"
                    else:
                        logger.info(i18n("Available Models:"))
                        for idx, m in enumerate(models):
                            logger.info(f"{idx+1}. {m}")
                        try:
                            m_idx = int(input(i18n("Select Model (Number): "))) - 1
                            if 0 <= m_idx < len(models):
                                args.ai_model_name = models[m_idx]
                            else:
                                logger.warning(i18n("Invalid selection. Using first model."))
                                args.ai_model_name = models[0]
                        except (ValueError, IndexError):
                            logger.warning(i18n("Invalid input. Using first model."))
                            args.ai_model_name = models[0]
                else:
                    ai_backend = "manual"

        api_key = args.api_key or os.getenv("GEMINI_API_KEY", "")
        if not api_key:
            api_key = loaded_cfg.get("api_key", "")

        if ai_backend == "gemini" and not api_key:
            if args.skip_prompts:
                logger.warning(i18n("Gemini API key missing, but skip-prompts is ON. Might fail."))
            else:
                logger.warning(i18n("Gemini API Key not found in api_config.json or arguments."))
                api_key = input(i18n("Enter your Gemini API Key: ")).strip()

        ctx.ai_backend = ai_backend
        ctx.api_key = api_key

    # If segments were loaded, ai_backend stayed "manual" — resolve from config
    if ctx.ai_backend == "manual":
        if args.ai_backend and args.ai_backend != "manual":
            ctx.ai_backend = args.ai_backend
        else:
            cfg_backend = loaded_cfg.get("backend", "")
            if cfg_backend and cfg_backend != "manual":
                ctx.ai_backend = cfg_backend
        if not ctx.api_key:
            ctx.api_key = loaded_cfg.get("api_key", "")
        if not args.ai_model_name:
            args.ai_model_name = loaded_cfg.get("model_name", "")

    # Face config
    ctx.face_model = args.face_model
    ctx.face_mode = args.face_mode

    # Detection intervals
    if args.face_detect_interval:
        try:
            parts = args.face_detect_interval.split(',')
            if len(parts) == 1:
                val = float(parts[0])
                ctx.detection_intervals = {'1': val, '2': val}
            elif len(parts) >= 2:
                val1 = float(parts[0])
                val2 = float(parts[1])
                ctx.detection_intervals = {'1': val1, '2': val2}
        except ValueError:
            pass

    if not args.burn_only and not args.skip_prompts:
        logger.info(i18n("--- Face Detection Settings ---"))
        logger.info(i18n("Current Face Model: {} | Mode: {}").format(ctx.face_model, ctx.face_mode))
        if ctx.detection_intervals:
            logger.info(i18n("Custom detection intervals: {}").format(ctx.detection_intervals))
        else:
            logger.info(i18n("Using dynamic intervals: 1s for 2-face, ~0.16s for 1-face."))

    return ctx
