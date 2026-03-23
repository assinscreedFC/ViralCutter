import os
import sys

# Suppress unnecessary logs before importing heavy libs
os.environ["ORT_LOGGING_LEVEL"] = "3" 
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import warnings
warnings.filterwarnings("ignore")

import json
import logging
import shutil
import subprocess
import argparse
import time
from typing import Optional

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger(__name__)

from scripts import (
    download_video,
    transcribe_video,
    create_viral_segments,
    cut_segments,
    edit_video,
    transcribe_cuts,
    adjust_subtitles,
    burn_subtitles,
    add_music,
    save_json,
    organize_output,
    translate_json,
)
from i18n.i18n import I18nAuto

# Inicializa sistema de tradução
i18n = I18nAuto()
#
# Configurações de Legenda (ASS Style)
# Cores no formato BGR (Blue-Green-Red) para o ASS
COLORS = {
    "red": "0000FF",  # Red
    "yellow": "00FFFF",   # Yellow
    "green": "00FF00",     # Green
    "white": "FFFFFF",    # White
    "black": "000000",     # Black
    "grey": "808080",     # Grey
}

def get_subtitle_config(config_path: Optional[str] = None) -> dict:
    """
    Returns the subtitle configuration dictionary.
    Can be expanded to load from a JSON/YAML file in the future.
    """
    # Default Config
    base_color_transparency = "00"
    outline_transparency = "FF" 
    highlight_color_transparency = "00"
    shadow_color_transparency = "00"
    
    config = {
        "font": "Montserrat-Regular",
        "base_size": 30,
        "base_color": f"&H{base_color_transparency}{COLORS['white']}&",
        "highlight_size": 35,
        "words_per_block": 3,
        "gap_limit": 0.5,
        "mode": 'highlight', # Options: 'no_highlight', 'word_by_word', 'highlight'
        "highlight_color": f"&H{highlight_color_transparency}{COLORS['green']}&",
        "vertical_position": 210, # 1=170(top), ... 4=60(default)
        "alignment": 2, # 2=Center
        "bold": 0,
        "italic": 0,
        "underline": 0,
        "strikeout": 0,
        "border_style": 2, # 1=outline, 3=box
        "outline_thickness": 1.5,
        "outline_color": f"&H{outline_transparency}{COLORS['grey']}&",
        "shadow_size": 2,
        "shadow_color": f"&H{shadow_color_transparency}{COLORS['black']}&",
        "remove_punctuation": True,
    }

    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                loaded_config = json.load(f)
                config.update(loaded_config)
                logger.info(i18n("Loaded subtitle config from {}").format(config_path))
        except Exception as e:
            logger.error(i18n("Error loading subtitle config: {}. Using defaults.").format(e))
    
    return config

def interactive_input_int(prompt_text: str) -> int:
    """Solicita um inteiro ao usuário via terminal."""
    while True:
        try:
            value = int(input(i18n(prompt_text)))
            if value > 0:
                return value
            logger.error(i18n("Error: Number must be greater than 0."))
        except ValueError:
            logger.error(i18n("Error: The value you entered is not an integer. Please try again."))

def main() -> None:
    # Configuração de Argumentos via Linha de Comando (CLI)
    parser = argparse.ArgumentParser(description="ViralCutter CLI")
    parser.add_argument("--url", help="YouTube Video URL")
    parser.add_argument("--segments", type=int, help="Number of segments to create")
    parser.add_argument("--viral", action="store_true", help="Enable viral mode")
    parser.add_argument("--themes", help="Comma-separated themes (if not viral mode)")
    parser.add_argument("--burn-only", action="store_true", help="Skip processing and only burn subtitles")
    parser.add_argument("--min-duration", type=int, default=15, help="Minimum segment duration (seconds)")
    parser.add_argument("--max-duration", type=int, default=90, help="Maximum segment duration (seconds)")
    parser.add_argument("--model", default="large-v3-turbo", help="Whisper model to use")
    
    parser.add_argument("--ai-backend", choices=["manual", "gemini", "g4f", "pleiade", "local"], help="AI backend for viral analysis")
    parser.add_argument("--api-key", help="Gemini API Key (required if ai-backend is gemini)")
    
    parser.add_argument("--chunk-size", help="Override Chunk Size")
    parser.add_argument("--ai-model-name", help="Override AI Model Name")

    parser.add_argument("--project-path", help="Path to existing project folder (overrides URL/Latest)")
    parser.add_argument("--workflow", choices=["1", "2", "3"], default="1", help="Workflow choice: 1=Full, 2=Cut Only, 3=Subtitles Only")
    parser.add_argument("--face-model", choices=["insightface", "mediapipe"], default="insightface", help="Face detection model")
    parser.add_argument("--face-mode", choices=["auto", "1", "2"], default="auto", help="Face tracking mode: auto, 1, 2")
    parser.add_argument("--subtitle-config", help="Path to subtitle configuration JSON file")
    parser.add_argument("--no-face-mode", choices=["padding", "zoom", "saliency", "motion"], default="padding", help="Method to handle segments with no face detected: 'padding' (black bars), 'zoom' (center crop), 'saliency' (spectral residual), 'motion' (motion tracking)")
    parser.add_argument("--face-detect-interval", type=str, default="0.17,1.0", help="Face detection interval in seconds. Single value or 'interval_1face,interval_2face'")
    parser.add_argument("--face-filter-threshold", type=float, default=0.35, help="Relative area threshold to ignore background faces (default: 0.35)")
    parser.add_argument("--face-two-threshold", type=float, default=0.60, help="Relative area threshold to trigger 2-face mode (default: 0.60)")
    parser.add_argument("--face-confidence-threshold", type=float, default=0.30, help="Face detection confidence threshold (0.0 - 1.0) (default: 0.30)")
    parser.add_argument("--face-dead-zone", type=str, default="40", help="Camera movement dead zone in pixels (default: 40)") # str to support future "auto"
    parser.add_argument("--focus-active-speaker", action="store_true", help="Enable experimental active speaker focus (InsightFace only)")
    parser.add_argument("--active-speaker-mar", type=float, default=0.03, help="Mouth Aspect Ratio threshold for active speaker (0.0 - 1.0) (default: 0.03)")
    parser.add_argument("--active-speaker-score-diff", type=float, default=1.5, help="Score difference to focus on active speaker (default: 1.5)")
    parser.add_argument("--include-motion", action="store_true", help="Include motion (body/head movement) in activity score")
    parser.add_argument("--active-speaker-motion-threshold", type=float, default=3.0, help="Motion deadzone in pixels (default: 3.0)")
    parser.add_argument("--active-speaker-motion-sensitivity", type=float, default=0.05, help="Motion sensitivity multiplier (default: 0.05)")
    parser.add_argument("--active-speaker-decay", type=float, default=2.0, help="Activity score decay rate (default: 2.0)")
    parser.add_argument("--skip-prompts", action="store_true", help="Skip interactive prompts and use defaults/existing files")
    parser.add_argument("--video-quality", choices=["best", "1080p", "720p", "480p"], default="best", help="Video download quality")
    parser.add_argument("--skip-youtube-subs", action="store_true", help="Skip downloading YouTube subtitles")
    parser.add_argument("--translate-target", help="Target language code for subtitle translation (e.g. 'pt', 'en').")
    parser.add_argument("--content-type", choices=["auto", "anime", "comedy", "commentary", "cooking", "education", "gaming", "manga", "motivation", "music", "news", "podcast", "sport", "talkshow", "vlog"], action="append", dest="content_type", default=None, help="Content type for adaptive prompts, repeatable for multi-label (e.g. --content-type gaming --content-type comedy)")
    parser.add_argument("--enable-scoring", action="store_true", help="Enable LLM scoring pass to filter low-quality segments")
    parser.add_argument("--min-score", type=int, default=70, help="Minimum viral score to keep a segment (0-100, default: 70)")
    parser.add_argument("--enable-validation", action="store_true", help="Enable LLM validation pass (hook strength, standalone test, narrative arc, viral value)")
    parser.add_argument("--zoom-out-factor", type=float, default=2.2, help="Zoom out factor for 2-face mode (default: 2.2)")
    parser.add_argument("--add-music", action="store_true", help="Add background music to final clips")
    parser.add_argument("--music-dir", help="Directory with background music files (default: music/)")
    parser.add_argument("--music-file", help="Specific music file to use")
    parser.add_argument("--music-volume", type=float, default=0.12, help="Background music volume (default: 0.12)")
    parser.add_argument("--add-distraction", action="store_true", help="Add split-screen distraction video (TikTok format)")
    parser.add_argument("--distraction-dir", help="Directory with distraction videos (default: distraction/)")
    parser.add_argument("--distraction-file", help="Specific distraction video to use")
    parser.add_argument("--distraction-no-fetch", action="store_true", help="Disable auto-fetch of distraction videos (use cache only)")
    parser.add_argument("--distraction-ratio", type=float, default=0.35,
                        help="Part de la hauteur de l'écran pour la distraction (0.20-0.50, défaut 0.35)")
    parser.add_argument("--remove-silence", action="store_true", help="Remove silent portions from cut segments (jump cuts)")
    parser.add_argument("--silence-threshold", type=float, default=-30, help="Silence detection threshold in dB (default: -30)")
    parser.add_argument("--silence-min-duration", type=float, default=0.5, help="Minimum silence duration to detect in seconds (default: 0.5)")
    parser.add_argument("--silence-max-keep", type=float, default=0.3, help="Maximum silence to keep in seconds, 0=remove all (default: 0.3)")
    # --- Video Quality (Phase 1) ---
    parser.add_argument("--smart-trim", action="store_true", help="Snap cuts to sentence boundaries using word timestamps")
    parser.add_argument("--trim-pad-start", type=float, default=0.3, help="Padding before start in seconds (default: 0.3)")
    parser.add_argument("--trim-pad-end", type=float, default=0.5, help="Padding after end in seconds (default: 0.5)")
    parser.add_argument("--scene-detection", action="store_true", help="Detect scene changes to avoid mid-scene cuts")
    parser.add_argument("--validate-clips", action="store_true", help="Validate clip boundaries for silence and compute speech ratio")
    parser.add_argument("--hook-detection", action="store_true", help="Score first 3 seconds of each clip for hook strength")
    parser.add_argument("--min-hook-score", type=int, default=40, help="Minimum hook score to keep clip (0-100, default: 40)")
    parser.add_argument("--blur-detection", action="store_true", help="Detect blurry frames in clips")
    parser.add_argument("--max-blur-ratio", type=float, default=0.3, help="Max ratio of blurry frames (0-1, default: 0.3)")
    # --- Scoring (Phase 2) ---
    parser.add_argument("--pacing-analysis", action="store_true", help="Analyze speech pacing and audio energy")
    parser.add_argument("--composite-scoring", action="store_true", help="Compute composite quality score from all signals")
    # --- Features (Phase 3) ---
    parser.add_argument("--remove-fillers", action="store_true", help="Detect and remove filler words (um, uh, like...)")
    parser.add_argument("--auto-thumbnail", action="store_true", help="Generate best-frame thumbnails for each clip")
    parser.add_argument("--auto-zoom", action="store_true", help="Apply dynamic zoom on LLM-generated zoom_cues")
    parser.add_argument("--speed-ramp", action="store_true", help="Speed up dead moments, slow down highlights")
    parser.add_argument("--speed-up-factor", type=float, default=1.5, help="Speed factor for dead moments (default: 1.5)")
    # --- Post-production (Phase 4) ---
    parser.add_argument("--progress-bar", action="store_true", help="Add animated progress bar overlay")
    parser.add_argument("--bar-color", type=str, default="white", help="Progress bar color (default: white)")
    parser.add_argument("--bar-position", type=str, default="top", choices=["top", "bottom"], help="Progress bar position")
    parser.add_argument("--ab-variants", action="store_true", help="Generate A/B caption variants")
    parser.add_argument("--num-variants", type=int, default=3, help="Number of caption variants (default: 3)")
    parser.add_argument("--layout", type=str, default=None, choices=["pip", "lower-third"], help="Visual layout template")
    parser.add_argument("--auto-broll", action="store_true", help="Auto-insert B-roll from Pexels")
    parser.add_argument("--transitions", type=str, default=None, choices=["fade", "wipeleft", "wiperight", "slideup", "slidedown"], help="Transition type between clips")
    parser.add_argument("--output-resolution", type=str, default="1080p", choices=["720p", "1080p", "4k"], help="Output resolution (default: 1080p)")
    parser.add_argument("--emoji-overlay", action="store_true", help="Add emoji overlays at key moments")
    parser.add_argument("--color-grade", type=str, default=None, help="Color grading LUT preset (cinematic, vintage, warm, cool, high_contrast)")
    parser.add_argument("--grade-intensity", type=float, default=0.7, help="Color grading intensity 0-1 (default: 0.7)")
    # --- Advanced AI (Phase 5) ---
    parser.add_argument("--engagement-prediction", action="store_true", help="Predict engagement score using ML model")
    parser.add_argument("--engagement-model", type=str, default=None, help="Path to trained XGBoost model file")
    parser.add_argument("--dubbing", action="store_true", help="AI dubbing: translate and voice-over in target language")
    parser.add_argument("--dubbing-language", type=str, default="en", help="Target language for dubbing (default: en)")
    parser.add_argument("--dubbing-original-volume", type=float, default=0.2, help="Original audio volume during dubbing (0-1, default: 0.2)")

    parser.add_argument("--enable-parts", action="store_true", help="Enable parts mode: long passages auto-split into multi-part series")
    parser.add_argument("--target-part-duration", type=int, default=55, help="Target duration for each part after splitting (seconds, default: 55)")

    args = parser.parse_args()
    
    # Workflow Logic
    workflow_choice = args.workflow
    
    # If Subtitles Only, checking project path
    if workflow_choice == "3" and not args.project_path and not args.url and not args.skip_prompts:
        # Prompt for project path or use latest if not provided?
        pass # Will handle in main flow

    # Modo Apenas Queimar Legenda (Legacy support, mapped to Workflow 3 internally if burn-only is set)
    # Verifica o argumento CLI ou uma variável local hardcoded (para compatibilidade)
    burn_only_mode = args.burn_only

    if burn_only_mode:
        logger.info(i18n("Burn only mode activated. Switching to Workflow 3..."))
        workflow_choice = "3"

    # Obtenção de Inputs (CLI ou Interativo)
    url = args.url
    project_path_arg = args.project_path
    input_video = None

    # Se project_path for fornecido, ignoramos URL
    if project_path_arg:
        if os.path.exists(project_path_arg):
             logger.info(i18n("Using provided project path: {}").format(project_path_arg))
             # Tentar achar o input.mp4 pra manter compatibilidade de variaveis, embora Workflow 3 não precise de download
             possible_input = os.path.join(project_path_arg, "input.mp4")
             if os.path.exists(possible_input):
                 input_video = possible_input
             else:
                 # Se não tiver input.mp4, tudo bem para workflow 3, mas definimos um dummy para não quebrar logica
                 input_video = os.path.join(project_path_arg, "dummy_input.mp4")
             
             # Se for workflow 3, não precisamos de URL
        else:
             logger.error(i18n("Error: Provided project path does not exist."))
             sys.exit(1)

    # Se não temos URL via CLI nem Project Path, pedimos agora
    if not url and not project_path_arg:
        if args.skip_prompts:
             logger.info(i18n("No URL provided and skipping prompts. Trying to load latest project..."))
             # Fallthrough to project loading logic
        else:
            user_input = input(i18n("Enter the YouTube video URL (or press Enter to use latest project): ")).strip()
            if user_input:
                url = user_input
    
    if not url and not input_video:
        # Usuário apertou Enter (Vazio) -> Tentar pegar último projeto
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
                    logger.error(i18n("Latest project found but 'input.mp4' is missing."))
                    sys.exit(1)
            else:
                logger.error(i18n("No existing projects found in VIRALS folder."))
                sys.exit(1)
        else:
             logger.error(i18n("VIRALS folder not found. Cannot load latest project."))
             sys.exit(1)

    # -------------------------------------------------------------------------
    # Checagem Antecipada de Segmentos Virais (Para pular configurações se já existirem)
    # -------------------------------------------------------------------------
    viral_segments = None
    project_folder_anticipated = None

    if input_video:
        # Se já temos o vídeo, podemos deduzir a pasta
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
                        logger.debug(f"Loaded {len(viral_segments['segments'])} segments from file.")
                    else:
                        logger.debug("Loaded JSON but 'segments' key is missing or empty.")
                except Exception as e:
                    logger.error(i18n("Error loading JSON: {}.").format(e))

    # Variaveis de config de IA (só necessárias se não tivermos os segmentos)
    num_segments = None
    viral_mode = False
    themes = ""
    ai_backend = "manual" # default
    api_key = None
    
    if not viral_segments:
        num_segments = args.segments
        if not num_segments:
            if args.skip_prompts:
                logger.info(i18n("No segments count provided and skip-prompts is ON. Using default 3."))
                num_segments = 3
            else:
                num_segments = interactive_input_int("Enter the number of viral segments to create: ")

        viral_mode = args.viral
        if not args.viral and not args.themes:
            if args.skip_prompts:
                logger.info(i18n("Viral mode not set, defaulting to True."))
                viral_mode = True
            else:
                response = input(i18n("Do you want viral mode? (yes/no): ")).lower()
                viral_mode = response in ['yes', 'y']
        
        themes = args.themes if args.themes else ""
        if not viral_mode and not themes:
            if not args.skip_prompts:
                 themes = input(i18n("Enter themes (comma-separated, leave blank if viral mode is True): "))

        # Duration Config
        logger.info(i18n("Current duration settings: {}s - {}s").format(args.min_duration, args.max_duration))
        if not args.skip_prompts:
            change_dur = input(i18n("Change duration? (y/n) [default: n]: ")).strip().lower()
            if change_dur in ['y', 'yes']:
                 try:
                     min_d = input(i18n("Minimum duration [{}]: ").format(args.min_duration)).strip()
                     if min_d: args.min_duration = int(min_d)
                     
                     max_d = input(i18n("Maximum duration [{}]: ").format(args.max_duration)).strip()
                     if max_d: args.max_duration = int(max_d)
                 except ValueError:
                     logger.warning(i18n("Invalid number. Using previous values."))

        # Load API Config
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'api_config.json')
        api_config = {}
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    api_config = json.load(f)
            except (json.JSONDecodeError, OSError):
                pass

        # Seleção do Backend de IA
        ai_backend = args.ai_backend
        
        # Try to load backend from config if not in args
        if not ai_backend and api_config.get("selected_api"):
            ai_backend = api_config.get("selected_api")
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
                    # Interactive model selection for local
                    # List models
                    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
                    if not os.path.exists(models_dir): os.makedirs(models_dir)
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
                                args.ai_model_name = models[m_idx] # Set global arg
                            else:
                                logger.warning(i18n("Invalid selection. Using first model."))
                                args.ai_model_name = models[0]
                        except (ValueError, IndexError):
                             logger.warning(i18n("Invalid input. Using first model."))
                             args.ai_model_name = models[0]
                             
                else:
                    ai_backend = "manual"

        api_key = args.api_key or os.getenv("GEMINI_API_KEY", "")
        # Check config for API Key if using Gemini
        if ai_backend == "gemini" and not api_key:
            cfg_key = api_config.get("gemini", {}).get("api_key", "")
            if cfg_key and cfg_key != "SUA_KEY_AQUI":
                api_key = cfg_key
        
        if ai_backend == "gemini" and not api_key:
             if args.skip_prompts:
                 logger.warning(i18n("Gemini API key missing, but skip-prompts is ON. Might fail."))
             else:
                 logger.warning(i18n("Gemini API Key not found in api_config.json or arguments."))
                 api_key = input(i18n("Enter your Gemini API Key: ")).strip()

    # Si les segments étaient déjà chargés, ai_backend est resté "manual" car le bloc de config a été sauté.
    # Priorité : 1) args.ai_backend (CLI / WebUI), 2) api_config.json selected_api
    if ai_backend == "manual":
        # 1) L'argument CLI/WebUI a la priorité absolue
        if args.ai_backend and args.ai_backend != "manual":
            ai_backend = args.ai_backend
            if ai_backend == "gemini" and not api_key:
                _cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'api_config.json')
                if os.path.exists(_cfg_path):
                    try:
                        with open(_cfg_path, 'r', encoding='utf-8') as _f:
                            _cfg = json.load(_f)
                        _key = _cfg.get("gemini", {}).get("api_key", "")
                        if _key and _key not in ("", "SUA_KEY_AQUI"):
                            api_key = _key
                        if not args.ai_model_name:
                            args.ai_model_name = _cfg.get("gemini", {}).get("model")
                    except Exception:
                        pass
        else:
            # 2) Fallback : lire api_config.json
            _cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'api_config.json')
            if os.path.exists(_cfg_path):
                try:
                    with open(_cfg_path, 'r', encoding='utf-8') as _f:
                        _cfg = json.load(_f)
                    _loaded_backend = _cfg.get("selected_api", "")
                    if _loaded_backend and _loaded_backend != "manual":
                        ai_backend = _loaded_backend
                        _key = _cfg.get(ai_backend, {}).get("api_key", "")
                        if _key and _key not in ("", "SUA_KEY_AQUI"):
                            api_key = _key
                        if not args.ai_model_name:
                            args.ai_model_name = _cfg.get(ai_backend, {}).get("model")
                except Exception:
                    pass

    # Workflow & Face Config Inputs
    workflow_choice = args.workflow
    face_model = args.face_model
    face_mode = args.face_mode

    # If args weren't provided and we are not skipping prompts, ask user
    # Note: argparse defaults are set, so they "are provided" effectively.
    # To truly detect "not provided", request default=None in argparse. 
    # But for "Simplified Mode", defaults are good.
    # Advanced users use params.
    # We will assume CLI defaults are what we want if skip_prompts is on.
    
    # Logic for detection intervals (Moved out of interactive block to support CLI/WebUI)
    detection_intervals = None
    if args.face_detect_interval:
        try:
            parts = args.face_detect_interval.split(',')
            if len(parts) == 1:
                val = float(parts[0])
                detection_intervals = {'1': val, '2': val}
            elif len(parts) >= 2:
                val1 = float(parts[0])
                val2 = float(parts[1])
                detection_intervals = {'1': val1, '2': val2}
        except ValueError:
            pass

    if not args.burn_only and not args.skip_prompts:
        # Interactive Face Config
        logger.info(i18n("--- Face Detection Settings ---"))
        logger.info(i18n("Current Face Model: {} | Mode: {}").format(face_model, face_mode))
        
        if detection_intervals:
             logger.info(i18n("Custom detection intervals: {}").format(detection_intervals))
        else:
             logger.info(i18n("Using dynamic intervals: 1s for 2-face, ~0.16s for 1-face."))


    # Pipeline Execution
    try:
        # 1. Download & Project Setup
        logger.debug(f"Checking input_video state. input_video={input_video}")
        
        if not input_video:
            if not url:
                logger.error(i18n("Error: No URL provided and no existing video selected."))
                sys.exit(1)
                
            logger.info(i18n("Starting download..."))
            download_subs = not args.skip_youtube_subs
            download_result = download_video.download(url, download_subs=download_subs, quality=args.video_quality)
            
            if isinstance(download_result, tuple):
                input_video, project_folder = download_result
            else:
                input_video = download_result
                project_folder = os.path.dirname(input_video)
                
            logger.debug(f"Download finished. input_video={input_video}, project_folder={project_folder}")
            
        else:
            # Reuso de video existente
            logger.debug("Using existing video logic.")
            project_folder = os.path.dirname(input_video)
            
        logger.info(f"Project Folder: {project_folder}")
        
        # 2. Transcribe
        if workflow_choice == "3":
            logger.info(i18n("Workflow 3: Skipping Transcribe."))
            # We assume transcription exists (SRT/JSON) or we won't need it for 'adjust_subtitles' if it uses 'subs/*.json' which are created by 'cut_segments'
            # Actually 'adjust_subtitles' reads from 'project_folder/subs'.
            # viral_segments = True # Removed to avoid overwritting dict loaded earlier
        else:
            logger.info(i18n("Transcribing with model {}...").format(args.model))
            # Se skip config, args.model é default
            srt_file, tsv_file = transcribe_video.transcribe(input_video, args.model, project_folder=project_folder)
 
        # 3. Create Viral Segments
        if workflow_choice != "3":
            # Se não carregamos 'viral_segments' lá em cima (ou se era download novo), checamos agora ou criamos
            if not viral_segments:
                # Checagem tardia para downloads novos que por acaso ja tenham json (Ex: URL repetida)
                viral_segments_file_late = os.path.join(project_folder, "viral_segments.txt")
                if os.path.exists(viral_segments_file_late):
                    logger.info(i18n("Found existing viral segments file at {}").format(viral_segments_file_late))
                    if args.skip_prompts:
                        logger.info(i18n("Skipping prompts enabled. Loading existing segments."))
                        try:
                            with open(viral_segments_file_late, 'r', encoding='utf-8') as f:
                                viral_segments = json.load(f)
                        except Exception as e:
                            logger.error(i18n("Error loading existing JSON: {}. Proceeding to create new segments.").format(e))
                    else:
                        logger.info(i18n("Loading existing viral segments found at {}").format(viral_segments_file_late))
                        try:
                            with open(viral_segments_file_late, 'r', encoding='utf-8') as f:
                                viral_segments = json.load(f)
                        except Exception as e:
                            logger.error(i18n("Error loading existing JSON: {}.").format(e))
                    
                if not viral_segments:
                    logger.info(i18n("Creating viral segments using {}...").format(ai_backend.upper()))
                    # args.content_type est None ou une liste (action="append")
                    # Filtrer "auto" et normaliser
                    raw_ct = args.content_type or []
                    content_type_arg = [ct for ct in raw_ct if ct != "auto"] or None
                    viral_segments = create_viral_segments.create(
                        num_segments,
                        viral_mode,
                        themes,
                        args.min_duration,
                        args.max_duration,
                        ai_mode=ai_backend,
                        api_key=api_key,
                        project_folder=project_folder,
                        chunk_size_arg=args.chunk_size,
                        model_name_arg=args.ai_model_name,
                        content_type=content_type_arg,
                        enable_scoring=args.enable_scoring,
                        min_score=args.min_score,
                        enable_validation=args.enable_validation,
                        enable_parts=args.enable_parts
                    )
                
                if not viral_segments or not viral_segments.get("segments"):
                    logger.error(i18n("Error: No viral segments were generated."))
                    logger.error(i18n("Possible reasons: API error, Model not found, or empty response."))
                    logger.error(i18n("Stopping execution."))
                    sys.exit(1)
                
                save_json.save_viral_segments(viral_segments, project_folder=project_folder) 

        # 3.5. Fix Raw Segments (missing timestamps)
        if workflow_choice != "3" and viral_segments and "segments" in viral_segments:
            segs = viral_segments.get("segments", [])
            if segs and len(segs) > 0:
                 # Check first segment for duration 0 but having start_time_ref or just check duration
                 first = segs[0]
                 # If duration is effectively 0 and we have a ref tag (or even if we dont, we cant cut 0s video)
                 # We assume if duration is 0, it is raw.
                 if first.get("duration", 0) == 0:
                      logger.info(i18n("Detected raw AI segments without timestamps (Duration 0). Running alignment..."))
                      try:
                          # Load transcript
                          transcript = create_viral_segments.load_transcript(project_folder)
                          # Process (Align)
                          # Use None for output_count to keep all found segments
                          viral_segments = create_viral_segments.process_segments(
                              segs, 
                              transcript, 
                              args.min_duration, 
                              args.max_duration, 
                              output_count=None 
                          )
                          save_json.save_viral_segments(viral_segments, project_folder=project_folder)
                          logger.info(i18n("Segments aligned and saved."))
                      except Exception as e:
                          logger.error(i18n("Failed to align raw segments: {}").format(e))
                          # If alignment fails, it might crash later, but we tried. 

        # 3.6. Génération des captions TikTok (seulement si au moins un segment n'en a pas)
        _segs_for_caption = viral_segments.get("segments", []) if viral_segments else []
        _needs_captions = any(not s.get("tiktok_caption") for s in _segs_for_caption)
        if viral_segments and workflow_choice != "3" and ai_backend in ("pleiade", "gemini", "g4f") and _needs_captions:
            logger.info(i18n("Generating TikTok captions..."))
            try:
                transcript_for_captions = create_viral_segments.load_transcript(project_folder)
                transcript_text = create_viral_segments.preprocess_transcript_for_ai(transcript_for_captions)
                viral_segments["segments"] = create_viral_segments.generate_tiktok_captions(
                    viral_segments["segments"],
                    transcript_text,
                    ai_mode=ai_backend,
                    api_key=api_key,
                    model_name=args.ai_model_name,
                    content_type=viral_segments.get("content_type")
                )
                # Validation des captions (si validation activée)
                if args.enable_validation:
                    logger.info(i18n("Validating TikTok captions..."))
                    viral_segments["segments"] = create_viral_segments.validate_captions(
                        viral_segments["segments"],
                        transcript_text,
                        ai_mode=ai_backend,
                        api_key=api_key,
                        model_name=args.ai_model_name,
                    )

                save_json.save_viral_segments(viral_segments, project_folder=project_folder)
            except Exception as e:
                logger.warning(f"TikTok caption generation failed: {e}")

        # 3.7 Split Long Segments into Parts (LLM-guided)
        if args.enable_parts and viral_segments and "segments" in viral_segments:
            from scripts.split_parts import split_long_segments
            logger.info(i18n("Splitting long segments into parts (AI-guided)..."))
            transcript_segments_for_split = create_viral_segments.load_transcript(project_folder)
            viral_segments = split_long_segments(
                viral_segments,
                transcript_json_path=os.path.join(project_folder, "input.json"),
                transcript_segments=transcript_segments_for_split,
                target_part_duration=args.target_part_duration,
                min_part_duration=max(args.min_duration, 30),
                max_normal_duration=args.max_duration,
                ai_mode=ai_backend,
                api_key=api_key,
                model_name=args.ai_model_name,
            )
            save_json.save_viral_segments(viral_segments, project_folder=project_folder)
            logger.info(i18n("{} segments after splitting into parts.").format(
                len(viral_segments.get("segments", []))))

        # 4. Cut Segments
        # Se workflow for 3, pulamos corte
        if workflow_choice == "3":
            logger.info(i18n("Workflow 3 (Subtitles Only): Skipping Cut and Edit."))
            # Deduzir cuts folder apenas para log
            cuts_folder = os.path.join(project_folder, "cuts")
        else:
            cuts_folder = os.path.join(project_folder, "cuts")
            skip_cutting = False
            
            if os.path.exists(cuts_folder) and os.listdir(cuts_folder):
                logger.info(i18n("Existing cuts found in: {}").format(cuts_folder))
                if args.skip_prompts:
                    cut_again_resp = 'no'
                else:
                    cut_again_resp = input(i18n("Cuts already exist. Cut again? (yes/no) [default: no]: ")).strip().lower()
                
                # Default is no (skip) if they just press enter or say no
                if cut_again_resp not in ['y', 'yes']:
                    skip_cutting = True
            
            if skip_cutting:
                logger.info(i18n("Skipping Video Rendering (using existing cuts), but updating Subtitle JSONs..."))
            else:
                logger.info(i18n("Cutting segments..."))

            cut_segments.cut(
                viral_segments,
                project_folder=project_folder,
                skip_video=skip_cutting,
                smart_trim=args.smart_trim,
                trim_pad_start=args.trim_pad_start,
                trim_pad_end=args.trim_pad_end,
                scene_detection=args.scene_detection,
            )

        # 4b. Remove Silence (Jump Cuts)
        if args.remove_silence and workflow_choice not in ("3",):
            from scripts import remove_silence
            logger.info(i18n("Removing silences (jump cuts)..."))
            remove_silence.process_project(
                project_folder=project_folder,
                noise_db=args.silence_threshold,
                min_silence_duration=args.silence_min_duration,
                max_silence_keep=args.silence_max_keep,
            )

        # 4d. Filler word removal
        if args.remove_fillers and workflow_choice not in ("3",):
            import glob as glob_mod
            from scripts.filler_removal import detect_fillers, remove_fillers_from_video, update_subtitle_json
            from scripts.smart_trim import load_whisperx_words

            cuts_folder = os.path.join(project_folder, "cuts")
            subs_folder = os.path.join(project_folder, "subs")
            video_files = sorted(glob_mod.glob(os.path.join(cuts_folder, "*_original_scale.mp4")))

            for video_path in video_files:
                filename = os.path.basename(video_path)
                base_name = filename.replace("_original_scale.mp4", "")
                json_path = os.path.join(subs_folder, f"{base_name}_processed.json")
                words = load_whisperx_words(json_path) if os.path.exists(json_path) else []
                if words:
                    fillers = detect_fillers(words)
                    if fillers:
                        logger.info(f"Found {len(fillers)} fillers in {filename}")
                        temp_out = video_path + ".tmp.mp4"
                        if remove_fillers_from_video(video_path, temp_out, fillers):
                            os.replace(temp_out, video_path)
                            update_subtitle_json(json_path, fillers, json_path)
                        elif os.path.exists(temp_out):
                            os.remove(temp_out)

        # 4e. Speed ramp
        if args.speed_ramp and workflow_choice not in ("3",):
            import glob as glob_mod
            from scripts.speed_ramp import apply_speed_ramp
            from scripts.remove_silence import detect_silences

            cuts_folder = os.path.join(project_folder, "cuts")
            video_files = sorted(glob_mod.glob(os.path.join(cuts_folder, "*_original_scale.mp4")))

            # Load zoom_cues for highlights
            segments_path = os.path.join(project_folder, "viral_segments.txt")
            all_zoom_cues = []
            if os.path.exists(segments_path):
                with open(segments_path, "r", encoding="utf-8") as f:
                    vs = json.load(f)
                all_zoom_cues = [s.get("zoom_cues", []) for s in vs.get("segments", [])]

            for idx, video_path in enumerate(video_files):
                silences = detect_silences(video_path, noise_db=-35, min_duration=0.8)
                highlights = None
                if idx < len(all_zoom_cues) and all_zoom_cues[idx]:
                    highlights = [{"timestamp": z.get("timestamp", 0), "duration": z.get("duration", 1.5)} for z in all_zoom_cues[idx]]
                if silences:
                    temp_out = video_path + ".tmp.mp4"
                    if apply_speed_ramp(video_path, temp_out, silences, speed_up_factor=args.speed_up_factor, highlights=highlights):
                        os.replace(temp_out, video_path)
                        logger.info(f"Speed ramp applied to {os.path.basename(video_path)}")
                    elif os.path.exists(temp_out):
                        os.remove(temp_out)

        # 4c. Clip Quality Validation (validate-clips, hook-detection, blur-detection, pacing, composite)
        if any([args.validate_clips, args.hook_detection, args.blur_detection, args.pacing_analysis, args.composite_scoring]) and workflow_choice not in ("3",):
            import glob as glob_mod
            from scripts.smart_trim import load_whisperx_words
            cuts_folder = os.path.join(project_folder, "cuts")
            subs_folder = os.path.join(project_folder, "subs")
            video_files = sorted(glob_mod.glob(os.path.join(cuts_folder, "*_original_scale.mp4")))

            if video_files and viral_segments and "segments" in viral_segments:
                logger.info(i18n("Validating clip quality..."))

                for idx, video_path in enumerate(video_files):
                    if idx >= len(viral_segments["segments"]):
                        logger.warning(f"Skipping validation for {os.path.basename(video_path)}: no matching segment data")
                        continue
                    seg = viral_segments["segments"][idx]
                    filename = os.path.basename(video_path)

                    # A2: Silence boundary validation
                    if args.validate_clips:
                        from scripts.clip_validator import validate_clip_boundaries
                        boundary = validate_clip_boundaries(video_path, noise_db=args.silence_threshold)
                        seg["speech_ratio"] = boundary["speech_ratio"]
                        seg["starts_on_silence"] = boundary["starts_on_silence"]
                        seg["ends_on_silence"] = boundary["ends_on_silence"]
                        if boundary["starts_on_silence"]:
                            logger.warning(f"  {filename}: starts on silence!")
                        logger.info(f"  {filename}: speech_ratio={boundary['speech_ratio']}")

                    # A3: Hook detection
                    if args.hook_detection:
                        from scripts.hook_scorer import score_hook
                        # Load words from the segment's subtitle JSON
                        base_name = filename.replace("_original_scale.mp4", "")
                        json_path = os.path.join(subs_folder, f"{base_name}_processed.json")
                        words = load_whisperx_words(json_path) if os.path.exists(json_path) else []
                        hook = score_hook(video_path, words)
                        seg["hook_score"] = hook["hook_score"]
                        seg["hook_audio_energy"] = hook["audio_energy"]
                        logger.info(f"  {filename}: hook_score={hook['hook_score']}")

                    # A5: Blur detection
                    if args.blur_detection:
                        from scripts.blur_detector import detect_blur_frames
                        blur = detect_blur_frames(video_path)
                        seg["blur_ratio"] = blur["blur_ratio"]
                        seg["avg_sharpness"] = blur["avg_sharpness"]
                        if blur["blur_ratio"] > args.max_blur_ratio:
                            logger.warning(f"  {filename}: high blur ratio {blur['blur_ratio']:.2f}")
                        logger.info(f"  {filename}: blur_ratio={blur['blur_ratio']}, sharpness={blur['avg_sharpness']}")

                    # A4: Pacing/energy analysis
                    if args.pacing_analysis:
                        from scripts.pacing_analyzer import analyze_pacing
                        base_name = filename.replace("_original_scale.mp4", "")
                        json_path = os.path.join(subs_folder, f"{base_name}_processed.json")
                        words = load_whisperx_words(json_path) if os.path.exists(json_path) else []
                        pacing = analyze_pacing(video_path, words)
                        seg["pacing_score"] = pacing["pacing_score"]
                        seg["words_per_sec"] = pacing["words_per_sec"]
                        seg["avg_rms_energy"] = pacing["avg_rms_energy"]
                        logger.info(f"  {filename}: pacing_score={pacing['pacing_score']}, wps={pacing['words_per_sec']}")

                    # A7+A8: Visual variety + Speaker activity
                    if args.validate_clips:
                        from scripts.clip_validator import score_visual_variety, analyze_speaker_activity
                        variety = score_visual_variety(video_path)
                        seg["visual_variety_score"] = variety["visual_variety_score"]
                        seg["scene_change_count"] = variety["scene_change_count"]
                        logger.info(f"  {filename}: visual_variety={variety['visual_variety_score']}")

                        base_name = filename.replace("_original_scale.mp4", "")
                        json_path = os.path.join(subs_folder, f"{base_name}_processed.json")
                        words = load_whisperx_words(json_path) if os.path.exists(json_path) else []
                        if words:
                            speaker = analyze_speaker_activity(words, 0, words[-1].get("end", 0))
                            seg["speaking_time_ratio"] = speaker["speaking_time_ratio"]
                            logger.info(f"  {filename}: speaking_ratio={speaker['speaking_time_ratio']}")

                    # A9: Composite score
                    if args.composite_scoring:
                        from scripts.composite_scorer import compute_composite_score
                        composite = compute_composite_score(
                            hook_score=seg.get("hook_score", 50.0),
                            speech_ratio=seg.get("speech_ratio", 0.8),
                            pacing_score=seg.get("pacing_score", 50.0),
                            blur_ratio=seg.get("blur_ratio", 0.0),
                            visual_variety_score=seg.get("visual_variety_score", 50.0),
                        )
                        seg["composite_quality_score"] = composite
                        logger.info(f"  {filename}: composite_score={composite}")

                    # A10: Engagement prediction
                    if args.engagement_prediction:
                        from scripts.engagement_predictor import predict_from_metadata
                        engagement = predict_from_metadata(seg, model_path=args.engagement_model)
                        seg["engagement_score"] = engagement
                        logger.info(f"  {filename}: engagement_score={engagement}")

                # Save updated metadata
                segments_path = os.path.join(project_folder, "viral_segments.txt")
                with open(segments_path, "w", encoding="utf-8") as f:
                    json.dump(viral_segments, f, indent=2, ensure_ascii=False)
                logger.info(i18n("Clip quality validation complete."))

        # 5. Workflow Check
        if workflow_choice == "2":
            logger.info(i18n("Cut Only selected. Skipping Face Crop and Subtitles."))
            logger.info(i18n(f"Process completed! Check your results in: {project_folder}"))
            sys.exit(0)

        # 5. Edit Video (Face Crop)
        if workflow_choice != "3":
            logger.info(i18n("Editing video with {} (Mode: {})...").format(face_model, face_mode))
            
            # Parse dead zone safely
            try:
                dead_zone_val = int(args.face_dead_zone)
            except (ValueError, TypeError):
                dead_zone_val = 40
                
            edit_video.edit(
                project_folder=project_folder, 
                face_model=face_model, 
                face_mode=face_mode, 
                detection_period=detection_intervals,
                filter_threshold=args.face_filter_threshold,
                two_face_threshold=args.face_two_threshold,
                confidence_threshold=args.face_confidence_threshold,
                dead_zone=dead_zone_val,
                focus_active_speaker=args.focus_active_speaker,
                active_speaker_mar=args.active_speaker_mar,
                active_speaker_score_diff=args.active_speaker_score_diff,
                include_motion=args.include_motion,
                active_speaker_motion_deadzone=args.active_speaker_motion_threshold,
                active_speaker_motion_sensitivity=args.active_speaker_motion_sensitivity,
                active_speaker_decay=args.active_speaker_decay,
                segments_data=viral_segments.get("segments", []) if viral_segments else None,
                no_face_mode=args.no_face_mode,
                zoom_out_factor=args.zoom_out_factor
            )


        else:
            logger.info(i18n("Workflow 3: Skipping Face Crop."))
            # Rename existing files if viral_segments available (since edit_video didn't run)
            if viral_segments and "segments" in viral_segments:
                 segments_data = viral_segments.get("segments", [])
                 final_folder = os.path.join(project_folder, "final")
                 subs_folder = os.path.join(project_folder, "subs")
                 
                 logger.info(i18n("Renaming existing files with titles..."))
                 for idx, segment in enumerate(segments_data):
                     title = segment.get("title", f"Segment_{idx}")
                     safe_title = "".join([c for c in title if c.isalnum() or c in " _-"]).strip()
                     safe_title = safe_title.replace(" ", "_")[:60]
                     
                     new_base_name = f"{idx:03d}_{safe_title}"
                     
                     # 1. MP4
                     old_mp4_name = f"final-output{idx:03d}_processed.mp4"
                     old_mp4_path = os.path.join(final_folder, old_mp4_name)
                     new_mp4_path = os.path.join(final_folder, f"{new_base_name}.mp4")
                     if os.path.exists(old_mp4_path) and not os.path.exists(new_mp4_path):
                         os.rename(old_mp4_path, new_mp4_path)
                         logger.info(f"Renamed (Workflow 3): {old_mp4_name} -> {new_base_name}.mp4")

                     # 2. JSON Sub
                     old_json_name = f"final-output{idx:03d}_processed.json"
                     old_json_path = os.path.join(subs_folder, old_json_name)
                     new_json_path = os.path.join(subs_folder, f"{new_base_name}_processed.json")
                     if os.path.exists(old_json_path) and not os.path.exists(new_json_path):
                         os.rename(old_json_path, new_json_path)
                         logger.info(f"Renamed (Workflow 3): {old_json_name} -> {new_base_name}_processed.json")
                         
                     # 3. Timeline
                     old_tl_name = f"temp_video_no_audio_{idx}_timeline.json"
                     old_tl_path = os.path.join(final_folder, old_tl_name)
                     new_tl_path = os.path.join(final_folder, f"{new_base_name}_timeline.json")
                     if os.path.exists(old_tl_path) and not os.path.exists(new_tl_path):
                         os.rename(old_tl_path, new_tl_path)
                         logger.info(f"Renamed (Workflow 3): {old_tl_name} -> {new_base_name}_timeline.json")

        # 6. Subtitles
        burn_subtitles_option = True 
        if burn_subtitles_option:
            logger.info(i18n("Processing subtitles..."))
            # transcribe_cuts removido: JSON de legenda já é gerado no corte
            # transcribe_cuts.transcribe(project_folder=project_folder)
            
            # --- Translation Integration ---
            if args.translate_target and args.translate_target.lower() != "none":
                 logger.info(i18n("Translating subtitles to: {}").format(args.translate_target))
                 import asyncio
                 try:
                    asyncio.run(translate_json.translate_project_subs(project_folder, args.translate_target))
                 except Exception as e:
                    logger.error(i18n("Translation failed: {}").format(e))
            # -------------------------------

            sub_config = get_subtitle_config(args.subtitle_config)

            # Split-screen auto-adjust: place subtitles exactement sur la ligne de coupure.
            # ASS PlayResY=640. Formule : int(620 * (1 - distraction_ratio))
            #   ratio=0.50 → 310 (split à y≈960px), ratio=0.35 → 403, ratio=0.30 → 434
            if getattr(args, 'add_distraction', False):
                _ratio = getattr(args, 'distraction_ratio', 0.5)
                sub_config['vertical_position'] = int(620 * (1 - _ratio))

            # Genera arquivos .ass
            try:
                adjust_subtitles.adjust(project_folder=project_folder, **sub_config)
            except FileNotFoundError as fnf_error:
                logger.error(i18n("[ERROR] Subtitle processing failed: {}").format(str(fnf_error)))
                logger.info(i18n("Tip: If you are using Workflow 3 (Subtitles Only), ensure the 'subs' folder exists and contains valid JSON files."))
                sys.exit(1)
            except Exception as e:
                logger.error(i18n("[ERROR] Unexpected error during subtitle processing: {}").format(str(e)))
                raise e
        else:
            logger.info(i18n("Subtitle burning skipped."))

        # --- Ajout musique de fond (optionnel) ---
        if args.add_music:
            logger.info(i18n("Adding background music..."))
            try:
                _segs = viral_segments.get("segments", []) if viral_segments else []
                add_music.add_music_to_project(
                    project_folder=project_folder,
                    music_dir=args.music_dir,
                    music_file=args.music_file,
                    music_volume=args.music_volume,
                    segments=_segs,
                )
            except Exception as e:
                logger.warning(f"Music addition failed: {e}")

        # --- Split-screen distraction (optionnel) ---
        # Note: distraction est appliquée AVANT burn_subtitles pour que les subs soient
        # brûlés directement sur la vidéo composite (évite le décalage dû au crop centré).
        if args.add_distraction:
            logger.info("Adding split-screen distraction video...")
            try:
                from scripts.add_distraction_video import add_distraction_to_project
                from scripts.add_distraction_video import DEFAULT_DISTRACTION_DIR as _DIST_DIR
                add_distraction_to_project(
                    project_folder=project_folder,
                    distraction_dir=args.distraction_dir or _DIST_DIR,
                    distraction_file=args.distraction_file,
                    no_fetch=args.distraction_no_fetch,
                    # Mode 2 faces : visage du haut à Y=480 → top crop (y=0) pour éviter qu'il soit coupé
                    # Mode 1 face : visage à Y=960 (centre) → crop centré par défaut (None)
                    main_crop_y=0 if getattr(args, 'face_mode', '1') == '2' else None,
                    distraction_ratio=getattr(args, 'distraction_ratio', 0.35),
                )
            except Exception as e:
                logger.warning(f"Split-screen distraction failed: {e}")

        # --- Burn subtitles ---
        if burn_subtitles_option:
            try:
                if args.add_distraction:
                    # Burn sur la vidéo composite split_screen/ (subs positionnés correctement sur 1920px)
                    split_screen_folder = os.path.join(project_folder, 'split_screen')
                    # Le suffix à supprimer dépend des étapes appliquées avant :
                    # avec musique : {name}_music_split.mp4 → strip "_music_split"
                    # sans musique : {name}_split.mp4 → strip "_split"
                    split_suffix = "_music_split" if args.add_music else "_split"
                    burn_subtitles.burn(
                        project_folder=project_folder,
                        source_folder=split_screen_folder,
                        name_suffix_strip=split_suffix,
                    )
                else:
                    burn_subtitles.burn(project_folder=project_folder)
            except Exception as e:
                logger.error(i18n("[ERROR] Unexpected error during subtitle burning: {}").format(str(e)))
                raise e

        # 9b. Auto thumbnails
        if args.auto_thumbnail and workflow_choice not in ("3",):
            import glob as glob_mod
            from scripts.auto_thumbnail import extract_best_frame, save_thumbnail

            burned_folder = os.path.join(project_folder, "burned_sub")
            if not os.path.isdir(burned_folder):
                burned_folder = os.path.join(project_folder, "cuts")
            video_files = sorted(glob_mod.glob(os.path.join(burned_folder, "*.mp4")))

            for video_path in video_files:
                frame, ts = extract_best_frame(video_path)
                if frame is not None:
                    thumb_path = video_path.rsplit(".", 1)[0] + "_thumbnail.jpg"
                    save_thumbnail(frame, thumb_path)
                    logger.info(f"Thumbnail saved: {os.path.basename(thumb_path)}")

        # 9c. Post-production overlays
        if workflow_choice not in ("3",):
            import glob as glob_mod

            # Determine the folder with final videos
            final_folder = os.path.join(project_folder, "burned_sub")
            if not os.path.isdir(final_folder):
                final_folder = os.path.join(project_folder, "cuts")
            video_files = sorted(glob_mod.glob(os.path.join(final_folder, "*.mp4")))

            for video_path in video_files:
                temp_out = video_path + ".tmp.mp4"

                # Color grading
                if args.color_grade:
                    from scripts.color_grading import apply_lut
                    if apply_lut(video_path, temp_out, lut_name=args.color_grade, intensity=args.grade_intensity):
                        os.replace(temp_out, video_path)
                        logger.info(f"Color grading applied: {os.path.basename(video_path)}")

                # Progress bar
                if args.progress_bar:
                    from scripts.overlay_effects import add_progress_bar
                    if add_progress_bar(video_path, temp_out, bar_color=args.bar_color, bar_position=args.bar_position):
                        os.replace(temp_out, video_path)
                        logger.info(f"Progress bar added: {os.path.basename(video_path)}")

                # Emoji overlay
                if args.emoji_overlay and viral_segments and "segments" in viral_segments:
                    from scripts.overlay_effects import add_emoji_overlay
                    idx = video_files.index(video_path)
                    if idx < len(viral_segments["segments"]):
                        emoji_cues = viral_segments["segments"][idx].get("emoji_cues", [])
                        if emoji_cues:
                            if add_emoji_overlay(video_path, temp_out, emoji_cues):
                                os.replace(temp_out, video_path)
                                logger.info(f"Emoji overlay added: {os.path.basename(video_path)}")

                # Clean up temp if it still exists
                if os.path.exists(temp_out):
                    os.remove(temp_out)

        # 10. AI Dubbing
        if args.dubbing and workflow_choice not in ("3",):
            import glob as glob_mod
            from scripts.ai_dubbing import dub_segment

            # Use burned_sub folder if available, else cuts
            dub_folder = os.path.join(project_folder, "burned_sub")
            if not os.path.isdir(dub_folder):
                dub_folder = os.path.join(project_folder, "cuts")
            video_files = sorted(glob_mod.glob(os.path.join(dub_folder, "*.mp4")))

            # Get transcript text from viral_segments
            segments_path = os.path.join(project_folder, "viral_segments.txt")
            segment_texts = []
            if os.path.exists(segments_path):
                with open(segments_path, "r", encoding="utf-8") as f:
                    vs = json.load(f)
                segment_texts = [s.get("description", s.get("title", "")) for s in vs.get("segments", [])]

            for idx, video_path in enumerate(video_files):
                text = segment_texts[idx] if idx < len(segment_texts) else ""
                if text:
                    dubbed_path = video_path.rsplit(".", 1)[0] + f"_dubbed_{args.dubbing_language}.mp4"
                    if dub_segment(video_path, text, args.dubbing_language, dubbed_path, original_volume=args.dubbing_original_volume):
                        logger.info(f"Dubbed: {os.path.basename(dubbed_path)}")

        # Organização Final (Opcional, pois agora já está tudo em project_folder)
        # organize_output.organize(project_folder=project_folder)

        # --- Save Processing Configuration ---
        try:
            # Determine AI Model used
            used_ai_model = args.ai_model_name
            if not used_ai_model and ai_backend != "manual":
                if ai_backend == "gemini":
                    used_ai_model = api_config.get("gemini", {}).get("model", "default")
                elif ai_backend == "g4f":
                    used_ai_model = api_config.get("g4f", {}).get("model", "default")
            
            # Ensure sub_config exists
            current_sub_config = sub_config if 'sub_config' in locals() else get_subtitle_config(args.subtitle_config)
            
            final_config = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "workflow": workflow_choice,
                "ai_config": {
                    "backend": ai_backend,
                    "model_name": used_ai_model,
                    "viral_mode": viral_mode,
                    "themes": themes,
                    "num_segments": num_segments,
                    "chunk_size": args.chunk_size
                },
                "face_config": {
                    "model": face_model,
                    "mode": face_mode,
                    "detect_interval": args.face_detect_interval,
                    "filter_threshold": args.face_filter_threshold,
                    "two_face_threshold": args.face_two_threshold,
                    "confidence_threshold": args.face_confidence_threshold,
                    "dead_zone": args.face_dead_zone,
                    "focus_active_speaker": args.focus_active_speaker,
                    "active_speaker_mar": args.active_speaker_mar,
                    "active_speaker_score_diff": args.active_speaker_score_diff,
                    "include_motion": args.include_motion
                },
                "video_config": {
                    "min_duration": args.min_duration,
                    "max_duration": args.max_duration,
                    "whisper_model": args.model
                },
                "subtitle_config": current_sub_config
            }

            config_save_path = os.path.join(project_folder, "process_config.json")
            with open(config_save_path, "w", encoding="utf-8") as f:
                json.dump(final_config, f, indent=4, ensure_ascii=False)
            logger.info(i18n("Configuration saved to: {}").format(config_save_path))
            
        except Exception as e:
            logger.error(i18n("Error saving configuration JSON: {}").format(e))
        # -------------------------------------

        logger.info(i18n("Process completed! Check your results in: {}").format(project_folder))

    except Exception as e:
        logger.error(i18n("An error occurred: {}").format(str(e)))
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
