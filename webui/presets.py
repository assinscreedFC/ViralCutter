"""WebUI constants, presets, and utility functions."""
from __future__ import annotations

import logging
import os
import re

import gradio as gr

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------
MAIN_SCRIPT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "main_improved.py",
)
WORKING_DIR = os.path.dirname(MAIN_SCRIPT_PATH)
VIRALS_DIR = os.path.join(WORKING_DIR, "VIRALS")
MODELS_DIR = os.path.join(WORKING_DIR, "models")

# Ensure directories exist
os.makedirs(VIRALS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Face-detection presets
# ---------------------------------------------------------------------------
FACE_PRESETS = {
    "Default (Balanced)": {"thresh": 0.35, "two_face": 0.60, "conf": 0.40, "dead_zone": 150},
    "Stable (Focus Main)": {"thresh": 0.60, "two_face": 0.80, "conf": 0.60, "dead_zone": 200},
    "Sensitive (Catch All)": {"thresh": 0.10, "two_face": 0.40, "conf": 0.30, "dead_zone": 100},
    "High Precision": {"thresh": 0.40, "two_face": 0.65, "conf": 0.75, "dead_zone": 150},
}

# ---------------------------------------------------------------------------
# Experimental presets
# ---------------------------------------------------------------------------
EXPERIMENTAL_PRESETS = {
    "Default (Off)": {"focus": False, "mar": 0.03, "score": 1.5, "motion": False, "motion_th": 3.0, "motion_sens": 0.05, "decay": 2.0},
    "Active Speaker (Balanced)": {"focus": True, "mar": 0.03, "score": 1.5, "motion": True, "motion_th": 3.0, "motion_sens": 0.05, "decay": 2.0},
    "Active Speaker (Sensitive)": {"focus": True, "mar": 0.02, "score": 1.0, "motion": True, "motion_th": 2.0, "motion_sens": 0.10, "decay": 1.0},
    "Active Speaker (Stable)": {"focus": True, "mar": 0.05, "score": 2.5, "motion": False, "motion_th": 5.0, "motion_sens": 0.02, "decay": 3.0},
}

# ---------------------------------------------------------------------------
# LLM model lists
# ---------------------------------------------------------------------------
GEMINI_MODELS = [
    'gemini-3-pro-preview',
    'gemini-2.5-flash',
    'gemini-2.5-flash-preview-09-2025',
    'gemini-2.5-flash-lite',
    'gemini-2.5-flash-lite-preview-09-2025',
    'gemini-2.5-pro',
    'gemini-2.0-flash',
    'gemini-2.0-flash-lite',
]

G4F_MODELS = [
    'gpt-4o',
    'gpt-4o-mini',
    'gpt-4',
    'o1-mini',
    'o1',
    'deepseek-r1',
    'deepseek-v3',
    'llama-3.3-70b',
    'llama-3.1-405b',
    'claude-3.5-sonnet',
    'claude-3.7-sonnet',
    'gemini-2.0-flash',
    'qwen-2.5-72b',
]

# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def get_local_models() -> list[str]:
    """Return .gguf model filenames found in MODELS_DIR."""
    if not os.path.exists(MODELS_DIR):
        return []
    return [f for f in os.listdir(MODELS_DIR) if f.endswith(".gguf")]


def apply_face_preset(preset_name: str):
    """Return face-detection parameter values for *preset_name*."""
    if preset_name not in FACE_PRESETS:
        return [gr.update() for _ in range(4)]

    p = FACE_PRESETS[preset_name]
    return p["thresh"], p["two_face"], p["conf"], p["dead_zone"]


def apply_experimental_preset(preset_name: str):
    """Return experimental parameter values for *preset_name*."""
    if preset_name not in EXPERIMENTAL_PRESETS:
        return [gr.update() for _ in range(7)]

    p = EXPERIMENTAL_PRESETS[preset_name]
    return p["focus"], p["mar"], p["score"], p["motion"], p["motion_th"], p["motion_sens"], p["decay"]


def convert_color_to_ass(hex_color: str, alpha: str = "00") -> str:
    """Convert a CSS colour (hex or rgb()) to ASS format ``&HAABBGGRR&``."""
    if not hex_color:
        return f"&H{alpha}FFFFFF&"

    hex_clean = hex_color.lstrip('#').strip()

    # Handle rgb/rgba format: rgb(255, 215, 0)
    if hex_clean.lower().startswith("rgb"):
        try:
            nums = re.findall(r"[\d\.]+", hex_clean)
            if len(nums) >= 3:
                r = max(0, min(255, int(float(nums[0]))))
                g = max(0, min(255, int(float(nums[1]))))
                b = max(0, min(255, int(float(nums[2]))))
                return f"&H{alpha}{b:02X}{g:02X}{r:02X}&".upper()
        except Exception:
            logger.debug("Failed to parse rgb color format", exc_info=True)

    # Handle 3-digit hex (e.g. F00 -> FF0000)
    if len(hex_clean) == 3:
        hex_clean = "".join([c * 2 for c in hex_clean])

    if len(hex_clean) == 6:
        r = hex_clean[0:2]
        g = hex_clean[2:4]
        b = hex_clean[4:6]
        return f"&H{alpha}{b}{g}{r}&".upper()

    return f"&H{alpha}FFFFFF&"
