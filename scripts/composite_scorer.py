"""Composite scorer: aggregate multi-modal quality signals into a single score."""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

DEFAULT_WEIGHTS = {
    "hook": 0.30,
    "speech": 0.15,
    "pacing": 0.20,
    "sharpness": 0.15,
    "variety": 0.20,
}


def compute_composite_score(
    hook_score: float = 50.0,
    speech_ratio: float = 0.8,
    pacing_score: float = 50.0,
    blur_ratio: float = 0.0,
    visual_variety_score: float = 50.0,
    weights: dict | None = None,
) -> float:
    """Compute a weighted composite quality score (0-100).

    Args:
        hook_score: Hook strength (0-100) from hook_scorer.
        speech_ratio: Fraction of clip with speech (0.0-1.0) from clip_validator.
        pacing_score: Pacing/energy score (0-100) from pacing_analyzer.
        blur_ratio: Fraction of blurry frames (0.0-1.0) from blur_detector.
        visual_variety_score: Visual variety (0-100) from clip_validator.
        weights: Custom weights dict. Keys: hook, speech, pacing, sharpness, variety.

    Returns:
        Composite score 0-100.
    """
    w = {**DEFAULT_WEIGHTS, **(weights or {})}

    # Normalize all inputs to 0-100 scale
    speech_score = speech_ratio * 100.0
    sharpness_score = (1.0 - blur_ratio) * 100.0  # Invert: low blur = high sharpness

    composite = (
        w.get("hook", 0.30) * hook_score
        + w.get("speech", 0.15) * speech_score
        + w.get("pacing", 0.20) * pacing_score
        + w.get("sharpness", 0.15) * sharpness_score
        + w.get("variety", 0.20) * visual_variety_score
    )

    return round(max(0.0, min(100.0, composite)), 1)
