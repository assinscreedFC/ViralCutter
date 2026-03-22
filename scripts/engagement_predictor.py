"""Engagement prediction using XGBoost on clip metadata features."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

FEATURE_NAMES = [
    "hook_score",
    "speech_ratio",
    "pacing_score",
    "visual_variety_score",
    "words_per_sec",
    "avg_rms_energy",
    "blur_ratio",
    "duration",
    "scene_change_count",
    "composite_quality_score",
]

FEATURE_DEFAULTS: dict[str, float] = {
    "hook_score": 50.0,
    "speech_ratio": 0.8,
    "pacing_score": 50.0,
    "visual_variety_score": 50.0,
    "words_per_sec": 2.0,
    "avg_rms_energy": 0.01,
    "blur_ratio": 0.0,
    "duration": 30.0,
    "scene_change_count": 0.0,
    "composite_quality_score": 50.0,
}


def extract_features(segment_metadata: dict) -> list[float]:
    """Extract feature vector from segment metadata dict.

    Uses sensible defaults for missing keys.
    """
    features: list[float] = []
    for name in FEATURE_NAMES:
        value = segment_metadata.get(name, FEATURE_DEFAULTS[name])
        features.append(float(value))
    return features


def predict_engagement(features: list[float], model_path: str) -> float:
    """Predict engagement score (0-100) using a trained XGBoost model.

    Returns composite_quality_score (last feature) or 50.0 if model unavailable.
    """
    if not Path(model_path).is_file():
        logger.warning("Model file not found: %s — using fallback score", model_path)
        # composite_quality_score is the last feature
        return features[-1] if len(features) == len(FEATURE_NAMES) else 50.0

    try:
        import xgboost as xgb
    except ImportError:
        logger.warning("xgboost not installed — using fallback score")
        return features[-1] if len(features) == len(FEATURE_NAMES) else 50.0

    try:
        model = xgb.Booster()
        model.load_model(model_path)
        dmatrix = xgb.DMatrix(np.array([features]), feature_names=FEATURE_NAMES)
        prediction = float(model.predict(dmatrix)[0])
        return max(0.0, min(100.0, prediction))
    except Exception:
        logger.error("XGBoost prediction failed", exc_info=True)
        return features[-1] if len(features) == len(FEATURE_NAMES) else 50.0


def predict_from_metadata(
    segment_metadata: dict, model_path: str | None = None
) -> float:
    """Convenience: extract features + predict engagement in one call.

    Falls back to composite_quality_score from metadata when model unavailable.
    """
    features = extract_features(segment_metadata)

    if model_path is None or not Path(model_path).is_file():
        return segment_metadata.get("composite_quality_score", 50.0)

    return predict_engagement(features, model_path)
