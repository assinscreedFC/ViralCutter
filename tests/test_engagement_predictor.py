"""Tests for scripts/engagement_predictor.py — pure logic, XGBoost mocked."""
from __future__ import annotations

import sys
import os
from unittest.mock import patch, MagicMock

import pytest
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from scripts.analysis.engagement_predictor import (
    extract_features,
    predict_engagement,
    predict_from_metadata,
    FEATURE_NAMES,
    FEATURE_DEFAULTS,
)


# ---------------------------------------------------------------------------
# extract_features / build_feature_vector
# ---------------------------------------------------------------------------

class TestExtractFeatures:
    def test_returns_list_of_floats(self):
        result = extract_features({})
        assert isinstance(result, list)
        assert all(isinstance(v, float) for v in result)

    def test_length_matches_feature_names(self):
        result = extract_features({})
        assert len(result) == len(FEATURE_NAMES)

    def test_correct_feature_ordering(self):
        metadata = {name: float(i) for i, name in enumerate(FEATURE_NAMES)}
        result = extract_features(metadata)
        for i, name in enumerate(FEATURE_NAMES):
            assert result[i] == pytest.approx(float(i))

    def test_missing_keys_filled_with_defaults(self):
        result = extract_features({})
        for i, name in enumerate(FEATURE_NAMES):
            assert result[i] == pytest.approx(FEATURE_DEFAULTS[name])

    def test_partial_metadata_uses_defaults_for_missing(self):
        metadata = {"hook_score": 80.0}
        result = extract_features(metadata)
        assert result[FEATURE_NAMES.index("hook_score")] == pytest.approx(80.0)
        assert result[FEATURE_NAMES.index("speech_ratio")] == pytest.approx(
            FEATURE_DEFAULTS["speech_ratio"]
        )

    def test_empty_dict_returns_all_defaults(self):
        result = extract_features({})
        expected = [FEATURE_DEFAULTS[n] for n in FEATURE_NAMES]
        for a, b in zip(result, expected):
            assert a == pytest.approx(b)

    def test_negative_values_preserved(self):
        """Negative values should be passed through unchanged."""
        metadata = {"hook_score": -10.0}
        result = extract_features(metadata)
        assert result[FEATURE_NAMES.index("hook_score")] == pytest.approx(-10.0)

    def test_zero_values_preserved(self):
        metadata = {name: 0.0 for name in FEATURE_NAMES}
        result = extract_features(metadata)
        assert all(v == 0.0 for v in result)

    def test_extra_keys_ignored(self):
        metadata = {"hook_score": 70.0, "not_a_feature": 999.9}
        result = extract_features(metadata)
        assert len(result) == len(FEATURE_NAMES)


# ---------------------------------------------------------------------------
# predict_engagement — model file not found fallback
# ---------------------------------------------------------------------------

class TestPredictEngagementFallback:
    def test_missing_model_file_returns_last_feature(self, tmp_path):
        features = [float(i) for i in range(len(FEATURE_NAMES))]
        fake_path = str(tmp_path / "nonexistent.json")
        result = predict_engagement(features, fake_path)
        assert result == pytest.approx(features[-1])

    def test_missing_model_wrong_length_returns_50(self, tmp_path):
        features = [1.0, 2.0]  # Wrong length
        fake_path = str(tmp_path / "nonexistent.json")
        result = predict_engagement(features, fake_path)
        assert result == pytest.approx(50.0)

    def test_xgboost_import_error_returns_fallback(self, tmp_path):
        """When xgboost is not installed, returns composite_quality_score."""
        # Create a dummy model file so the file-check passes
        model_path = str(tmp_path / "model.json")
        with open(model_path, "w") as f:
            f.write("{}")

        features = [float(i) for i in range(len(FEATURE_NAMES))]

        with patch.dict(sys.modules, {"xgboost": None}):
            result = predict_engagement(features, model_path)

        assert result == pytest.approx(features[-1])

    def test_result_clamped_between_0_and_100(self, tmp_path):
        """XGBoost model returns out-of-range value — must be clamped."""
        model_path = str(tmp_path / "model.json")
        with open(model_path, "w") as f:
            f.write("{}")

        mock_xgb = MagicMock()
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([150.0])  # Way above 100
        mock_xgb.Booster.return_value = mock_model
        mock_xgb.DMatrix.return_value = MagicMock()

        features = [50.0] * len(FEATURE_NAMES)

        with patch.dict(sys.modules, {"xgboost": mock_xgb}):
            with patch("scripts.analysis.engagement_predictor._get_model", return_value=mock_model):
                result = predict_engagement(features, model_path)

        assert result <= 100.0

    def test_result_clamped_non_negative(self, tmp_path):
        model_path = str(tmp_path / "model.json")
        with open(model_path, "w") as f:
            f.write("{}")

        mock_xgb = MagicMock()
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([-30.0])  # Below 0
        mock_xgb.DMatrix.return_value = MagicMock()

        features = [50.0] * len(FEATURE_NAMES)

        with patch.dict(sys.modules, {"xgboost": mock_xgb}):
            with patch("scripts.analysis.engagement_predictor._get_model", return_value=mock_model):
                result = predict_engagement(features, model_path)

        assert result >= 0.0


# ---------------------------------------------------------------------------
# predict_from_metadata
# ---------------------------------------------------------------------------

class TestPredictFromMetadata:
    def test_no_model_path_returns_composite_score(self):
        metadata = {"composite_quality_score": 72.5}
        result = predict_from_metadata(metadata, model_path=None)
        assert result == pytest.approx(72.5)

    def test_missing_composite_score_defaults_to_50(self):
        result = predict_from_metadata({}, model_path=None)
        assert result == pytest.approx(50.0)

    def test_nonexistent_model_path_returns_composite_score(self, tmp_path):
        metadata = {"composite_quality_score": 65.0}
        fake_path = str(tmp_path / "nope.json")
        result = predict_from_metadata(metadata, model_path=fake_path)
        assert result == pytest.approx(65.0)

    def test_full_metadata_extracts_all_features(self):
        """extract_features should be called inside predict_from_metadata."""
        metadata = {name: 42.0 for name in FEATURE_NAMES}
        # No model file → uses fallback
        result = predict_from_metadata(metadata, model_path=None)
        # composite_quality_score is 42.0
        assert result == pytest.approx(42.0)
