"""Tests for scripts/composite_scorer.py — pure logic, no external dependencies."""
from __future__ import annotations

import sys
import os

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from scripts.composite_scorer import compute_composite_score, DEFAULT_WEIGHTS


# ---------------------------------------------------------------------------
# Happy path — default weights
# ---------------------------------------------------------------------------

class TestCompositeScoreDefaults:
    def test_midpoint_inputs_returns_near_50(self):
        score = compute_composite_score(
            hook_score=50.0,
            speech_ratio=0.5,
            pacing_score=50.0,
            blur_ratio=0.5,
            visual_variety_score=50.0,
        )
        # speech_score=50, sharpness_score=50 → everything at 50
        assert score == pytest.approx(50.0, abs=0.1)

    def test_perfect_inputs_returns_100(self):
        score = compute_composite_score(
            hook_score=100.0,
            speech_ratio=1.0,
            pacing_score=100.0,
            blur_ratio=0.0,
            visual_variety_score=100.0,
        )
        assert score == 100.0

    def test_zero_inputs_returns_0(self):
        score = compute_composite_score(
            hook_score=0.0,
            speech_ratio=0.0,
            pacing_score=0.0,
            blur_ratio=1.0,       # max blur → sharpness=0
            visual_variety_score=0.0,
        )
        assert score == 0.0

    def test_default_call_no_args(self):
        """Called with all defaults should return a valid 0-100 float."""
        score = compute_composite_score()
        assert 0.0 <= score <= 100.0

    def test_return_type_is_float(self):
        score = compute_composite_score()
        assert isinstance(score, float)

    def test_result_is_rounded_to_one_decimal(self):
        # With non-round weights the result should still have at most 1 decimal place
        score = compute_composite_score(
            hook_score=33.3,
            speech_ratio=0.333,
            pacing_score=33.3,
            blur_ratio=0.333,
            visual_variety_score=33.3,
        )
        assert score == round(score, 1)


# ---------------------------------------------------------------------------
# Weight behaviour
# ---------------------------------------------------------------------------

class TestCompositeScoreWeights:
    def test_custom_weights_change_result(self):
        base = compute_composite_score(hook_score=100.0)
        custom = compute_composite_score(
            hook_score=100.0,
            weights={"hook": 0.0, "speech": 0.25, "pacing": 0.25,
                     "sharpness": 0.25, "variety": 0.25},
        )
        assert base != custom

    def test_hook_only_weight(self):
        """Weight everything on hook; hook=80 → score=80."""
        score = compute_composite_score(
            hook_score=80.0,
            speech_ratio=0.0,
            pacing_score=0.0,
            blur_ratio=1.0,
            visual_variety_score=0.0,
            weights={"hook": 1.0, "speech": 0.0, "pacing": 0.0,
                     "sharpness": 0.0, "variety": 0.0},
        )
        assert score == pytest.approx(80.0, abs=0.1)

    def test_partial_custom_weights_merge_with_defaults(self):
        """Passing only one custom weight merges with the remaining defaults."""
        score_default = compute_composite_score(hook_score=100.0)
        score_boosted = compute_composite_score(
            hook_score=100.0,
            weights={"hook": 0.50},   # only override hook
        )
        # boosted hook weight → higher score when hook is high
        assert score_boosted > score_default

    def test_default_weights_sum_to_one(self):
        total = sum(DEFAULT_WEIGHTS.values())
        assert total == pytest.approx(1.0, abs=1e-9)

    def test_weights_none_uses_defaults(self):
        score_none = compute_composite_score(hook_score=75.0, weights=None)
        score_default = compute_composite_score(hook_score=75.0)
        assert score_none == score_default


# ---------------------------------------------------------------------------
# Normalization / clamping
# ---------------------------------------------------------------------------

class TestCompositeScoreClamping:
    def test_over_range_inputs_clamped_to_100(self):
        score = compute_composite_score(
            hook_score=200.0,
            speech_ratio=2.0,       # speech_score = 200
            pacing_score=200.0,
            blur_ratio=-1.0,        # sharpness_score = 200
            visual_variety_score=200.0,
        )
        assert score == 100.0

    def test_negative_hook_score_clamped_to_0(self):
        score = compute_composite_score(
            hook_score=-50.0,
            speech_ratio=0.0,
            pacing_score=0.0,
            blur_ratio=1.0,
            visual_variety_score=0.0,
        )
        assert score == 0.0

    def test_blur_ratio_inverted_to_sharpness(self):
        """blur_ratio=1 → sharpness=0; blur_ratio=0 → sharpness=100."""
        high_blur = compute_composite_score(
            blur_ratio=1.0,
            weights={"hook": 0.0, "speech": 0.0, "pacing": 0.0,
                     "sharpness": 1.0, "variety": 0.0},
        )
        low_blur = compute_composite_score(
            blur_ratio=0.0,
            weights={"hook": 0.0, "speech": 0.0, "pacing": 0.0,
                     "sharpness": 1.0, "variety": 0.0},
        )
        assert high_blur == pytest.approx(0.0, abs=0.1)
        assert low_blur == pytest.approx(100.0, abs=0.1)

    def test_speech_ratio_scaled_to_100(self):
        """speech_ratio=0.5 should contribute 50 to the speech component."""
        score = compute_composite_score(
            speech_ratio=0.5,
            weights={"hook": 0.0, "speech": 1.0, "pacing": 0.0,
                     "sharpness": 0.0, "variety": 0.0},
        )
        assert score == pytest.approx(50.0, abs=0.1)
