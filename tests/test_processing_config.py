"""Tests for scripts/pipeline/config.py — ProcessingConfig dataclass."""
from __future__ import annotations

import os
import sys
from argparse import Namespace
from unittest.mock import patch

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from scripts.pipeline.config import (
    ProcessingConfig,
    InputConfig,
    AIConfig,
    SegmentConfig,
    FaceConfig,
    SubtitleConfig,
    AudioConfig,
    QualityConfig,
    PostProductionConfig,
    AdvancedAIConfig,
    DistractionConfig,
    PublishConfig,
    FLAT_KEY_MAP,
)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

class TestDefaults:
    def test_default_config_has_sane_values(self):
        cfg = ProcessingConfig()
        assert cfg.model == "large-v3-turbo"
        assert cfg.workflow == "1"
        assert cfg.skip_prompts is False
        assert cfg.input.video_quality == "best"
        assert cfg.segment.segments == 3
        assert cfg.face.face_model == "insightface"
        assert cfg.audio.music_volume == 0.12
        assert cfg.post_production.output_resolution == "1080p"

    def test_sub_configs_are_independent_instances(self):
        cfg1 = ProcessingConfig()
        cfg2 = ProcessingConfig()
        cfg1.face.ema_alpha = 0.99
        assert cfg2.face.ema_alpha == 0.18  # default, not mutated


# ---------------------------------------------------------------------------
# Flat dict round-trip
# ---------------------------------------------------------------------------

class TestFlatDictRoundTrip:
    def test_round_trip_preserves_all_values(self):
        cfg = ProcessingConfig(
            model="medium",
            workflow="2",
        )
        cfg.segment.segments = 5
        cfg.segment.min_duration = 20
        cfg.face.ema_alpha = 0.25
        cfg.post_production.progress_bar = True
        cfg.post_production.bar_color = "red"

        d = cfg.to_flat_dict()
        cfg2 = ProcessingConfig.from_flat_dict(d)

        assert cfg2.model == "medium"
        assert cfg2.workflow == "2"
        assert cfg2.segment.segments == 5
        assert cfg2.segment.min_duration == 20
        assert cfg2.face.ema_alpha == 0.25
        assert cfg2.post_production.progress_bar is True
        assert cfg2.post_production.bar_color == "red"

    def test_round_trip_all_defaults(self):
        """Default -> flat -> reconstruct should equal defaults."""
        cfg = ProcessingConfig()
        d = cfg.to_flat_dict()
        cfg2 = ProcessingConfig.from_flat_dict(d)
        assert cfg2.to_flat_dict() == d

    def test_api_key_excluded_from_flat_dict(self):
        cfg = ProcessingConfig()
        cfg.ai.api_key = "secret-key-123"
        d = cfg.to_flat_dict()
        assert "api_key" not in d

    def test_from_flat_dict_ignores_unknown_keys(self):
        d = {"model": "small", "unknown_key_xyz": True}
        cfg = ProcessingConfig.from_flat_dict(d)
        assert cfg.model == "small"
        # Should not raise, unknown key silently ignored

    def test_from_flat_dict_missing_keys_use_defaults(self):
        cfg = ProcessingConfig.from_flat_dict({})
        assert cfg.model == "large-v3-turbo"
        assert cfg.segment.segments == 3
        assert cfg.face.face_model == "insightface"

    def test_api_key_from_env(self):
        with patch.dict(os.environ, {"GEMINI_API_KEY": "env-key-42"}):
            cfg = ProcessingConfig.from_flat_dict({})
        assert cfg.ai.api_key == "env-key-42"

    def test_api_key_from_env_openai_fallback(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "oai-key"}, clear=False):
            os.environ.pop("GEMINI_API_KEY", None)
            cfg = ProcessingConfig.from_flat_dict({})
        assert cfg.ai.api_key == "oai-key"


# ---------------------------------------------------------------------------
# from_namespace
# ---------------------------------------------------------------------------

class TestFromNamespace:
    def test_namespace_basic(self):
        ns = Namespace(
            model="small",
            workflow="3",
            segments=7,
            viral=True,
            min_duration=30,
            max_duration=60,
            face_model="mediapipe",
            face_mode="2",
            skip_prompts=True,
            url="https://youtube.com/watch?v=abc",
        )
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("GEMINI_API_KEY", None)
            os.environ.pop("OPENAI_API_KEY", None)
            cfg = ProcessingConfig.from_namespace(ns)

        assert cfg.model == "small"
        assert cfg.workflow == "3"
        assert cfg.segment.segments == 7
        assert cfg.segment.min_duration == 30
        assert cfg.face.face_model == "mediapipe"
        assert cfg.input.url == "https://youtube.com/watch?v=abc"
        assert cfg.skip_prompts is True

    def test_namespace_none_values_ignored(self):
        ns = Namespace(
            model="large-v3-turbo",
            workflow="1",
            segments=None,
            url=None,
        )
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("GEMINI_API_KEY", None)
            os.environ.pop("OPENAI_API_KEY", None)
            cfg = ProcessingConfig.from_namespace(ns)

        assert cfg.segment.segments == 3  # default, not None
        assert cfg.input.url is None  # this is the default anyway


# ---------------------------------------------------------------------------
# FLAT_KEY_MAP coverage
# ---------------------------------------------------------------------------

class TestFlatKeyMap:
    def test_all_sub_config_fields_in_map(self):
        """Every field in every sub-config must be in FLAT_KEY_MAP."""
        from dataclasses import fields
        from scripts.pipeline.config import _SUB_CONFIGS

        for attr, cls in _SUB_CONFIGS:
            for f in fields(cls):
                assert f.name in FLAT_KEY_MAP, f"Missing from FLAT_KEY_MAP: {cls.__name__}.{f.name}"
                assert FLAT_KEY_MAP[f.name] == (attr, f.name)

    def test_no_key_collision_between_sub_configs(self):
        """No two sub-configs should share a flat key."""
        from dataclasses import fields
        from scripts.pipeline.config import _SUB_CONFIGS

        seen: dict[str, str] = {}
        for attr, cls in _SUB_CONFIGS:
            for f in fields(cls):
                if f.name in seen:
                    pytest.fail(f"Key collision: '{f.name}' in both {seen[f.name]} and {attr}")
                seen[f.name] = attr


# ---------------------------------------------------------------------------
# Integration with PipelineContext.cfg
# ---------------------------------------------------------------------------

class TestPipelineContextCfg:
    def test_cfg_property_lazy_build(self):
        from scripts.pipeline.context import PipelineContext
        ns = Namespace(
            model="medium",
            workflow="1",
            skip_prompts=True,
            url=None,
            project_path=None,
            burn_only=False,
            segments=5,
            viral=True,
            themes="",
            min_duration=15,
            max_duration=90,
            face_model="insightface",
            face_mode="auto",
            face_detect_interval="0.17,1.0",
        )
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("GEMINI_API_KEY", None)
            os.environ.pop("OPENAI_API_KEY", None)
            ctx = PipelineContext(args=ns)
            cfg = ctx.cfg

        assert isinstance(cfg, ProcessingConfig)
        assert cfg.model == "medium"
        assert cfg.segment.segments == 5
        # Second access returns same instance
        assert ctx.cfg is cfg

    def test_cfg_property_with_explicit_config(self):
        from scripts.pipeline.context import PipelineContext
        ns = Namespace()
        explicit_cfg = ProcessingConfig(model="tiny")
        ctx = PipelineContext(args=ns, config=explicit_cfg)
        assert ctx.cfg is explicit_cfg
        assert ctx.cfg.model == "tiny"


# ---------------------------------------------------------------------------
# to_flat_dict coverage
# ---------------------------------------------------------------------------

class TestToFlatDict:
    def test_all_expected_keys_present(self):
        cfg = ProcessingConfig()
        d = cfg.to_flat_dict()
        # Check a representative key from each sub-config
        assert "url" in d  # InputConfig
        assert "ai_backend" in d  # AIConfig
        assert "segments" in d  # SegmentConfig
        assert "face_model" in d  # FaceConfig
        assert "subtitle_config" in d  # SubtitleConfig
        assert "add_music" in d  # AudioConfig
        assert "smart_trim" in d  # QualityConfig
        assert "progress_bar" in d  # PostProductionConfig
        assert "engagement_prediction" in d  # AdvancedAIConfig
        assert "add_distraction" in d  # DistractionConfig
        assert "post_youtube" in d  # PublishConfig
        # Top-level
        assert "model" in d
        assert "workflow" in d
        assert "skip_prompts" in d

    def test_content_type_list_preserved(self):
        cfg = ProcessingConfig()
        cfg.ai.content_type = ["comedy", "podcast"]
        d = cfg.to_flat_dict()
        assert d["content_type"] == ["comedy", "podcast"]

    def test_none_values_preserved(self):
        cfg = ProcessingConfig()
        d = cfg.to_flat_dict()
        assert d["url"] is None
        assert d["translate_target"] is None
