"""Typed dataclass configuration for the ViralCutter pipeline.

Replaces untyped argparse.Namespace with a structured, serializable config.
Two entry points:
  - ProcessingConfig.from_flat_dict(d)  -- WebUI / multiprocessing Queue
  - ProcessingConfig.from_namespace(ns) -- CLI argparse
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from typing import Any

__all__ = [
    "InputConfig",
    "AIConfig",
    "SegmentConfig",
    "FaceConfig",
    "SubtitleConfig",
    "AudioConfig",
    "QualityConfig",
    "PostProductionConfig",
    "AdvancedAIConfig",
    "DistractionConfig",
    "PublishConfig",
    "ProcessingConfig",
]

# ---------------------------------------------------------------------------
# Sub-configs (12 domain groups)
# ---------------------------------------------------------------------------


@dataclass
class InputConfig:
    url: str | None = None
    project_path: str | None = None
    video_quality: str = "best"
    skip_youtube_subs: bool = False


@dataclass
class AIConfig:
    ai_backend: str = "manual"
    api_key: str = ""  # never serialized
    ai_model_name: str | None = None
    chunk_size: int | None = None
    content_type: list[str] | None = None
    enable_scoring: bool = False
    min_score: int = 70
    enable_validation: bool = False


@dataclass
class SegmentConfig:
    segments: int = 3
    viral: bool = True
    themes: str = ""
    min_duration: int = 15
    max_duration: int = 90
    enable_parts: bool = False
    target_part_duration: int = 55


@dataclass
class FaceConfig:
    face_model: str = "insightface"
    face_mode: str = "auto"
    face_detect_interval: str = "0.17,1.0"
    no_face_mode: str = "padding"
    face_filter_threshold: float = 0.35
    face_two_threshold: float = 0.60
    face_confidence_threshold: float = 0.30
    face_dead_zone: str = "40"
    zoom_out_factor: float = 2.2
    vertical_offset: float = 0.0
    single_face_zoom: float = 1.0
    ema_alpha: float = 0.18
    detection_resolution: int = 480
    focus_active_speaker: bool = False
    active_speaker_mar: float = 0.03
    active_speaker_score_diff: float = 1.5
    include_motion: bool = False
    active_speaker_motion_threshold: float = 3.0
    active_speaker_motion_sensitivity: float = 0.05
    active_speaker_decay: float = 2.0


@dataclass
class SubtitleConfig:
    subtitle_config: str | None = None


@dataclass
class AudioConfig:
    add_music: bool = False
    music_dir: str | None = None
    music_file: str | None = None
    music_volume: float = 0.12
    remove_silence: bool = False
    silence_threshold: float = -30.0
    silence_min_duration: float = 0.5
    silence_max_keep: float = 0.3


@dataclass
class QualityConfig:
    smart_trim: bool = False
    trim_pad_start: float = 0.3
    trim_pad_end: float = 0.5
    scene_detection: bool = False
    validate_clips: bool = False
    hook_detection: bool = False
    min_hook_score: int = 40
    blur_detection: bool = False
    max_blur_ratio: float = 0.3
    pacing_analysis: bool = False
    composite_scoring: bool = False


@dataclass
class PostProductionConfig:
    remove_fillers: bool = False
    auto_thumbnail: bool = False
    auto_zoom: bool = False
    speed_ramp: bool = False
    speed_up_factor: float = 1.5
    progress_bar: bool = False
    bar_color: str = "white"
    bar_position: str = "top"
    ab_variants: bool = False
    num_variants: int = 3
    layout: str | None = None
    auto_broll: bool = False
    transitions: str | None = None
    output_resolution: str = "1080p"
    emoji_overlay: bool = False
    color_grade: str | None = None
    grade_intensity: float = 0.7


@dataclass
class AdvancedAIConfig:
    engagement_prediction: bool = False
    engagement_model: str | None = None
    dubbing: bool = False
    dubbing_language: str = "en"
    dubbing_original_volume: float = 0.2


@dataclass
class DistractionConfig:
    add_distraction: bool = False
    distraction_dir: str | None = None
    distraction_file: str | None = None
    distraction_no_fetch: bool = False
    distraction_ratio: float = 0.35


@dataclass
class PublishConfig:
    post_youtube: bool = False
    post_tiktok: bool = False
    youtube_privacy: str = "private"
    post_interval_minutes: float = 0.0
    post_first_time: str | None = None


# ---------------------------------------------------------------------------
# Flat-key mapping: flat_key -> (sub_config_attr, field_name)
# ---------------------------------------------------------------------------

_SUB_CONFIGS: list[tuple[str, type]] = [
    ("input", InputConfig),
    ("ai", AIConfig),
    ("segment", SegmentConfig),
    ("face", FaceConfig),
    ("subtitle", SubtitleConfig),
    ("audio", AudioConfig),
    ("quality", QualityConfig),
    ("post_production", PostProductionConfig),
    ("advanced_ai", AdvancedAIConfig),
    ("distraction", DistractionConfig),
    ("publish", PublishConfig),
]

# Top-level fields that do NOT belong to any sub-config
_TOP_LEVEL_FIELDS = {"model", "workflow", "skip_prompts", "translate_target", "burn_only"}


def _build_flat_key_map() -> dict[str, tuple[str, str]]:
    """Build {flat_key: (sub_config_attr, field_name)} from dataclass fields."""
    mapping: dict[str, tuple[str, str]] = {}
    for attr, cls in _SUB_CONFIGS:
        for f in fields(cls):
            mapping[f.name] = (attr, f.name)
    return mapping


FLAT_KEY_MAP: dict[str, tuple[str, str]] = _build_flat_key_map()


# ---------------------------------------------------------------------------
# Main config
# ---------------------------------------------------------------------------


@dataclass
class ProcessingConfig:
    """Top-level pipeline configuration, aggregates all sub-configs."""

    # Top-level fields
    model: str = "large-v3-turbo"
    workflow: str = "1"
    skip_prompts: bool = False
    translate_target: str | None = None
    burn_only: bool = False

    # Sub-configs
    input: InputConfig = field(default_factory=InputConfig)
    ai: AIConfig = field(default_factory=AIConfig)
    segment: SegmentConfig = field(default_factory=SegmentConfig)
    face: FaceConfig = field(default_factory=FaceConfig)
    subtitle: SubtitleConfig = field(default_factory=SubtitleConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    post_production: PostProductionConfig = field(default_factory=PostProductionConfig)
    advanced_ai: AdvancedAIConfig = field(default_factory=AdvancedAIConfig)
    distraction: DistractionConfig = field(default_factory=DistractionConfig)
    publish: PublishConfig = field(default_factory=PublishConfig)

    # -- Serialization -------------------------------------------------------

    def to_flat_dict(self) -> dict[str, Any]:
        """Serialize to flat dict compatible with process_runner / CLI args.

        Keys use underscores, no sub-config prefix.  ``api_key`` is excluded
        for security (pass it via environment variable instead).
        """
        d: dict[str, Any] = {}
        # Top-level
        for key in _TOP_LEVEL_FIELDS:
            d[key] = getattr(self, key)
        # Sub-configs
        for attr, cls in _SUB_CONFIGS:
            sub = getattr(self, attr)
            for f in fields(cls):
                if f.name == "api_key":
                    continue
                d[f.name] = getattr(sub, f.name)
        return d

    # -- Deserialization (class methods) -------------------------------------

    @classmethod
    def from_flat_dict(cls, d: dict[str, Any]) -> ProcessingConfig:
        """Reconstruct from a flat dict (WebUI / multiprocessing path).

        Missing keys fall back to dataclass defaults.
        """
        top_kwargs: dict[str, Any] = {}
        sub_kwargs: dict[str, dict[str, Any]] = {attr: {} for attr, _ in _SUB_CONFIGS}

        for key, value in d.items():
            if key in _TOP_LEVEL_FIELDS:
                top_kwargs[key] = value
            elif key in FLAT_KEY_MAP:
                attr, fname = FLAT_KEY_MAP[key]
                sub_kwargs[attr][fname] = value

        # api_key from environment only
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("OPENAI_API_KEY") or ""
        if api_key:
            sub_kwargs["ai"]["api_key"] = api_key

        config = cls(**top_kwargs)
        for attr, sub_cls in _SUB_CONFIGS:
            setattr(config, attr, sub_cls(**sub_kwargs[attr]))
        return config

    @classmethod
    def from_namespace(cls, ns: Any) -> ProcessingConfig:
        """Convert an argparse.Namespace to ProcessingConfig.

        Attribute names on Namespace use underscores (argparse converts
        hyphens automatically).
        """
        d: dict[str, Any] = {}
        ns_dict = vars(ns) if hasattr(ns, "__dict__") else {}
        for key, value in ns_dict.items():
            if value is not None:
                d[key] = value
        return cls.from_flat_dict(d)
