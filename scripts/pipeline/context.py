from __future__ import annotations

from dataclasses import dataclass, field
from argparse import Namespace
from typing import Optional

from scripts.pipeline.config import ProcessingConfig


@dataclass
class PipelineContext:
    """Mutable state shared across pipeline stages."""
    args: Namespace
    config: ProcessingConfig | None = None
    project_folder: Optional[str] = None
    input_video: Optional[str] = None
    viral_segments: Optional[dict] = None
    sub_config: Optional[dict] = None
    workflow_choice: str = "1"
    url: Optional[str] = None
    ai_backend: str = "manual"
    api_key: str = ""
    num_segments: Optional[int] = None
    viral_mode: bool = False
    themes: str = ""
    content_type_arg: Optional[list] = None
    face_model: str = "insightface"
    face_mode: str = "auto"
    detection_intervals: Optional[dict] = None
    api_config: dict = field(default_factory=dict)

    @property
    def cfg(self) -> ProcessingConfig:
        """Typed config accessor — builds from Namespace on first access."""
        if self.config is not None:
            return self.config
        if self.args is not None:
            self.config = ProcessingConfig.from_namespace(self.args)
            return self.config
        raise ValueError("No config available: neither config nor args is set")
