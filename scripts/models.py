from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Segment:
    """Represents a viral clip segment extracted by the LLM."""
    start_time: str          # "HH:MM:SS" or seconds as string
    end_time: str            # "HH:MM:SS" or seconds as string
    duration: float          # seconds
    title: str = ""
    description: str = ""
    tiktok_caption: str = ""
    zoom_cues: list[dict] = field(default_factory=list)
    power_words: list[str] = field(default_factory=list)
    caption_variants: list[str] = field(default_factory=list)
    score: float = 0.0

    @property
    def start_seconds(self) -> float:
        """Convert start_time to seconds."""
        return _parse_time(self.start_time)

    @property
    def end_seconds(self) -> float:
        """Convert end_time to seconds."""
        return _parse_time(self.end_time)

    @classmethod
    def from_dict(cls, d: dict) -> Segment:
        """Create a Segment from a raw LLM response dict."""
        return cls(
            start_time=str(d.get("start_time", "0")),
            end_time=str(d.get("end_time", "0")),
            duration=float(d.get("duration", 0)),
            title=d.get("title", ""),
            description=d.get("description", ""),
            tiktok_caption=d.get("tiktok_caption", ""),
            zoom_cues=d.get("zoom_cues", []),
            power_words=d.get("power_words", []),
            caption_variants=d.get("caption_variants", []),
            score=float(d.get("score", 0)),
        )

    def to_dict(self) -> dict:
        """Serialize back to dict for backward compatibility."""
        return {
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "title": self.title,
            "description": self.description,
            "tiktok_caption": self.tiktok_caption,
            "zoom_cues": list(self.zoom_cues),
            "power_words": list(self.power_words),
            "caption_variants": list(self.caption_variants),
            "score": self.score,
        }


def _parse_time(t: str) -> float:
    """Parse 'HH:MM:SS', 'MM:SS', or plain seconds string to float seconds."""
    t = t.strip()
    parts = t.split(":")
    try:
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        elif len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
        else:
            return float(t)
    except (ValueError, IndexError):
        return 0.0
