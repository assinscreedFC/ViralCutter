"""Unit tests for scripts/models.py — Segment dataclass."""
from __future__ import annotations

import dataclasses

import pytest

from scripts.models import Segment, _parse_time


# ===========================================================================
# from_dict
# ===========================================================================

class TestFromDict:
    def test_from_dict_complete(self):
        d = {
            "start_time": "00:01:30",
            "end_time": "00:02:15",
            "duration": 45.0,
            "title": "Test Segment 1",
            "description": "First test",
            "tiktok_caption": "#test",
            "zoom_cues": [],
            "power_words": ["amazing"],
            "score": 75.0,
        }
        seg = Segment.from_dict(d)
        assert seg.start_time == "00:01:30"
        assert seg.end_time == "00:02:15"
        assert seg.duration == 45.0
        assert seg.title == "Test Segment 1"
        assert seg.description == "First test"
        assert seg.tiktok_caption == "#test"
        assert seg.zoom_cues == []
        assert seg.power_words == ["amazing"]
        assert seg.score == 75.0

    def test_from_dict_minimal(self):
        """Only required fields provided; optional fields fall back to defaults."""
        d = {"start_time": "00:00:10", "end_time": "00:00:40", "duration": 30.0}
        seg = Segment.from_dict(d)
        assert seg.title == ""
        assert seg.description == ""
        assert seg.tiktok_caption == ""
        assert seg.zoom_cues == []
        assert seg.power_words == []
        assert seg.score == 0.0

    def test_from_dict_missing_keys(self):
        """Empty dict uses all defaults — must not raise."""
        seg = Segment.from_dict({})
        assert seg.start_time == "0"
        assert seg.end_time == "0"
        assert seg.duration == 0.0
        assert seg.title == ""
        assert seg.zoom_cues == []
        assert seg.score == 0.0


# ===========================================================================
# to_dict / roundtrip
# ===========================================================================

class TestToDict:
    def test_to_dict_roundtrip(self):
        """from_dict -> to_dict must reproduce the original normalised dict."""
        d = {
            "start_time": "00:01:00",
            "end_time": "00:01:45",
            "duration": 45.0,
            "title": "Roundtrip",
            "description": "desc",
            "tiktok_caption": "#rt",
            "zoom_cues": [{"time": 5.0, "scale": 1.2}],
            "power_words": ["wow"],
            "score": 90.0,
        }
        assert Segment.from_dict(d).to_dict() == d


# ===========================================================================
# start_seconds / end_seconds properties
# ===========================================================================

class TestTimeProperties:
    def test_start_seconds_hhmmss(self):
        seg = Segment.from_dict({"start_time": "01:30:45", "end_time": "0", "duration": 0})
        assert seg.start_seconds == pytest.approx(5445.0)

    def test_start_seconds_mmss(self):
        seg = Segment.from_dict({"start_time": "02:30", "end_time": "0", "duration": 0})
        assert seg.start_seconds == pytest.approx(150.0)

    def test_start_seconds_plain(self):
        seg = Segment.from_dict({"start_time": "90.5", "end_time": "0", "duration": 0})
        assert seg.start_seconds == pytest.approx(90.5)

    def test_end_seconds(self):
        seg = Segment.from_dict({"start_time": "0", "end_time": "00:05:00", "duration": 0})
        assert seg.end_seconds == pytest.approx(300.0)


# ===========================================================================
# _parse_time edge cases
# ===========================================================================

class TestParseTime:
    def test_parse_time_invalid(self):
        assert _parse_time("abc") == pytest.approx(0.0)

    def test_parse_time_empty(self):
        assert _parse_time("") == pytest.approx(0.0)


# ===========================================================================
# Immutability & isolation
# ===========================================================================

class TestImmutability:
    def test_frozen_immutable(self):
        """Assigning to any field must raise FrozenInstanceError."""
        seg = Segment.from_dict({"start_time": "0", "end_time": "0", "duration": 0})
        with pytest.raises(dataclasses.FrozenInstanceError):
            seg.title = "mutated"  # type: ignore[misc]

    def test_zoom_cues_isolated(self):
        """Mutating the list returned by to_dict must not affect the Segment."""
        original_cues = [{"time": 1.0, "scale": 1.1}]
        seg = Segment.from_dict({
            "start_time": "0", "end_time": "0", "duration": 0,
            "zoom_cues": original_cues,
        })
        exported = seg.to_dict()
        exported["zoom_cues"].append({"time": 99.0, "scale": 2.0})
        # The segment's internal list must be unchanged
        assert len(seg.zoom_cues) == 1
