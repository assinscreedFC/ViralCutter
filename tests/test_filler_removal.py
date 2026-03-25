"""Tests for scripts/filler_removal.py — pure logic, no ffmpeg dependency."""
from __future__ import annotations

import json
import os
import sys

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from scripts.quality.filler_removal import detect_fillers, update_subtitle_json, FILLER_WORDS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _word(text: str, start: float, end: float) -> dict:
    return {"word": text, "start": start, "end": end}


# ---------------------------------------------------------------------------
# detect_fillers
# ---------------------------------------------------------------------------

class TestDetectFillers:
    def test_detects_known_english_fillers(self):
        words = [
            _word("Hello", 0.0, 0.5),
            _word("um", 0.5, 0.8),
            _word("world", 0.8, 1.2),
            _word("uh", 1.2, 1.5),
        ]
        result = detect_fillers(words, language="en")
        detected_words = {r["word"] for r in result}
        assert "um" in detected_words
        assert "uh" in detected_words
        assert "hello" not in detected_words
        assert "world" not in detected_words

    def test_detects_known_french_fillers(self):
        words = [
            _word("euh", 0.0, 0.4),
            _word("bonjour", 0.4, 0.9),
            _word("voilà", 0.9, 1.3),
        ]
        result = detect_fillers(words, language="fr")
        detected_words = {r["word"] for r in result}
        assert "euh" in detected_words
        assert "voilà" in detected_words

    def test_auto_language_detects_all_languages(self):
        words = [
            _word("um", 0.0, 0.4),       # English
            _word("euh", 0.4, 0.8),      # French
            _word("ähm", 0.8, 1.2),      # German
        ]
        result = detect_fillers(words, language="auto")
        detected_words = {r["word"] for r in result}
        assert detected_words == {"um", "euh", "ähm"}

    def test_empty_transcript_returns_empty(self):
        result = detect_fillers([], language="en")
        assert result == []

    def test_no_fillers_returns_empty(self):
        words = [_word("Hello", 0.0, 0.5), _word("world", 0.5, 1.0)]
        result = detect_fillers(words, language="en")
        assert result == []

    def test_all_words_are_fillers(self):
        words = [
            _word("um", 0.0, 0.4),
            _word("uh", 0.4, 0.8),
            _word("like", 0.8, 1.2),
        ]
        result = detect_fillers(words, language="en")
        assert len(result) == 3

    def test_very_short_detection_skipped(self):
        """Duration < 0.1 s is skipped."""
        words = [_word("um", 0.0, 0.05)]   # 0.05 s — too short
        result = detect_fillers(words, language="en")
        assert result == []

    def test_minimum_duration_boundary(self):
        """Duration exactly 0.1 s is kept."""
        words = [_word("um", 0.0, 0.1)]
        result = detect_fillers(words, language="en")
        assert len(result) == 1

    def test_punctuation_stripped_before_matching(self):
        """Trailing punctuation must not prevent filler detection."""
        words = [_word("um,", 0.0, 0.4)]
        result = detect_fillers(words, language="en")
        assert len(result) == 1
        assert result[0]["word"] == "um"

    def test_case_insensitive_matching(self):
        words = [_word("UM", 0.0, 0.4), _word("Uh", 0.4, 0.8)]
        result = detect_fillers(words, language="en")
        assert len(result) == 2

    def test_custom_fillers_added(self):
        words = [_word("basically", 0.0, 0.5), _word("myCustomFiller", 0.5, 1.0)]
        result = detect_fillers(words, language="en", custom_fillers={"mycustomfiller"})
        detected = {r["word"] for r in result}
        assert "basically" in detected
        assert "mycustomfiller" in detected

    def test_unknown_language_no_builtin_fillers(self):
        """Unknown language code → no builtin fillers, only custom ones."""
        words = [_word("um", 0.0, 0.4)]
        result = detect_fillers(words, language="zz")
        assert result == []

    def test_result_contains_word_start_end_keys(self):
        words = [_word("um", 1.0, 1.5)]
        result = detect_fillers(words, language="en")
        assert result[0] == {"word": "um", "start": 1.0, "end": 1.5}


# ---------------------------------------------------------------------------
# update_subtitle_json
# ---------------------------------------------------------------------------

class TestUpdateSubtitleJson:
    def _write_subtitle(self, tmp_path, segments: list) -> str:
        data = {"segments": segments}
        path = str(tmp_path / "subtitles.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)
        return path

    def test_removes_filler_words_from_segment(self, tmp_path):
        segments = [
            {
                "start": 0.0,
                "end": 2.0,
                "text": "um hello world",
                "words": [
                    {"word": "um", "start": 0.0, "end": 0.3},
                    {"word": "hello", "start": 0.3, "end": 0.8},
                    {"word": "world", "start": 0.8, "end": 1.2},
                ],
            }
        ]
        json_path = self._write_subtitle(tmp_path, segments)
        out_path = str(tmp_path / "out.json")
        fillers = [{"word": "um", "start": 0.0, "end": 0.3}]

        update_subtitle_json(json_path, fillers, out_path)

        with open(out_path, encoding="utf-8") as f:
            result = json.load(f)

        words = result["segments"][0]["words"]
        assert all(w["word"] != "um" for w in words)
        assert any(w["word"] == "hello" for w in words)

    def test_rebuilds_segment_text_after_removal(self, tmp_path):
        segments = [
            {
                "start": 0.0,
                "end": 2.0,
                "text": "um hello world",
                "words": [
                    {"word": "um", "start": 0.0, "end": 0.3},
                    {"word": "hello", "start": 0.3, "end": 0.8},
                    {"word": "world", "start": 0.8, "end": 1.2},
                ],
            }
        ]
        json_path = self._write_subtitle(tmp_path, segments)
        out_path = str(tmp_path / "out.json")
        fillers = [{"word": "um", "start": 0.0, "end": 0.3}]

        update_subtitle_json(json_path, fillers, out_path)

        with open(out_path, encoding="utf-8") as f:
            result = json.load(f)

        assert "um" not in result["segments"][0]["text"]
        assert "hello" in result["segments"][0]["text"]

    def test_adjusts_timestamps_after_filler_removal(self, tmp_path):
        """Words after a removed filler should have shifted timestamps."""
        segments = [
            {
                "start": 0.0,
                "end": 3.0,
                "words": [
                    {"word": "um", "start": 0.0, "end": 0.5},
                    {"word": "hello", "start": 0.5, "end": 1.0},
                ],
            }
        ]
        json_path = self._write_subtitle(tmp_path, segments)
        out_path = str(tmp_path / "out.json")
        fillers = [{"word": "um", "start": 0.0, "end": 0.5}]

        update_subtitle_json(json_path, fillers, out_path)

        with open(out_path, encoding="utf-8") as f:
            result = json.load(f)

        # "hello" was at 0.5 — after removing 0.5 s filler it should be at 0.0
        remaining = result["segments"][0]["words"]
        assert len(remaining) == 1
        assert remaining[0]["word"] == "hello"
        assert remaining[0]["start"] == pytest.approx(0.0, abs=0.01)

    def test_no_fillers_leaves_data_unchanged(self, tmp_path):
        segments = [
            {
                "start": 1.0,
                "end": 2.0,
                "text": "hello world",
                "words": [
                    {"word": "hello", "start": 1.0, "end": 1.5},
                    {"word": "world", "start": 1.5, "end": 2.0},
                ],
            }
        ]
        json_path = self._write_subtitle(tmp_path, segments)
        out_path = str(tmp_path / "out.json")

        update_subtitle_json(json_path, [], out_path)

        with open(out_path, encoding="utf-8") as f:
            result = json.load(f)

        assert result["segments"][0]["words"][0]["start"] == pytest.approx(1.0, abs=0.001)

    def test_missing_json_path_does_nothing(self, tmp_path):
        """If input file does not exist, output file should not be created."""
        missing = str(tmp_path / "nonexistent.json")
        out_path = str(tmp_path / "out.json")
        update_subtitle_json(missing, [], out_path)
        assert not os.path.exists(out_path)

    def test_empty_segments_list(self, tmp_path):
        json_path = self._write_subtitle(tmp_path, [])
        out_path = str(tmp_path / "out.json")
        fillers = [{"word": "um", "start": 0.0, "end": 0.3}]
        update_subtitle_json(json_path, fillers, out_path)

        with open(out_path, encoding="utf-8") as f:
            result = json.load(f)
        assert result["segments"] == []

    def test_all_words_are_fillers_leaves_empty_words(self, tmp_path):
        segments = [
            {
                "start": 0.0,
                "end": 1.0,
                "text": "um uh",
                "words": [
                    {"word": "um", "start": 0.0, "end": 0.4},
                    {"word": "uh", "start": 0.4, "end": 0.8},
                ],
            }
        ]
        json_path = self._write_subtitle(tmp_path, segments)
        out_path = str(tmp_path / "out.json")
        fillers = [
            {"word": "um", "start": 0.0, "end": 0.4},
            {"word": "uh", "start": 0.4, "end": 0.8},
        ]
        update_subtitle_json(json_path, fillers, out_path)

        with open(out_path, encoding="utf-8") as f:
            result = json.load(f)
        assert result["segments"][0]["words"] == []
        assert result["segments"][0]["text"] == ""
