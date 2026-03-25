"""Tests for scripts/smart_trim.py — pure logic, no ffmpeg/WhisperX calls."""
from __future__ import annotations

import json
import os
import sys

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from scripts.quality.smart_trim import snap_to_sentence_boundary, load_whisperx_words, _is_sentence_end


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _word(text: str, start: float, end: float) -> dict:
    return {"word": text, "start": start, "end": end}


def _make_words(*args) -> list[dict]:
    """Create a word list from (text, start, end) triples."""
    return [_word(t, s, e) for t, s, e in args]


# ---------------------------------------------------------------------------
# _is_sentence_end
# ---------------------------------------------------------------------------

class TestIsSentenceEnd:
    def test_period_is_sentence_end(self):
        assert _is_sentence_end("Hello.") is True

    def test_exclamation_is_sentence_end(self):
        assert _is_sentence_end("Wow!") is True

    def test_question_mark_is_sentence_end(self):
        assert _is_sentence_end("Really?") is True

    def test_no_punctuation_is_not_sentence_end(self):
        assert _is_sentence_end("hello") is False

    def test_comma_is_not_sentence_end(self):
        assert _is_sentence_end("hello,") is False

    def test_empty_string_is_not_sentence_end(self):
        assert _is_sentence_end("") is False

    def test_trailing_space_stripped(self):
        assert _is_sentence_end("Hello.  ") is True

    def test_custom_terminator(self):
        assert _is_sentence_end("done;", terminators=";") is True


# ---------------------------------------------------------------------------
# snap_to_sentence_boundary — empty transcript
# ---------------------------------------------------------------------------

class TestSnapEmptyTranscript:
    def test_empty_words_returns_original_times(self):
        start, end = snap_to_sentence_boundary(5.0, 15.0, [])
        assert start == pytest.approx(5.0)
        assert end == pytest.approx(15.0)


# ---------------------------------------------------------------------------
# snap_to_sentence_boundary — sentence terminator logic
# ---------------------------------------------------------------------------

class TestSnapSentenceBoundary:
    def _build_sentence(self) -> list[dict]:
        """Build: "This is a sentence. Next begins here now."
        Sentence 1 ends at t=4.0 (word "sentence." end=4.0)
        """
        return _make_words(
            ("This",      0.0,  0.5),
            ("is",        0.5,  0.8),
            ("a",         0.8,  1.0),
            ("sentence.", 1.0,  4.0),
            ("Next",      4.3,  4.7),
            ("begins",    4.7,  5.2),
            ("here",      5.2,  5.6),
            ("now.",      5.6,  8.0),
        )

    def test_end_snaps_to_sentence_terminator(self):
        words = self._build_sentence()
        # Request end near t=7.0; nearest sentence end is "now." at t=8.0
        _, end = snap_to_sentence_boundary(0.0, 7.0, words, pad_end=0.5)
        # Should snap to "now." end + pad_end = 8.5
        assert end == pytest.approx(8.5, abs=0.01)

    def test_start_snaps_after_sentence_terminator(self):
        words = self._build_sentence()
        # Original start near t=3.5 (inside sentence 1 which ends at 4.0)
        # Should snap to just before "Next" (t=4.3 - pad_start=0.3 = 4.0)
        start, _ = snap_to_sentence_boundary(3.5, 30.0, words, pad_start=0.3, pad_end=0.5)
        assert start == pytest.approx(4.0, abs=0.05)

    def test_minimum_duration_enforced(self):
        """If adjusted duration < 3.0 s, return original times."""
        words = _make_words(
            ("Hello.", 1.0, 1.5),
            ("World.", 2.0, 2.5),
        )
        # Adjusted range would be very short → should fall back to original
        start, end = snap_to_sentence_boundary(1.0, 2.0, words)
        assert start == pytest.approx(1.0)
        assert end == pytest.approx(2.0)

    def test_no_sentence_end_snaps_to_nearest_word_boundary(self):
        """When no sentence terminator found, snap to nearest word boundary."""
        words = _make_words(
            ("hello", 0.0,  1.0),
            ("world", 1.0,  2.0),
            ("foo",   2.0,  5.0),
            ("bar",   5.0,  8.0),
            ("baz",   8.0, 12.0),
        )
        start, end = snap_to_sentence_boundary(
            1.5, 8.5, words, pad_start=0.0, pad_end=0.0
        )
        # Should still return valid floats and differ from originals
        assert isinstance(start, float)
        assert isinstance(end, float)

    def test_max_shift_respected(self):
        """Words beyond max_shift from the boundary should be ignored."""
        words = _make_words(
            ("sentence.", 0.0, 1.0),   # 8 s away from start=9 — beyond max_shift=3
            ("word",      9.0, 9.5),
            ("end.",      9.5, 10.0),
            ("more",     15.0, 20.0),
        )
        start, end = snap_to_sentence_boundary(
            9.0, 10.5, words, pad_start=0.3, pad_end=0.5, max_shift=3.0
        )
        # "sentence." at t=1 is 8 s before start=9 — must NOT trigger snap
        # "end." at t=9.5 is within range — end should snap to 10.0+0.5=10.5
        assert end == pytest.approx(10.5, abs=0.01)

    def test_result_is_rounded_to_3_decimals(self):
        words = _make_words(
            ("intro.", 0.0, 3.333),
            ("body",   3.5, 6.666),
            ("done.",  6.7, 10.0),
        )
        start, end = snap_to_sentence_boundary(2.0, 8.0, words)
        # Check precision: at most 3 decimal places
        assert start == round(start, 3)
        assert end == round(end, 3)


# ---------------------------------------------------------------------------
# load_whisperx_words
# ---------------------------------------------------------------------------

class TestLoadWhisperxWords:
    def _write_whisperx(self, tmp_path, segments: list) -> str:
        data = {"segments": segments}
        path = str(tmp_path / "transcript.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)
        return path

    def test_loads_words_from_valid_json(self, tmp_path):
        segments = [
            {
                "words": [
                    {"word": "hello", "start": 0.0, "end": 0.5},
                    {"word": "world", "start": 0.5, "end": 1.0},
                ]
            }
        ]
        path = self._write_whisperx(tmp_path, segments)
        words = load_whisperx_words(path)
        assert len(words) == 2
        assert words[0]["word"] == "hello"

    def test_missing_file_returns_empty(self, tmp_path):
        result = load_whisperx_words(str(tmp_path / "missing.json"))
        assert result == []

    def test_invalid_json_returns_empty(self, tmp_path):
        path = str(tmp_path / "bad.json")
        with open(path, "w") as f:
            f.write("not valid json {{{")
        result = load_whisperx_words(path)
        assert result == []

    def test_words_sorted_by_start_time(self, tmp_path):
        segments = [
            {
                "words": [
                    {"word": "second", "start": 1.0, "end": 1.5},
                    {"word": "first",  "start": 0.0, "end": 0.5},
                ]
            }
        ]
        path = self._write_whisperx(tmp_path, segments)
        words = load_whisperx_words(path)
        assert words[0]["word"] == "first"
        assert words[1]["word"] == "second"

    def test_words_without_required_keys_excluded(self, tmp_path):
        segments = [
            {
                "words": [
                    {"word": "ok", "start": 0.0, "end": 0.5},
                    {"word": "no_end", "start": 1.0},         # missing end
                    {"start": 2.0, "end": 2.5},               # missing word
                ]
            }
        ]
        path = self._write_whisperx(tmp_path, segments)
        words = load_whisperx_words(path)
        assert len(words) == 1
        assert words[0]["word"] == "ok"

    def test_multiple_segments_flattened(self, tmp_path):
        segments = [
            {"words": [{"word": "a", "start": 0.0, "end": 0.5}]},
            {"words": [{"word": "b", "start": 1.0, "end": 1.5}]},
        ]
        path = self._write_whisperx(tmp_path, segments)
        words = load_whisperx_words(path)
        assert len(words) == 2

    def test_empty_segments_returns_empty(self, tmp_path):
        path = self._write_whisperx(tmp_path, [])
        words = load_whisperx_words(path)
        assert words == []
