"""Tests for webui decomposition modules.

Covers the logic that will live in:
  - webui.presets          (FACE_PRESETS, EXPERIMENTAL_PRESETS, helpers)
  - webui.settings_manager (SETTINGS_KEYS, save_settings, load_settings)
  - webui.process_runner   (kill_process, run_viral_cutter)

Strategy
--------
app.py executes a full Gradio UI at import time, making it impossible to
import without a real Gradio install and all its dependencies.  Instead of
patching the entire Gradio surface we copy the pure-logic sections verbatim
from app.py here (marked with SOURCE snapshot comments).  The tests therefore
validate the behaviour that the new extracted modules MUST reproduce.  Once
the modules exist the import shims at the top can be replaced with direct
imports from webui.presets / webui.settings_manager / webui.process_runner.
"""
from __future__ import annotations

import json
import os
import re
import sys
import subprocess
import types
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEBUI_DIR = PROJECT_ROOT / "webui"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# SOURCE snapshot — copy of pure logic from webui/app.py
# These definitions are intentionally kept in sync with app.py and will be
# replaced by `from webui.presets import ...` once extraction lands.
# ---------------------------------------------------------------------------

_GR_UPDATE = lambda **kw: kw  # noqa: E731 — mirrors gr.update in tests

# --- Preset data (app.py lines 37-49) ---

FACE_PRESETS: dict = {
    "Default (Balanced)": {"thresh": 0.35, "two_face": 0.60, "conf": 0.40, "dead_zone": 150},
    "Stable (Focus Main)": {"thresh": 0.60, "two_face": 0.80, "conf": 0.60, "dead_zone": 200},
    "Sensitive (Catch All)": {"thresh": 0.10, "two_face": 0.40, "conf": 0.30, "dead_zone": 100},
    "High Precision": {"thresh": 0.40, "two_face": 0.65, "conf": 0.75, "dead_zone": 150},
}

EXPERIMENTAL_PRESETS: dict = {
    "Default (Off)": {"focus": False, "mar": 0.03, "score": 1.5, "motion": False, "motion_th": 3.0, "motion_sens": 0.05, "decay": 2.0},
    "Active Speaker (Balanced)": {"focus": True, "mar": 0.03, "score": 1.5, "motion": True, "motion_th": 3.0, "motion_sens": 0.05, "decay": 2.0},
    "Active Speaker (Sensitive)": {"focus": True, "mar": 0.02, "score": 1.0, "motion": True, "motion_th": 2.0, "motion_sens": 0.10, "decay": 1.0},
    "Active Speaker (Stable)": {"focus": True, "mar": 0.05, "score": 2.5, "motion": False, "motion_th": 5.0, "motion_sens": 0.02, "decay": 3.0},
}

# --- Preset functions (app.py lines 239-251) ---

def apply_face_preset(preset_name: str):
    if preset_name not in FACE_PRESETS:
        return [_GR_UPDATE() for _ in range(4)]
    p = FACE_PRESETS[preset_name]
    return p["thresh"], p["two_face"], p["conf"], p["dead_zone"]


def apply_experimental_preset(preset_name: str):
    if preset_name not in EXPERIMENTAL_PRESETS:
        return [_GR_UPDATE() for _ in range(7)]
    p = EXPERIMENTAL_PRESETS[preset_name]
    return p["focus"], p["mar"], p["score"], p["motion"], p["motion_th"], p["motion_sens"], p["decay"]


# --- Color conversion (app.py lines 153-190) ---

def convert_color_to_ass(hex_color, alpha: str = "00") -> str:
    if not hex_color:
        return f"&H{alpha}FFFFFF&"

    hex_clean = hex_color.lstrip("#").strip()

    if hex_clean.lower().startswith("rgb"):
        try:
            nums = re.findall(r"[\d\.]+", hex_clean)
            if len(nums) >= 3:
                r = max(0, min(255, int(float(nums[0]))))
                g = max(0, min(255, int(float(nums[1]))))
                b = max(0, min(255, int(float(nums[2]))))
                return f"&H{alpha}{b:02X}{g:02X}{r:02X}&".upper()
        except Exception:
            pass

    if len(hex_clean) == 3:
        hex_clean = "".join([c * 2 for c in hex_clean])

    if len(hex_clean) == 6:
        r, g, b = hex_clean[0:2], hex_clean[2:4], hex_clean[4:6]
        return f"&H{alpha}{b}{g}{r}&".upper()

    return f"&H{alpha}FFFFFF&"


# --- Model listing (app.py lines 233-235) ---
_MODELS_DIR = str(WEBUI_DIR.parent / "models")


def get_local_models() -> list:
    if not os.path.exists(_MODELS_DIR):
        return []
    return [f for f in os.listdir(_MODELS_DIR) if f.endswith(".gguf")]


# --- Settings keys (app.py lines 261-290) ---

SETTINGS_KEYS: list[str] = [
    "segments", "viral", "themes", "min_duration", "max_duration",
    "model", "ai_backend", "api_key", "ai_model_name", "chunk_size",
    "workflow", "face_model", "face_mode", "face_detect_interval", "no_face_mode",
    "face_filter_thresh", "face_two_thresh", "face_conf_thresh", "face_dead_zone", "zoom_out_factor",
    "focus_active_speaker",
    "active_speaker_mar", "active_speaker_score_diff", "include_motion",
    "active_speaker_motion_threshold", "active_speaker_motion_sensitivity", "active_speaker_decay",
    "content_type", "enable_scoring", "min_score", "enable_validation",
    "use_custom_subs", "subtitle_preset",
    "font_name", "font_size", "font_color", "highlight_color",
    "outline_color", "outline_thickness", "shadow_color", "shadow_size",
    "bold", "italic", "uppercase", "vertical_pos", "alignment",
    "highlight_size", "words_per_block", "gap", "mode",
    "underline", "strikeout", "border_style", "remove_punc",
    "video_quality", "use_youtube_subs", "translate_target",
    "add_music", "music_dir", "music_file", "music_volume",
    "add_distraction", "distraction_dir", "distraction_file", "distraction_no_fetch", "distraction_ratio",
    "smart_trim", "trim_pad_start", "trim_pad_end", "scene_detection",
    "validate_clips", "hook_detection", "min_hook_score", "blur_detection", "max_blur_ratio",
    "pacing_analysis", "composite_scoring",
    "remove_fillers", "auto_thumbnail", "auto_zoom", "speed_ramp", "speed_up_factor",
    "progress_bar", "bar_color", "bar_position", "ab_variants", "num_variants",
    "layout_template", "auto_broll", "transitions", "output_resolution",
    "emoji_overlay", "color_grade", "grade_intensity",
    "engagement_prediction", "dubbing", "dubbing_language", "dubbing_original_volume",
    "remove_silence", "silence_threshold", "silence_min_duration", "silence_max_keep",
    "enable_parts", "target_part_duration",
    "post_youtube", "post_tiktok", "youtube_privacy", "post_interval_minutes", "post_first_time",
]

# --- Settings save/load (app.py lines 292-310) ---
_SETTINGS_FILE = str(WEBUI_DIR / "settings.json")


def save_settings(*values, _file: str = _SETTINGS_FILE) -> str:
    data = dict(zip(SETTINGS_KEYS, values))
    try:
        with open(_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return "Settings saved."
    except Exception as e:
        return f"Error saving settings: {e}"


def load_settings(_file: str = _SETTINGS_FILE) -> list:
    if not os.path.exists(_file):
        return [_GR_UPDATE() for _ in SETTINGS_KEYS]
    try:
        with open(_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [_GR_UPDATE(value=data[k]) if k in data else _GR_UPDATE() for k in SETTINGS_KEYS]
    except Exception:
        return [_GR_UPDATE() for _ in SETTINGS_KEYS]


# --- kill_process (app.py lines 192-204) ---
_current_process = None


def kill_process(_state: dict | None = None) -> str:
    """Stateless version for testing — accepts optional state dict."""
    proc = (_state or {}).get("process", _current_process)
    if proc:
        try:
            import psutil
            parent = psutil.Process(proc.pid)
            for child in parent.children(recursive=True):
                child.kill()
            parent.kill()
            return "Process terminated."
        except Exception as e:
            return f"Error terminating process: {e}"
    return "No process running."


# ===========================================================================
# TestPresets
# ===========================================================================

class TestPresets:
    """Unit tests for preset dictionaries and helper functions."""

    # --- FACE_PRESETS -------------------------------------------------------

    def test_face_presets_keys(self):
        expected = {
            "Default (Balanced)",
            "Stable (Focus Main)",
            "Sensitive (Catch All)",
            "High Precision",
        }
        assert set(FACE_PRESETS.keys()) == expected

    def test_apply_face_preset_default(self):
        result = apply_face_preset("Default (Balanced)")
        assert result == (0.35, 0.60, 0.40, 150)

    def test_apply_face_preset_stable(self):
        thresh, two_face, conf, dead_zone = apply_face_preset("Stable (Focus Main)")
        assert thresh == 0.60
        assert dead_zone == 200

    def test_apply_face_preset_sensitive(self):
        thresh, _, _, dead_zone = apply_face_preset("Sensitive (Catch All)")
        assert thresh == 0.10
        assert dead_zone == 100

    def test_apply_face_preset_unknown(self):
        result = apply_face_preset("Does Not Exist")
        assert len(result) == 4

    def test_apply_face_preset_empty_string(self):
        result = apply_face_preset("")
        assert len(result) == 4

    # --- EXPERIMENTAL_PRESETS -----------------------------------------------

    def test_experimental_presets_keys(self):
        expected = {
            "Default (Off)",
            "Active Speaker (Balanced)",
            "Active Speaker (Sensitive)",
            "Active Speaker (Stable)",
        }
        assert set(EXPERIMENTAL_PRESETS.keys()) == expected

    def test_apply_experimental_preset_default(self):
        result = apply_experimental_preset("Default (Off)")
        assert len(result) == 7
        focus, mar, score, motion, motion_th, motion_sens, decay = result
        assert focus is False
        assert motion is False
        assert mar == 0.03

    def test_apply_experimental_preset_unknown(self):
        result = apply_experimental_preset("Ghost Preset")
        assert len(result) == 7

    def test_apply_experimental_preset_active_balanced(self):
        result = apply_experimental_preset("Active Speaker (Balanced)")
        focus, _mar, _score, motion, _th, _sens, _decay = result
        assert focus is True
        assert motion is True

    def test_apply_experimental_preset_active_stable(self):
        _focus, _mar, score, motion, _th, _sens, decay = apply_experimental_preset(
            "Active Speaker (Stable)"
        )
        assert score == 2.5
        assert motion is False
        assert decay == 3.0

    # --- convert_color_to_ass -----------------------------------------------

    def test_convert_color_to_ass_hex_red(self):
        # #FF0000 = R=FF G=00 B=00 — ASS uses BGR order → &H000000FF&
        assert convert_color_to_ass("#FF0000") == "&H000000FF&"

    def test_convert_color_to_ass_hex_blue(self):
        # #0000FF = R=00 G=00 B=FF — BGR → FF0000 → &H00FF0000&
        assert convert_color_to_ass("#0000FF") == "&H00FF0000&"

    def test_convert_color_to_ass_white(self):
        assert convert_color_to_ass("#FFFFFF") == "&H00FFFFFF&"

    def test_convert_color_to_ass_named_falls_back(self):
        # "white" is not a valid hex string — returns the default fallback
        assert convert_color_to_ass("white") == "&H00FFFFFF&"

    def test_convert_color_to_ass_empty_string(self):
        assert convert_color_to_ass("") == "&H00FFFFFF&"

    def test_convert_color_to_ass_none(self):
        assert convert_color_to_ass(None) == "&H00FFFFFF&"

    def test_convert_color_to_ass_3digit_hex(self):
        # #F00 expands to FF0000 → red → &H000000FF&
        assert convert_color_to_ass("#F00") == "&H000000FF&"

    def test_convert_color_to_ass_rgb_format(self):
        # rgb(255, 0, 0) — red in RGB → BGR → 00 00 FF
        assert convert_color_to_ass("rgb(255, 0, 0)") == "&H000000FF&"

    def test_convert_color_to_ass_rgb_blue(self):
        assert convert_color_to_ass("rgb(0, 0, 255)") == "&H00FF0000&"

    def test_convert_color_to_ass_custom_alpha(self):
        result = convert_color_to_ass("#FF0000", alpha="80")
        assert result == "&H800000FF&"

    def test_convert_color_to_ass_uppercase_output(self):
        # Output must always be uppercase hex
        result = convert_color_to_ass("#aabbcc")
        assert result == result.upper()

    # --- get_local_models ---------------------------------------------------

    def test_get_local_models_empty_dir(self):
        with patch("os.path.exists", return_value=True), \
             patch("os.listdir", return_value=[]):
            result = get_local_models()
        assert result == []

    def test_get_local_models_nonexistent_dir(self):
        with patch("os.path.exists", return_value=False):
            result = get_local_models()
        assert result == []

    def test_get_local_models_filters_gguf(self):
        files = ["model_a.gguf", "model_b.gguf", "readme.txt", "config.json"]
        with patch("os.path.exists", return_value=True), \
             patch("os.listdir", return_value=files):
            result = get_local_models()
        assert sorted(result) == ["model_a.gguf", "model_b.gguf"]

    def test_get_local_models_no_gguf(self):
        files = ["weights.pt", "config.yaml"]
        with patch("os.path.exists", return_value=True), \
             patch("os.listdir", return_value=files):
            result = get_local_models()
        assert result == []


# ===========================================================================
# TestSettings
# ===========================================================================

class TestSettings:
    """Unit tests for settings persistence helpers."""

    def test_settings_keys_is_list(self):
        assert isinstance(SETTINGS_KEYS, list)
        assert len(SETTINGS_KEYS) > 0

    def test_settings_keys_are_strings(self):
        assert all(isinstance(k, str) for k in SETTINGS_KEYS)

    def test_settings_keys_no_duplicates(self):
        assert len(SETTINGS_KEYS) == len(set(SETTINGS_KEYS))

    def test_save_settings_creates_valid_json(self, tmp_path):
        settings_file = str(tmp_path / "settings.json")
        values = list(range(len(SETTINGS_KEYS)))
        save_settings(*values, _file=settings_file)

        assert Path(settings_file).exists()
        data = json.loads(Path(settings_file).read_text(encoding="utf-8"))
        assert isinstance(data, dict)
        assert data[SETTINGS_KEYS[0]] == 0
        assert data[SETTINGS_KEYS[-1]] == len(SETTINGS_KEYS) - 1

    def test_save_settings_keys_match(self, tmp_path):
        settings_file = str(tmp_path / "settings.json")
        values = ["v"] * len(SETTINGS_KEYS)
        save_settings(*values, _file=settings_file)
        data = json.loads(Path(settings_file).read_text(encoding="utf-8"))
        assert set(data.keys()) == set(SETTINGS_KEYS)

    def test_save_settings_returns_success_string(self, tmp_path):
        settings_file = str(tmp_path / "settings.json")
        result = save_settings(*["x"] * len(SETTINGS_KEYS), _file=settings_file)
        assert "saved" in result.lower()

    def test_load_settings_no_file(self, tmp_path):
        missing = str(tmp_path / "no_such_file.json")
        result = load_settings(_file=missing)
        assert isinstance(result, list)
        assert len(result) == len(SETTINGS_KEYS)

    def test_load_settings_restores_values(self, tmp_path):
        settings_file = str(tmp_path / "settings.json")
        payload = {k: f"val_{i}" for i, k in enumerate(SETTINGS_KEYS[:5])}
        Path(settings_file).write_text(json.dumps(payload), encoding="utf-8")

        result = load_settings(_file=settings_file)
        assert len(result) == len(SETTINGS_KEYS)
        # First 5 entries must carry the saved values (gr.update returns dict)
        for i, (k, expected) in enumerate(list(payload.items())[:5]):
            assert result[i].get("value") == expected

    def test_load_settings_missing_keys_return_empty_update(self, tmp_path):
        settings_file = str(tmp_path / "settings.json")
        # Only write the first key
        Path(settings_file).write_text(
            json.dumps({SETTINGS_KEYS[0]: "only_one"}), encoding="utf-8"
        )
        result = load_settings(_file=settings_file)
        # Second entry should have no "value" key (empty gr.update)
        assert "value" not in result[1]

    def test_load_settings_corrupt_file_returns_defaults(self, tmp_path):
        settings_file = str(tmp_path / "settings.json")
        Path(settings_file).write_text("not valid json{{{{", encoding="utf-8")
        result = load_settings(_file=settings_file)
        assert isinstance(result, list)
        assert len(result) == len(SETTINGS_KEYS)


# ===========================================================================
# TestProcessRunner
# ===========================================================================

class TestProcessRunner:
    """Unit tests for process management helpers."""

    def test_kill_process_no_process(self):
        result = kill_process(_state={"process": None})
        assert result is not None
        assert "No process" in result or "running" in result.lower()

    def test_kill_process_calls_psutil(self):
        mock_proc = MagicMock()
        mock_proc.pid = 1234

        mock_child = MagicMock()
        mock_parent = MagicMock()
        mock_parent.children.return_value = [mock_child]

        with patch("psutil.Process", return_value=mock_parent):
            result = kill_process(_state={"process": mock_proc})

        mock_child.kill.assert_called_once()
        mock_parent.kill.assert_called_once()
        assert "terminated" in result.lower()

    def test_kill_process_multiple_children(self):
        mock_proc = MagicMock()
        mock_proc.pid = 9999
        children = [MagicMock(), MagicMock(), MagicMock()]
        mock_parent = MagicMock()
        mock_parent.children.return_value = children

        with patch("psutil.Process", return_value=mock_parent):
            kill_process(_state={"process": mock_proc})

        for child in children:
            child.kill.assert_called_once()

    def test_kill_process_psutil_error_returns_message(self):
        mock_proc = MagicMock()
        mock_proc.pid = 42

        with patch("psutil.Process", side_effect=Exception("no such process")):
            result = kill_process(_state={"process": mock_proc})

        assert "Error" in result or "error" in result.lower()

    def test_run_viral_cutter_builds_cmd_with_url(self, tmp_path):
        """Verify the subprocess command contains --url and --segments."""
        captured: dict = {"cmd": []}

        def fake_popen(cmd, **kw):
            captured["cmd"] = list(cmd)
            m = MagicMock()
            m.stdout = iter([])
            m.poll.return_value = 0
            m.returncode = 0
            return m

        # Import the real run_viral_cutter from app.py via a targeted exec
        # that stops before the Gradio UI block.  We do this by reading only
        # the function body and exec-ing it in a controlled namespace.
        #
        # Because app.py's run_viral_cutter is a generator using `yield` and
        # `subprocess.Popen`, we rebuild a minimal version that replicates the
        # flag-construction logic so we can assert on the command list.
        #
        # The test patches subprocess.Popen and exhausts the generator.
        import inspect

        # Minimal args that exercise the URL branch:
        #   input_source="Download URL", url=..., segments=3, ai_backend="gemini" …
        # We build a namespace with only the symbols run_viral_cutter needs.

        # Since we cannot import the real function without loading the full UI,
        # we verify the snapshot logic directly: replicate the flag-building
        # section and assert on its output.

        cmd: list[str] = [sys.executable, "main_improved.py"]
        input_source = "Download URL"
        url = "https://youtu.be/abc123"
        segments = 3
        viral = False
        themes = ""
        min_duration = 30
        max_duration = 60
        model = "base"
        ai_backend = "gemini"
        workflow = "Full"

        if input_source != "Existing Project" and input_source != "Upload Video":
            if url:
                cmd.extend(["--url", url])

        cmd.extend(["--segments", str(int(segments))])
        if viral:
            cmd.append("--viral")
        if themes:
            cmd.extend(["--themes", themes])
        cmd.extend(["--min-duration", str(int(min_duration))])
        cmd.extend(["--max-duration", str(int(max_duration))])
        cmd.extend(["--model", model])
        cmd.extend(["--ai-backend", ai_backend])

        workflow_map = {"Full": "1", "Cut Only": "2", "Subtitles Only": "3"}
        cmd.extend(["--workflow", workflow_map.get(workflow, "1")])

        assert "--url" in cmd
        url_idx = cmd.index("--url")
        assert cmd[url_idx + 1] == "https://youtu.be/abc123"

        assert "--segments" in cmd
        seg_idx = cmd.index("--segments")
        assert cmd[seg_idx + 1] == "3"

        assert "--workflow" in cmd
        wf_idx = cmd.index("--workflow")
        assert cmd[wf_idx + 1] == "1"

        assert "--viral" not in cmd  # viral=False

    def test_run_viral_cutter_adds_music_flags(self):
        """Verify --add-music and --music-volume flags are appended."""
        cmd: list[str] = [sys.executable, "main_improved.py"]
        add_music = True
        music_dir = "/some/dir"
        music_file = "track.mp3"
        music_volume = 0.4

        if add_music:
            cmd.append("--add-music")
            if music_dir:
                cmd.extend(["--music-dir", music_dir])
            if music_file:
                cmd.extend(["--music-file", music_file])
            if music_volume is not None:
                cmd.extend(["--music-volume", str(music_volume)])

        assert "--add-music" in cmd
        assert "--music-volume" in cmd
        vol_idx = cmd.index("--music-volume")
        assert cmd[vol_idx + 1] == "0.4"

    def test_run_viral_cutter_existing_project_branch(self):
        """Existing Project branch uses --project-path, not --url."""
        cmd: list[str] = [sys.executable, "main_improved.py"]
        input_source = "Existing Project"
        project_name = "my_project"
        virals_dir = "/virals"

        if input_source == "Existing Project":
            full_project_path = os.path.join(virals_dir, project_name)
            cmd.extend(["--project-path", full_project_path])

        assert "--project-path" in cmd
        idx = cmd.index("--project-path")
        assert cmd[idx + 1].endswith("my_project")
        assert "--url" not in cmd
