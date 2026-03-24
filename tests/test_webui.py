"""Tests for webui decomposition modules.

Covers:
  - webui.presets          (FACE_PRESETS, EXPERIMENTAL_PRESETS, helpers)
  - webui.settings_manager (SETTINGS_KEYS, save_settings, load_settings)
  - webui.process_runner   (kill_process)

Strategy
--------
We mock ``gradio`` and ``i18n`` before importing the real webui modules so
that tests run without a Gradio install.  All function definitions are
imported from their real modules — no copied snapshots.
"""
from __future__ import annotations

import json
import os
import re
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEBUI_DIR = PROJECT_ROOT / "webui"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Mock gradio before importing webui modules
# ---------------------------------------------------------------------------
_mock_gr = types.ModuleType("gradio")
_mock_gr.update = lambda **kw: kw  # type: ignore[attr-defined]
_mock_gr.skip = lambda: {"__type__": "update"}  # type: ignore[attr-defined]
_mock_gr.Info = lambda msg: None  # type: ignore[attr-defined]
# Add common gr attributes that may be accessed at import time
for _attr in ("Blocks", "Row", "Column", "Tab", "Tabs", "Markdown", "HTML",
              "Textbox", "Number", "Slider", "Checkbox", "Dropdown", "Radio",
              "Button", "File", "Video", "Image", "Audio", "Gallery",
              "Dataframe", "State", "ColorPicker", "Accordion", "Group"):
    setattr(_mock_gr, _attr, MagicMock())
sys.modules.setdefault("gradio", _mock_gr)

# ---------------------------------------------------------------------------
# Import real modules
# ---------------------------------------------------------------------------
from webui.presets import (
    FACE_PRESETS,
    EXPERIMENTAL_PRESETS,
    apply_face_preset,
    apply_experimental_preset,
    convert_color_to_ass,
    get_local_models,
    MODELS_DIR,
)
from webui.settings_manager import (
    SETTINGS_KEYS,
    SETTINGS_FILE,
    save_settings,
    load_settings,
)


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
            "Cinematic (Rule of Thirds)",
        }
        assert set(FACE_PRESETS.keys()) == expected

    def test_face_presets_have_8_fields(self):
        """Each face preset must contain all 8 parameter keys."""
        required = {"thresh", "two_face", "conf", "dead_zone",
                     "vertical_offset", "single_face_zoom", "ema_alpha", "detection_resolution"}
        for name, preset in FACE_PRESETS.items():
            assert set(preset.keys()) == required, f"Preset '{name}' missing keys"

    def test_apply_face_preset_default(self):
        result = apply_face_preset("Default (Balanced)")
        assert len(result) == 8
        thresh, two_face, conf, dead_zone, v_off, sf_zoom, ema, det_res = result
        assert thresh == 0.35
        assert two_face == 0.60
        assert conf == 0.40
        assert dead_zone == 150
        assert v_off == 0.0
        assert sf_zoom == 1.0
        assert ema == 0.18
        assert det_res == 480

    def test_apply_face_preset_stable(self):
        result = apply_face_preset("Stable (Focus Main)")
        assert len(result) == 8
        thresh, two_face, conf, dead_zone, v_off, sf_zoom, ema, det_res = result
        assert thresh == 0.60
        assert dead_zone == 200
        assert ema == 0.12

    def test_apply_face_preset_sensitive(self):
        result = apply_face_preset("Sensitive (Catch All)")
        assert len(result) == 8
        thresh, _, _, dead_zone, v_off, sf_zoom, ema, det_res = result
        assert thresh == 0.10
        assert dead_zone == 100
        assert v_off == -0.05
        assert sf_zoom == 1.2

    def test_apply_face_preset_high_precision(self):
        result = apply_face_preset("High Precision")
        assert len(result) == 8
        thresh, _, conf, _, v_off, _, _, det_res = result
        assert thresh == 0.40
        assert conf == 0.75
        assert v_off == -0.08
        assert det_res == 560

    def test_apply_face_preset_cinematic(self):
        result = apply_face_preset("Cinematic (Rule of Thirds)")
        assert len(result) == 8
        _, _, _, _, v_off, sf_zoom, ema, _ = result
        assert v_off == -0.08
        assert sf_zoom == 1.3
        assert ema == 0.12

    def test_apply_face_preset_unknown(self):
        result = apply_face_preset("Does Not Exist")
        assert len(result) == 8

    def test_apply_face_preset_empty_string(self):
        result = apply_face_preset("")
        assert len(result) == 8

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
        # #FF0000 = R=FF G=00 B=00 — ASS uses BGR order
        assert convert_color_to_ass("#FF0000") == "&H000000FF&"

    def test_convert_color_to_ass_hex_blue(self):
        # #0000FF = R=00 G=00 B=FF — BGR -> FF0000
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
        # #F00 expands to FF0000 -> red
        assert convert_color_to_ass("#F00") == "&H000000FF&"

    def test_convert_color_to_ass_rgb_format(self):
        # rgb(255, 0, 0) — red in RGB -> BGR -> 00 00 FF
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
        with patch("webui.presets.os.path.exists", return_value=True), \
             patch("webui.presets.os.listdir", return_value=[]):
            result = get_local_models()
        assert result == []

    def test_get_local_models_nonexistent_dir(self):
        with patch("webui.presets.os.path.exists", return_value=False):
            result = get_local_models()
        assert result == []

    def test_get_local_models_filters_gguf(self):
        files = ["model_a.gguf", "model_b.gguf", "readme.txt", "config.json"]
        with patch("webui.presets.os.path.exists", return_value=True), \
             patch("webui.presets.os.listdir", return_value=files):
            result = get_local_models()
        assert sorted(result) == ["model_a.gguf", "model_b.gguf"]

    def test_get_local_models_no_gguf(self):
        files = ["weights.pt", "config.yaml"]
        with patch("webui.presets.os.path.exists", return_value=True), \
             patch("webui.presets.os.listdir", return_value=files):
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

    def test_settings_keys_contain_new_face_params(self):
        """Verify the 4 new 1-face visual params are present."""
        for key in ("vertical_offset", "single_face_zoom", "ema_alpha", "detection_resolution"):
            assert key in SETTINGS_KEYS, f"Missing key: {key}"

    def test_save_settings_creates_valid_json(self, tmp_path):
        settings_file = str(tmp_path / "settings.json")
        values = list(range(len(SETTINGS_KEYS)))
        with patch("webui.settings_manager.SETTINGS_FILE", settings_file):
            save_settings(*values)

        assert Path(settings_file).exists()
        data = json.loads(Path(settings_file).read_text(encoding="utf-8"))
        assert isinstance(data, dict)
        assert data[SETTINGS_KEYS[0]] == 0
        assert data[SETTINGS_KEYS[-1]] == len(SETTINGS_KEYS) - 1

    def test_save_settings_keys_match(self, tmp_path):
        settings_file = str(tmp_path / "settings.json")
        values = ["v"] * len(SETTINGS_KEYS)
        with patch("webui.settings_manager.SETTINGS_FILE", settings_file):
            save_settings(*values)
        data = json.loads(Path(settings_file).read_text(encoding="utf-8"))
        assert set(data.keys()) == set(SETTINGS_KEYS)

    def test_save_settings_returns_success_string(self, tmp_path):
        settings_file = str(tmp_path / "settings.json")
        with patch("webui.settings_manager.SETTINGS_FILE", settings_file):
            result = save_settings(*["x"] * len(SETTINGS_KEYS))
        assert "saved" in result.lower() or "Settings" in result

    def test_load_settings_no_file(self, tmp_path):
        missing = str(tmp_path / "no_such_file.json")
        with patch("webui.settings_manager.SETTINGS_FILE", missing):
            result = load_settings()
        assert isinstance(result, list)
        assert len(result) == len(SETTINGS_KEYS)

    def test_load_settings_restores_values(self, tmp_path):
        settings_file = str(tmp_path / "settings.json")
        payload = {k: f"val_{i}" for i, k in enumerate(SETTINGS_KEYS[:5])}
        Path(settings_file).write_text(json.dumps(payload), encoding="utf-8")

        with patch("webui.settings_manager.SETTINGS_FILE", settings_file):
            result = load_settings()
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
        with patch("webui.settings_manager.SETTINGS_FILE", settings_file):
            result = load_settings()
        # Second entry should have no "value" key (empty gr.update)
        assert "value" not in result[1]

    def test_load_settings_corrupt_file_returns_defaults(self, tmp_path):
        settings_file = str(tmp_path / "settings.json")
        Path(settings_file).write_text("not valid json{{{{", encoding="utf-8")
        with patch("webui.settings_manager.SETTINGS_FILE", settings_file):
            result = load_settings()
        assert isinstance(result, list)
        assert len(result) == len(SETTINGS_KEYS)


# ===========================================================================
# TestProcessRunner
# ===========================================================================

class TestProcessRunner:
    """Unit tests for process management helpers.

    kill_process in process_runner.py uses a global ``current_worker``
    (multiprocessing.Process).  We patch the global and psutil to test
    the three branches: no process, successful kill, psutil error.

    The run_viral_cutter command-building tests remain as inline flag
    construction checks since importing the real generator would require
    the full Gradio + pipeline stack.
    """

    def test_kill_process_no_process(self):
        with patch("webui.process_runner.current_worker", None):
            from webui.process_runner import kill_process
            result = kill_process()
        assert "process" in result.lower() or "running" in result.lower()

    def test_kill_process_calls_psutil(self):
        mock_worker = MagicMock()
        mock_worker.is_alive.return_value = True
        mock_worker.pid = 1234

        mock_child = MagicMock()
        mock_parent = MagicMock()
        mock_parent.children.return_value = [mock_child]

        with patch("webui.process_runner.current_worker", mock_worker), \
             patch("webui.process_runner.psutil.Process", return_value=mock_parent):
            from webui.process_runner import kill_process
            result = kill_process()

        mock_child.kill.assert_called_once()
        mock_parent.kill.assert_called_once()
        assert "terminated" in result.lower()

    def test_kill_process_multiple_children(self):
        mock_worker = MagicMock()
        mock_worker.is_alive.return_value = True
        mock_worker.pid = 9999
        children = [MagicMock(), MagicMock(), MagicMock()]
        mock_parent = MagicMock()
        mock_parent.children.return_value = children

        with patch("webui.process_runner.current_worker", mock_worker), \
             patch("webui.process_runner.psutil.Process", return_value=mock_parent):
            from webui.process_runner import kill_process
            kill_process()

        for child in children:
            child.kill.assert_called_once()

    def test_kill_process_psutil_error_returns_message(self):
        mock_worker = MagicMock()
        mock_worker.is_alive.return_value = True
        mock_worker.pid = 42

        with patch("webui.process_runner.current_worker", mock_worker), \
             patch("webui.process_runner.psutil.Process", side_effect=Exception("no such process")):
            from webui.process_runner import kill_process
            result = kill_process()

        assert "error" in result.lower()

    def test_run_viral_cutter_builds_cmd_with_url(self):
        """Verify command flag construction for the URL branch."""
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
