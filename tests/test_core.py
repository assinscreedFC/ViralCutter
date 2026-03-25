"""
Tests unitaires pour les fonctions utilitaires critiques de ViralCutter.

Aucune dependance externe requise (pas de reseau, GPU, ffmpeg).
"""

import json
import os
import sys
import tempfile
import textwrap
from unittest.mock import patch, MagicMock

import pytest

# ---------------------------------------------------------------------------
# Setup path so we can import project modules without install
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# We import the functions directly from their modules.
# Some modules have heavy imports (cv2, mediapipe) so we mock them.

# --- create_viral_segments: pure stdlib, safe to import directly -----------
from scripts.analysis.create_viral_segments import clean_json_response, clean_json_response_simple

# --- subtitle_editor: parse_timestamp (needs json/os only) ----------------
from webui.subtitle_editor import parse_timestamp


# ===========================================================================
# 1. clean_json_response — critical JSON recovery (134 lines)
# ===========================================================================
class TestCleanJsonResponse:
    """Tests pour clean_json_response() — recuperation JSON robuste depuis LLM."""

    # -- Cas nominal --------------------------------------------------------
    def test_valid_json_passthrough(self):
        """JSON valide avec segments -> retourne le dict tel quel."""
        data = {"segments": [{"title": "Intro", "start": 0, "end": 10}]}
        raw = json.dumps(data)
        result = clean_json_response(raw)
        assert result == data
        assert isinstance(result["segments"], list)

    def test_valid_json_multiple_segments(self):
        data = {"segments": [
            {"title": "A", "start": 0, "end": 5},
            {"title": "B", "start": 5, "end": 12},
        ]}
        result = clean_json_response(json.dumps(data))
        assert len(result["segments"]) == 2

    # -- Markdown code block wrapping --------------------------------------
    def test_markdown_json_block(self):
        """JSON enveloppe dans un bloc ```json ... ```."""
        inner = {"segments": [{"title": "test"}]}
        raw = f"Here is the result:\n```json\n{json.dumps(inner)}\n```\nDone."
        result = clean_json_response(raw)
        assert result["segments"] == inner["segments"]

    def test_markdown_block_no_language_tag(self):
        """Le bloc markdown peut ne pas avoir 'json' apres les backticks."""
        inner = {"segments": [{"title": "x"}]}
        # La fonction cherche "segments" d'abord, donc ca marche quand meme
        raw = f"```\n{json.dumps(inner)}\n```"
        result = clean_json_response(raw)
        assert "segments" in result

    # -- <think> blocks (DeepSeek R1 pattern) ------------------------------
    def test_think_tags_removed(self):
        """Les balises <think>...</think> sont supprimees avant parsing."""
        inner = {"segments": [{"title": "clip"}]}
        raw = f'<think>Let me analyze this...</think>\n{json.dumps(inner)}'
        result = clean_json_response(raw)
        assert result["segments"][0]["title"] == "clip"

    def test_think_tags_multiline(self):
        inner = {"segments": [{"title": "a"}]}
        raw = f'<think>\nStep 1: think\nStep 2: more think\n</think>\n{json.dumps(inner)}'
        result = clean_json_response(raw)
        assert len(result["segments"]) == 1

    # -- Extra text before/after -------------------------------------------
    def test_text_before_json(self):
        """Du texte avant le JSON ne devrait pas poser probleme."""
        data = {"segments": [{"title": "hook"}]}
        raw = f"Sure! Here is the analysis:\n\n{json.dumps(data)}\n\nI hope this helps."
        result = clean_json_response(raw)
        assert result["segments"][0]["title"] == "hook"

    def test_text_after_json(self):
        data = {"segments": [{"title": "end"}]}
        raw = f'{json.dumps(data)}\n\nLet me know if you need changes.'
        result = clean_json_response(raw)
        assert result["segments"][0]["title"] == "end"

    # -- JSON tronque (LLM coupe la reponse) -------------------------------
    def test_truncated_json_missing_closing_brackets(self):
        """JSON tronque : manque ] et } finaux. Le fragment parser doit recuperer."""
        raw = textwrap.dedent('''\
        {"segments": [
            {"title": "Segment 1", "start": "00:00", "end": "00:30"},
            {"title": "Segment 2", "start": "00:30", "end": "01:00"}
        ''')
        # Missing ]} at the end
        result = clean_json_response(raw)
        assert len(result["segments"]) >= 1  # At least partial recovery

    def test_truncated_json_mid_object(self):
        """JSON tronque en plein milieu d'un objet segment."""
        raw = textwrap.dedent('''\
        {"segments": [
            {"title": "Complete", "start": "0", "end": "30"},
            {"title": "Incom
        ''')
        result = clean_json_response(raw)
        # Should recover at least the first complete segment
        assert len(result["segments"]) >= 1
        assert result["segments"][0]["title"] == "Complete"

    # -- Entree vide / invalide --------------------------------------------
    def test_empty_string(self):
        result = clean_json_response("")
        assert result == {"segments": []}

    def test_none_input(self):
        """None est converti en str, puis retourne vide (pas de 'segments')."""
        result = clean_json_response(None)
        assert result == {"segments": []}

    def test_completely_invalid_input(self):
        """Texte sans JSON du tout."""
        result = clean_json_response("This is just a plain text response with no JSON at all.")
        assert result == {"segments": []}

    def test_json_without_segments_key(self):
        """JSON valide mais sans cle 'segments'."""
        raw = json.dumps({"data": [1, 2, 3]})
        result = clean_json_response(raw)
        assert result == {"segments": []}

    # -- Cas edge ----------------------------------------------------------
    def test_escaped_characters_in_response(self):
        """Reponse avec des escapes excessifs (\\n, \\\")."""
        inner = {"segments": [{"title": "test \"quote\""}]}
        # Simule double-escaping
        raw = json.dumps(json.dumps(inner))  # Double-encoded
        # La fonction tente de nettoyer, le resultat depend de la profondeur
        result = clean_json_response(raw)
        # Should at least not crash
        assert isinstance(result, dict)

    def test_non_string_input(self):
        """Entree numerique -> convertie en str."""
        result = clean_json_response(12345)
        assert result == {"segments": []}

    def test_segments_not_a_list(self):
        """segments est un dict au lieu d'une list -> devrait etre rejete."""
        raw = json.dumps({"segments": {"bad": True}})
        result = clean_json_response(raw)
        # La fonction verifie isinstance(obj['segments'], list)
        assert result == {"segments": []}


# ===========================================================================
# 2. clean_json_response_simple
# ===========================================================================
class TestCleanJsonResponseSimple:
    """Tests pour clean_json_response_simple() — parsing JSON simple."""

    def test_valid_json(self):
        data = {"score": 85, "reasoning": "Great hook"}
        result = clean_json_response_simple(json.dumps(data))
        assert result == data

    def test_nested_json(self):
        data = {"scores": {"hook": 90, "pacing": 80}, "total": 85}
        result = clean_json_response_simple(json.dumps(data))
        assert result["scores"]["hook"] == 90

    def test_markdown_wrapped(self):
        data = {"caption": "Check this out!"}
        raw = f"```json\n{json.dumps(data)}\n```"
        result = clean_json_response_simple(raw)
        assert result == data

    def test_think_tags_stripped(self):
        data = {"result": "ok"}
        raw = f"<think>hmm</think>{json.dumps(data)}"
        result = clean_json_response_simple(raw)
        assert result == data

    def test_text_before_json(self):
        data = {"key": "value"}
        raw = f"Here is the output: {json.dumps(data)}"
        result = clean_json_response_simple(raw)
        assert result == data

    def test_empty_input(self):
        assert clean_json_response_simple("") == {}
        assert clean_json_response_simple(None) == {}

    def test_invalid_json(self):
        result = clean_json_response_simple("no json here at all")
        assert result == {}

    def test_only_list_not_dict(self):
        """Si le JSON est une liste (pas dict), retourne {}."""
        raw = json.dumps([1, 2, 3])
        result = clean_json_response_simple(raw)
        # raw_decode trouve un '[' pas un '{', et isinstance check dict
        assert result == {}


# ===========================================================================
# 3. parse_srt — parsing de fichiers SRT
# ===========================================================================
class TestParseSrt:
    """Tests pour parse_srt() depuis transcribe_video.py."""

    def _write_srt(self, content: str) -> str:
        """Ecrit un fichier SRT temporaire et retourne le chemin."""
        fd, path = tempfile.mkstemp(suffix=".srt")
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            f.write(content)
        return path

    def test_standard_srt(self):
        from scripts.transcription.transcribe_video import parse_srt

        content = textwrap.dedent("""\
        1
        00:00:01,000 --> 00:00:03,500
        Hello world

        2
        00:00:04,000 --> 00:00:06,200
        This is a test
        """)
        path = self._write_srt(content)
        try:
            result = parse_srt(path)
            assert result is not None
            assert len(result) == 2
            assert result[0]["text"] == "Hello world"
            assert result[0]["start"] == pytest.approx(1.0)
            assert result[0]["end"] == pytest.approx(3.5)
            assert result[1]["text"] == "This is a test"
        finally:
            os.unlink(path)

    def test_srt_with_html_tags(self):
        """Les balises HTML dans le texte SRT doivent etre supprimees."""
        from scripts.transcription.transcribe_video import parse_srt

        content = textwrap.dedent("""\
        1
        00:00:00,000 --> 00:00:02,000
        <b>Bold text</b> and <i>italic</i>
        """)
        path = self._write_srt(content)
        try:
            result = parse_srt(path)
            assert result is not None
            assert "Bold text" in result[0]["text"]
            assert "<b>" not in result[0]["text"]
        finally:
            os.unlink(path)

    def test_srt_multiline_text(self):
        """Un bloc SRT peut avoir du texte sur plusieurs lignes."""
        from scripts.transcription.transcribe_video import parse_srt

        content = textwrap.dedent("""\
        1
        00:00:00,000 --> 00:00:05,000
        Line one
        Line two
        """)
        path = self._write_srt(content)
        try:
            result = parse_srt(path)
            assert result is not None
            assert "Line one" in result[0]["text"]
            assert "Line two" in result[0]["text"]
        finally:
            os.unlink(path)

    def test_empty_srt(self):
        from scripts.transcription.transcribe_video import parse_srt
        path = self._write_srt("")
        try:
            result = parse_srt(path)
            assert result is not None
            assert len(result) == 0
        finally:
            os.unlink(path)

    def test_malformed_srt_no_timestamp(self):
        """Bloc sans timestamp -> ignore sans crash."""
        from scripts.transcription.transcribe_video import parse_srt

        content = textwrap.dedent("""\
        1
        This has no timestamp line

        2
        00:00:01,000 --> 00:00:02,000
        Valid entry
        """)
        path = self._write_srt(content)
        try:
            result = parse_srt(path)
            assert result is not None
            assert len(result) == 1
            assert result[0]["text"] == "Valid entry"
        finally:
            os.unlink(path)

    def test_srt_windows_line_endings(self):
        """SRT avec \\r\\n (CRLF) doit etre parse correctement."""
        from scripts.transcription.transcribe_video import parse_srt

        content = "1\r\n00:00:00,000 --> 00:00:01,500\r\nCRLF test\r\n\r\n"
        fd, path = tempfile.mkstemp(suffix=".srt")
        with os.fdopen(fd, 'wb') as f:
            f.write(content.encode('utf-8'))
        try:
            result = parse_srt(path)
            assert result is not None
            assert len(result) == 1
        finally:
            os.unlink(path)


# ===========================================================================
# 4. parse_vtt — parsing de fichiers WebVTT
# ===========================================================================
class TestParseVtt:
    """Tests pour parse_vtt() depuis transcribe_video.py."""

    def _write_vtt(self, content: str) -> str:
        fd, path = tempfile.mkstemp(suffix=".vtt")
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            f.write(content)
        return path

    def test_standard_vtt(self):
        from scripts.transcription.transcribe_video import parse_vtt

        content = textwrap.dedent("""\
        WEBVTT

        00:00:01.000 --> 00:00:03.500
        Hello world

        00:00:04.000 --> 00:00:06.200
        Second line
        """)
        path = self._write_vtt(content)
        try:
            result = parse_vtt(path)
            assert result is not None
            assert len(result) == 2
            assert result[0]["text"] == "Hello world"
            assert result[0]["start"] == pytest.approx(1.0)
            assert result[0]["end"] == pytest.approx(3.5)
        finally:
            os.unlink(path)

    def test_vtt_with_webvtt_header(self):
        """Le header WEBVTT doit etre ignore."""
        from scripts.transcription.transcribe_video import parse_vtt

        content = textwrap.dedent("""\
        WEBVTT
        X-TIMESTAMP-MAP=LOCAL:00:00:00.000,MPEGTS:0

        00:00:00.000 --> 00:00:02.000
        First subtitle
        """)
        path = self._write_vtt(content)
        try:
            result = parse_vtt(path)
            assert result is not None
            assert len(result) == 1
            assert result[0]["text"] == "First subtitle"
        finally:
            os.unlink(path)

    def test_vtt_with_html_tags(self):
        from scripts.transcription.transcribe_video import parse_vtt

        content = textwrap.dedent("""\
        WEBVTT

        00:00:00.000 --> 00:00:02.000
        <v Speaker>Hello</v> there
        """)
        path = self._write_vtt(content)
        try:
            result = parse_vtt(path)
            assert result is not None
            assert "<v" not in result[0]["text"]
        finally:
            os.unlink(path)

    def test_vtt_mm_ss_format(self):
        """VTT avec format MM:SS.mmm au lieu de HH:MM:SS.mmm."""
        from scripts.transcription.transcribe_video import parse_vtt

        content = textwrap.dedent("""\
        WEBVTT

        01:30.000 --> 02:00.000
        Short format
        """)
        path = self._write_vtt(content)
        try:
            result = parse_vtt(path)
            assert result is not None
            assert len(result) == 1
            assert result[0]["start"] == pytest.approx(90.0)
            assert result[0]["end"] == pytest.approx(120.0)
        finally:
            os.unlink(path)

    def test_vtt_timestamp_with_settings(self):
        """VTT avec settings apres le timestamp (position, align)."""
        from scripts.transcription.transcribe_video import parse_vtt

        content = textwrap.dedent("""\
        WEBVTT

        00:00:01.000 --> 00:00:05.000 position:10% align:start
        Positioned text
        """)
        path = self._write_vtt(content)
        try:
            result = parse_vtt(path)
            assert result is not None
            assert len(result) == 1
            assert result[0]["end"] == pytest.approx(5.0)
        finally:
            os.unlink(path)

    def test_empty_vtt(self):
        from scripts.transcription.transcribe_video import parse_vtt
        path = self._write_vtt("WEBVTT\n\n")
        try:
            result = parse_vtt(path)
            assert result is not None
            assert len(result) == 0
        finally:
            os.unlink(path)


# ===========================================================================
# 5. parse_timestamp — parsing de timestamps (subtitle_editor)
# ===========================================================================
class TestParseTimestamp:
    """Tests pour parse_timestamp() depuis webui/subtitle_editor.py."""

    def test_hhmmss_format(self):
        assert parse_timestamp("01:23:45.000") == pytest.approx(5025.0)

    def test_hhmmss_comma_separator(self):
        """Format SRT avec virgule: 01:23:45,678."""
        assert parse_timestamp("01:23:45,678") == pytest.approx(5025.678)

    def test_mmss_format(self):
        assert parse_timestamp("02:30.000") == pytest.approx(150.0)

    def test_zero_timestamp(self):
        assert parse_timestamp("00:00:00.000") == pytest.approx(0.0)

    def test_invalid_input(self):
        """Entree invalide -> retourne 0.0 sans crash."""
        assert parse_timestamp("not_a_time") == 0.0

    def test_empty_string(self):
        assert parse_timestamp("") == 0.0

    def test_single_number(self):
        """Un seul nombre sans ':' -> 0.0 (pas de split en parties)."""
        assert parse_timestamp("123") == 0.0


# ===========================================================================
# 6. get_best_encoder — detection encodeur video
# ===========================================================================
class TestGetBestEncoder:
    """Tests pour get_best_encoder() avec mock de subprocess/ffmpeg."""

    def _import_fresh(self):
        """Importe le module ffmpeg_utils et reset le cache."""
        import importlib

        if "scripts.core.ffmpeg_utils" in sys.modules:
            importlib.reload(sys.modules["scripts.core.ffmpeg_utils"])
        else:
            import scripts.core.ffmpeg_utils

        mod = sys.modules["scripts.core.ffmpeg_utils"]
        mod.CACHED_ENCODER = None  # Reset cache
        return mod

    def test_nvidia_detected(self):
        mod = self._import_fresh()
        try:
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(stdout="... h264_nvenc ... h264_amf ...")
                result = mod.get_best_encoder()
                assert result[0] == "h264_nvenc"
                assert result[1] == "p1"
        finally:
            mod.CACHED_ENCODER = None

    def test_amd_detected(self):
        mod = self._import_fresh()
        try:
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(stdout="... h264_amf ...")
                result = mod.get_best_encoder()
                assert result[0] == "h264_amf"
                assert result[1] == "balanced"
        finally:
            mod.CACHED_ENCODER = None

    def test_intel_qsv_detected(self):
        mod = self._import_fresh()
        try:
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(stdout="... h264_qsv ...")
                result = mod.get_best_encoder()
                assert result[0] == "h264_qsv"
                assert result[1] == "faster"
        finally:
            mod.CACHED_ENCODER = None

    def test_mac_videotoolbox_detected(self):
        mod = self._import_fresh()
        try:
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(stdout="... h264_videotoolbox ...")
                result = mod.get_best_encoder()
                assert result[0] == "h264_videotoolbox"
        finally:
            mod.CACHED_ENCODER = None

    def test_cpu_fallback(self):
        mod = self._import_fresh()
        try:
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(stdout="... libx264 ...")
                result = mod.get_best_encoder()
                assert result[0] == "libx264"
                assert result[1] == "fast"
        finally:
            mod.CACHED_ENCODER = None

    def test_ffmpeg_not_found(self):
        mod = self._import_fresh()
        try:
            with patch("subprocess.run", side_effect=FileNotFoundError("ffmpeg not found")):
                result = mod.get_best_encoder()
                assert result[0] == "libx264"  # Fallback CPU
        finally:
            mod.CACHED_ENCODER = None

    def test_encoder_cached(self):
        """Apres le premier appel, le cache est utilise."""
        mod = self._import_fresh()
        try:
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(stdout="... h264_nvenc ...")
                result1 = mod.get_best_encoder()
                result2 = mod.get_best_encoder()
                assert result1 == result2
                # subprocess.run appele une seule fois grace au cache
                mock_run.assert_called_once()
        finally:
            mod.CACHED_ENCODER = None


# ===========================================================================
# 6b. build_quality_params + create_ffmpeg_pipe
# ===========================================================================
class TestBuildQualityParams:
    """Tests pour build_quality_params() — flags qualite par encodeur."""

    def test_nvenc_params(self):
        from scripts.core.ffmpeg_utils import build_quality_params
        params = build_quality_params("h264_nvenc")
        assert "-rc:v" in params
        assert "-cq" in params
        assert "19" in params
        assert "-bufsize" in params

    def test_amf_params(self):
        from scripts.core.ffmpeg_utils import build_quality_params
        params = build_quality_params("h264_amf")
        assert "-rc" in params
        assert "vbr_peak" in params
        assert "-qp_i" in params

    def test_qsv_params(self):
        from scripts.core.ffmpeg_utils import build_quality_params
        params = build_quality_params("h264_qsv")
        assert "-global_quality" in params

    def test_videotoolbox_params(self):
        from scripts.core.ffmpeg_utils import build_quality_params
        params = build_quality_params("h264_videotoolbox")
        assert "-q:v" in params
        assert "65" in params

    def test_cpu_params(self):
        from scripts.core.ffmpeg_utils import build_quality_params
        params = build_quality_params("libx264")
        assert "-crf" in params
        assert "18" in params

    def test_unknown_encoder_uses_cpu_defaults(self):
        from scripts.core.ffmpeg_utils import build_quality_params
        params = build_quality_params("some_unknown_encoder")
        assert "-crf" in params


class TestBuildPresetFlags:
    """Tests pour _build_preset_flags() — flags preset adaptes par encodeur."""

    def test_nvenc_uses_preset(self):
        from scripts.core.ffmpeg_utils import _build_preset_flags
        flags = _build_preset_flags("h264_nvenc", "p1")
        assert flags == ["-preset", "p1"]

    def test_amf_uses_quality(self):
        from scripts.core.ffmpeg_utils import _build_preset_flags
        flags = _build_preset_flags("h264_amf", "balanced")
        assert flags == ["-quality", "balanced"]

    def test_videotoolbox_empty(self):
        from scripts.core.ffmpeg_utils import _build_preset_flags
        flags = _build_preset_flags("h264_videotoolbox", "default")
        assert flags == []

    def test_qsv_uses_preset(self):
        from scripts.core.ffmpeg_utils import _build_preset_flags
        flags = _build_preset_flags("h264_qsv", "faster")
        assert flags == ["-preset", "faster"]

    def test_libx264_uses_preset(self):
        from scripts.core.ffmpeg_utils import _build_preset_flags
        flags = _build_preset_flags("libx264", "fast")
        assert flags == ["-preset", "fast"]


class TestCreateFfmpegPipe:
    """Tests pour create_ffmpeg_pipe() — verifie la commande FFmpeg generee."""

    def test_pipe_creates_popen(self):
        """create_ffmpeg_pipe doit appeler subprocess.Popen avec les bons args."""
        import scripts.core.ffmpeg_utils as mod
        old_cache = mod.CACHED_ENCODER
        mod.CACHED_ENCODER = ("libx264", "fast")
        try:
            with patch("scripts.core.ffmpeg_utils.subprocess.Popen") as mock_popen:
                mock_popen.return_value = MagicMock()
                proc = mod.create_ffmpeg_pipe("/tmp/out.mp4", 30.0)
                mock_popen.assert_called_once()
                cmd = mock_popen.call_args[0][0]
                assert "ffmpeg" in cmd[0]
                assert "-c:v" in cmd
                idx = cmd.index("-c:v")
                assert cmd[idx + 1] == "libx264"
                assert "-crf" in cmd
                assert "1080x1920" in cmd[cmd.index("-s") + 1]
        finally:
            mod.CACHED_ENCODER = old_cache

    def test_pipe_custom_dimensions(self):
        """create_ffmpeg_pipe respecte les dimensions personnalisees."""
        import scripts.core.ffmpeg_utils as mod
        old_cache = mod.CACHED_ENCODER
        mod.CACHED_ENCODER = ("h264_nvenc", "p1")
        try:
            with patch("scripts.core.ffmpeg_utils.subprocess.Popen") as mock_popen:
                mock_popen.return_value = MagicMock()
                mod.create_ffmpeg_pipe("/tmp/out.mp4", 25.0, width=720, height=1280)
                cmd = mock_popen.call_args[0][0]
                assert "720x1280" in cmd[cmd.index("-s") + 1]
                assert "-rc:v" in cmd  # nvenc quality params
        finally:
            mod.CACHED_ENCODER = old_cache


# ===========================================================================
# 7. Path traversal validation (library.py)
# ===========================================================================
class TestPathTraversalValidation:
    """Tests pour la validation anti-path-traversal dans library.py gallery."""

    def test_normal_project_name(self):
        """Un nom de projet normal ne devrait pas etre bloque."""
        # On teste la logique directement sans importer tout gradio/i18n
        virals_dir = tempfile.mkdtemp()
        project_name = "my_project_2024"
        project_path = os.path.join(virals_dir, project_name)
        os.makedirs(project_path, exist_ok=True)

        real_path = os.path.realpath(project_path)
        real_virals = os.path.realpath(virals_dir)
        assert real_path.startswith(real_virals)

        os.rmdir(project_path)
        os.rmdir(virals_dir)

    def test_path_traversal_blocked(self):
        """../../../etc/passwd ne doit PAS etre sous VIRALS_DIR."""
        virals_dir = tempfile.mkdtemp()
        malicious_name = "../../../etc/passwd"
        project_path = os.path.join(virals_dir, malicious_name)

        real_path = os.path.realpath(project_path)
        real_virals = os.path.realpath(virals_dir)
        assert not real_path.startswith(real_virals)

        os.rmdir(virals_dir)

    def test_absolute_path_injection(self):
        """Un chemin absolu ne doit pas passer la validation."""
        virals_dir = tempfile.mkdtemp()

        if sys.platform == "win32":
            injected = "C:\\Windows\\System32"
        else:
            injected = "/etc/passwd"

        project_path = os.path.join(virals_dir, injected)
        real_path = os.path.realpath(project_path)
        real_virals = os.path.realpath(virals_dir)

        # Sur Windows os.path.join avec un chemin absolu le prend tel quel
        # Sur Linux il concatene. Dans les deux cas, ce n'est pas sous virals_dir
        # OU c'est un chemin bizarre qui n'existe pas
        is_safe = real_path.startswith(real_virals + os.sep) or real_path == real_virals
        # C:\Windows\System32 ne commence pas par le tmpdir
        assert not is_safe

        os.rmdir(virals_dir)

    def test_dot_dot_in_middle(self):
        """project/../../../secret ne doit pas etre sous VIRALS_DIR."""
        virals_dir = tempfile.mkdtemp()
        malicious = "legit_project/../../../secret"
        project_path = os.path.join(virals_dir, malicious)

        real_path = os.path.realpath(project_path)
        real_virals = os.path.realpath(virals_dir)
        assert not real_path.startswith(real_virals + os.sep)

        os.rmdir(virals_dir)

    def test_symlink_escape(self):
        """Un symlink qui pointe hors de VIRALS_DIR doit etre detecte."""
        virals_dir = tempfile.mkdtemp()
        target = tempfile.mkdtemp()  # Dossier hors virals
        link_path = os.path.join(virals_dir, "sneaky_link")

        try:
            os.symlink(target, link_path)
        except (OSError, NotImplementedError):
            # Symlinks peuvent necessiter des privileges sur Windows
            pytest.skip("Symlink creation not supported in this environment")

        real_path = os.path.realpath(link_path)
        real_virals = os.path.realpath(virals_dir)
        assert not real_path.startswith(real_virals + os.sep)

        os.unlink(link_path)
        os.rmdir(target)
        os.rmdir(virals_dir)


# ===========================================================================
# 8. validate_url — SSRF protection for yt-dlp downloads
# ===========================================================================
from scripts.download.download_video import validate_url


class TestValidateUrl:
    """Tests pour validate_url() — protection SSRF avant yt-dlp."""

    # -- URLs valides (ne doivent PAS lever d'exception) --------------------
    def test_youtube_url(self):
        validate_url("https://www.youtube.com/watch?v=dQw4w9WgXcQ")

    def test_youtu_be_short(self):
        validate_url("https://youtu.be/dQw4w9WgXcQ")

    def test_tiktok_url(self):
        validate_url("https://www.tiktok.com/@user/video/123456")

    def test_vimeo_url(self):
        validate_url("https://vimeo.com/123456")

    def test_http_allowed(self):
        validate_url("http://www.dailymotion.com/video/x123")

    def test_instagram_url(self):
        validate_url("https://www.instagram.com/reel/ABC123/")

    def test_twitter_url(self):
        validate_url("https://twitter.com/user/status/123456")

    def test_x_dot_com_url(self):
        validate_url("https://x.com/user/status/123456")

    def test_reddit_url(self):
        validate_url("https://www.reddit.com/r/sub/comments/abc/title/")

    def test_twitch_url(self):
        validate_url("https://www.twitch.tv/videos/123456")

    def test_facebook_url(self):
        validate_url("https://www.facebook.com/watch/?v=123456")

    # -- Schemes interdits --------------------------------------------------
    def test_file_scheme_rejected(self):
        with pytest.raises(ValueError, match="scheme"):
            validate_url("file:///etc/passwd")

    def test_ftp_scheme_rejected(self):
        with pytest.raises(ValueError, match="scheme"):
            validate_url("ftp://ftp.example.com/file.mp4")

    def test_empty_scheme_rejected(self):
        with pytest.raises(ValueError, match="scheme"):
            validate_url("/etc/passwd")

    def test_no_scheme_bare_path(self):
        with pytest.raises(ValueError, match="scheme"):
            validate_url("C:\\Windows\\System32\\cmd.exe")

    # -- Hostname manquant ou interdit --------------------------------------
    def test_no_hostname_rejected(self):
        with pytest.raises(ValueError, match="hostname"):
            validate_url("https://")

    def test_localhost_rejected(self):
        with pytest.raises(ValueError, match="not allowed"):
            validate_url("https://localhost/admin")

    def test_localhost_localdomain_rejected(self):
        with pytest.raises(ValueError, match="not allowed"):
            validate_url("http://localhost.localdomain/")

    # -- IP privees / reservees ---------------------------------------------
    def test_loopback_ip_rejected(self):
        with pytest.raises(ValueError, match="private"):
            validate_url("http://127.0.0.1/admin")

    def test_private_class_a_rejected(self):
        with pytest.raises(ValueError, match="private"):
            validate_url("http://10.0.0.1/internal")

    def test_private_class_b_rejected(self):
        with pytest.raises(ValueError, match="private"):
            validate_url("http://172.16.0.1/internal")

    def test_private_class_c_rejected(self):
        with pytest.raises(ValueError, match="private"):
            validate_url("http://192.168.1.1/router")

    def test_ipv6_loopback_rejected(self):
        with pytest.raises(ValueError, match="private"):
            validate_url("http://[::1]/admin")


# ===========================================================================
# 9. build_lut_filter — LUT filter string builder
# ===========================================================================
class TestBuildLutFilter:
    """Tests pour build_lut_filter() — genere un filtre LUT sans executer FFmpeg."""

    def test_returns_none_when_lut_file_missing(self):
        from scripts.editing.color_grading import build_lut_filter
        result = build_lut_filter(lut_name="nonexistent.cube", lut_dir=tempfile.mkdtemp())
        assert result is None

    def test_returns_filter_string_with_valid_lut(self):
        from scripts.editing.color_grading import build_lut_filter
        tmp_dir = tempfile.mkdtemp()
        lut_file = os.path.join(tmp_dir, "test.cube")
        with open(lut_file, "w") as f:
            f.write("# dummy lut\n")
        try:
            result = build_lut_filter(lut_name="test.cube", intensity=0.5, lut_dir=tmp_dir)
            assert result is not None
            assert "lut3d=" in result
            assert "blend=all_mode=normal:all_opacity=0.5" in result
            assert "split[__lut_a][__lut_b]" in result
        finally:
            os.unlink(lut_file)
            os.rmdir(tmp_dir)

    def test_intensity_clamped(self):
        from scripts.editing.color_grading import build_lut_filter
        tmp_dir = tempfile.mkdtemp()
        lut_file = os.path.join(tmp_dir, "test.cube")
        with open(lut_file, "w") as f:
            f.write("# dummy\n")
        try:
            result = build_lut_filter(lut_name="test.cube", intensity=5.0, lut_dir=tmp_dir)
            assert result is not None
            assert "all_opacity=1.0" in result
        finally:
            os.unlink(lut_file)
            os.rmdir(tmp_dir)

    def test_path_traversal_blocked(self):
        from scripts.editing.color_grading import build_lut_filter
        result = build_lut_filter(lut_name="../../etc/passwd", lut_dir=tempfile.mkdtemp())
        assert result is None

    def test_backslash_escaping(self):
        """LUT path with backslashes should be converted to forward slashes."""
        from scripts.editing.color_grading import build_lut_filter
        tmp_dir = tempfile.mkdtemp()
        lut_file = os.path.join(tmp_dir, "my_lut.cube")
        with open(lut_file, "w") as f:
            f.write("# dummy\n")
        try:
            result = build_lut_filter(lut_name="my_lut.cube", lut_dir=tmp_dir)
            assert result is not None
            # No backslashes in the filter string
            assert "\\" not in result
        finally:
            os.unlink(lut_file)
            os.rmdir(tmp_dir)


# ===========================================================================
# 9b. apply_lut — path traversal validation
# ===========================================================================
class TestApplyLutPathTraversal:
    """Tests pour apply_lut() — validation anti-path-traversal."""

    def test_apply_lut_traversal_blocked(self):
        """apply_lut() doit bloquer les noms LUT avec ../."""
        from scripts.editing.color_grading import apply_lut
        tmp_dir = tempfile.mkdtemp()
        try:
            result = apply_lut(
                "/fake/input.mp4", "/fake/output.mp4",
                lut_name="../../etc/passwd", lut_dir=tmp_dir,
            )
            assert result is False
        finally:
            os.rmdir(tmp_dir)

    def test_apply_lut_absolute_path_blocked(self):
        """apply_lut() doit bloquer un chemin absolu comme nom LUT."""
        from scripts.editing.color_grading import apply_lut
        tmp_dir = tempfile.mkdtemp()
        if sys.platform == "win32":
            malicious = "C:\\Windows\\System32\\evil.cube"
        else:
            malicious = "/etc/evil.cube"
        try:
            result = apply_lut(
                "/fake/input.mp4", "/fake/output.mp4",
                lut_name=malicious, lut_dir=tmp_dir,
            )
            # Either blocked by traversal check or file not found -> False
            assert result is False
        finally:
            os.rmdir(tmp_dir)

    def test_build_lut_filter_absolute_path_blocked(self):
        """build_lut_filter() doit bloquer un chemin absolu comme nom LUT."""
        from scripts.editing.color_grading import build_lut_filter
        tmp_dir = tempfile.mkdtemp()
        if sys.platform == "win32":
            malicious = "C:\\Windows\\System32\\evil.cube"
        else:
            malicious = "/etc/evil.cube"
        try:
            result = build_lut_filter(lut_name=malicious, lut_dir=tmp_dir)
            assert result is None
        finally:
            os.rmdir(tmp_dir)


# ===========================================================================
# 10. apply_post_production — combined single-pass post-production
# ===========================================================================
class TestApplyPostProduction:
    """Tests pour apply_post_production() — filtre combine en un seul pass FFmpeg."""

    def test_no_effects_returns_true_no_encode(self):
        """Aucun effet active -> True sans appel FFmpeg."""
        from scripts.editing.overlay_effects import apply_post_production
        result = apply_post_production("/fake/input.mp4", "/fake/output.mp4")
        assert result is True

    def test_progress_bar_invalid_color_returns_false(self):
        from scripts.editing.overlay_effects import apply_post_production
        result = apply_post_production(
            "/fake/input.mp4", "/fake/output.mp4",
            progress_bar=True, bar_color="invalid;color",
        )
        assert result is False

    def test_progress_bar_invalid_position_returns_false(self):
        from scripts.editing.overlay_effects import apply_post_production
        result = apply_post_production(
            "/fake/input.mp4", "/fake/output.mp4",
            progress_bar=True, bar_position="middle",
        )
        assert result is False

    def test_single_effect_bar_only(self):
        """Progress bar seul: verifie la commande FFmpeg generee."""
        from scripts.editing.overlay_effects import apply_post_production
        import scripts.core.ffmpeg_utils as ffu
        old_cache = ffu.CACHED_ENCODER
        ffu.CACHED_ENCODER = ("libx264", "fast")
        try:
            with patch("scripts.editing.overlay_effects.get_video_duration", return_value=10.0), \
                 patch("scripts.editing.overlay_effects.run_cmd") as mock_run:
                result = apply_post_production(
                    "/fake/input.mp4", "/fake/output.mp4",
                    progress_bar=True, bar_color="white", bar_position="bottom",
                )
                assert result is True
                mock_run.assert_called_once()
                cmd = mock_run.call_args[0][0]
                assert "-filter_complex" in cmd
                fc_idx = cmd.index("-filter_complex")
                fc = cmd[fc_idx + 1]
                assert "overlay" in fc
                assert "t/10.0" in fc
                assert "-c:a" in cmd
                assert "copy" in cmd[cmd.index("-c:a") + 1]
        finally:
            ffu.CACHED_ENCODER = old_cache

    def test_lut_only_with_missing_file(self):
        """LUT active mais fichier absent -> LUT saute, pas de filtre, False."""
        from scripts.editing.overlay_effects import apply_post_production
        with patch("scripts.editing.overlay_effects.get_video_duration", return_value=10.0):
            result = apply_post_production(
                "/fake/input.mp4", "/fake/output.mp4",
                lut_name="nonexistent_lut.cube", lut_dir=tempfile.mkdtemp(),
            )
            # LUT skipped -> no filters -> returns False (requested effect failed)
            assert result is False

    def test_combined_bar_and_emoji(self):
        """Progress bar + emoji: verifie que filter_complex contient les deux."""
        from scripts.editing.overlay_effects import apply_post_production
        import scripts.core.ffmpeg_utils as ffu
        old_cache = ffu.CACHED_ENCODER
        ffu.CACHED_ENCODER = ("libx264", "fast")
        try:
            fake_png = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
            emojis = [{"emoji": "fire", "timestamp": 1.0, "duration": 2.0, "position": "center"}]
            with patch("scripts.editing.overlay_effects.get_video_duration", return_value=10.0), \
                 patch("scripts.editing.overlay_effects._render_emoji_png", return_value=fake_png), \
                 patch("scripts.editing.overlay_effects.run_cmd") as mock_run:
                result = apply_post_production(
                    "/fake/input.mp4", "/fake/output.mp4",
                    progress_bar=True, bar_color="white", bar_position="top",
                    emojis=emojis,
                )
                assert result is True
                cmd = mock_run.call_args[0][0]
                fc = cmd[cmd.index("-filter_complex") + 1]
                # Both progress bar (color+overlay) and emoji overlay should be present
                assert "color=c=white" in fc
                assert "overlay=" in fc
                assert "between(t,1.0,3.0)" in fc
        finally:
            ffu.CACHED_ENCODER = old_cache
            if os.path.exists(fake_png):
                os.unlink(fake_png)

    def test_all_three_effects(self):
        """LUT + progress bar + emoji: 3 effets dans un seul filter_complex."""
        from scripts.editing.overlay_effects import apply_post_production
        import scripts.core.ffmpeg_utils as ffu
        old_cache = ffu.CACHED_ENCODER
        ffu.CACHED_ENCODER = ("libx264", "fast")
        tmp_dir = tempfile.mkdtemp()
        lut_file = os.path.join(tmp_dir, "test.cube")
        with open(lut_file, "w") as f:
            f.write("# dummy\n")
        try:
            fake_png = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
            emojis = [{"emoji": "star", "timestamp": 0.5, "duration": 1.0, "position": "top-right"}]
            with patch("scripts.editing.overlay_effects.get_video_duration", return_value=15.0), \
                 patch("scripts.editing.overlay_effects._render_emoji_png", return_value=fake_png), \
                 patch("scripts.editing.overlay_effects.run_cmd") as mock_run:
                result = apply_post_production(
                    "/fake/input.mp4", "/fake/output.mp4",
                    lut_name="test.cube", lut_intensity=0.7, lut_dir=tmp_dir,
                    progress_bar=True, bar_color="red", bar_position="bottom",
                    emojis=emojis,
                )
                assert result is True
                cmd = mock_run.call_args[0][0]
                fc = cmd[cmd.index("-filter_complex") + 1]
                # All three effects present
                assert "lut3d=" in fc
                assert "blend=" in fc
                assert "color=c=red" in fc
                assert "overlay=" in fc
                # Verify chain: LUT tags -> bar tag -> emoji tag (without final tag)
                assert "[__pp_lut]" in fc
                assert "[__pp_bar]" in fc
        finally:
            ffu.CACHED_ENCODER = old_cache
            os.unlink(lut_file)
            os.rmdir(tmp_dir)
            if os.path.exists(fake_png):
                os.unlink(fake_png)

    def test_invalid_duration_returns_false(self):
        """Duree video invalide -> False."""
        from scripts.editing.overlay_effects import apply_post_production
        with patch("scripts.editing.overlay_effects.get_video_duration", return_value=0.0):
            result = apply_post_production(
                "/fake/input.mp4", "/fake/output.mp4",
                progress_bar=True,
            )
            assert result is False

    def test_ffmpeg_failure_returns_false(self):
        """Erreur FFmpeg -> False."""
        from scripts.editing.overlay_effects import apply_post_production
        import scripts.core.ffmpeg_utils as ffu
        old_cache = ffu.CACHED_ENCODER
        ffu.CACHED_ENCODER = ("libx264", "fast")
        try:
            with patch("scripts.editing.overlay_effects.get_video_duration", return_value=10.0), \
                 patch("scripts.editing.overlay_effects.run_cmd", side_effect=RuntimeError("ffmpeg crash")):
                result = apply_post_production(
                    "/fake/input.mp4", "/fake/output.mp4",
                    progress_bar=True,
                )
                assert result is False
        finally:
            ffu.CACHED_ENCODER = old_cache

    def test_emoji_png_cleanup_on_success(self):
        """Les PNG emoji temporaires sont supprimes meme en cas de succes."""
        from scripts.editing.overlay_effects import apply_post_production
        import scripts.core.ffmpeg_utils as ffu
        old_cache = ffu.CACHED_ENCODER
        ffu.CACHED_ENCODER = ("libx264", "fast")
        try:
            fake_png = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
            assert os.path.exists(fake_png)
            emojis = [{"emoji": "fire", "timestamp": 0, "duration": 1}]
            with patch("scripts.editing.overlay_effects.get_video_duration", return_value=5.0), \
                 patch("scripts.editing.overlay_effects._render_emoji_png", return_value=fake_png), \
                 patch("scripts.editing.overlay_effects.run_cmd"):
                apply_post_production(
                    "/fake/input.mp4", "/fake/output.mp4",
                    emojis=emojis,
                )
            # PNG should be cleaned up
            assert not os.path.exists(fake_png)
        finally:
            ffu.CACHED_ENCODER = old_cache

    def test_emoji_png_cleanup_on_failure(self):
        """Les PNG emoji temporaires sont supprimes meme en cas d'erreur."""
        from scripts.editing.overlay_effects import apply_post_production
        import scripts.core.ffmpeg_utils as ffu
        old_cache = ffu.CACHED_ENCODER
        ffu.CACHED_ENCODER = ("libx264", "fast")
        try:
            fake_png = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
            emojis = [{"emoji": "fire", "timestamp": 0, "duration": 1}]
            with patch("scripts.editing.overlay_effects.get_video_duration", return_value=5.0), \
                 patch("scripts.editing.overlay_effects._render_emoji_png", return_value=fake_png), \
                 patch("scripts.editing.overlay_effects.run_cmd", side_effect=RuntimeError("boom")):
                apply_post_production(
                    "/fake/input.mp4", "/fake/output.mp4",
                    emojis=emojis,
                )
            assert not os.path.exists(fake_png)
        finally:
            ffu.CACHED_ENCODER = old_cache


# ===========================================================================
# A/B Caption Variants
# ===========================================================================
class TestABVariants:
    """Tests for A/B caption variant generation."""

    def test_modify_hook_text_list_format(self):
        """Test hook text modification with list format."""
        from scripts.postprod.ab_variants import _modify_hook_text
        sub_data = [
            {"word": "Hello", "start": 0.0, "end": 0.5},
            {"word": "world", "start": 0.5, "end": 1.0},
            {"word": "test", "start": 1.0, "end": 1.5},
        ]
        result = _modify_hook_text(sub_data, "Bonjour monde")
        assert result[0]["word"] == "Bonjour"
        assert result[1]["word"] == "monde"
        assert result[2]["word"] == "test"  # Unchanged
        # Timing preserved
        assert result[0]["start"] == 0.0
        assert result[0]["end"] == 0.5

    def test_modify_hook_text_dict_format(self):
        """Test hook text modification with dict format."""
        from scripts.postprod.ab_variants import _modify_hook_text
        sub_data = {
            "words": [
                {"word": "Hello", "start": 0.0, "end": 0.5},
                {"word": "world", "start": 0.5, "end": 1.0},
            ]
        }
        result = _modify_hook_text(sub_data, "New hook text here")
        assert result["words"][0]["word"] == "New"
        assert result["words"][1]["word"] == "hook"

    def test_modify_hook_text_empty(self):
        """Test with empty data."""
        from scripts.postprod.ab_variants import _modify_hook_text
        result = _modify_hook_text([], "Test")
        assert result == []

    def test_modify_hook_text_does_not_mutate_original(self):
        """La fonction ne doit pas muter les donnees originales."""
        from scripts.postprod.ab_variants import _modify_hook_text
        original = [
            {"word": "Hello", "start": 0.0, "end": 0.5},
            {"word": "world", "start": 0.5, "end": 1.0},
        ]
        _modify_hook_text(original, "Changed")
        assert original[0]["word"] == "Hello"

    def test_modify_hook_text_variant_longer_than_words(self):
        """Variant text plus long que le nombre de mots -> pas de crash."""
        from scripts.postprod.ab_variants import _modify_hook_text
        sub_data = [{"word": "Hello", "start": 0.0, "end": 0.5}]
        result = _modify_hook_text(sub_data, "One two three four")
        assert result[0]["word"] == "One"
        assert len(result) == 1

    def test_variant_labels(self):
        """Test variant label generation A, B, C..."""
        labels = [chr(65 + i) for i in range(5)]
        assert labels == ["A", "B", "C", "D", "E"]

    def test_generate_variants_no_segments_file(self):
        """generate_variants retourne [] si pas de viral_segments.txt."""
        from scripts.postprod.ab_variants import generate_variants
        with tempfile.TemporaryDirectory() as tmpdir:
            result = generate_variants(tmpdir)
            assert result == []

    def test_generate_variants_no_caption_variants(self):
        """generate_variants retourne [] si aucun segment n'a caption_variants."""
        from scripts.postprod.ab_variants import generate_variants
        with tempfile.TemporaryDirectory() as tmpdir:
            vs = {"segments": [{"title": "Test", "start_time": 0, "end_time": 10}]}
            with open(os.path.join(tmpdir, "viral_segments.txt"), "w") as f:
                json.dump(vs, f)
            result = generate_variants(tmpdir)
            assert result == []

    def test_find_subtitle_json_by_prefix(self):
        """_find_subtitle_json trouve par prefixe d'index."""
        from scripts.postprod.ab_variants import _find_subtitle_json
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a file matching the pattern
            test_file = os.path.join(tmpdir, "000_Some_Title_processed.json")
            with open(test_file, "w") as f:
                f.write("{}")
            result = _find_subtitle_json(tmpdir, 0, {"title": "Different Title"})
            assert result == test_file


# ===========================================================================
# Tests: _get_animation_tags (Fix 1 — fs instead of fscx/fscy)
# ===========================================================================

class TestGetAnimationTags:
    def test_pop_uses_fs_not_fscx(self):
        from scripts.editing.adjust_subtitles import _get_animation_tags
        tags = _get_animation_tags("pop", 24)
        assert "\\fs27" in tags
        assert "\\fs24" in tags
        assert "fscx" not in tags
        assert "fscy" not in tags

    def test_bounce_uses_fs_not_fscx(self):
        from scripts.editing.adjust_subtitles import _get_animation_tags
        tags = _get_animation_tags("bounce", 24)
        assert "\\fs28" in tags  # hl_size + 4
        assert "\\fs22" in tags  # hl_size - 2
        assert "\\fs24" in tags  # back to normal
        assert "fscx" not in tags
        assert "fscy" not in tags

    def test_fade_pop_uses_fs_not_fscx(self):
        from scripts.editing.adjust_subtitles import _get_animation_tags
        tags = _get_animation_tags("fade_pop", 20)
        assert "\\fs23" in tags
        assert "\\fs20" in tags
        assert "\\alpha" in tags
        assert "fscx" not in tags

    def test_none_returns_empty(self):
        from scripts.editing.adjust_subtitles import _get_animation_tags
        assert _get_animation_tags("none", 24) == ""

    def test_unknown_returns_empty(self):
        from scripts.editing.adjust_subtitles import _get_animation_tags
        assert _get_animation_tags("unknown_anim", 24) == ""



# ===========================================================================
# Tests: face_start_snap stricter params (Fix 3)
# ===========================================================================

class TestFaceStartSnap:
    def test_returns_start_time_for_missing_file(self):
        from scripts.vision.face_start_snap import snap_to_first_face
        result = snap_to_first_face("/nonexistent/video.mp4", 5.0)
        assert result == 5.0

    def test_cascade_params_are_strict(self):
        """Verify the source code uses stricter Haar params (minNeighbors>=6)."""
        import inspect
        from scripts.vision import face_start_snap
        source = inspect.getsource(face_start_snap.snap_to_first_face)
        assert "minNeighbors=6" in source
        assert "minSize=(120, 120)" in source
        assert "scaleFactor=1.1" in source

    def test_aspect_ratio_filter_present(self):
        """Verify aspect ratio filtering is in the source."""
        import inspect
        from scripts.vision import face_start_snap
        source = inspect.getsource(face_start_snap.snap_to_first_face)
        assert "0.7" in source
        assert "1.4" in source
        assert "min_face_h" in source
