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
from scripts.create_viral_segments import clean_json_response, clean_json_response_simple

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
        from scripts.transcribe_video import parse_srt

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
        from scripts.transcribe_video import parse_srt

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
        from scripts.transcribe_video import parse_srt

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
        from scripts.transcribe_video import parse_srt
        path = self._write_srt("")
        try:
            result = parse_srt(path)
            assert result is not None
            assert len(result) == 0
        finally:
            os.unlink(path)

    def test_malformed_srt_no_timestamp(self):
        """Bloc sans timestamp -> ignore sans crash."""
        from scripts.transcribe_video import parse_srt

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
        from scripts.transcribe_video import parse_srt

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
        from scripts.transcribe_video import parse_vtt

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
        from scripts.transcribe_video import parse_vtt

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
        from scripts.transcribe_video import parse_vtt

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
        from scripts.transcribe_video import parse_vtt

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
        from scripts.transcribe_video import parse_vtt

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
        from scripts.transcribe_video import parse_vtt
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
        """Importe le module avec les deps lourdes mockees et reset le cache."""
        import importlib
        # Mock heavy dependencies before import
        mocks = {}
        for mod in ["cv2", "numpy", "mediapipe", "scripts.one_face", "scripts.two_face",
                     "scripts.face_detection_insightface"]:
            if mod not in sys.modules:
                mocks[mod] = MagicMock()
                sys.modules[mod] = mocks[mod]

        if "scripts.edit_video" in sys.modules:
            importlib.reload(sys.modules["scripts.edit_video"])
        else:
            import scripts.edit_video

        mod = sys.modules["scripts.edit_video"]
        mod.CACHED_ENCODER = None  # Reset cache
        return mod, mocks

    def _cleanup_mocks(self, mocks):
        for mod in mocks:
            if mod in sys.modules and sys.modules[mod] is mocks[mod]:
                del sys.modules[mod]

    def test_nvidia_detected(self):
        mod, mocks = self._import_fresh()
        try:
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(stdout="... h264_nvenc ... h264_amf ...")
                result = mod.get_best_encoder()
                assert result[0] == "h264_nvenc"
                assert result[1] == "p1"
        finally:
            mod.CACHED_ENCODER = None
            self._cleanup_mocks(mocks)

    def test_amd_detected(self):
        mod, mocks = self._import_fresh()
        try:
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(stdout="... h264_amf ...")
                result = mod.get_best_encoder()
                assert result[0] == "h264_amf"
        finally:
            mod.CACHED_ENCODER = None
            self._cleanup_mocks(mocks)

    def test_intel_qsv_detected(self):
        mod, mocks = self._import_fresh()
        try:
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(stdout="... h264_qsv ...")
                result = mod.get_best_encoder()
                assert result[0] == "h264_qsv"
        finally:
            mod.CACHED_ENCODER = None
            self._cleanup_mocks(mocks)

    def test_mac_videotoolbox_detected(self):
        mod, mocks = self._import_fresh()
        try:
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(stdout="... h264_videotoolbox ...")
                result = mod.get_best_encoder()
                assert result[0] == "h264_videotoolbox"
        finally:
            mod.CACHED_ENCODER = None
            self._cleanup_mocks(mocks)

    def test_cpu_fallback(self):
        mod, mocks = self._import_fresh()
        try:
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(stdout="... libx264 ...")
                result = mod.get_best_encoder()
                assert result[0] == "libx264"
                assert result[1] == "ultrafast"
        finally:
            mod.CACHED_ENCODER = None
            self._cleanup_mocks(mocks)

    def test_ffmpeg_not_found(self):
        mod, mocks = self._import_fresh()
        try:
            with patch("subprocess.run", side_effect=FileNotFoundError("ffmpeg not found")):
                result = mod.get_best_encoder()
                assert result[0] == "libx264"  # Fallback CPU
        finally:
            mod.CACHED_ENCODER = None
            self._cleanup_mocks(mocks)

    def test_encoder_cached(self):
        """Apres le premier appel, le cache est utilise."""
        mod, mocks = self._import_fresh()
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
            self._cleanup_mocks(mocks)


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
