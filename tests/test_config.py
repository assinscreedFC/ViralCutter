"""Tests for scripts/config.py — load_api_config and validate_api_config."""
from __future__ import annotations

import json
import os
import sys
from unittest.mock import patch

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from scripts.config import (
    load_api_config,
    validate_api_config,
    _api_key_is_empty,
    _env_api_key,
    _to_positive_int,
    VALID_BACKENDS,
)


# ---------------------------------------------------------------------------
# validate_api_config
# ---------------------------------------------------------------------------

class TestValidateApiConfig:
    def test_valid_config_returns_no_errors(self):
        config = {"backend": "g4f", "chunk_size": 15000, "model_name": "", "api_key": ""}
        # g4f has no required env vars
        errors = validate_api_config(config)
        assert errors == []

    def test_missing_backend_reports_error(self):
        config = {"chunk_size": 15000}
        errors = validate_api_config(config)
        assert any("backend" in e for e in errors)

    def test_invalid_backend_reports_error(self):
        config = {"backend": "openai", "chunk_size": 10000}
        errors = validate_api_config(config)
        assert any("backend" in e for e in errors)

    def test_missing_chunk_size_reports_error(self):
        config = {"backend": "g4f"}
        errors = validate_api_config(config)
        assert any("chunk_size" in e for e in errors)

    def test_non_positive_chunk_size_reports_error(self):
        config = {"backend": "g4f", "chunk_size": -1}
        errors = validate_api_config(config)
        assert any("chunk_size" in e for e in errors)

    def test_zero_chunk_size_reports_error(self):
        config = {"backend": "g4f", "chunk_size": 0}
        errors = validate_api_config(config)
        assert any("chunk_size" in e for e in errors)

    def test_string_chunk_size_reports_error(self):
        config = {"backend": "g4f", "chunk_size": "big"}
        errors = validate_api_config(config)
        assert any("chunk_size" in e for e in errors)

    def test_gemini_missing_env_var_reports_warning(self):
        config = {"backend": "gemini", "chunk_size": 15000}
        with patch.dict(os.environ, {}, clear=False):
            # Ensure GEMINI_API_KEY is absent
            os.environ.pop("GEMINI_API_KEY", None)
            errors = validate_api_config(config)
        assert any("GEMINI_API_KEY" in e for e in errors)

    def test_valid_backends_constant(self):
        assert "pleiade" in VALID_BACKENDS
        assert "gemini" in VALID_BACKENDS
        assert "g4f" in VALID_BACKENDS


# ---------------------------------------------------------------------------
# _api_key_is_empty
# ---------------------------------------------------------------------------

class TestApiKeyIsEmpty:
    def test_empty_string_is_empty(self):
        assert _api_key_is_empty("") is True

    def test_placeholder_is_empty(self):
        assert _api_key_is_empty("SUA_KEY_AQUI") is True

    def test_none_is_empty(self):
        # None is falsy so should be treated as empty
        assert _api_key_is_empty(None) is True  # type: ignore[arg-type]

    def test_real_key_is_not_empty(self):
        assert _api_key_is_empty("sk-abc123") is False


# ---------------------------------------------------------------------------
# _to_positive_int
# ---------------------------------------------------------------------------

class TestToPositiveInt:
    def test_positive_int_returned_as_is(self):
        assert _to_positive_int(5000) == 5000

    def test_string_number_coerced(self):
        assert _to_positive_int("8000") == 8000

    def test_zero_returns_default(self):
        assert _to_positive_int(0, default=15000) == 15000

    def test_negative_returns_default(self):
        assert _to_positive_int(-100, default=15000) == 15000

    def test_none_returns_default(self):
        assert _to_positive_int(None, default=15000) == 15000  # type: ignore[arg-type]

    def test_non_numeric_string_returns_default(self):
        assert _to_positive_int("abc", default=15000) == 15000


# ---------------------------------------------------------------------------
# _env_api_key
# ---------------------------------------------------------------------------

class TestEnvApiKey:
    def test_pleiade_reads_pleiade_env(self):
        with patch.dict(os.environ, {"PLEIADE_API_KEY": "test-key"}):
            assert _env_api_key("pleiade") == "test-key"

    def test_gemini_reads_gemini_env(self):
        with patch.dict(os.environ, {"GEMINI_API_KEY": "gem-key"}, clear=False):
            assert _env_api_key("gemini") == "gem-key"

    def test_gemini_falls_back_to_openai_env(self):
        env = {"OPENAI_API_KEY": "oai-key"}
        # Remove GEMINI_API_KEY if present
        with patch.dict(os.environ, env, clear=False):
            os.environ.pop("GEMINI_API_KEY", None)
            result = _env_api_key("gemini")
        assert result == "oai-key"

    def test_unknown_backend_returns_empty(self):
        assert _env_api_key("unknown") == ""


# ---------------------------------------------------------------------------
# load_api_config — filesystem-level integration
# ---------------------------------------------------------------------------

class TestLoadApiConfig:
    def _make_api_config(self, tmp_path, data: dict) -> str:
        path = tmp_path / "api_config.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        return str(tmp_path)

    def test_missing_api_config_returns_defaults(self, tmp_path):
        """When no config files exist at all, we should get a valid dict."""
        # Point project root to a dir that has no config files
        with patch("scripts.config._PROJECT_ROOT", str(tmp_path)):
            cfg = load_api_config()
        assert "backend" in cfg
        assert "chunk_size" in cfg
        assert isinstance(cfg["chunk_size"], int)
        assert cfg["chunk_size"] > 0

    def test_api_config_backend_used(self, tmp_path):
        data = {
            "selected_api": "g4f",
            "g4f": {"model": "gpt-4o", "chunk_size": 20000},
        }
        with patch("scripts.config._PROJECT_ROOT", str(tmp_path)):
            (tmp_path / "api_config.json").write_text(json.dumps(data), encoding="utf-8")
            cfg = load_api_config()
        assert cfg["backend"] == "g4f"
        assert cfg["chunk_size"] == 20000

    def test_project_folder_process_config_overrides_backend(self, tmp_path):
        api_data = {
            "selected_api": "gemini",
            "gemini": {"api_key": "", "model": "gemini-pro", "chunk_size": 12000},
            "g4f": {"model": "gpt-4o", "chunk_size": 15000},
        }
        proc_data = {"ai_config": {"backend": "g4f", "model_name": "gpt-4o"}}

        with patch("scripts.config._PROJECT_ROOT", str(tmp_path)):
            (tmp_path / "api_config.json").write_text(json.dumps(api_data), encoding="utf-8")
            project_dir = tmp_path / "proj"
            project_dir.mkdir()
            (project_dir / "process_config.json").write_text(
                json.dumps(proc_data), encoding="utf-8"
            )
            cfg = load_api_config(project_folder=str(project_dir))

        assert cfg["backend"] == "g4f"
        assert cfg["model_name"] == "gpt-4o"

    def test_chunk_size_fallback_to_default(self, tmp_path):
        """If chunk_size is absent from all config files, use 15000."""
        data = {"selected_api": "g4f", "g4f": {"model": "gpt-4o"}}
        with patch("scripts.config._PROJECT_ROOT", str(tmp_path)):
            (tmp_path / "api_config.json").write_text(json.dumps(data), encoding="utf-8")
            cfg = load_api_config()
        assert cfg["chunk_size"] == 15000

    def test_returned_dict_has_all_keys(self, tmp_path):
        with patch("scripts.config._PROJECT_ROOT", str(tmp_path)):
            cfg = load_api_config()
        for key in ("backend", "model_name", "api_key", "chunk_size"):
            assert key in cfg

    def test_process_config_manual_backend_ignored(self, tmp_path):
        """backend='manual' in process_config should be ignored."""
        api_data = {"selected_api": "g4f", "g4f": {"model": "gpt-4o", "chunk_size": 10000}}
        proc_data = {"ai_config": {"backend": "manual"}}

        with patch("scripts.config._PROJECT_ROOT", str(tmp_path)):
            (tmp_path / "api_config.json").write_text(json.dumps(api_data), encoding="utf-8")
            project_dir = tmp_path / "proj"
            project_dir.mkdir()
            (project_dir / "process_config.json").write_text(
                json.dumps(proc_data), encoding="utf-8"
            )
            cfg = load_api_config(project_folder=str(project_dir))

        # Should stay with g4f, not switch to manual
        assert cfg["backend"] == "g4f"
