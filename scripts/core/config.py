"""Centralized config loading with validation."""

import json
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

VALID_BACKENDS = ("pleiade", "gemini", "g4f")

# Env vars required per backend
_BACKEND_ENV_VARS: dict[str, list[str]] = {
    "pleiade": ["PLEIADE_API_URL", "PLEIADE_API_KEY"],
    "gemini": ["GEMINI_API_KEY"],
    "g4f": [],
}

# Root of the project (parent of scripts/)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _read_json(path: str) -> dict:
    """Read a JSON file, return empty dict on failure."""
    if not os.path.exists(path):
        logger.debug("Config file not found: %s", path)
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to read %s: %s", path, e)
        return {}


def load_api_config(project_folder: Optional[str] = None) -> dict:
    """Load merged API config with fallback chain.

    Priority (highest to lowest):
      1. project_folder/process_config.json  (ai_config section)
      2. webui/settings.json                 (ai_backend / ai_model_name / chunk_size)
      3. api_config.json                     (base config)

    Returns a flat dict with keys: backend, model_name, api_key, chunk_size.
    """
    # --- Base: api_config.json ---
    api_cfg_path = os.path.join(_PROJECT_ROOT, "api_config.json")
    api_cfg = _read_json(api_cfg_path)

    backend = api_cfg.get("selected_api", "gemini")
    backend_section = api_cfg.get(backend, {})
    api_key = backend_section.get("api_key", "")
    model_name = backend_section.get("model", "")
    chunk_size = backend_section.get("chunk_size", 15000)

    # --- Fallback: webui/settings.json ---
    settings_path = os.path.join(_PROJECT_ROOT, "webui", "settings.json")
    settings = _read_json(settings_path)

    if settings:
        settings_backend = settings.get("ai_backend", "")
        # Use settings backend if api_config key looks invalid
        if _api_key_is_empty(api_key) and backend not in ("pleiade", "g4f"):
            if settings_backend and settings_backend != "manual":
                backend = settings_backend
                api_key = api_cfg.get(backend, {}).get("api_key", "")
                model_name = settings.get("ai_model_name") or api_cfg.get(backend, {}).get("model", "")

        # settings.json chunk_size as fallback
        if not chunk_size:
            chunk_size = settings.get("chunk_size", 15000)

    # --- Override: process_config.json (per-project) ---
    if project_folder:
        proc_cfg = _read_json(os.path.join(project_folder, "process_config.json"))
        ai_section = proc_cfg.get("ai_config", {})
        proj_backend = ai_section.get("backend", "")
        if proj_backend and proj_backend != "manual":
            backend = proj_backend
            model_name = ai_section.get("model_name") or model_name
            api_key = api_cfg.get(backend, {}).get("api_key") or api_key

    # Resolve api_key from env if still empty
    if _api_key_is_empty(api_key):
        api_key = _env_api_key(backend)

    result = {
        "backend": backend,
        "model_name": model_name,
        "api_key": api_key,
        "chunk_size": _to_positive_int(chunk_size, default=15000),
    }

    errors = validate_api_config(result)
    if errors:
        for err in errors:
            logger.warning("Config validation: %s", err)

    return result


def validate_api_config(config: dict) -> list[str]:
    """Validate an API config dict. Returns a list of error strings (empty = valid)."""
    errors: list[str] = []

    backend = config.get("backend", "")
    if not backend:
        errors.append("'backend' is missing")
    elif backend not in VALID_BACKENDS:
        errors.append(f"'backend' must be one of {VALID_BACKENDS}, got '{backend}'")

    chunk_size = config.get("chunk_size")
    if chunk_size is None:
        errors.append("'chunk_size' is missing")
    elif not isinstance(chunk_size, int) or chunk_size <= 0:
        errors.append(f"'chunk_size' must be a positive integer, got {chunk_size!r}")

    # Warn about missing env vars for the selected backend
    if backend in _BACKEND_ENV_VARS:
        for var in _BACKEND_ENV_VARS[backend]:
            if not os.getenv(var):
                errors.append(f"Env var '{var}' is not set (required for backend '{backend}')")

    return errors


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _api_key_is_empty(key: str) -> bool:
    """Check if an API key value is effectively empty."""
    return not key or key in ("", "SUA_KEY_AQUI")


def _env_api_key(backend: str) -> str:
    """Try to resolve an API key from environment variables."""
    if backend == "pleiade":
        return os.getenv("PLEIADE_API_KEY", "")
    if backend == "gemini":
        return os.getenv("GEMINI_API_KEY", "") or os.getenv("OPENAI_API_KEY", "")
    return ""


def _to_positive_int(value, default: int = 15000) -> int:
    """Coerce value to a positive int, fallback to default."""
    try:
        val = int(value)
        return val if val > 0 else default
    except (TypeError, ValueError):
        return default
