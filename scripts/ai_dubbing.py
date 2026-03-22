"""AI dubbing: translate transcript + generate TTS via edge-tts + mix audio."""
from __future__ import annotations

import asyncio
import logging
import subprocess
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

VOICE_MAP: dict[str, str] = {
    "en": "en-US-AriaNeural",
    "fr": "fr-FR-DeniseNeural",
    "es": "es-ES-ElviraNeural",
    "de": "de-DE-KatjaNeural",
    "pt": "pt-BR-FranciscaNeural",
    "tr": "tr-TR-EmelNeural",
    "ja": "ja-JP-NanamiNeural",
    "zh": "zh-CN-XiaoxiaoNeural",
}


def get_voice_for_language(language: str) -> str:
    """Return the edge-tts voice ID for a language code.

    Falls back to en-US-AriaNeural for unknown languages.
    """
    return VOICE_MAP.get(language, "en-US-AriaNeural")


def translate_text(text: str, target_language: str) -> str:
    """Translate text to target_language using deep_translator.

    Returns original text on failure.
    """
    if not text or not text.strip():
        return text
    try:
        from deep_translator import GoogleTranslator

        translated = GoogleTranslator(source="auto", target=target_language).translate(text)
        return translated or text
    except Exception:
        logger.error("Translation to '%s' failed", target_language, exc_info=True)
        return text


async def generate_tts(
    text: str, output_path: str, voice: str = "en-US-AriaNeural"
) -> bool:
    """Generate TTS audio via edge-tts and save as MP3.

    Returns True on success, False otherwise.
    """
    if not text or not text.strip():
        logger.warning("Empty text — skipping TTS generation")
        return False

    try:
        import edge_tts
    except ImportError:
        logger.error("edge_tts not installed — run: pip install edge-tts")
        return False

    try:
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(output_path)
        logger.info("TTS audio saved to %s", output_path)
        return True
    except Exception:
        logger.error("TTS generation failed", exc_info=True)
        return False


def dub_segment(
    video_path: str,
    transcript_text: str,
    target_language: str,
    output_path: str,
    original_volume: float = 0.2,
) -> bool:
    """Dub a video segment: translate, generate TTS, mix audio.

    Steps:
        1. Translate transcript to target language
        2. Generate TTS audio to a temp file
        3. Mix original audio (lowered) with TTS overlay via ffmpeg

    Returns True on success, False otherwise.
    """
    if not Path(video_path).is_file():
        logger.error("Video file not found: %s", video_path)
        return False

    # 1. Translate
    translated = translate_text(transcript_text, target_language)
    if not translated or not translated.strip():
        logger.warning("Translation produced empty text — aborting dub")
        return False

    # 2. Generate TTS to temp file
    voice = get_voice_for_language(target_language)
    tts_temp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    tts_path = tts_temp.name
    tts_temp.close()

    try:
        success = asyncio.run(generate_tts(translated, tts_path, voice))
        if not success:
            return False

        # 3. Mix audio via ffmpeg
        filter_complex = (
            f"[0:a]volume={original_volume}[a1];"
            f"[a1][1:a]amix=inputs=2:duration=longest[aout]"
        )
        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-i", tts_path,
            "-filter_complex", filter_complex,
            "-map", "0:v",
            "-map", "[aout]",
            "-c:v", "copy",
            "-c:a", "aac",
            output_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            logger.error("ffmpeg mixing failed: %s", result.stderr[:500])
            return False

        logger.info("Dubbed video saved to %s", output_path)
        return True
    except Exception:
        logger.error("Dubbing failed", exc_info=True)
        return False
    finally:
        Path(tts_path).unlink(missing_ok=True)
