"""Per-language MiniMax voices: ``tts.minimax.voices_by_language`` maps an ISO 639-1 code to a voice.

The language comes from the model (``text_to_speech(language=...)``) or, when it names none (the
gateway's auto-voice of a reply), from lingua over the spoken text. An unknown or ambiguous
language keeps the configured ``voice_id``."""

from __future__ import annotations

import contextlib
import logging
from functools import lru_cache
from typing import Any, Optional

logger = logging.getLogger(__name__)

MINIMAX_LANGUAGES = {
    "ru": "Russian", "uk": "Ukrainian", "en": "English", "pl": "Polish", "cs": "Czech", "de": "German",
    "fr": "French", "es": "Spanish", "pt": "Portuguese", "it": "Italian", "nl": "Dutch", "ro": "Romanian",
    "fi": "Finnish", "el": "Greek", "tr": "Turkish", "ar": "Arabic", "hi": "Hindi", "zh": "Chinese",
    "ja": "Japanese", "ko": "Korean", "th": "Thai", "id": "Indonesian", "vi": "Vietnamese",
}

_MIN_RELATIVE_DISTANCE = 0.1


def voice_table(mm_config: dict[str, Any]) -> dict[str, str]:
    table = mm_config.get("voices_by_language")
    if not isinstance(table, dict):
        return {}
    voices = {str(code).strip().lower(): str(voice or "").strip() for code, voice in table.items()}
    return {code: voice for code, voice in voices.items() if code in MINIMAX_LANGUAGES and voice}


@lru_cache(maxsize=1)
def _detector():
    with contextlib.suppress(Exception):
        from pm import ensure_import
        ensure_import("lang-detect")
    from lingua import IsoCode639_1, LanguageDetectorBuilder
    codes = [IsoCode639_1.from_str(code) for code in MINIMAX_LANGUAGES]
    return (LanguageDetectorBuilder.from_iso_codes_639_1(*codes)
            .with_minimum_relative_distance(_MIN_RELATIVE_DISTANCE).build())


def detect_language(text: str) -> Optional[str]:
    try:
        detector = _detector()
    except ImportError:
        logger.warning("lingua is not installed; TTS language falls back to the script of the text")
        from tools.tts_ru_normalize import detect_speech_language
        return detect_speech_language(text)
    found = detector.detect_language_of(text)
    return found.iso_code_639_1.name.lower() if found is not None else None


def resolve_language_voice(mm_config: dict[str, Any], text: str,
                           language: Optional[str] = None) -> tuple[Optional[str], Optional[str]]:
    """``(voice_id, language_boost)`` for this text; ``(None, None)`` without a voice table."""
    table = voice_table(mm_config)
    if not table:
        return None, None
    code = (language or "").strip().lower()
    if code not in MINIMAX_LANGUAGES:
        code = detect_language(text) or ""
    voice = table.get(code)
    logger.info("TTS language %r -> voice %s", code or None, voice or "default")
    return voice, MINIMAX_LANGUAGES.get(code)
