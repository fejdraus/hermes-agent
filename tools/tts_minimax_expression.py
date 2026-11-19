"""Expressive MiniMax speech (``speech-2.8-*``): an emotion per call, sound tags and stress marks
on Russian homographs.

The model that writes the reply picks the ``emotion`` of ``text_to_speech`` and writes the sound
tags into the text; without an emotion MiniMax picks one from the text. Stress marks come from
ruaccent (``tts_ru_stress``), never from a language model. Older MiniMax models would read the
tags aloud, so all of this applies only to 2.8."""

from __future__ import annotations

import re
from typing import Any, Optional

EXPRESSIVE_MODEL_PREFIX = "speech-2.8"

EMOTIONS = ("happy", "sad", "angry", "fearful", "disgusted", "surprised", "calm", "fluent")

SOUND_TAGS = (
    "laughs", "chuckle", "coughs", "clear-throat", "groans", "breath", "pant", "inhale", "exhale",
    "gasps", "sniffs", "sighs", "snorts", "burps", "lip-smacking", "humming", "hissing", "emm",
    "sneezes",
)

_SOUND_TAG_RE = re.compile(rf"\((?:{'|'.join(re.escape(tag) for tag in SOUND_TAGS)})\)\s*")

EMOTION_SCHEMA = {
    "type": "string",
    "enum": list(EMOTIONS),
    "description": (
        "The emotion the whole text is spoken with, matching its meaning and mood. Omit it "
        "when unsure: the voice then picks one from the text."),
}

TEXT_GUIDANCE = (
    " Make the voice sound alive: put a sound tag where a person would make that sound, e.g. "
    "\"(sighs) Ну вот, опять дождь. (laughs) Зато поливать не надо.\" Tags: "
    + ", ".join(f"({tag})" for tag in SOUND_TAGS)
    + ". Use them rarely and only where the meaning calls for it, never as decoration. Do not "
    "write stress marks: they are added for you.")


def is_expressive(tts_config: dict[str, Any], provider: str) -> bool:
    """True when the active MiniMax model speaks sound tags and takes the 2.8 emotions."""
    if provider != "minimax":
        return False
    model = str((tts_config.get("minimax") or {}).get("model") or "")
    return model.startswith(EXPRESSIVE_MODEL_PREFIX)


def apply_emotion(tts_config: dict[str, Any], provider: str, emotion: Optional[str]) -> dict[str, Any]:
    """The model's emotion on a copy of the config; an unknown one keeps the configured emotion."""
    emotion = (emotion or "").strip().lower()
    if emotion not in EMOTIONS or not is_expressive(tts_config, provider):
        return tts_config
    section = dict(tts_config.get("minimax") or {})
    section["emotion"] = emotion
    return {**tts_config, "minimax": section}


def strip_sound_tags(text: str) -> str:
    """The text without sound tags, so English tag words do not sway language detection."""
    return _SOUND_TAG_RE.sub("", text)


def apply_stress(text: str, tts_config: dict[str, Any], provider: str, language: Optional[str]) -> str:
    """Russian text for a 2.8 voice gets ruaccent's stress marks on its homographs."""
    if not is_expressive(tts_config, provider):
        return text
    language = (language or "").strip().lower()
    if not language:
        from tools.tts_voice_language import detect_language
        language = detect_language(text)
    if language != "ru":
        return text
    from tools.tts_ru_stress import stress_homographs
    return stress_homographs(text)


def warm_up_voice_models() -> None:
    """At gateway start, load the Russian stress model in the background when the voice is 2.8."""
    from tools.tts_tool import _get_provider, _load_tts_config
    tts_config = _load_tts_config()
    if is_expressive(tts_config, _get_provider(tts_config)):
        from tools.tts_ru_stress import warm_up
        warm_up()
