"""Expressive MiniMax speech (``speech-2.8-*``): an emotion per call, sound tags and stress marks.

The model picks the ``emotion`` of each text and writes the sound tags and stress marks into the
text itself; without an emotion the configured one (or MiniMax's own choice) applies. Older
MiniMax models would read the tags aloud, so the model is told about them only for 2.8."""

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
        "The emotion the whole text is spoken with, matching its meaning and mood. Omit for "
        "plain informational text: the voice then keeps its usual tone."),
}

TEXT_GUIDANCE = (
    " Make the voice sound alive: put a sound tag where a person would make that sound, e.g. "
    "\"(sighs) Ну вот, опять дождь. (laughs) Зато поливать не надо.\" Tags: "
    + ", ".join(f"({tag})" for tag in SOUND_TAGS)
    + ". Use them rarely and only where the meaning calls for it, never as decoration. Put the "
    "stress mark U+0301 right after the stressed vowel of a word whose stress is ambiguous or "
    "often misread: homographs (за́мок / замо́к, до́рога / дорога́), names, "
    "surnames, place names, rare words. Do not mark ordinary words.")


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
