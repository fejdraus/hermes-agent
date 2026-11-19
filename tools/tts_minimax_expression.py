"""Expressive MiniMax speech (``speech-2.8-*``): emotions, sound tags and stress marks on Russian
homographs.

The model that writes the reply knows it will be spoken (``spoken_reply_note``) and marks the
emotion of each part, ``[[sad]] ... [[happy]] ...``, and the sounds, ``(sighs)``; each part is
then spoken with its own emotion, and the user reads the reply without the marks
(``strip_voice_marks``). Without a mark MiniMax picks the emotion from the text. Stress marks
come from ruaccent (``tts_ru_stress``), never from a language model. Older MiniMax models would
read the marks aloud, so all of this applies only to 2.8."""

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


_EMOTION_MARK_RE = re.compile(r"\[\[\s*(" + "|".join(EMOTIONS) + r")\s*\]\]\s*", re.IGNORECASE)

VOICE_TEXT_RULES = (
    "write it to be listened to: numbers, dates, times, decimals and units as words in the "
    "grammatical form the sentence needs; connected speech without lists, headings, emoji, links "
    "or markdown. Mark the emotion of each part with [[emotion]] at its start, one of "
    + ", ".join(EMOTIONS) + ", and mark it again where the mood changes; leave a part unmarked "
    "and the voice picks its emotion. Where a person would make a sound, rarely and only where "
    "the meaning calls for it, write it: " + ", ".join(f"({tag})" for tag in SOUND_TAGS)
    + ". Stress and the voice are added for you.")

SPOKEN_REPLY_NOTE = (
    "[The text of your reply will be turned into speech and sent as a voice message, so "
    + VOICE_TEXT_RULES + " The marks are removed from the text the user reads.]")

TEXT_GUIDANCE = " For a voice, " + VOICE_TEXT_RULES


def emotion_segments(text: str) -> list[tuple[Optional[str], str]]:
    """The reply cut at its ``[[emotion]]`` marks: ``(emotion or None, part)`` in order."""
    parts: list[tuple[Optional[str], str]] = []
    emotion, start = None, 0
    for mark in _EMOTION_MARK_RE.finditer(text):
        if text[start:mark.start()].strip():
            parts.append((emotion, text[start:mark.start()].strip()))
        emotion, start = mark.group(1).lower(), mark.end()
    if text[start:].strip():
        parts.append((emotion, text[start:].strip()))
    return parts


def strip_voice_marks(text: str) -> str:
    """The reply as the user reads it: without ``[[emotion]]`` marks and sound tags."""
    return _SOUND_TAG_RE.sub("", _EMOTION_MARK_RE.sub("", text))


def spoken_reply_note() -> Optional[str]:
    """The note on a voice message the gateway will answer by voice, for a 2.8 voice; ``None``
    keeps the gateway's plain note."""
    from tools.tts_tool import _get_provider, _load_tts_config
    tts_config = _load_tts_config()
    return SPOKEN_REPLY_NOTE if is_expressive(tts_config, _get_provider(tts_config)) else None


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
