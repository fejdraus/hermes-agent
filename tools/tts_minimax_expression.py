"""Expressive MiniMax speech (``speech-2.8-*``): an emotion per call, sound tags, numbers as words
and stress marks on Russian homographs.

When the model calls ``text_to_speech`` it picks the ``emotion`` and writes the sound tags into
the text; when the gateway voices a reply on its own, the auxiliary model does both and marks
the form of each number, whose words the code then writes (``tts_number_words``). Stress marks
come from ruaccent on both paths (``tts_ru_stress``), never from a language model. Older MiniMax
models would read the tags aloud, so all of this applies only to 2.8."""

from __future__ import annotations

import logging
import re
from typing import Any, Optional

logger = logging.getLogger(__name__)

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


REWRITE_TASK = "tts_audio_tags"
REWRITE_MAX_CHARS = 3000
_WORD_RE = re.compile(r"[^\W\d_]+")
_UNIT_AFTER_NUMBER_RE = re.compile(r"(\d)\s*[-°]?\s*[^\W\d_]{1,3}(?:/[^\W\d_]{1,3})?(?![^\W\d_])")
_THINK_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL | re.IGNORECASE)
_WORDS_PER_NUMBER = 6
_EXTRA_WORDS = 3


def _rewrite_prompt(with_digits: bool) -> str:
    """The rewrite instructions; the number marks are asked for only when the reply has digits,
    since a model asked for marks also turns numbers already written as words into marks."""
    from tools.tts_number_words import MARKUP_GUIDE
    numbers = MARKUP_GUIDE if with_digits else "Numbers are already words: leave them exactly as they are."
    return (
        "You prepare a chat reply to be spoken by a MiniMax speech-2.8 voice. Return the same "
        "text with only these changes:\n"
        "- " + numbers + "\n"
        "- A sound tag where a person would make that sound, rarely and only where the meaning "
        "calls for it: " + ", ".join(f"({tag})" for tag in SOUND_TAGS) + ".\n"
        "Keep every other word, its order and the meaning; do not shorten, add or explain "
        "anything, and write no stress marks.\n"
        "Answer in two parts: the first line is the emotion of the whole text, one of "
        + ", ".join(EMOTIONS) + ", or none for plain informational text; then the rewritten text.")


def _words(text: str) -> list[str]:
    return [word.lower() for word in _WORD_RE.findall(strip_sound_tags(text).replace("\u0301", ""))]


def keeps_the_text(original: str, rewritten: str) -> bool:
    """True when the rewrite kept every word of the original in order and added little beyond
    the numbers it spelled out (a unit abbreviated after a number may become a word): a
    shortened, extended or reworded reply, or one misspelled word, fails."""
    from difflib import SequenceMatcher
    from tools.tts_number_words import number_values
    before, after = _words(_UNIT_AFTER_NUMBER_RE.sub(r"\1 ", original)), _words(rewritten)
    if not after:
        return False
    kept = sum(block.size for block in SequenceMatcher(None, before, after, autojunk=False).get_matching_blocks())
    numbers = len(number_values(original))
    return kept == len(before) and len(after) - kept <= numbers * _WORDS_PER_NUMBER + _EXTRA_WORDS


def _parse_rewrite(reply: str) -> tuple[Optional[str], str]:
    """``(emotion, text)``: the first line is the emotion slot and is never spoken, even when the
    model wrote something else there; the reply it rewrites is one line."""
    reply = _THINK_RE.sub("", reply).strip()
    first, newline, rest = reply.partition("\n")
    if not newline:
        return None, reply
    emotion = first.strip().strip("*:. ").lower()
    return (emotion if emotion in EMOTIONS else None), rest.strip()


def rewrite_for_voice(text: str) -> tuple[str, Optional[str]]:
    """``(spoken_text, emotion)`` for a reply the gateway voices on its own: the auxiliary model
    adds the emotion and sound tags and marks the numbers, the code writes the numbers as words.
    Any failure, marks whose values differ from the reply's numbers, or a rewrite that changed
    the words keeps the original text with no emotion."""
    if len(text) > REWRITE_MAX_CHARS:
        return text, None
    try:
        from pm import ensure_import
        ensure_import("ru-voice")
        from agent.auxiliary_client import call_llm
        from tools.tts_number_words import number_values, render_numbers
        from tools.tts_tool_providers import _auxiliary_reply_text
        from tools.tts_voice_language import detect_language
        response = call_llm(
            task=REWRITE_TASK, temperature=0.3, max_tokens=len(text) * 3 + 200,
            extra_body={"thinking": {"type": "disabled"}},
            messages=[{"role": "system", "content": _rewrite_prompt(bool(number_values(text)))},
                      {"role": "user", "content": text}])
        emotion, marked = _parse_rewrite(_auxiliary_reply_text(response))
        spoken = render_numbers(text, marked.replace("\u0301", ""), detect_language(text) or "")
    except Exception as exc:
        logger.warning("Voice rewrite failed; speaking the reply as written: %s", exc)
        return text, None
    if spoken is None:
        logger.warning("Voice rewrite marked other numbers than the reply has; speaking it as written")
        return text, None
    if not keeps_the_text(text, spoken):
        logger.warning("Voice rewrite changed the reply; speaking it as written")
        return text, None
    return spoken, emotion
