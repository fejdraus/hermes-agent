# pyright: basic
"""Russian / Ukrainian symbol + unit expansion for TTS (fork addition).

Kept separate from tts_text_normalize (which is escape-only by convention) so
Cyrillic here can be written as readable literal UTF-8. The English pipeline in
tts_text_normalize expands units to English words ("degrees Celsius", "percent")
which, injected into Russian text, makes MiniMax code-switch ru<->en. For
Cyrillic-dominant text we expand to Russian/Ukrainian words instead.
"""
from __future__ import annotations

import re

_CYRILLIC_RE = re.compile(r"[Ѐ-ӿ]")
_LATIN_RE = re.compile(r"[A-Za-z]")
_UK_RE = re.compile(r"[іїєґІЇЄҐ]")


def detect_speech_language(text: str):
    """Return 'uk'/'ru' when text is predominantly Cyrillic, else None."""
    if not text:
        return None
    cyr = len(_CYRILLIC_RE.findall(text))
    if cyr == 0:
        return None
    if cyr < len(_LATIN_RE.findall(text)):
        return None
    return "uk" if _UK_RE.search(text) else "ru"


def _sign(num: str, plus: str, minus: str) -> str:
    num = num.strip()
    if num.startswith("+"):
        return plus + " " + num[1:]
    if num.startswith("-"):
        return minus + " " + num[1:]
    return num


def normalize_symbols_cyrillic(text: str, lang: str) -> str:
    """Expand symbols/units into Russian or Ukrainian words for TTS."""
    uk = lang == "uk"
    deg = "градусів" if uk else "градусов"
    pct = "відсотків" if uk else "процентов"
    kmh = "кілометрів на годину" if uk else "километров в час"
    mps = "метрів за секунду" if uk else "метров в секунду"
    mm = "міліметрів" if uk else "миллиметров"
    cm = "сантиметрів" if uk else "сантиметров"
    plus = "плюс"
    minus = "мінус" if uk else "минус"
    to = "до"

    text = str(text)
    text = re.sub("[   ]", " ", text)
    text = text.replace("−", "-")

    # temperature range: "N–M °C" -> "N to M degrees"
    text = re.sub(
        r"(?<!\w)([+\-]?\d+(?:[.,]\d+)?)\s*[–—…\-]+\s*([+\-]?\d+(?:[.,]\d+)?)\s*°\s*[CСcс]",
        lambda m: f"{_sign(m.group(1), plus, minus)} {to} {_sign(m.group(2), plus, minus)} {deg}",
        text,
    )
    # single temperature: "N °C"
    text = re.sub(
        r"(?<!\w)([+\-]?\d+(?:[.,]\d+)?)\s*°\s*[CСcс]",
        lambda m: f"{_sign(m.group(1), plus, minus)} {deg}",
        text,
    )
    # bare degree
    text = re.sub(r"°\s*[CСcс]", deg, text)
    text = text.replace("°", " " + deg)

    # weather / travel units (latin + cyrillic forms)
    text = re.sub(r"(?<=\d)\s*(?:km\s*/\s*h|км\s*/\s*год|км\s*/\s*ч)\b", " " + kmh, text, flags=re.IGNORECASE)
    text = re.sub(r"(?<=\d)\s*(?:m\s*/\s*s|м\s*/\s*с)\b", " " + mps, text, flags=re.IGNORECASE)
    text = re.sub(r"(?<=\d)\s*(?:mm|мм)\b", " " + mm, text, flags=re.IGNORECASE)
    text = re.sub(r"(?<=\d)\s*(?:cm|см)\b", " " + cm, text, flags=re.IGNORECASE)
    text = re.sub(r"(?<=\d)\s*%", " " + pct, text)

    # numeric range without unit: "+18…+20"
    text = re.sub(
        r"(?<!\w)([+\-]\d+(?:[.,]\d+)?)\s*[–—…]+\s*([+\-]?\d+(?:[.,]\d+)?)",
        lambda m: f"{_sign(m.group(1), plus, minus)} {to} {_sign(m.group(2), plus, minus)}",
        text,
    )
    # standalone sign before a number (not after a word/digit -> keeps dates safe)
    text = re.sub(r"(?<![\w])\+(?=\d)", plus + " ", text)
    text = re.sub(r"(?<![\w])-(?=\d)", minus + " ", text)

    # leftovers
    text = text.replace("…", " " + to + " ")
    text = re.sub("[•◦▪▫]", " ", text)
    text = text.replace("→", " " + to + " ")
    text = text.replace("&", " і " if uk else " и ")
    text = text.replace("~", " ")
    return text
