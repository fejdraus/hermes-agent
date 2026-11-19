"""Numbers as words whose value the code guarantees.

The auxiliary model only marks each number with the form the sentence needs,
``{{21|ord gen n}}``; this module checks that the marked values are exactly the numbers of the
original text, in order, and writes the words with num2words. A number the library cannot put in
the asked form keeps its digits, which the voice then reads itself."""

from __future__ import annotations

import logging
import re
from decimal import Decimal, InvalidOperation
from typing import Optional

logger = logging.getLogger(__name__)

MARK_RE = re.compile(r"\{\{\s*([0-9]+(?:[.,:][0-9]+)*)\s*(?:\|([^{}]*))?\}\}")
_THOUSANDS_RE = re.compile(r"(?<=\d)[ \u00a0\u202f](?=\d{3}(?!\d))")
_DIGITS_RE = re.compile(r"\d+")

CASES = {"nom": "nominative", "gen": "genitive", "dat": "dative", "acc": "accusative",
         "ins": "instrumental", "prep": "prepositional"}
GENDERS = {"m", "f", "n"}
CASED_LANGUAGES = {"ru", "uk"}

MARKUP_GUIDE = (
    "Replace every number written in digits with a mark {{value|form}}: value is the number as "
    "written (\"5,6\"), form is the words that say how it is read in this sentence: card (amount) "
    "or ord (ordinal), the case nom, gen, dat, acc, ins or prep, the gender m, f or n, pl for "
    "a plural ordinal, and anim when it counts people or animals in the accusative (\"вижу "
    "{{3|card acc anim}} человек\"). A time is one mark (\"в 18:30\" -> \"в {{18:30|time}}\"). Write a unit or sign after the number as a word in the "
    "form the number needs (\"150 г\" -> \"{{150|card nom}} граммов\", \"+12 °C\" -> \"плюс "
    "{{12|card nom}} градусов\", \"21 ноября\" -> \"{{21|ord gen n}} ноября\", \"на 3 порции\" -> "
    "\"на {{3|card acc f}} порции\").")

_FRACTION_STEMS = {1: "десят", 2: "сот", 3: "тысячн", 4: "десятитысячн"}
_ADJECTIVE_ENDINGS = {
    "nominative": ("ая", "ых"), "genitive": ("ой", "ых"), "dative": ("ой", "ым"),
    "accusative": ("ую", "ых"), "instrumental": ("ой", "ыми"), "prepositional": ("ой", "ых"),
}


def number_values(text: str) -> list[str]:
    """The digit groups of a text, with thousands separators joined (``1 000`` is one number)."""
    return _DIGITS_RE.findall(_THOUSANDS_RE.sub("", text))


def _form(spec: str) -> tuple[bool, str, Optional[str], bool, bool]:
    tokens = (spec or "").replace(",", " ").lower().split()
    ordinal = "ord" in tokens
    case = next((CASES[t] for t in tokens if t in CASES), "nominative")
    gender = next((t for t in tokens if t in GENDERS), None)
    return ordinal, case, gender, "pl" in tokens, "anim" in tokens


def _ru_adjective(stem: str, number: int, case: str) -> str:
    singular = number % 10 == 1 and number % 100 != 11
    return stem + _ADJECTIVE_ENDINGS[case][0 if singular else 1]


def _decimal_words(value: str, case: str, lang: str) -> Optional[str]:
    from num2words import num2words
    whole, _, fraction = value.replace(".", ",").partition(",")
    if lang != "ru":
        return num2words(Decimal(f"{whole}.{fraction}"), lang=lang) if lang not in CASED_LANGUAGES or case == "nominative" else None
    stem = _FRACTION_STEMS.get(len(fraction))
    if stem is None:
        return None
    whole_n, fraction_n = int(whole), int(fraction)
    return " ".join((num2words(whole_n, lang="ru", case=case, gender="f"), _ru_adjective("цел", whole_n, case),
                     num2words(fraction_n, lang="ru", case=case, gender="f"), _ru_adjective(stem, fraction_n, case)))


def _leading_zeros(value: str, lang: str) -> tuple[str, str]:
    """``("ноль ", "5")`` for ``05``: a time or code reads its leading zeros aloud."""
    digits = value.lstrip("0") or "0"
    zeros = len(value) - len(digits)
    if not zeros:
        return "", value
    from num2words import num2words
    return (num2words(0, lang=lang) + " ") * zeros, digits


def _time_words(value: str, lang: str) -> str:
    """``18:30`` as "восемнадцать тридцать", ``7:05`` as "семь ноль пять"."""
    from num2words import num2words
    words = []
    for part in value.split(":"):
        prefix, digits = _leading_zeros(part, lang)
        words.append(prefix + num2words(int(digits), lang=lang))
    return " ".join(words)


def number_words(value: str, spec: str, lang: str) -> Optional[str]:
    """The words of one marked number in ``lang``, or ``None`` when that form is not available."""
    ordinal, case, gender, plural, animate = _form(spec)
    try:
        from num2words import num2words
        if ":" in value:
            return _time_words(value, lang)
        if "," in value or "." in value:
            return None if ordinal else _decimal_words(value, case, lang)
        prefix, digits = _leading_zeros(value, lang)
        number = int(digits)
        if lang == "ru":
            kwargs = {"case": case, "gender": gender or "m", "plural": plural, "animate": animate}
            return prefix + num2words(number, lang="ru", to="ordinal" if ordinal else "cardinal", **kwargs)
        if lang == "uk":
            if ordinal:
                return prefix + num2words(number, lang="uk", to="ordinal") if case == "nominative" and gender in (None, "m") and not plural else None
            return prefix + num2words(number, lang="uk", case=case, **({"gender": gender} if gender else {}))
        return prefix + num2words(number, lang=lang, to="ordinal" if ordinal else "cardinal")
    except (ImportError, NotImplementedError, InvalidOperation, ValueError, TypeError, KeyError) as exc:
        logger.debug("num2words cannot say %r (%s, %s): %s", value, spec, lang, exc)
        return None


def render_numbers(original: str, marked: str, lang: str) -> Optional[str]:
    """``marked`` with every mark written as words, or ``None`` when its values are not exactly
    the numbers of ``original`` in order, or digits stand outside the marks."""
    values = [group for match in MARK_RE.finditer(marked) for group in number_values(match.group(1))]
    if values != number_values(original):
        return None
    if _DIGITS_RE.search(MARK_RE.sub("", marked)):
        return None
    return MARK_RE.sub(lambda m: number_words(m.group(1), m.group(2) or "", lang) or m.group(1), marked)
