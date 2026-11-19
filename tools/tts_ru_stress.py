"""Russian text made easier to read aloud: ё restored from ruaccent's dictionary, and a stress
mark on each homograph (за́мок / замо́к), chosen from the words around it.

Only the homograph model and two word lists are loaded (about 0.5 GB of memory), not ruaccent's
full stress dictionary (about 2 GB): the voice already stresses ordinary words right, it is the
words with two readings it gets wrong. ё comes from the list of words that always have it, not
from ruaccent's ё model, which turns "небо" into "нёбо". The files (about 360 MB) are downloaded
once into the machine-wide models directory and shared by every profile."""

from __future__ import annotations

import gzip
import json
import logging
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

STRESS = "́"
REPO = "ruaccent/accentuator"
MODEL = "nn/nn_omograph/turbo3.1"
MODEL_FILES = ("model.onnx", "config.json", "tokenizer.json", "tokenizer_config.json",
               "special_tokens_map.json", "added_tokens.json", "vocab.json", "merges.txt")
HOMOGRAPHS = "dictionary/omographs.json.gz"
YO_WORDS = "dictionary/yo_words.json.gz"
CONTEXT_WORDS = 10

_TOKEN_RE = re.compile(r"[а-яё]+(?:-[а-яё]+)*|[^а-яё\s]+", re.IGNORECASE)
_SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+(?=[,.?!:;…])")
_VOWELS = "аеёиоуыэюяАЕЁИОУЫЭЮЯ"


@dataclass(frozen=True)
class _Engine:
    model: Any
    homographs: dict[str, list[str]]
    yo_words: dict[str, str]


_lock = threading.Lock()
_engine: Optional[_Engine] = None
_failed = False


def models_dir() -> Path:
    from hermes_constants import get_default_hermes_root
    return get_default_hermes_root() / "models" / "ruaccent"


def _download(root: Path) -> None:
    from huggingface_hub import hf_hub_download
    for name in [HOMOGRAPHS, YO_WORDS] + [f"{MODEL}/{file}" for file in MODEL_FILES]:
        if not (root / name).exists():
            hf_hub_download(repo_id=REPO, filename=name, local_dir=root)


def _read_json(path: Path) -> Any:
    with gzip.open(path) as handle:
        return json.load(handle)


def _load() -> Optional[_Engine]:
    """The homograph model and word lists, loaded once per process; ``None`` when unavailable."""
    global _engine, _failed
    with _lock:
        if _engine is not None or _failed:
            return _engine
        try:
            from pm import ensure_import
            ensure_import("ru-voice")
            from ruaccent.omograph_model import OmographModel
            root = models_dir()
            _download(root)
            model = OmographModel()
            model.load(str(root / MODEL))
            _engine = _Engine(model, _read_json(root / HOMOGRAPHS), _read_json(root / YO_WORDS))
        except Exception as exc:
            _failed = True
            logger.warning("Russian stress marks are unavailable: %s", exc)
        return _engine


def warm_up() -> None:
    """Load the model on a background thread, so the first voice reply does not wait for it."""
    threading.Thread(target=_load, name="ru-stress-warm-up", daemon=True).start()


def _same_case(source: str, word: str) -> str:
    if source.isupper() and len(source) > 1:
        return word.upper()
    return word[:1].upper() + word[1:] if source[:1].isupper() else word


def _marked(variant: str, word: str) -> str:
    """``з+амок`` as the original word with U+0301 after its stressed vowel, keeping its case."""
    plus = variant.find("+")
    if plus < 0 or plus + 1 > len(word) or word[plus] not in _VOWELS:
        return word
    return word[:plus + 1] + STRESS + word[plus + 1:]


def prepare_russian(text: str) -> str:
    """``text`` with ё restored where a word always has it and a stress mark on each homograph,
    chosen from the words around it. Earlier stress marks are dropped first: a guessed stress on
    an ordinary word is often wrong."""
    text = text.replace(STRESS, "")
    engine = _load()
    if engine is None:
        return text
    tokens = [token.group(0) for token in _TOKEN_RE.finditer(text)]
    spans = [token.span() for token in _TOKEN_RE.finditer(text)]
    tokens = [_same_case(token, engine.yo_words[token.lower()]) if token.lower() in engine.yo_words else token
              for token in tokens]
    words = [token.lower() for token in tokens]
    found = [(i, engine.homographs[word]) for i, word in enumerate(words) if word in engine.homographs]
    chosen: list[str] = []
    if found:
        batch, hypotheses, counts = [], [], []
        for i, variants in found:
            context = words.copy()
            context[i] = f" <w>{context[i]}</w> "
            context = context[max(0, i - CONTEXT_WORDS):i + CONTEXT_WORDS + 1]
            batch += [_SPACE_BEFORE_PUNCT_RE.sub("", " ".join(context))] * len(variants)
            hypotheses += variants
            counts.append(len(variants))
        try:
            chosen = engine.model.classify(batch, hypotheses, counts)
        except Exception as exc:
            logger.warning("Russian stress marks failed: %s", exc)
            found = []
    for (i, _), variant in zip(found, chosen):
        tokens[i] = _marked(variant, tokens[i])
    pieces, last = [], 0
    for (start, end), token in zip(spans, tokens):
        pieces += [text[last:start], token]
        last = end
    return "".join(pieces) + text[last:]
