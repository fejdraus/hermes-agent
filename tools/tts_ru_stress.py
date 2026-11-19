"""Stress marks on Russian homographs (за\u0301мок / замо\u0301к), chosen by ruaccent from the ten words
on each side of it: on real replies that reads more homographs right than the whole reply or
its sentence.

Only the homograph model and its word list are loaded (about 0.5 GB of memory), not ruaccent's
full stress dictionary (about 2 GB): the voice already stresses ordinary words right, it is the
words with two readings it gets wrong. The model files (351 MB) are downloaded once into the
machine-wide models directory and shared by every profile."""

from __future__ import annotations

import gzip
import json
import logging
import re
import threading
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

STRESS = "\u0301"
REPO = "ruaccent/accentuator"
MODEL = "nn/nn_omograph/turbo3.1"
MODEL_FILES = ("model.onnx", "config.json", "tokenizer.json", "tokenizer_config.json",
               "special_tokens_map.json", "added_tokens.json", "vocab.json", "merges.txt")
HOMOGRAPHS = "dictionary/omographs.json.gz"
CONTEXT_WORDS = 10

_TOKEN_RE = re.compile(r"[а-яё]+(?:-[а-яё]+)*|[^а-яё\s]+", re.IGNORECASE)
_SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+(?=[,.?!:;…])")
_VOWELS = "аеёиоуыэюяАЕЁИОУЫЭЮЯ"

_lock = threading.Lock()
_engine: Optional[tuple[Any, dict[str, list[str]]]] = None
_failed = False


def models_dir() -> Path:
    from hermes_constants import get_default_hermes_root
    return get_default_hermes_root() / "models" / "ruaccent"


def _download(root: Path) -> None:
    from huggingface_hub import hf_hub_download
    for name in [HOMOGRAPHS] + [f"{MODEL}/{file}" for file in MODEL_FILES]:
        if not (root / name).exists():
            hf_hub_download(repo_id=REPO, filename=name, local_dir=root)


def _load() -> Optional[tuple[Any, dict[str, list[str]]]]:
    """The homograph model and word list, loaded once per process; ``None`` when unavailable."""
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
            with gzip.open(root / HOMOGRAPHS) as handle:
                homographs = json.load(handle)
            _engine = (model, homographs)
        except Exception as exc:
            _failed = True
            logger.warning("Russian stress marks are unavailable: %s", exc)
        return _engine


def warm_up() -> None:
    """Load the model on a background thread, so the first voice reply does not wait for it."""
    threading.Thread(target=_load, name="ru-stress-warm-up", daemon=True).start()


def _marked(variant: str, word: str) -> str:
    """``з+амок`` as the original word with U+0301 after its stressed vowel, keeping its case."""
    plus = variant.find("+")
    if plus < 0 or plus + 1 > len(word) or word[plus] not in _VOWELS:
        return word
    return word[:plus + 1] + STRESS + word[plus + 1:]


def stress_homographs(text: str) -> str:
    """``text`` with a stress mark on each Russian homograph, chosen from the words around it. Earlier
    stress marks are dropped first: a guessed stress on an ordinary word is often wrong."""
    text = text.replace(STRESS, "")
    engine = _load()
    if engine is None:
        return text
    model, homographs = engine
    tokens = list(_TOKEN_RE.finditer(text))
    words = [token.group(0).lower() for token in tokens]
    found = [(i, homographs[word]) for i, word in enumerate(words) if word in homographs]
    if not found:
        return text
    batch, hypotheses, counts = [], [], []
    for i, variants in found:
        context = words.copy()
        context[i] = f" <w>{context[i]}</w> "
        context = context[max(0, i - CONTEXT_WORDS):i + CONTEXT_WORDS + 1]
        batch += [_SPACE_BEFORE_PUNCT_RE.sub("", " ".join(context))] * len(variants)
        hypotheses += variants
        counts.append(len(variants))
    try:
        chosen = model.classify(batch, hypotheses, counts)
    except Exception as exc:
        logger.warning("Russian stress marks failed: %s", exc)
        return text
    pieces, last = [], 0
    for (i, _), variant in zip(found, chosen):
        token = tokens[i]
        pieces += [text[last:token.start()], _marked(variant, token.group(0))]
        last = token.end()
    return "".join(pieces) + text[last:]
