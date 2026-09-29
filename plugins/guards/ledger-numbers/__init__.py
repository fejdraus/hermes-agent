"""ledger-number-guard — позначка на відповіді з числами, не звіреними з обліком.

Правило «число про їжу — лише після виклику обліку» записане в навичці, тобто це
прохання до моделі, і раз на тиждень вона його порушує: меню дня з калоріями з пам'яті,
калорії круасана «на око». Тут перевірка в коді, без повтору ходу й без правки ядра:

* плагін запам'ятовує, чи був у цьому ході виклик обліку (скрипти з ``ledger_markers``);
* якщо не був, а у відповіді є кількості їжі, Jev (TypeSafe) вирішує, чи подає відповідь
  калорії, вагу порцій або залишки як факт, визначений самим асистентом, — а не числа
  людини, її денну ціль чи цифри зі скриншота;
* так — до відповіді дописується ``footer``; ні або будь-який збій — відповідь іде як є.

Кронів не торкається (там свій cron-delivery-guard).

Config у config.yaml профілю:
  plugins:
    ledger-number-guard:
      enabled: true
      ledger_markers: [inventory.py, menu_, day.py, nutrition_search.py]   # необов'язково
      footer: "..."                                                         # необов'язково
"""

from __future__ import annotations

import json
import logging
import os
import re
import urllib.request
from collections import OrderedDict
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_JEV_URL = "https://api.typesafe.ai/v1/systemone"
_TIMEOUT = 15.0
_CLAIM_MIN = 0.7
_TRACK_MAX = 256
_DEFAULT_MARKERS = ("inventory.py", "menu_", "day.py", "nutrition_search.py")
_DEFAULT_FOOTER = ("⚠️ Цифри в цій відповіді не звірені з базою — напиши «перевір», "
                   "і я порахую через облік.")
_FOOD_NUMBER = re.compile(r"\d+(?:[.,]\d+)?\s?(?:г|гр|кг|мл|л|ккал|kcal|шт|порц)(?![а-яіїєґa-z])", re.I)
_Q_CLAIM = ("The assistant's reply states calories, protein/fat/carb grams, portion weights or amounts "
            "of food at home as facts the assistant itself determined. Numbers the user gave in their "
            "message, the user's own daily targets, and numbers read from an image or screenshot the "
            "user sent do not count.")

_ledger_turns: "OrderedDict[tuple, bool]" = OrderedDict()
_user_messages: "OrderedDict[tuple, str]" = OrderedDict()


def _remember(store: OrderedDict, key: tuple, value: Any) -> None:
    store[key] = value
    store.move_to_end(key)
    while len(store) > _TRACK_MAX:
        store.popitem(last=False)


def _home() -> str:
    try:
        from hermes_cli.config import get_hermes_home
        return str(get_hermes_home())
    except Exception:
        return os.environ.get("HERMES_HOME") or os.path.expanduser("~/.hermes")


def _settings() -> Dict[str, Any]:
    try:
        from hermes_cli.config import load_config
        return ((load_config() or {}).get("plugins") or {}).get("ledger-number-guard") or {}
    except Exception:
        return {}


def _key(home: str) -> Optional[str]:
    try:
        with open(os.path.join(home, ".env"), encoding="utf-8") as fh:
            m = re.search(r"^TYPESAFE_API_KEY=[\"']?([^\"'\s]+)", fh.read(), re.M)
        return m.group(1) if m else None
    except OSError:
        return None


def _prob(a: Any) -> float:
    if isinstance(a, dict):
        for k in ("noul", "probability", "value"):
            v = a.get(k)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                return float(v)
    return 0.0


def claim_probability(key: str, user_message: str, reply: str) -> float:
    state = f"User message:\n{user_message[-1500:]}\n\nAssistant reply:\n{reply[:3500]}"
    body = json.dumps({"model": "jev-latest", "state": state,
                       "questions": {"c": {"type": "noul", "instructions": _Q_CLAIM}}}).encode()
    req = urllib.request.Request(_JEV_URL, data=body, headers={
        "Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=_TIMEOUT) as r:
        resp = json.load(r)
    answers = resp.get("answers") or (resp.get("result") or {}).get("answers") or {}
    return _prob(answers.get("c"))


def on_pre_llm_call(session_id: str = "", turn_id: str = "", user_message: Any = "", **_: Any) -> None:
    if session_id and turn_id:
        _remember(_user_messages, (session_id, turn_id), str(user_message or ""))


def on_post_tool_call(args: Any = None, session_id: str = "", turn_id: str = "", **_: Any) -> None:
    if not (session_id and turn_id):
        return
    markers = _settings().get("ledger_markers") or _DEFAULT_MARKERS
    text = args if isinstance(args, str) else json.dumps(args, ensure_ascii=False, default=str)
    if any(m in text for m in markers):
        _remember(_ledger_turns, (session_id, turn_id), True)


def on_transform_llm_output(response_text: str = "", session_id: str = "", turn_id: str = "",
                            **_: Any) -> Optional[str]:
    if not response_text or not session_id or str(session_id).startswith("cron_"):
        return None
    turn = (session_id, turn_id)
    if _ledger_turns.pop(turn, False):
        _user_messages.pop(turn, None)
        return None
    user_message = _user_messages.pop(turn, "")
    if not _FOOD_NUMBER.search(response_text):
        return None
    settings = _settings()
    if not settings.get("enabled"):
        return None
    key = _key(_home())
    if not key:
        return None
    try:
        p = claim_probability(key, user_message, response_text)
    except Exception as e:
        logger.info("ledger-number-guard: jev unavailable (%s) — reply as is", e)
        return None
    if p < _CLAIM_MIN:
        return None
    logger.warning("ledger-number-guard: unverified food numbers (%.2f) — footer added", p)
    return f"{response_text.rstrip()}\n\n{settings.get('footer') or _DEFAULT_FOOTER}"


def register(ctx) -> None:
    ctx.register_hook("pre_llm_call", on_pre_llm_call)
    ctx.register_hook("post_tool_call", on_post_tool_call)
    ctx.register_hook("transform_llm_output", on_transform_llm_output)
