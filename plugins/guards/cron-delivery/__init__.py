"""cron-delivery-guard — Jev перевіряє відповідь крона перед доставкою.

Агентський крон інколи надсилає замість повідомлення робочу кухню: міркування про
заблоковані інструменти, звіт про кроки, назви скриптів і таблиць. Правило в промпті
цього не зупиняє, тому перевірка стоїть у коді: Jev (TypeSafe) оцінює, чи відповідь — саме
те повідомлення, яке просило завдання, маючи промпт завдання як контекст.

* пройшло — відповідь іде без змін;
* не пройшло, але є абзац, явно звернений до людини, — іде лише він;
* інакше — ``[SILENT]`` (планувальник таке не надсилає) і запис у лог.

Штатні формати не перевіряються: ``MEDIA:`` (голос), ``[CRON_FAILURE]``, ``[SILENT]``, а також
кронам зі ``skip_jobs`` технічний звіт і потрібен. Будь-який збій — відповідь іде як є.

Config у config.yaml профілю (шлюз мультиплексний, тож рішення — на кожен виклик):
  plugins:
    cron-delivery-guard:
      enabled: true
      skip_jobs: [memory-hygiene-paul]
"""

from __future__ import annotations

import json
import logging
import os
import re
import urllib.request
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_JEV_URL = "https://api.typesafe.ai/v1/systemone"
_TIMEOUT = 20.0
_READY_MIN = 0.5
_PARA_MIN = 0.85
_PASS_PREFIXES = ("MEDIA:", "[CRON_FAILURE]", "[SILENT]")
_Q_READY = ("The response is the message the job asked to send to the user, ready to deliver: it "
            "addresses the user and contains no leaked working notes — no reasoning about tools, "
            "commands, blocked actions, errors, the job's own steps, or internal file, script or "
            "database names that the user did not ask about.")
_Q_PARA = ("This paragraph is written to the user as part of the message the job asked to send "
           "(not the assistant's working notes or reasoning).")


def _home() -> str:
    try:
        from hermes_cli.config import get_hermes_home
        return str(get_hermes_home())
    except Exception:
        return os.environ.get("HERMES_HOME") or os.path.expanduser("~/.hermes")


def _settings() -> Dict[str, Any]:
    try:
        from hermes_cli.config import load_config
        return ((load_config() or {}).get("plugins") or {}).get("cron-delivery-guard") or {}
    except Exception:
        return {}


def _key(home: str) -> Optional[str]:
    try:
        with open(os.path.join(home, ".env"), encoding="utf-8") as fh:
            m = re.search(r"^TYPESAFE_API_KEY=[\"']?([^\"'\s]+)", fh.read(), re.M)
        return m.group(1) if m else None
    except OSError:
        return None


def _job(home: str, session_id: str) -> Optional[Dict[str, Any]]:
    m = re.match(r"cron_([0-9a-f]+)_", session_id or "")
    if not m:
        return None
    try:
        with open(os.path.join(home, "cron", "jobs.json"), encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return None
    jobs = data.get("jobs", []) if isinstance(data, dict) else data
    return next((j for j in jobs if isinstance(j, dict) and j.get("id") == m.group(1)), None)


def _ask(key: str, state: str, questions: Dict[str, Any]) -> Dict[str, Any]:
    body = json.dumps({"model": "jev-latest", "state": state[:6000], "questions": questions}).encode()
    req = urllib.request.Request(_JEV_URL, data=body, headers={
        "Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=_TIMEOUT) as r:
        resp = json.load(r)
    return resp.get("answers") or (resp.get("result") or {}).get("answers") or {}


def _prob(a: Any) -> float:
    if isinstance(a, dict):
        for k in ("noul", "probability", "value"):
            v = a.get(k)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                return float(v)
    return 0.0


def guard(response_text: str, prompt: str, key: str) -> Optional[str]:
    """None — надіслати як є; рядок — надіслати його замість відповіді."""
    state = f"Job instructions:\n{prompt[-1500:]}\n\nResponse to deliver:\n{response_text[:3000]}"
    ready = _prob(_ask(key, state, {"r": {"type": "noul", "instructions": _Q_READY}}).get("r"))
    if ready >= _READY_MIN:
        return None
    paras: List[str] = [p.strip() for p in re.split(r"\n\s*\n", response_text) if p.strip()][:12]
    scores = _ask(key, state, {f"p{i}": {"type": "noul", "instructions": {
        "question": _Q_PARA, "paragraph": p[:800]}} for i, p in enumerate(paras)}) if paras else {}
    keep = [p for i, p in enumerate(paras) if _prob(scores.get(f"p{i}")) >= _PARA_MIN]
    logger.warning("cron-delivery-guard: response not ready (%.2f) — %s", ready,
                   f"kept {len(keep)} of {len(paras)} paragraph(s)" if keep else "suppressed")
    return "\n\n".join(keep) if keep else "[SILENT]"


def on_transform_llm_output(response_text: str = "", session_id: str = "", **_: Any) -> Optional[str]:
    if not response_text or not str(session_id).startswith("cron_"):
        return None
    if response_text.lstrip().startswith(_PASS_PREFIXES):
        return None
    settings = _settings()
    if not settings.get("enabled"):
        return None
    home = _home()
    job = _job(home, session_id)
    if job is None or job.get("name") in (settings.get("skip_jobs") or []):
        return None
    key = _key(home)
    if not key:
        return None
    try:
        return guard(response_text, str(job.get("prompt") or ""), key)
    except Exception as e:
        logger.info("cron-delivery-guard: jev unavailable (%s) — delivering as is", e)
        return None


def register(ctx) -> None:
    ctx.register_hook("transform_llm_output", on_transform_llm_output)
