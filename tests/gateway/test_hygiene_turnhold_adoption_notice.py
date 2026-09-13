"""Отложенное сжатие сообщает о завершении.

Пользователю говорят «ход продолжается без сжатия», а работа идёт дальше в фоне и
может занять минуты. Молчание после этого неотличимо от зависшей операции, поэтому
принятие сводки должно отзываться — и ровно один раз, только когда она принята.
"""
from __future__ import annotations

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def test_every_locale_defines_the_adoption_notice():
    missing = []
    for path in sorted((ROOT / "locales").glob("*.yaml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        compress = ((data.get("gateway") or {}).get("compress") or {})
        if "turnhold_deferred" in compress and not str(compress.get("turnhold_adopted") or "").strip():
            missing.append(path.name)
    assert missing == [], f"locales missing turnhold_adopted: {missing}"


def test_adoption_notice_is_translated_not_english_fallback():
    en = yaml.safe_load((ROOT / "locales" / "en.yaml").read_text(encoding="utf-8"))
    english = en["gateway"]["compress"]["turnhold_adopted"]
    for lang in ("ru", "uk", "de", "fr"):
        data = yaml.safe_load((ROOT / "locales" / f"{lang}.yaml").read_text(encoding="utf-8"))
        assert data["gateway"]["compress"]["turnhold_adopted"] != english, lang


def test_notice_is_sent_only_on_the_committed_branch():
    source = (ROOT / "gateway" / "run_turn.py").read_text(encoding="utf-8")
    assert source.count('t("gateway.compress.turnhold_adopted")') == 1
    callback = source[source.index("def _hyg_adopt_or_space_retry"):]
    callback = callback[:callback.index("attempt.future.add_done_callback")]
    branch = callback[callback.index("                if _committed:"):]
    committed, _, not_committed = branch.partition("                else:")
    assert "turnhold_adopted" in committed
    assert "turnhold_adopted" not in not_committed


def test_in_progress_reply_promises_nothing_it_cannot_deliver():
    from agent.manual_compression_feedback import describe_compression_lock_skip

    reply = describe_compression_lock_skip("pid=1:tid=2")
    assert "pid=1:tid=2" in reply
    assert not re.search(r"wait for it to finish", reply, re.I)
