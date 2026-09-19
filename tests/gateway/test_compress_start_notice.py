"""Ручной /compress сообщает о начале работы.

Команда выполняется inline и отвечает только когда сумматор вернул результат — на
длинном транскрипте это минуты. Без сообщения о старте это неотличимо от потерянной
команды, поэтому уведомление обязано уйти до начала работы и не обязано её ломать.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]


class _Adapter:
    def __init__(self, fail=False):
        self.sent = []
        self.fail = fail

    async def send(self, chat_id, text, metadata=None):
        if self.fail:
            raise RuntimeError("transport down")
        self.sent.append((chat_id, text))


class _Source:
    chat_id = "chat-1"


class _Gateway:
    def __init__(self, adapter):
        self._adapter = adapter

    def _intake_adapter_for(self, source):
        return self._adapter

    def _reply_metadata(self, event):
        return {"thread": 1}

    _announce_manual_compression = None


@pytest.fixture
def announce():
    from gateway.slash_commands_session import GatewaySessionCommandsMixin

    return GatewaySessionCommandsMixin._announce_manual_compression


def test_notice_is_sent_before_work(announce):
    adapter = _Adapter()
    asyncio.run(announce(_Gateway(adapter), object(), _Source()))
    assert len(adapter.sent) == 1
    chat_id, text = adapter.sent[0]
    assert chat_id == "chat-1" and text.strip()


def test_transport_failure_does_not_propagate(announce):
    asyncio.run(announce(_Gateway(_Adapter(fail=True)), object(), _Source()))


def test_missing_adapter_is_tolerated(announce):
    class _NoAdapter(_Gateway):
        def _intake_adapter_for(self, source):
            return None

    asyncio.run(announce(_NoAdapter(None), object(), _Source()))


def test_every_locale_defines_the_start_notice():
    missing = []
    for path in sorted((ROOT / "locales").glob("*.yaml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        compress = ((data.get("gateway") or {}).get("compress") or {})
        if "not_enough" in compress and not str(compress.get("started") or "").strip():
            missing.append(path.name)
    assert missing == [], f"locales missing compress.started: {missing}"


def test_start_notice_is_translated():
    en = yaml.safe_load((ROOT / "locales" / "en.yaml").read_text(encoding="utf-8"))
    english = en["gateway"]["compress"]["started"]
    for lang in ("ru", "uk", "de"):
        data = yaml.safe_load((ROOT / "locales" / f"{lang}.yaml").read_text(encoding="utf-8"))
        assert data["gateway"]["compress"]["started"] != english, lang
