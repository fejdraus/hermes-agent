"""Ход, доеденный из очереди, тоже показывает индикатор работы.

Индикатор поднимается там, где сообщение ОТКРЫВАЕТ ход. Сообщение, пришедшее во время
работы, паркуется и доедается другим путём — и там индикатора не было. Для пачки вложений,
которую шлюз намеренно поглощает вместо прерывания, это ровно тот случай: минуты распознавания
выглядят как молчащий бот. Сигнал косметический, поэтому он не имеет права стоить хода.
"""
from __future__ import annotations

import asyncio

import pytest

from gateway.run_turn_followup_ack import _start_followup_typing, _stop_followup_typing


class _Event:
    message_id = 7


class _Adapter:
    def __init__(self, *, raises=False, no_indicator=False):
        self.started = []
        self.raises = raises
        if no_indicator:
            return
        self._start_typing_refresh = self._start  # type: ignore[assignment]

    def _start(self, event, stop_event, metadata):
        if self.raises:
            raise RuntimeError("transport down")
        self.started.append((event, metadata))
        return asyncio.get_event_loop().create_future()

    def _reply_metadata(self, event):
        return {"thread": 3}


def test_indicator_is_raised_for_a_drained_followup():
    async def go():
        adapter = _Adapter()
        task = _start_followup_typing(adapter, _Event())
        assert adapter.started, "the follow-up ran with no working indicator"
        assert adapter.started[0][1] == {"thread": 3}
        _stop_followup_typing(task)

    asyncio.run(go())


def test_missing_adapter_or_event_is_tolerated():
    assert _start_followup_typing(None, _Event()) is None
    assert _start_followup_typing(_Adapter(), None) is None


def test_adapter_without_indicator_is_tolerated():
    assert _start_followup_typing(_Adapter(no_indicator=True), _Event()) is None


def test_a_failing_indicator_never_breaks_the_turn():
    async def go():
        assert _start_followup_typing(_Adapter(raises=True), _Event()) is None

    asyncio.run(go())


def test_stop_is_safe_on_nothing():
    _stop_followup_typing(None)


def test_drain_path_raises_and_clears_the_indicator():
    """Every exit of the drain — success, failure, cancel — must drop the indicator."""
    import inspect

    from gateway.run_turn import TurnRunner

    src = inspect.getsource(TurnRunner._run_agent_queued_followup)
    assert src.count("_stop_followup_typing(") == 3, "an exit path leaves the indicator running"
    assert src.index("_start_followup_typing(") < src.index("_stop_followup_typing(")
