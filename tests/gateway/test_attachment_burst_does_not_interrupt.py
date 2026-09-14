"""Пачка вложений не прерывает идущий ход.

Двадцать один PDF приходит двадцатью одним сообщением. Раньше в очередь попадал только
текст и фото, а документ прерывал ход — вместе с ходом рвался поток к провайдеру, и это
всплывало как APIConnectionError, то есть как сбой сети, а не как следствие отправки.
Голос из правила исключён намеренно: речью пользователь прерывает осознанно.
"""
from __future__ import annotations

import pytest

from gateway.platforms.event import MessageType
from gateway.run_inbound import _ATTACHMENT_BURST_TYPES


@pytest.mark.parametrize("kind", [MessageType.PHOTO, MessageType.DOCUMENT, MessageType.VIDEO])
def test_file_attachments_are_absorbed(kind):
    assert kind in _ATTACHMENT_BURST_TYPES


@pytest.mark.parametrize("kind", [MessageType.VOICE, MessageType.TEXT])
def test_speech_and_text_still_interrupt(kind):
    assert kind not in _ATTACHMENT_BURST_TYPES


def test_documents_were_the_gap():
    """The regression this closes: DOCUMENT was missing while PHOTO was already handled."""
    assert MessageType.DOCUMENT in _ATTACHMENT_BURST_TYPES
    assert MessageType.PHOTO in _ATTACHMENT_BURST_TYPES


def test_burst_branch_requires_actual_media():
    """A typed-but-empty event must fall through rather than be swallowed silently."""
    import inspect

    from gateway.run_inbound import GatewayInboundMixin

    src = inspect.getsource(GatewayInboundMixin._hm_busy_slash_or_photo)
    assert "_ATTACHMENT_BURST_TYPES" in src
    assert 'getattr(event, "media_urls", None)' in src
