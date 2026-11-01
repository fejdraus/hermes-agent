"""A voice note that queued behind a running turn is answered through ``deliver_response``. Its reply is
voiced like the first message's (``_process_message_background``); the text still goes out through the
ledgered send."""
import pytest

from gateway.config import Platform
from gateway.platforms.base import BasePlatformAdapter, _ExtractedResponse
from gateway.platforms.event import MessageEvent, MessageType
from gateway.session import SessionSource


class _Adapter:
    _wants_auto_tts = BasePlatformAdapter._wants_auto_tts

    def __init__(self, tmp_path, auto_tts=True):
        self.tmp_path = tmp_path
        self.auto_tts = auto_tts
        self.sent = []

    def _should_auto_tts_for_chat(self, chat_id):
        return self.auto_tts

    def _streaming_tts_turn_completed(self, session_key, turn_marker=None, *, event=None):
        return False

    def _unwrap_ephemeral(self, response):
        return response, 0

    async def _extract_response_content(self, text, event, session_key, *, is_ephemeral_response):
        return _ExtractedResponse(text, [], [], [], False, text)

    async def _synthesize_auto_tts(self, text):
        path = self.tmp_path / "reply.ogg"
        path.write_bytes(b"ogg")
        return [str(path)], str(path)

    async def play_tts(self, chat_id, audio_path, caption=None, metadata=None):
        self.sent.append(("voice", caption))

    async def _send_final_text(self, event, session_key, text, metadata, ephemeral, ttl, record):
        self.sent.append(("text", text))

    async def _deliver_attachments(self, event, extracted, metadata, *, anything_sent, record_delivery):
        return None


def _event(kind):
    return MessageEvent(text="Не на Боливии, а в Львове.", message_type=kind,
                        source=SessionSource(platform=Platform.TELEGRAM, chat_id="c"), message_id="m2")


@pytest.mark.asyncio
async def test_reply_to_a_queued_voice_note_is_voiced_and_still_sent_as_text(tmp_path):
    from gateway.session_ingress import deliver_response
    adapter = _Adapter(tmp_path)
    await deliver_response(adapter, _event(MessageType.VOICE), "s", "Во Львове пасмурно.")
    assert adapter.sent == [("voice", None), ("text", "Во Львове пасмурно.")]
    assert not (tmp_path / "reply.ogg").exists()


@pytest.mark.asyncio
async def test_reply_to_queued_text_is_not_voiced(tmp_path):
    from gateway.session_ingress import deliver_response
    adapter = _Adapter(tmp_path)
    await deliver_response(adapter, _event(MessageType.TEXT), "s", "Во Львове пасмурно.")
    assert adapter.sent == [("text", "Во Львове пасмурно.")]


@pytest.mark.asyncio
async def test_queued_voice_note_stays_unvoiced_when_the_chat_has_voice_off(tmp_path):
    from gateway.session_ingress import deliver_response
    adapter = _Adapter(tmp_path, auto_tts=False)
    await deliver_response(adapter, _event(MessageType.VOICE), "s", "Во Львове пасмурно.")
    assert adapter.sent == [("text", "Во Львове пасмурно.")]
