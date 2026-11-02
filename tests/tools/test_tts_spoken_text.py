"""Text spoken by TTS carries numbers as words: the model's ``text_to_speech`` call is sent back to
rewrite digits, and a voice note answered by auto-TTS tells the model its reply is spoken."""

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from gateway.platforms.event import MessageType
from gateway.run_inbound_transcription import GatewayInboundTranscriptionMixin, _SPOKEN_REPLY_NOTE
from tools import tts_tool


def test_model_call_with_digits_is_sent_back_and_words_are_spoken():
    with patch("tools.tts_tool.text_to_speech_tool", return_value='{"success": true}') as speak:
        refused = json.loads(tts_tool._model_text_to_speech({"text": "Завтра 3 ноября, +5,6°C"}))
        spoken = json.loads(tts_tool._model_text_to_speech(
            {"text": "Завтра третьего ноября, плюс пять целых шесть десятых градуса", "language": "ru"}))
    assert not refused["success"] and "digits" in refused["error"]
    assert spoken["success"]
    assert speak.call_count == 1 and speak.call_args.kwargs["language"] == "ru"


class _Runner(GatewayInboundTranscriptionMixin):
    def __init__(self, auto_tts: bool):
        self.adapter = SimpleNamespace(_should_auto_tts_for_chat=lambda _chat_id: auto_tts)

    def _delivery_adapter_for(self, _source):
        return self.adapter


@pytest.mark.parametrize("auto_tts, message_type, transcripts, noted", [
    (True, MessageType.VOICE, ["какая погода"], True),
    (False, MessageType.VOICE, ["какая погода"], False),
    (True, MessageType.TEXT, ["какая погода"], False),
    (True, MessageType.VOICE, [], False),
])
def test_spoken_reply_note_only_when_the_reply_will_be_voiced(auto_tts, message_type, transcripts, noted):
    event = SimpleNamespace(message_type=message_type, source=SimpleNamespace(chat_id="42"))
    text = _Runner(auto_tts)._note_spoken_reply(event, '"какая погода"', transcripts)
    assert text.startswith('"какая погода"')
    assert (_SPOKEN_REPLY_NOTE in text) is noted
