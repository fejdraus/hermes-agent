"""Expressive MiniMax 2.8 speech: the model picks an emotion per call and is told about sound tags
and stress marks only when the configured model speaks them."""

import json
from unittest.mock import patch

import pytest

from tools import tts_tool


def _config(model):
    return {"provider": "minimax", "minimax": {
        "model": model, "voice_id": "Deep_Voice_Man", "emotion": "neutral", "language_boost": "",
        "voices_by_language": {"ru": "Russian_ReliableMan", "en": "Deep_Voice_Man"}}}


def _speak(tmp_path, config, text, **kwargs):
    seen = []

    def _fake_minimax(_text, output_path, tts_config):
        seen.append((tts_config["minimax"]["emotion"], tts_config["minimax"]["voice_id"]))
        with open(output_path, "wb") as handle:
            handle.write(b"ID3\x03\x00\x00\x00\x00\x00\x00")
        return output_path

    with patch("tools.tts_tool._generate_minimax_tts", side_effect=_fake_minimax), \
         patch("tools.tts_tool._load_tts_config", return_value=config):
        result = json.loads(tts_tool.text_to_speech_tool(text, output_path=str(tmp_path / "a.mp3"), **kwargs))
    assert result["success"]
    return seen


@pytest.mark.parametrize("model, emotion, expected", [
    ("speech-2.8-hd", "happy", "happy"),
    ("speech-2.8-turbo", "sad", "sad"),
    ("speech-2.8-hd", "whisper", "neutral"),
    ("speech-2.8-hd", None, "neutral"),
    ("speech-02-hd", "happy", "neutral"),
])
def test_the_emotion_of_the_call_is_spoken_by_a_2_8_model(tmp_path, model, emotion, expected):
    seen = _speak(tmp_path, _config(model), "Ну вот, опять дождь.", language="ru", emotion=emotion)
    assert seen == [(expected, "Russian_ReliableMan")]


def test_emotion_and_tag_guidance_are_offered_only_for_a_2_8_model():
    with patch("tools.tts_tool._load_tts_config", return_value=_config("speech-2.8-hd")):
        props = tts_tool._tts_schema_overrides()["parameters"]["properties"]
    assert "happy" in props["emotion"]["enum"]
    assert "(sighs)" in props["text"]["description"] and "U+0301" in props["text"]["description"]
    with patch("tools.tts_tool._load_tts_config", return_value=_config("speech-02-hd")):
        props = tts_tool._tts_schema_overrides()["parameters"]["properties"]
    assert "emotion" not in props
    assert "(sighs)" not in props["text"]["description"]


def test_sound_tags_do_not_change_the_voice_of_the_language(tmp_path):
    pytest.importorskip("lingua")
    seen = _speak(tmp_path, _config("speech-2.8-hd"), "(sighs) (laughs) (coughs) Ну да.")
    assert seen == [("neutral", "Russian_ReliableMan")]
