"""Per-language MiniMax voices: the model names the language, or lingua detects it for the
gateway's auto-voice; an unknown language keeps the configured voice."""

import json
from unittest.mock import patch

import pytest

from tools import tts_tool

_MM_CONFIG = {"provider": "minimax", "minimax": {
    "voice_id": "Deep_Voice_Man", "language_boost": "",
    "voices_by_language": {"ru": "Russian_ReliableMan", "uk": "Ukrainian_WiseScholar", "es": "Spanish_MaturePartner"}}}


def _speak(tmp_path, text, **kwargs):
    seen = []

    def _fake_minimax(_text, output_path, tts_config):
        seen.append((tts_config["minimax"]["voice_id"], tts_config["minimax"].get("language_boost")))
        with open(output_path, "wb") as handle:
            handle.write(b"ID3\x03\x00\x00\x00\x00\x00\x00")
        return output_path

    with patch("tools.tts_tool._generate_minimax_tts", side_effect=_fake_minimax), \
         patch("tools.tts_tool._load_tts_config", return_value=_MM_CONFIG):
        result = json.loads(tts_tool.text_to_speech_tool(text, output_path=str(tmp_path / "a.mp3"), **kwargs))
    assert result["success"]
    return seen


@pytest.mark.parametrize("text, language, expected", [
    ("Hola, ¿cómo estás? Hoy compré leche y pan.", "es", ("Spanish_MaturePartner", "Spanish")),
    ("Привіт, як справи? Сьогодні купив молоко і хліб.", None, ("Ukrainian_WiseScholar", "Ukrainian")),
    ("Hola, ¿cómo estás? Hoy compré leche y pan.", None, ("Spanish_MaturePartner", "Spanish")),
    ("Hello, how are you today? I bought milk and bread.", None, ("Deep_Voice_Man", "English")),
    ("Bonjour, je suis parti.", "fr", ("Deep_Voice_Man", "French")),
])
def test_language_picks_the_voice_of_that_language(tmp_path, text, language, expected):
    pytest.importorskip("lingua")
    assert _speak(tmp_path, text, language=language) == [expected]
    assert _MM_CONFIG["minimax"]["voice_id"] == "Deep_Voice_Man"


def test_language_field_is_offered_only_with_a_voice_table():
    with patch("tools.tts_tool._load_tts_config", return_value=_MM_CONFIG):
        props = tts_tool._tts_schema_overrides()["parameters"]["properties"]
    assert props["language"]["enum"] == ["es", "ru", "uk"]
    with patch("tools.tts_tool._load_tts_config", return_value={"provider": "minimax", "minimax": {}}):
        assert "language" not in tts_tool._tts_schema_overrides()["parameters"]["properties"]
