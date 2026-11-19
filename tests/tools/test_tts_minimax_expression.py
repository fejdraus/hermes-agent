"""Expressive MiniMax 2.8 speech: the model picks an emotion per call and is told about sound tags
only when the configured model speaks them; Russian homographs get ruaccent's stress; a reply the
gateway voices on its own is prepared by the auxiliary model, with numbers the code writes."""

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tools import tts_ru_stress, tts_tool


def _config(model):
    return {"provider": "minimax", "minimax": {
        "model": model, "voice_id": "Deep_Voice_Man", "emotion": "neutral", "language_boost": "",
        "voices_by_language": {"ru": "Russian_ReliableMan", "en": "Deep_Voice_Man"}}}


def _speak(tmp_path, config, text, **kwargs):
    seen = []

    def _fake_minimax(spoken, output_path, tts_config):
        seen.append((tts_config["minimax"]["emotion"], tts_config["minimax"]["voice_id"], spoken))
        with open(output_path, "wb") as handle:
            handle.write(b"ID3\x03\x00\x00\x00\x00\x00\x00")
        return output_path

    with patch("tools.tts_tool._generate_minimax_tts", side_effect=_fake_minimax), \
         patch("tools.tts_tool._load_tts_config", return_value=config), \
         patch("tools.tts_ru_stress.stress_homographs", side_effect=lambda t: t.replace("замке", "за́мке")):
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
    assert [(e, v) for e, v, _ in seen] == [(expected, "Russian_ReliableMan")]


def test_tag_guidance_is_offered_only_for_a_2_8_model():
    with patch("tools.tts_tool._load_tts_config", return_value=_config("speech-2.8-hd")):
        props = tts_tool._tts_schema_overrides()["parameters"]["properties"]
    assert "happy" in props["emotion"]["enum"]
    assert "(sighs)" in props["text"]["description"]
    with patch("tools.tts_tool._load_tts_config", return_value=_config("speech-02-hd")):
        props = tts_tool._tts_schema_overrides()["parameters"]["properties"]
    assert "emotion" not in props
    assert "(sighs)" not in props["text"]["description"]


def test_sound_tags_do_not_change_the_voice_of_the_language(tmp_path):
    pytest.importorskip("lingua")
    seen = _speak(tmp_path, _config("speech-2.8-hd"), "(sighs) (laughs) (coughs) Ну да.")
    assert [(e, v) for e, v, _ in seen] == [("neutral", "Russian_ReliableMan")]


@pytest.mark.parametrize("model, language, spoken", [
    ("speech-2.8-hd", "ru", "Встреча в за́мке."),
    ("speech-2.8-hd", "en", "Встреча в замке."),
    ("speech-02-hd", "ru", "Встреча в замке."),
])
def test_russian_homographs_are_stressed_for_a_2_8_voice(tmp_path, model, language, spoken):
    seen = _speak(tmp_path, _config(model), "Встреча в замке.", language=language)
    assert [s for _, _, s in seen] == [spoken]


def test_homograph_stress_follows_the_chosen_reading_and_replaces_guessed_marks():
    class _Model:
        def classify(self, texts, hypotheses, counts):
            assert "<w>замке</w>" in texts[0] and "<w>замка</w>" in texts[-1]
            return ["з+амке", "замк+а"]

    engine = (_Model(), {"замке": ["з+амке", "замк+е"], "замка": ["з+амка", "замк+а"]})
    with patch("tools.tts_ru_stress._load", return_value=engine):
        spoken = tts_ru_stress.stress_homographs("В Замке ключ от за́мка, кури́ца.")
    assert spoken == "В За́мке ключ от замка́, курица."


def _auto_voice(model, reply_text, rewritten):
    from gateway.platforms import base_auto_tts

    calls = []

    def _fake_llm(**_kwargs):
        message = SimpleNamespace(content=rewritten)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    def _fake_tool(text, output_path, emotion=None):
        calls.append((text, emotion))
        return json.dumps({"success": False})

    with patch("tools.tts_tool._load_tts_config", return_value=_config(model)), \
         patch("pm.ensure_import"), \
         patch("tools.tts_voice_language.detect_language", return_value="ru"), \
         patch("agent.auxiliary_client.call_llm", side_effect=_fake_llm), \
         patch("tools.tts_tool.text_to_speech_tool", side_effect=_fake_tool):
        base_auto_tts._synthesize(reply_text, "out.ogg")
    return calls


def test_an_auto_voice_reply_is_spoken_with_numbers_the_code_wrote():
    pytest.importorskip("num2words")
    reply = "Через 5 минут, 21 ноября в 18:30, будет 150 г сыра и 5,6 °C."
    marked = ("happy\n(sighs) Через {{5|card acc}} минут, {{21|ord gen n}} ноября в {{18|card nom}} "
              "{{30|card nom}}, будет {{150|card nom}} граммов сыра и {{5,6|card nom}} градуса.")
    assert _auto_voice("speech-2.8-hd", reply, marked) == [(
        "(sighs) Через пять минут, двадцать первого ноября в восемнадцать тридцать, будет сто "
        "пятьдесят граммов сыра и пять целых шесть десятых градуса.", "happy")]


@pytest.mark.parametrize("marked", [
    "calm\nВстреча в {{17|card nom}} {{30|card nom}}.",
    "calm\nВстреча в {{18|card nom}}.",
    "calm\nВстреча в 18 {{30|card nom}}.",
    "calm\nВстреча в половине седьмого.",
])
def test_an_auto_voice_rewrite_with_other_numbers_is_not_spoken(marked):
    reply = "Встреча в 18:30."
    assert _auto_voice("speech-2.8-hd", reply, marked) == [(reply, None)]


@pytest.mark.parametrize("rewritten", [
    "calm\nЕсть курица.",
    "calm\nЕсть курица и рис, могу предложить плов, а ещё борщ, салат и компот с пирогом.",
    "calm\nЕсть курица и рис, могу предложить плова.",
])
def test_an_auto_voice_rewrite_that_changed_the_words_is_not_spoken(rewritten):
    reply = "Есть курица и рис, могу предложить плов."
    assert _auto_voice("speech-2.8-hd", reply, rewritten) == [(reply, None)]


def test_the_first_line_of_the_rewrite_is_never_spoken():
    reply = "Есть курица и рис."
    assert _auto_voice("speech-2.8-hd", reply, "reminder\nЕсть курица и рис.") == [(reply, None)]


def test_an_older_minimax_model_speaks_the_auto_voice_reply_as_written():
    reply = "Есть курица и рис."
    assert _auto_voice("speech-02-hd", reply, "happy\n(laughs) Есть курица и рис.") == [(reply, None)]


@pytest.mark.parametrize("value, form, lang, words", [
    ("5,6", "card gen", "ru", "пяти целых шести десятых"),
    ("1,1", "card nom", "ru", "одна целая одна десятая"),
    ("05", "card nom", "ru", "ноль пять"),
    ("7:05", "time", "ru", "семь ноль пять"),
    ("18:30", "time", "uk", "вісімнадцять тридцять"),
    ("3", "card acc f", "ru", "три"),
    ("3", "card acc anim", "ru", "трёх"),
    ("2", "ord nom m", "ru", "второй"),
    ("21", "card gen", "uk", "двадцяти одного"),
    ("21", "ord gen n", "uk", None),
])
def test_a_marked_number_is_written_in_its_form_or_keeps_its_digits(value, form, lang, words):
    pytest.importorskip("num2words")
    from tools.tts_number_words import number_words
    assert number_words(value, form, lang) == words
