"""Expressive MiniMax 2.8 speech: the model that writes the reply picks an emotion per call and
is told about sound tags only when the configured model speaks them; without an emotion MiniMax
picks one itself; Russian homographs get ruaccent's stress."""

import json
from unittest.mock import patch

import pytest

from tools import tts_ru_stress, tts_tool
from tools.tts_tool_providers import _generate_minimax_tts


def _config(model, emotion="neutral"):
    section = {"model": model, "voice_id": "Deep_Voice_Man", "language_boost": "",
               "voices_by_language": {"ru": "Russian_ReliableMan", "en": "Deep_Voice_Man"}}
    if emotion:
        section["emotion"] = emotion
    return {"provider": "minimax", "minimax": section}


def _speak(tmp_path, config, text, **kwargs):
    seen = []

    def _fake_minimax(spoken, output_path, tts_config):
        seen.append((tts_config["minimax"].get("emotion"), tts_config["minimax"]["voice_id"], spoken))
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


class _Sent(Exception):
    pass


@pytest.mark.parametrize("configured, expected", [(None, None), ("calm", "calm")])
def test_minimax_picks_the_emotion_itself_when_none_is_set(monkeypatch, configured, expected):
    monkeypatch.setattr("hermes_cli.config.get_env_value",
                        lambda name, default=None: "FAKE_KEY" if name == "MINIMAX_API_KEY" else default)
    sent = []

    def _capture(url, payload, headers, **_kw):
        sent.append(payload)
        raise _Sent

    with patch("tools.tts_tool_providers._post_json", side_effect=_capture), pytest.raises(_Sent):
        _generate_minimax_tts("Ну вот, опять дождь.", "out.mp3", _config("speech-2.8-hd", configured))
    assert sent[0]["voice_setting"].get("emotion") == expected


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


def test_each_marked_part_is_spoken_with_its_own_emotion(tmp_path):
    reply = "[[sad]] Ковалев стоял у ворот. [[happy]] (laughs) А бабушка смеялась!"
    seen = _speak(tmp_path, _config("speech-2.8-hd", None), reply, language="ru")
    assert [(e, s) for e, _, s in seen] == [
        ("sad", "Ковалев стоял у ворот."), ("happy", "(laughs) А бабушка смеялась!")]


def test_an_unmarked_reply_leaves_the_emotion_to_the_voice(tmp_path):
    seen = _speak(tmp_path, _config("speech-2.8-hd", None), "Ковалев стоял у ворот.", language="ru")
    assert [(e, s) for e, _, s in seen] == [(None, "Ковалев стоял у ворот.")]


def test_the_user_reads_the_reply_without_voice_marks():
    from tools.tts_minimax_expression import strip_voice_marks
    assert strip_voice_marks("[[sad]] Ковалев стоял. [[happy]] (laughs) Бабушка смеялась!") == \
        "Ковалев стоял. Бабушка смеялась!"


def test_the_spoken_reply_note_asks_for_marks_only_for_a_2_8_voice():
    from tools.tts_minimax_expression import spoken_reply_note
    with patch("tools.tts_tool._load_tts_config", return_value=_config("speech-2.8-hd")):
        note = spoken_reply_note()
    assert "[[emotion]]" in note and "(sighs)" in note and "as words" in note
    with patch("tools.tts_tool._load_tts_config", return_value=_config("speech-02-hd")):
        assert spoken_reply_note() is None


def test_a_homograph_is_judged_by_the_words_around_it_not_the_whole_reply():
    seen = []

    class _Model:
        def classify(self, texts, hypotheses, counts):
            seen.extend(texts)
            return ["зам+ок"]

    far = "Утром мы долго гуляли по парку у реки и кормили уток, потом пили чай. "
    with patch("tools.tts_ru_stress._load", return_value=(_Model(), {"замок": ["з+амок", "зам+ок"]})):
        spoken = tts_ru_stress.stress_homographs(far + "На двери висит новый замок.")
    assert spoken.endswith("висит новый замо́к.")
    assert "утром" not in seen[0] and "висит новый" in seen[0] and "<w>замок</w>" in seen[0]
