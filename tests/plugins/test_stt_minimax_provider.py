"""Плагин MiniMax STT: конверт ответа, ключ, подсказка языка.

Сеть не трогается — ``requests.post`` подменяется. Проверяется то, ради чего плагин
написан: распознанный текст доходит до вызывающего, сбой становится конвертом ошибки,
а не исключением, и язык не выдумывается там, где его не задали.
"""
from __future__ import annotations

import sys
import types

import pytest

from plugins.stt.minimax import MiniMaxTranscriptionProvider


@pytest.fixture
def audio(tmp_path):
    path = tmp_path / "voice.mp3"
    path.write_bytes(b"\xff\xfb" + b"\x00" * 64)
    return str(path)


@pytest.fixture
def provider(monkeypatch):
    monkeypatch.setattr("plugins.stt.minimax._api_key", lambda: "test-key")
    monkeypatch.setattr("plugins.stt.minimax._config_section", dict)
    return MiniMaxTranscriptionProvider()


def _fake_requests(monkeypatch, *, status=200, payload=None, sent=None, boom=None):
    def post(url, **kwargs):
        if sent is not None:
            sent.update({"url": url, **kwargs})
        if boom is not None:
            raise boom
        return types.SimpleNamespace(
            status_code=status, text="body", json=lambda: payload or {},
        )

    monkeypatch.setitem(sys.modules, "requests", types.SimpleNamespace(post=post))


def test_transcript_is_returned(provider, audio, monkeypatch):
    _fake_requests(monkeypatch, payload={"text": " курка з рисом "})
    assert provider.transcribe(audio) == {
        "success": True, "transcript": "курка з рисом", "provider": "minimax",
    }


def test_default_model_is_sent(provider, audio, monkeypatch):
    sent = {}
    _fake_requests(monkeypatch, payload={"text": "ok"}, sent=sent)
    provider.transcribe(audio)
    assert sent["data"]["model"] == "asr-1.0"
    assert sent["url"].endswith("/v1/speech_to_text")


def test_language_is_forwarded_only_when_given(provider, audio, monkeypatch):
    sent = {}
    _fake_requests(monkeypatch, payload={"text": "ok"}, sent=sent)
    provider.transcribe(audio)
    assert "language" not in sent["data"]
    provider.transcribe(audio, language="uk")
    assert sent["data"]["language"] == "uk"


def test_http_error_becomes_envelope(provider, audio, monkeypatch):
    _fake_requests(monkeypatch, status=400)
    result = provider.transcribe(audio)
    assert result["success"] is False and result["transcript"] == ""
    assert "400" in result["error"]


def test_exception_becomes_envelope(provider, audio, monkeypatch):
    _fake_requests(monkeypatch, boom=OSError("connection reset"))
    result = provider.transcribe(audio)
    assert result["success"] is False and "connection reset" in result["error"]


def test_empty_transcript_is_failure(provider, audio, monkeypatch):
    _fake_requests(monkeypatch, payload={"base_resp": {"status_code": 2013}})
    result = provider.transcribe(audio)
    assert result["success"] is False and "2013" in result["error"]


def test_missing_key_is_reported(audio, monkeypatch):
    monkeypatch.setattr("plugins.stt.minimax._api_key", lambda: "")
    result = MiniMaxTranscriptionProvider().transcribe(audio)
    assert result["success"] is False and "MINIMAX_API_KEY" in result["error"]
    assert MiniMaxTranscriptionProvider().is_available() is False
