"""Изображение, названное путём в тексте, идёт тем же маршрутом, что и вложение.

Вложение шлюз прогоняет через vision до хода модели; путь, напечатанный в тексте,
раньше не проходил ничего — и текстовая основная модель отвечала выдумкой. Здесь
проверяется, что такой путь попадает в список вложений, и что граница доступа у него
та же, что у ``@``-ссылок: существующий обычный файл под корнем сессии.
"""
import os

import pytest

from gateway.run_inbound import InboundMixin


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
    return tmp_path


def _png(path):
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)
    return str(path)


def test_path_in_text_joins_attachments(root):
    shot = _png(root / "shot.png")
    merged = InboundMixin._merge_text_image_refs(f"что на {shot} ?", [])
    assert merged == [os.path.realpath(shot)]


def test_existing_attachment_is_not_duplicated(root):
    shot = _png(root / "shot.png")
    resolved = os.path.realpath(shot)
    assert InboundMixin._merge_text_image_refs(f"смотри {shot}", [resolved]) == [resolved]


def test_path_outside_session_root_is_ignored(root, tmp_path_factory):
    outside = _png(tmp_path_factory.mktemp("elsewhere") / "secret.png")
    assert InboundMixin._merge_text_image_refs(f"открой {outside}", []) == []


def test_missing_file_is_ignored(root):
    assert InboundMixin._merge_text_image_refs(f"смотри {root}/nope.png", []) == []


def test_remote_url_is_not_fetched(root):
    assert InboundMixin._merge_text_image_refs("см. https://example.com/a.png", []) == []


def test_merge_is_capped(root):
    text = " ".join(_png(root / f"s{i}.png") for i in range(9))
    assert len(InboundMixin._merge_text_image_refs(text, [])) == 4


def test_empty_text_returns_attachments_unchanged(root):
    assert InboundMixin._merge_text_image_refs("", ["/a.png"]) == ["/a.png"]
