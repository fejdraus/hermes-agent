"""video_url -> Anthropic video block on the MiniMax-native surface."""

from agent.anthropic_adapter import (
    _convert_content_part_to_anthropic,
    _video_source_from_openai_url,
)


def test_base64_video_source():
    src = _video_source_from_openai_url("data:video/mp4;base64,AAAA")
    assert src == {"type": "base64", "media_type": "video/mp4", "data": "AAAA"}


def test_plain_url_video_source():
    src = _video_source_from_openai_url("https://example.com/clip.mp4")
    assert src == {"type": "url", "url": "https://example.com/clip.mp4"}


def test_webm_container_preserved():
    src = _video_source_from_openai_url("data:video/webm;base64,AAAA")
    assert src["media_type"] == "video/webm"


def test_part_conversion_emits_video_block():
    block = _convert_content_part_to_anthropic({
        "type": "video_url",
        "video_url": {"url": "data:video/mp4;base64,AAAA", "fps": 2},
    })
    assert block["type"] == "video"
    assert block["source"]["type"] == "base64"
    assert block["fps"] == 2


def test_fps_omitted_when_unset():
    block = _convert_content_part_to_anthropic({
        "type": "video_url",
        "video_url": {"url": "data:video/mp4;base64,AAAA"},
    })
    assert "fps" not in block


def test_image_conversion_still_works():
    block = _convert_content_part_to_anthropic({
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64,BBBB"},
    })
    assert block["type"] == "image"
