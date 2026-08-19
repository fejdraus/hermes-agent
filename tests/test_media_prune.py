"""Media payloads leave the history at compression; their meaning stays."""

from agent.context_compressor import (
    _media_part_reference,
    _strip_image_parts_from_parts,
)


def test_video_payload_replaced_by_reference():
    parts = [
        {"type": "text", "text": "what is here"},
        {"type": "video_url", "video_url": {"url": "data:video/mp4;base64,AAAA"},
         "_meta": {"path": "/cache/videos/clip.mp4"}},
    ]
    out = _strip_image_parts_from_parts(parts)

    assert out[0] == {"type": "text", "text": "what is here"}
    assert out[1]["type"] == "text"
    assert "video" in out[1]["text"]
    assert "/cache/videos/clip.mp4" in out[1]["text"]
    assert "base64" not in out[1]["text"]


def test_anthropic_native_video_block_handled():
    parts = [{"type": "video", "source": {"type": "base64", "media_type": "video/mp4", "data": "AAAA"}}]
    out = _strip_image_parts_from_parts(parts)
    assert out[0]["type"] == "text"
    assert "video" in out[0]["text"]


def test_image_keeps_its_long_standing_wording():
    """Screenshots are the common image case; existing behaviour keys off it."""
    parts = [{"type": "image_url", "image_url": {"url": "data:image/png;base64,BBBB"}}]
    out = _strip_image_parts_from_parts(parts)
    assert out[0]["type"] == "text"
    assert "screenshot removed" in out[0]["text"]
    assert "base64" not in out[0]["text"]


def test_url_source_kept_as_handle():
    """A real URL survives — the agent can fetch it again; a data: URL cannot."""
    part = {"type": "video", "source": {"type": "url", "url": "https://x/clip.mp4"}}
    assert "https://x/clip.mp4" in _media_part_reference(part)

    part_b64 = {"type": "video", "source": {"type": "base64", "data": "AAAA"}}
    assert "base64" not in _media_part_reference(part_b64)


def test_parts_without_media_are_left_alone():
    parts = [{"type": "text", "text": "plain turn"}]
    assert _strip_image_parts_from_parts(parts) is None
