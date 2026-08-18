"""Video attachments reach the model only when the profile opts in.

``video_url`` is not part of the shared OpenAI-compatible surface: a provider
that does not know the block rejects the whole request, so video stays a path
reference unless ``agent.video_input`` is set.
"""

from __future__ import annotations

import base64

from agent.image_routing import (
    build_native_video_parts,
    video_input_enabled,
    video_input_fps,
)


def _write_video(tmp_path, name="clip.mp4", size=32):
    path = tmp_path / name
    path.write_bytes(b"\x00" * size)
    return str(path)


def test_disabled_without_config():
    assert video_input_enabled(None) is False
    assert video_input_enabled({}) is False
    assert video_input_enabled({"agent": {}}) is False


def test_enabled_by_agent_flag():
    assert video_input_enabled({"agent": {"video_input": True}}) is True


def test_fps_read_and_range_checked():
    assert video_input_fps({"agent": {"video_fps": 2}}) == 2.0
    assert video_input_fps({"agent": {}}) is None
    # outside MiniMax's documented 0.2-5 window: fall back to provider default
    assert video_input_fps({"agent": {"video_fps": 9}}) is None
    assert video_input_fps({"agent": {"video_fps": "nonsense"}}) is None


def test_builds_video_url_data_block(tmp_path):
    video = _write_video(tmp_path, size=64)

    parts, skipped = build_native_video_parts([video], fps=2.0)

    assert skipped == []
    assert len(parts) == 1
    block = parts[0]
    assert block["type"] == "video_url"
    assert block["video_url"]["fps"] == 2.0
    url = block["video_url"]["url"]
    assert url.startswith("data:video/mp4;base64,")
    assert base64.b64decode(url.split(",", 1)[1]) == b"\x00" * 64


def test_fps_omitted_when_not_given(tmp_path):
    parts, _ = build_native_video_parts([_write_video(tmp_path)])
    assert "fps" not in parts[0]["video_url"]


def test_missing_file_is_skipped_not_raised(tmp_path):
    parts, skipped = build_native_video_parts([str(tmp_path / "gone.mp4")])
    assert parts == []
    assert len(skipped) == 1


def test_oversized_video_is_skipped(tmp_path, monkeypatch):
    """A body over the provider cap cannot succeed — do not spend the upload."""
    monkeypatch.setattr("agent.image_routing._MAX_INLINE_VIDEO_BYTES", 16)
    video = _write_video(tmp_path, size=64)

    parts, skipped = build_native_video_parts([video])

    assert parts == []
    assert skipped == [video]
