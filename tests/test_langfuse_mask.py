"""The mask strips base64 data URIs before the Langfuse SDK decodes them."""

import importlib

lf = importlib.import_module("plugins.observability.langfuse")


def test_bare_data_uri_is_replaced():
    out = lf._mask_data_uris(data="data:image/png;base64,AAAA")
    assert out["type"] == "data_uri"
    assert out["media_type"] == "image/png"
    assert out["omitted"] is True


def test_nested_beyond_serializer_depth_is_still_masked():
    """The SDK walks the whole payload — our own depth limit does not apply."""
    payload = {"a": {"b": {"c": {"d": {"e": {"f": [
        {"image_url": {"url": "data:image/jpeg;base64,BBBB"}}
    ]}}}}}}
    out = lf._mask_data_uris(data=payload)
    leaf = out["a"]["b"]["c"]["d"]["e"]["f"][0]["image_url"]["url"]
    assert leaf["omitted"] is True


def test_video_data_uri_masked_too():
    out = lf._mask_data_uris(data=[{"video_url": {"url": "data:video/mp4;base64,CCCC"}}])
    assert out[0]["video_url"]["url"]["media_type"] == "video/mp4"


def test_truncated_uri_masked_not_decoded():
    """A cut URI is exactly what produced the padding errors."""
    out = lf._mask_data_uris(data="data:image/png;base64,AAA")
    assert out["omitted"] is True


def test_ordinary_content_passes_through_untouched():
    payload = {"role": "user", "content": [{"type": "text", "text": "What is on this?"}]}
    assert lf._mask_data_uris(data=payload) == payload


def test_masking_never_raises_on_odd_input():
    class Weird:
        def __iter__(self):
            raise RuntimeError("boom")

    weird = Weird()
    assert lf._mask_data_uris(data=weird) is weird


def test_anthropic_payload_dict_is_masked():
    """The SDK decodes a bare payload dict too, not only a data: string."""
    out = lf._mask_data_uris(data={"type": "base64", "media_type": "image/png", "data": "AAA"})
    assert out["type"] == "data_uri"
    assert out["media_type"] == "image/png"
    assert out["omitted"] is True and out["length"] == 3


def test_vertex_payload_dict_is_masked():
    out = lf._mask_data_uris(data={"type": "media", "mime_type": "video/mp4", "data": "BBBB"})
    assert out["media_type"] == "video/mp4" and out["omitted"] is True


def test_nested_anthropic_image_block_is_masked():
    """Shape hermes actually sends to MiniMax's Anthropic endpoint."""
    payload = {"messages": [{"role": "user", "content": [
        {"type": "text", "text": "что на фото?"},
        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "Zm9vYmFy"}},
    ]}]}
    block = lf._mask_data_uris(data=payload)["messages"][0]["content"][1]
    assert block["type"] == "image"
    assert block["source"]["omitted"] is True and block["source"]["media_type"] == "image/jpeg"


def test_unrelated_dicts_survive_untouched():
    payload = {"type": "base64", "media_type": "image/png"}
    assert lf._mask_data_uris(data=payload) == payload
    other = {"type": "text", "data": "plain text"}
    assert lf._mask_data_uris(data=other) == other


def test_missing_declared_type_still_masks_the_payload():
    out = lf._mask_data_uris(data={"type": "base64", "data": "AAA"})
    assert out["omitted"] is True and out["media_type"] is None
