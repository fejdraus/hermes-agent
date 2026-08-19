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
    out = lf._mask_data_uris(data="data:image/png;base64,AAA")  # invalid padding
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
