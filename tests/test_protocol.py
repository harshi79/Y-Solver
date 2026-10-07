"""Wire-protocol parsing: the compatibility surface clients depend on."""

from __future__ import annotations

import base64

import pytest

from ysolver.protocol import (
    ProtocolError,
    classify_method,
    decode_base64_payload,
    enforce_length,
    extract_options,
    first,
    job_id_is_valid,
    mask_key,
    normalize_id,
    parse_task,
    truthy,
)


def test_first_is_case_insensitive_and_skips_blanks():
    assert first({"ClientKey": "abc"}, ("clientkey",)) == "abc"
    assert first({"key": "  "}, ("key", "clientkey")) is None
    assert first({}, ("key",)) is None


def test_truthy_variants():
    for value in ("1", "true", "YES", "on", True):
        assert truthy(value)
    for value in ("0", "false", "", None, False):
        assert not truthy(value)


def test_mask_key_hides_the_secret():
    assert mask_key("1234567890") == "12***90"
    assert mask_key("abc") == "***"
    assert mask_key(None) == ""


def test_decode_base64_plain_and_data_url():
    raw = b"\x89PNG\r\n\x1a\n"
    encoded = base64.b64encode(raw).decode()
    assert decode_base64_payload(encoded) == raw
    assert decode_base64_payload(f"data:image/png;base64,{encoded}") == raw


def test_decode_base64_tolerates_urlsafe_and_padding():
    raw = bytes(range(256)) * 3
    encoded = base64.urlsafe_b64encode(raw).decode().rstrip("=")
    assert decode_base64_payload(encoded) == raw


def test_decode_base64_rejects_nonsense():
    with pytest.raises(ProtocolError) as excinfo:
        decode_base64_payload("!!! not base64 !!!")
    assert excinfo.value.code.startswith("ERROR_")


def test_classify_method_routes_the_three_families():
    assert classify_method("base64", None) == "image"
    assert classify_method("post", None) == "image"
    assert classify_method(None, None) == "image"
    assert classify_method("userrecaptcha", None) == "interactive"
    assert classify_method("HCaptchaTaskProxyless", None) == "interactive"
    assert classify_method("TurnstileTask", None) == "interactive"
    assert classify_method("mysteryThing", None) == "unknown"


def test_normalize_id_accepts_numeric_ids_only():
    assert normalize_id(" 123456789 ") == "123456789"
    for bad in ("", None, "abc", "12", "1;2"):
        with pytest.raises(ProtocolError):
            normalize_id(bad)
    assert job_id_is_valid("999999999")
    assert not job_id_is_valid("nope")


def test_extract_options_reads_aliases():
    options = extract_options({"numeric": "1", "min_len": "4", "max_len": "6"})
    assert options["numeric"] is True
    assert options["min_len"] == 4
    assert options["max_len"] == 6
    assert extract_options({"charset": "abc"})["charset"] == "abc"


def test_enforce_length_pads_and_trims():
    assert enforce_length("abc", 5, None) == "abccc"
    assert enforce_length("abcdef", None, 4) == "abcd"
    assert enforce_length("abc", None, None) == "abc"


def test_parse_task_accepts_object_and_json_string():
    assert parse_task({"type": "ImageToTextTask", "body": "zzz"})["image_base64"] == "zzz"
    parsed = parse_task('{"type": "ImageToTextTask", "body": "yyy"}')
    assert parsed["image_base64"] == "yyy"
    assert parse_task(None) == {}


def test_parse_task_rejects_bad_input():
    with pytest.raises(ProtocolError):
        parse_task("{not json}")
    with pytest.raises(ProtocolError):
        parse_task(42)


def test_decode_base64_survives_urlencoded_plus_pollution():
    """`+` in a base64 payload arrives as a space when a client posts urlencoded."""
    from ysolver.protocol import looks_like_image

    # Build a payload whose base64 contains '+' characters.
    payload = bytes(range(40, 220)) * 2
    encoded = base64.b64encode(payload).decode()
    assert "+" in encoded, "test vector must exercise plus signs"
    mangled = encoded.replace("+", " ")
    assert decode_base64_payload(mangled) == payload
    assert looks_like_image(b"\x89PNG\r\n\x1a\n") is True
    assert looks_like_image(b"nope") is False
