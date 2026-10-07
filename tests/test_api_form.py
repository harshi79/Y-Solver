"""The /in.php + /res.php form API."""

from __future__ import annotations

import base64

from conftest import STUB_ANSWER, poll_form


def test_post_multipart_upload_round_trip(client, sample_png):
    created = client.post(
        "/in.php",
        data={"key": "testkey", "method": "post"},
        files={"file": ("captcha.png", sample_png, "image/png")},
    )
    assert created.status_code == 200
    body = created.text
    assert body.startswith("OK|")
    job_id = body.split("|", 1)[1]
    assert job_id.isdigit()

    settled = poll_form(client, job_id)
    assert settled == f"OK|{STUB_ANSWER}"


def test_base64_body_round_trip(client, sample_b64):
    created = client.post(
        "/in.php",
        data={"key": "testkey", "method": "base64", "body": sample_b64},
    )
    assert created.text.startswith("OK|")
    job_id = created.text.split("|", 1)[1]
    assert poll_form(client, job_id) == f"OK|{STUB_ANSWER}"


def test_base64_accepts_data_url_and_urlsafe(client, sample_png):
    data_url = "data:image/png;base64," + base64.b64encode(sample_png).decode()
    created = client.post("/in.php", data={"key": "testkey", "body": data_url})
    assert created.text.startswith("OK|")

    urlsafe = base64.urlsafe_b64encode(sample_png).decode().rstrip("=")
    created = client.post("/in.php", data={"key": "testkey", "body": urlsafe})
    assert created.text.startswith("OK|")


def test_raw_binary_body_is_accepted(client, sample_png):
    created = client.post(
        "/in.php?key=testkey&method=base64",
        content=sample_png,
        headers={"Content-Type": "application/octet-stream"},
    )
    assert created.text.startswith("OK|")


def test_get_in_php_with_body_param(client, sample_b64):
    created = client.get(
        "/in.php", params={"key": "testkey", "method": "base64", "body": sample_b64}
    )
    assert created.text.startswith("OK|")


def test_missing_key_is_rejected(client, sample_b64):
    response = client.post("/in.php", data={"method": "base64", "body": sample_b64})
    assert response.text.strip() == "ERROR_WRONG_USER_KEY"


def test_wrong_key_is_rejected(client, sample_b64):
    response = client.post(
        "/in.php", data={"key": "nope", "method": "base64", "body": sample_b64}
    )
    assert response.text.strip() == "ERROR_WRONG_USER_KEY"


def test_missing_image_is_rejected(client):
    response = client.post("/in.php", data={"key": "testkey", "method": "base64"})
    assert response.text.strip().startswith("ERROR_")


def test_interactive_methods_are_refused_honestly(client):
    for method in ("userrecaptcha", "hcaptcha", "turnstile", "geetest"):
        response = client.post(
            "/in.php",
            data={"key": "testkey", "method": method, "googlekey": "x", "pageurl": "y"},
        )
        assert response.text.strip() == "ERROR_METHOD_NOT_SUPPORTED"


def test_res_php_unknown_id(client):
    response = client.get("/res.php", params={"key": "testkey", "action": "get", "id": "1"})
    assert response.text.strip() == "ERROR_WRONG_CAPTCHA_ID"


def test_res_php_bad_id_format(client):
    response = client.get(
        "/res.php", params={"key": "testkey", "action": "get", "id": "not-an-id"}
    )
    assert response.text.strip() == "ERROR_WRONG_CAPTCHA_ID"


def test_res_php_balance(client):
    response = client.get("/res.php", params={"key": "testkey", "action": "getbalance"})
    assert response.text.strip().startswith("OK|$")


def test_res_php_reportbad_is_a_noop(client, sample_b64):
    job_id = (
        client.post("/in.php", data={"key": "testkey", "body": sample_b64})
        .text.split("|", 1)[1]
    )
    response = client.get(
        "/res.php", params={"key": "testkey", "action": "reportbad", "id": job_id}
    )
    assert response.text.strip() == "OK|OK"


def test_res_php_unknown_action(client):
    response = client.get("/res.php", params={"key": "testkey", "action": "frobnicate"})
    assert response.text.strip() == "ERROR_BAD_ACTION"


def test_key_can_be_disabled(tmp_path, sample_b64):
    from conftest import build_app
    from conftest import poll_form as _poll
    from fastapi.testclient import TestClient

    app = build_app(tmp_path, require_key=False)
    with TestClient(app) as client:
        created = client.post("/in.php", data={"method": "base64", "body": sample_b64})
        assert created.text.startswith("OK|")
        job_id = created.text.split("|", 1)[1]
        assert _poll(client, job_id) == f"OK|{STUB_ANSWER}"


def test_queue_overflow_reports_no_slot(tmp_path, sample_b64):
    from conftest import build_app
    from fastapi.testclient import TestClient

    app = build_app(tmp_path, max_queue=1, workers=1)

    def slow_worker(store, item):
        import time

        if not store.mark_processing(item.job_id):
            return
        time.sleep(0.3)
        store.mark_ready(item.job_id, STUB_ANSWER, 1.0, "stub", 5)

    app.state.pool.worker_fn = slow_worker
    with TestClient(app) as client:
        first = client.post("/in.php", data={"key": "testkey", "body": sample_b64})
        assert first.text.startswith("OK|")
        second = client.post("/in.php", data={"key": "testkey", "body": sample_b64})
        assert second.text.strip() == "ERROR_NO_SLOT_AVAILABLE"
