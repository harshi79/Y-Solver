"""The JSON /createTask + /getTaskResult API."""

from __future__ import annotations

from conftest import STUB_ANSWER, poll_task


def test_create_task_plain_json(client, sample_b64):
    response = client.post(
        "/createTask",
        json={
            "clientKey": "testkey",
            "task": {"type": "ImageToTextTask", "body": sample_b64},
        },
    )
    payload = response.json()
    assert payload["errorId"] == 0
    assert payload["taskId"].isdigit()

    result = poll_task(client, payload["taskId"])
    assert result["errorId"] == 0
    assert result["status"] == "ready"
    assert result["solution"]["text"] == STUB_ANSWER


def test_create_task_with_flat_body_param(client, sample_b64):
    response = client.post(
        "/createTask",
        json={"clientKey": "testkey", "type": "ImageToTextTask", "body": sample_b64},
    )
    payload = response.json()
    assert payload["errorId"] == 0
    assert poll_task(client, payload["taskId"])["status"] == "ready"


def test_create_task_accepts_qip_style_camel_case(client, sample_b64):
    """Anti-Captcha uses 'taskId'; Q-like services use 'task_id' in the response."""
    created = client.post(
        "/createTask",
        json={"clientKey": "testkey", "task": {"type": "ImageToTextTask", "body": sample_b64}},
    ).json()
    result = client.post(
        "/getTaskResult", json={"clientKey": "testkey", "taskId": created["taskId"]}
    ).json()
    assert result["status"] in ("processing", "ready")


def test_wrong_key_is_rejected(client, sample_b64):
    payload = client.post(
        "/createTask",
        json={"clientKey": "bad", "task": {"type": "ImageToTextTask", "body": sample_b64}},
    ).json()
    assert payload["errorId"] == 1
    assert payload["errorCode"] == "ERROR_WRONG_USER_KEY"


def test_interactive_task_type_is_refused(client):
    payload = client.post(
        "/createTask",
        json={
            "clientKey": "testkey",
            "task": {
                "type": "RecaptchaV2TaskProxyless",
                "websiteURL": "https://example.com",
                "websiteKey": "site-key",
            },
        },
    ).json()
    assert payload["errorId"] == 1
    assert payload["errorCode"] == "ERROR_METHOD_NOT_SUPPORTED"
    assert "image" in payload["errorDescription"]


def test_missing_image_is_rejected(client):
    payload = client.post(
        "/createTask", json={"clientKey": "testkey", "task": {"type": "ImageToTextTask"}}
    ).json()
    assert payload["errorId"] == 1
    assert payload["errorCode"].startswith("ERROR_")


def test_get_task_result_unknown_id(client):
    payload = client.get("/getTaskResult", params={"clientKey": "testkey", "taskId": "123"}).json()
    assert payload["errorId"] == 1
    assert payload["errorCode"] == "ERROR_WRONG_CAPTCHA_ID"


def test_get_task_result_requires_key(client):
    payload = client.get("/getTaskResult", params={"taskId": "123"}).json()
    assert payload["errorCode"] == "ERROR_WRONG_USER_KEY"


def test_processing_status_before_the_worker_finishes(tmp_path, sample_b64):
    import time

    from conftest import build_app
    from fastapi.testclient import TestClient

    app = build_app(tmp_path)

    def slow_worker(store, item):
        if not store.mark_processing(item.job_id):
            return
        time.sleep(0.2)
        store.mark_ready(item.job_id, "LATE", 1.0, "stub", 200)

    app.state.pool.worker_fn = slow_worker
    with TestClient(app) as client:
        created = client.post(
            "/createTask",
            json={"clientKey": "testkey", "task": {"type": "ImageToTextTask", "body": sample_b64}},
        ).json()
        immediate = client.get(
            "/getTaskResult", params={"clientKey": "testkey", "taskId": created["taskId"]}
        ).json()
        assert immediate == {"errorId": 0, "status": "processing"}
        assert poll_task(client, created["taskId"])["solution"]["text"] == "LATE"


def test_failed_job_surfaces_error_code(tmp_path, sample_b64):
    from conftest import build_app
    from fastapi.testclient import TestClient

    from ysolver.models import JobStatus

    job_id = "555000111"
    app = build_app(tmp_path)
    app.state.store.create(method="base64", charset="abc", job_id=job_id)
    app.state.store.mark_processing(job_id)
    app.state.store.mark_failed(job_id, "ERROR_CAPTCHA_UNSOLVABLE", "could not read it")

    with TestClient(app) as client:
        payload = client.get(
            "/getTaskResult", params={"clientKey": "testkey", "taskId": job_id}
        ).json()
        assert payload["errorId"] == 1
        assert payload["errorCode"] == "ERROR_CAPTCHA_UNSOLVABLE"
        assert app.state.store.get(job_id).status is JobStatus.FAILED
