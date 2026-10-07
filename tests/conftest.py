"""Shared test fixtures: an isolated app instance and a stub solver."""

from __future__ import annotations

import base64
import os
import sys
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ysolver.api import create_app
from ysolver.config import Settings
from ysolver.protocol import first
from ysolver.synth import make_captcha, to_png

STUB_ANSWER = "AB123"


def make_settings(tmp_path, **overrides) -> Settings:
    defaults = dict(
        api_keys=("testkey",),
        require_key=True,
        backend="template",
        workers=1,
        db_path=str(tmp_path / "jobs.db"),
        result_ttl=60,
        max_queue=4,
        solve_timeout=5.0,
        dashboard=False,
        port=8123,
    )
    defaults.update(overrides)
    return Settings(**defaults)


@pytest.fixture()
def app(tmp_path):
    """App wired to a one-shot stub worker so tests never run real OCR."""
    return build_app(tmp_path)


def build_app(tmp_path, wait: float = 2.0, **overrides):
    from ysolver.worker import WorkItem

    def stub_worker(store, item: WorkItem) -> None:
        if not store.mark_processing(item.job_id):
            return
        time.sleep(0.01)  # let polling code observe the "processing" state
        store.mark_ready(item.job_id, STUB_ANSWER, 0.99, "stub", 10)

    return create_app(make_settings(tmp_path, **overrides), worker_fn=stub_worker)


@pytest.fixture()
def client(app):
    from fastapi.testclient import TestClient

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def sample_png() -> bytes:
    image, _ = make_captcha(text="abc23", seed=1)
    return to_png(image)


@pytest.fixture()
def sample_b64(sample_png: bytes) -> str:
    return base64.b64encode(sample_png).decode("ascii")


def poll_form(client, job_id: str, timeout: float = 3.0) -> str:
    """Poll the form API until the job settles."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        response = client.get("/res.php", params={"key": "testkey", "action": "get", "id": job_id})
        body = response.text.strip()
        if body != "CAPCHA_NOT_READY":
            return body
        time.sleep(0.02)
    raise AssertionError("job never settled")


def poll_task(client, task_id: str, timeout: float = 3.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        payload = client.get("/getTaskResult", params={"clientKey": "testkey", "taskId": task_id})
        data = payload.json()
        if data.get("status") != "processing":
            return data
        time.sleep(0.02)
    raise AssertionError("task never settled")


def key_of(payload: dict, *names) -> str:
    return first(payload, names) or ""
