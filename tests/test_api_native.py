"""Native REST API: one-shot solve, stats, health, samples."""

from __future__ import annotations

import base64


def test_healthz_reports_engines(client):
    payload = client.get("/healthz").json()
    assert payload["status"] in ("ok", "degraded")
    assert "engines" in payload and "backend" in payload
    assert payload["engines"]["template"] is True  # always available


def test_stats_counts_jobs(client, sample_b64):
    client.post("/in.php", data={"key": "testkey", "body": sample_b64})
    client.get("/healthz")
    payload = client.get("/api/stats").json()
    assert payload["total"] >= 1
    assert payload["workers"] >= 1
    assert payload["requireKey"] is True
    assert isinstance(payload["engines"], list)


def test_api_solve_one_shot(client, sample_b64):
    payload = client.get("/api/solve", params={"key": "testkey", "body": sample_b64}).json()
    assert payload["status"] == "ready"
    assert payload["text"] == "AB123"
    assert payload["backend"] == "stub"


def test_api_solve_async_returns_202(client, sample_b64):
    response = client.get(
        "/api/solve", params={"key": "testkey", "body": sample_b64, "wait": "false"}
    )
    assert response.status_code == 202
    assert response.json()["status"] == "pending"


def test_api_solve_bad_key_is_401(client, sample_b64):
    response = client.get("/api/solve", params={"key": "bad", "body": sample_b64})
    assert response.status_code == 401


def test_api_jobs_listing(client, sample_b64):
    client.get("/api/solve", params={"key": "testkey", "body": sample_b64})
    payload = client.get("/api/jobs", params={"limit": 5}).json()
    assert payload["jobs"]
    assert payload["jobs"][0]["text"] == "AB123"


def test_api_job_by_id(client, sample_b64):
    solved = client.get("/api/solve", params={"key": "testkey", "body": sample_b64}).json()
    payload = client.get(f"/api/jobs/{solved['id']}").json()
    assert payload["id"] == solved["id"]
    assert client.get("/api/jobs/000000001").status_code == 404


def test_api_sample_endpoint_generates_truthful_labels(client):
    payload = client.get("/api/sample", params={"length": 4, "seed": 99}).json()
    assert payload["label"]
    assert len(payload["label"]) == 4
    assert base64.b64decode(payload["pngBase64"])[:4] == b"\x89PNG"


def test_openapi_schema_is_served(client):
    schema = client.get("/openapi.json").json()
    assert schema["info"]["title"] == "Y-Solver"
    assert "/api/solve" in schema["paths"]


def test_solve_wakes_immediately_when_the_job_is_already_done(client, sample_b64):
    """The waiter must not hang if the job settled before we began waiting."""
    payload = client.get("/api/solve", params={"key": "testkey", "body": sample_b64}).json()
    assert payload["status"] == "ready"

    # A second read of the same finished job goes through the same code path.
    job = client.get(f"/api/jobs/{payload['id']}").json()
    assert job["status"] == "ready"


def test_solve_returns_202_after_the_timeout(client, sample_b64):
    """A job still in flight past the caller's timeout yields 202, not a hang."""
    import time

    app = client.app

    def slow_worker(store, item):
        if not store.mark_processing(item.job_id):
            return
        time.sleep(0.6)
        store.mark_ready(item.job_id, "SLOW", 1.0, "stub", 600)

    app.state.pool.worker_fn = slow_worker
    started = time.monotonic()
    response = client.get(
        "/api/solve", params={"key": "testkey", "body": sample_b64, "timeout": "0.1"}
    )
    elapsed = time.monotonic() - started
    assert response.status_code == 202
    assert response.json()["status"] == "pending"
    assert elapsed < 0.5, "the endpoint should give up at its own timeout, not the worker's"
