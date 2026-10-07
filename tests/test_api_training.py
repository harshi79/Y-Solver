"""HTTP surface of the human-in-the-loop loop, plus the scope statement."""

from __future__ import annotations

import base64

from ysolver.synth import make_captcha, to_png


def _png(seed: int = 1) -> bytes:
    return to_png(make_captcha(seed=seed)[0])


def test_scope_endpoint_states_what_is_supported(client):
    payload = client.get("/api/scope").json()
    assert "text-in-image" in payload["solves"]
    assert "recaptcha" in payload["unsupported"]
    assert payload["errorForUnsupported"] == "ERROR_METHOD_NOT_SUPPORTED"
    assert payload["supported"]["base64"]


def test_interactive_refusal_carries_actionable_suggestions(client):
    payload = client.post(
        "/createTask",
        json={"clientKey": "testkey", "task": {"type": "RecaptchaV2TaskProxyless"}},
    ).json()
    assert payload["errorCode"] == "ERROR_METHOD_NOT_SUPPORTED"
    assert "reCAPTCHA" in payload["errorDescription"] or "recaptcha" in payload["errorDescription"]
    assert payload["suggestions"], "a refusal should tell the caller what to do instead"


def test_label_queue_lifecycle_over_http(client, app):
    app.state.labels.add_candidate(_png(3), model_text="wrong", confidence=0.2,
                                  reason="uncertain")

    pending = client.get("/api/labels/pending", params={"limit": 1}).json()
    assert pending["pending"] == 1
    item = pending["items"][0]
    assert item["modelText"] == "wrong"
    assert base64.b64decode(item["pngBase64"])[:4] == b"\x89PNG"

    submission = client.post(
        "/api/labels", data={"key": "testkey", "id": item["id"], "text": "ab12c"}
    ).json()
    assert submission["status"] == "labeled"
    assert submission["text"] == "ab12c"
    assert submission["stats"]["labeled"] == 1

    assert client.get("/api/labels/pending").json()["pending"] == 0


def test_label_submission_can_skip_an_unreadable_sample(client, app):
    app.state.labels.add_candidate(_png(4), reason="failed")
    item = client.get("/api/labels/pending").json()["items"][0]
    payload = client.post(
        "/api/labels", data={"key": "testkey", "id": item["id"], "skip": "1"}
    ).json()
    assert payload["status"] == "skipped"
    assert client.get("/api/labels/stats").json()["skipped"] == 1


def test_a_captcha_that_really_reads_skip_is_still_a_label(client, app):
    """'skip' as text must file a label, not silently discard the sample."""
    app.state.labels.add_candidate(_png(7), reason="uncertain")
    item = client.get("/api/labels/pending").json()["items"][0]
    payload = client.post(
        "/api/labels", data={"key": "testkey", "id": item["id"], "text": "skip"}
    ).json()
    assert payload["status"] == "labeled"
    assert payload["text"] == "skip"


def test_label_submission_requires_a_key(client, app):
    app.state.labels.add_candidate(_png(5))
    item = client.get("/api/labels/pending").json()["items"][0]
    response = client.post("/api/labels", data={"id": item["id"], "text": "abc"})
    assert response.status_code == 401


def test_label_submission_validates_its_input(client, app):
    app.state.labels.add_candidate(_png(6))
    item = client.get("/api/labels/pending").json()["items"][0]
    too_long = client.post(
        "/api/labels", data={"key": "testkey", "id": item["id"], "text": "x" * 40}
    )
    assert too_long.status_code == 400
    missing = client.post("/api/labels", data={"key": "testkey", "text": "abc"})
    assert missing.status_code == 400


def test_label_stats_report_model_state(client):
    payload = client.get("/api/labels/stats").json()
    assert payload["learningEnabled"] is True
    assert "labeled" in payload and "pending" in payload
    assert payload["captureConfidenceBelow"] > 0
    assert payload["model"] is None or "glyphs" in payload["model"]


def test_failed_solves_are_queued_for_labelling(tmp_path):
    """The loop must capture failures without anyone configuring it.

    Exercised at the worker level: the HTTP fixtures swap in a stub worker, so
    this asserts the real processing path does the capturing.
    """
    from ysolver.config import Settings
    from ysolver.labels import LabelStore
    from ysolver.solvers import CaptchaSolver
    from ysolver.store import JobStore
    from ysolver.worker import WorkItem, process_item

    jobs = JobStore(str(tmp_path / "jobs.db"))
    labels = LabelStore(str(tmp_path / "jobs.db"))
    solver = CaptchaSolver(Settings(backend="template", charset="abc"))
    job = jobs.create(method="base64", charset="abc")
    process_item(
        jobs,
        WorkItem(job.id, b"definitely not an image", "abc", False),
        solver,
        labels=labels,
        learn=True,
        learn_rate=1.0,
    )
    assert jobs.get(job.id).status.value == "failed"
    pending = labels.pending(1)
    assert len(pending) == 1 and pending[0].reason == "failed"
    jobs.close()
    labels.close()


def test_capture_can_be_switched_off(tmp_path):
    from ysolver.config import Settings
    from ysolver.labels import LabelStore
    from ysolver.solvers import CaptchaSolver
    from ysolver.store import JobStore
    from ysolver.worker import WorkItem, process_item

    jobs = JobStore(str(tmp_path / "jobs.db"))
    labels = LabelStore(str(tmp_path / "jobs.db"))
    solver = CaptchaSolver(Settings(backend="template", charset="abc"))
    job = jobs.create(method="base64", charset="abc")
    process_item(
        jobs,
        WorkItem(job.id, b"definitely not an image", "abc", False),
        solver,
        labels=labels,
        learn=False,
    )
    assert labels.pending_count() == 0
    jobs.close()
    labels.close()


def test_confident_traffic_is_sampled_not_flooded(tmp_path):
    """With a high sample rate everything lands in the queue; at 0 nothing does."""
    from ysolver.config import Settings
    from ysolver.labels import LabelStore
    from ysolver.solvers import CaptchaSolver
    from ysolver.store import JobStore
    from ysolver.synth import make_captcha, to_png
    from ysolver.worker import WorkItem, process_item

    image = to_png(make_captcha(text="abc", seed=12, rotation=0, noise=0.0)[0])
    for rate, expected in ((0.0, 0), (1.0, 1)):
        jobs = JobStore(str(tmp_path / f"jobs{rate}.db"))
        labels = LabelStore(str(tmp_path / f"jobs{rate}.db"))
        solver = CaptchaSolver(Settings(backend="template", charset="abcdefghjkmnpqrstuvwxyz"))
        job = jobs.create(method="base64", charset="abc")
        process_item(
            jobs, WorkItem(job.id, image, "", False), solver,
            labels=labels, learn=True, learn_rate=rate, learn_confidence=-1.0,
        )
        assert jobs.get(job.id).status.value == "ready"
        assert labels.pending_count() == expected
        jobs.close()
        labels.close()
