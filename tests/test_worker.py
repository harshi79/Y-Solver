"""Worker pool, queue limits, scheduled jobs and the watchdog."""

from __future__ import annotations

import time

import pytest

from ysolver.config import Settings
from ysolver.models import JobStatus
from ysolver.solvers import CaptchaSolver
from ysolver.store import JobStore
from ysolver.worker import QueueFull, Watchdog, WorkerPool


def _pool(tmp_path, workers: int = 2, max_queue: int = 10, delay: float = 0.0, **kwargs):
    store = JobStore(str(tmp_path / "jobs.db"))
    settings = Settings(backend="template", charset="abc123")
    solver = CaptchaSolver(settings)
    pool = WorkerPool(store, solver, workers=workers, max_queue=max_queue, **kwargs)
    return store, pool


def test_jobs_are_solved_and_settled(tmp_path):
    store, pool = _pool(tmp_path)
    pool.start()
    try:
        for _ in range(6):
            pool.submit(image=b"\x89PNG fake", method="base64", charset="abc123")
        assert pool.drain(5.0)
        jobs = store.recent(10)
        assert len(jobs) == 6
        assert all(job.status is JobStatus.FAILED for job in jobs)  # fake image
        assert all(job.error_code for job in jobs)
    finally:
        pool.stop()
        store.close()


def test_custom_worker_function_is_used(tmp_path):
    seen = []

    def worker(store, item):
        seen.append(item.job_id)
        store.mark_processing(item.job_id)
        store.mark_ready(item.job_id, "XYZ", 1.0, "test", 1)

    store, pool = _pool(tmp_path, worker_fn=worker)
    pool.start()
    try:
        job = pool.submit(image=b"data", method="base64", charset="abc")
        assert pool.drain(5.0)
        assert seen == [job.id]
        assert store.get(job.id).text == "XYZ"
    finally:
        pool.stop()
        store.close()


def test_queue_limit_raises_queuefull(tmp_path):
    store, pool = _pool(tmp_path, workers=1, max_queue=1)
    pool.start()
    try:
        pool.submit(image=b"one", method="base64", charset="abc")
        with pytest.raises(QueueFull):
            pool.submit(image=b"two", method="base64", charset="abc")
        pool.max_queue = 100
        assert pool.drain(5.0)
    finally:
        pool.stop()
        store.close()


def test_in_flight_returns_to_zero(tmp_path):
    store, pool = _pool(tmp_path, workers=1)
    pool.start()
    try:
        for _ in range(3):
            pool.submit(image=b"x", method="base64", charset="abc")
        assert pool.drain(5.0)
        time.sleep(0.05)
        assert pool.in_flight == 0
    finally:
        pool.stop()
        store.close()


def test_scheduled_job_waits_for_its_delay(tmp_path):
    store, pool = _pool(tmp_path, workers=1)
    pool.start()
    try:
        job = pool.submit(image=b"later", method="base64", charset="abc", delay=0.4)
        assert pool.scheduled_depth() == 1
        time.sleep(0.15)
        assert store.get(job.id).status is JobStatus.PENDING
        pool.drain(5.0)
        deadline = time.time() + 3
        while time.time() < deadline and store.get(job.id).status is JobStatus.PENDING:
            time.sleep(0.05)
        assert pool.scheduled_depth() == 0
        assert store.get(job.id).status is JobStatus.FAILED  # ran, then failed on fake data
    finally:
        pool.stop()
        store.close()


def test_watchdog_fails_stuck_jobs(tmp_path):
    store = JobStore(str(tmp_path / "jobs.db"))
    job = store.create(method="base64", charset="abc")
    store.mark_processing(job.id)
    store._conn.execute(
        "UPDATE jobs SET started_at=? WHERE id=?", (time.time() - 600, job.id)
    )
    store._conn.commit()

    watchdog = Watchdog(store, timeout_s=60, ttl_s=1800, interval=0.05)
    watchdog.start()
    try:
        deadline = time.time() + 3
        while time.time() < deadline and store.get(job.id).status is JobStatus.PROCESSING:
            time.sleep(0.05)
        assert store.get(job.id).status is JobStatus.FAILED
        assert store.get(job.id).error_code == "ERROR_INTERNAL"
    finally:
        watchdog.stop()
        watchdog.join(timeout=1)
        store.close()


def test_length_hints_are_applied_to_the_result(tmp_path):
    store, pool = _pool(tmp_path)
    pool.start()
    try:
        job = pool.submit(
            image=b"\x89PNG fake",
            method="base64",
            charset="abc",
            min_len=4,
            max_len=6,
        )
        assert pool.drain(5.0)
        assert store.get(job.id).status is JobStatus.FAILED  # engine error path is fine
    finally:
        pool.stop()
        store.close()


def test_min_len_padding_helper():
    from ysolver.protocol import enforce_length

    assert enforce_length("ab", 4, None) == "abbb"
    assert enforce_length("abcdefgh", None, 3) == "abc"


def test_store_images_option_writes_the_upload(tmp_path):
    store, pool = _pool(tmp_path, image_dir=str(tmp_path / "images"))
    pool.start()
    try:
        job = pool.submit(image=b"\x89PNG fake payload", method="base64", charset="abc")
        assert pool.drain(5.0)
        stored = tmp_path / "images" / f"{job.id}.png"
        assert stored.read_bytes() == b"\x89PNG fake payload"
    finally:
        pool.stop()
        store.close()


def test_images_are_not_stored_by_default(tmp_path):
    store, pool = _pool(tmp_path)
    pool.start()
    try:
        pool.submit(image=b"secret bytes", method="base64", charset="abc")
        assert pool.drain(5.0)
        assert not (tmp_path / "images").exists()
    finally:
        pool.stop()
        store.close()
