"""Job store behaviour."""

from __future__ import annotations

import time

from ysolver.models import JobStatus
from ysolver.store import JobStore


def _store(tmp_path) -> JobStore:
    return JobStore(str(tmp_path / "jobs.db"))


def test_create_returns_a_numeric_id_and_pending_status(tmp_path):
    store = _store(tmp_path)
    job = store.create(method="base64", charset="abc")
    try:
        assert job.id.isdigit() and 6 <= len(job.id) <= 20
        assert job.status is JobStatus.PENDING
        assert store.get(job.id).id == job.id
    finally:
        store.close()


def test_lifecycle_transitions_are_guarded(tmp_path):
    store = _store(tmp_path)
    job = store.create(method="base64", charset="abc")
    try:
        assert store.mark_processing(job.id) is True
        assert store.mark_processing(job.id) is False  # already claimed
        assert store.mark_ready(job.id, "hello", 0.9, "template", 12) is True
        assert store.mark_failed(job.id, "ERROR_INTERNAL", "late") is False
        settled = store.get(job.id)
        assert settled.status is JobStatus.READY
        assert settled.text == "hello"
        assert settled.solve_ms == 12
        assert settled.backend == "template"
    finally:
        store.close()


def test_unknown_job_is_none(tmp_path):
    store = _store(tmp_path)
    try:
        assert store.get("000000001") is None
    finally:
        store.close()


def test_stats_aggregate_by_status(tmp_path):
    store = _store(tmp_path)
    try:
        ready = store.create(method="base64", charset="abc")
        failed = store.create(method="base64", charset="abc")
        store.create(method="base64", charset="abc")  # left pending
        store.mark_ready(ready.id, "aaa", 0.5, "template", 20)
        store.mark_failed(failed.id, "ERROR_CAPTCHA_UNSOLVABLE", "nope")
        stats = store.stats()
        assert stats["total"] == 3
        assert stats["solved"] == 1
        assert stats["failed"] == 1
        assert stats["pending"] == 1
        assert stats["lastHour"] == 3
        assert stats["avgSolveMs"] == 20
    finally:
        store.close()


def test_queue_depth_counts_pending_work(tmp_path):
    store = _store(tmp_path)
    try:
        assert store.queue_depth() == 0
        store.create(method="base64", charset="abc")
        assert store.queue_depth() == 1
    finally:
        store.close()


def test_expire_stale_reports_only_overdue_jobs(tmp_path):
    store = _store(tmp_path)
    try:
        job = store.create(method="base64", charset="abc")
        store.mark_processing(job.id)
        store._conn.execute(  # simulate a worker that started long ago
            "UPDATE jobs SET started_at=? WHERE id=?", (time.time() - 999, job.id)
        )
        store._conn.commit()
        assert store.expire_stale(60) == [job.id]
    finally:
        store.close()


def test_cleanup_deletes_only_finished_jobs(tmp_path):
    store = _store(tmp_path)
    try:
        done = store.create(method="base64", charset="abc")
        store.mark_ready(done.id, "x", 0.1, "template", 5)
        store._conn.execute(
            "UPDATE jobs SET finished_at=? WHERE id=?", (time.time() - 9999, done.id)
        )
        store._conn.commit()
        pending = store.create(method="base64", charset="abc")
        assert store.cleanup(60) == 1
        assert store.get(done.id) is None
        assert store.get(pending.id) is not None
    finally:
        store.close()


def test_recent_returns_newest_first(tmp_path):
    store = _store(tmp_path)
    try:
        first_job = store.create(method="base64", charset="abc")
        time.sleep(0.01)
        second = store.create(method="base64", charset="abc")
        ids = [job.id for job in store.recent(10)]
        assert ids[0] == second.id and ids[1] == first_job.id
    finally:
        store.close()


def test_job_to_public_hides_internals(tmp_path):
    store = _store(tmp_path)
    try:
        job = store.create(method="base64", charset="abc", key_hint="te***ey")
        public = job.to_public()
        assert public["status"] == "pending"
        assert public["keyHint"] == "te***ey"
        assert "image_sha" not in public
    finally:
        store.close()
