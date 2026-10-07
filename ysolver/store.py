"""SQLite-backed job store.

A single connection guarded by a lock is plenty for this workload (OCR is the
bottleneck, not the bookkeeping) and keeps Y-Solver dependency-free — no Redis,
no Postgres, no external queue.
"""

from __future__ import annotations

import os
import random
import sqlite3
import threading
import time
from typing import Dict, List, Optional

from .models import Job, JobStatus

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id          TEXT PRIMARY KEY,
    method      TEXT NOT NULL,
    charset     TEXT NOT NULL,
    status      TEXT NOT NULL,
    text        TEXT,
    confidence  REAL,
    backend     TEXT,
    error_code  TEXT,
    error_text  TEXT,
    created_at  REAL NOT NULL,
    started_at  REAL,
    finished_at REAL,
    solve_ms    INTEGER,
    tries       INTEGER NOT NULL DEFAULT 0,
    key_hint    TEXT NOT NULL DEFAULT '',
    image_sha   TEXT
);
CREATE INDEX IF NOT EXISTS idx_jobs_created ON jobs (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs (status);
"""

_COLUMNS = [
    "id", "method", "charset", "status", "text", "confidence", "backend",
    "error_code", "error_text", "created_at", "started_at", "finished_at",
    "solve_ms", "tries", "key_hint", "image_sha",
]


class JobStore:
    def __init__(self, path: str = "data/ysolver.db"):
        self.path = path
        if path != ":memory:":
            parent = os.path.dirname(os.path.abspath(path))
            os.makedirs(parent, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(path, check_same_thread=False, timeout=30.0)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            if path != ":memory:":
                self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    # ---------------------------------------------------------------- writes
    def create(self, method: str, charset: str, key_hint: str = "",
               image_sha: Optional[str] = None, job_id: Optional[str] = None) -> Job:
        last_error: Optional[Exception] = None
        for _ in range(10):
            job = Job(
                id=job_id or _new_id(),
                method=method,
                charset=charset,
                key_hint=key_hint,
                image_sha=image_sha,
            )
            with self._lock:
                try:
                    self._conn.execute(
                        f"INSERT INTO jobs ({','.join(_COLUMNS)}) "
                        f"VALUES ({','.join('?' * len(_COLUMNS))})",
                        (
                            job.id, job.method, job.charset, job.status.value, None,
                            None, None, None, None, job.created_at, None, None, None,
                            0, job.key_hint, job.image_sha,
                        ),
                    )
                    self._conn.commit()
                    return job
                except sqlite3.IntegrityError as exc:  # id collision
                    last_error = exc
                    if job_id is not None:
                        break
        raise RuntimeError(f"could not allocate a unique job id: {last_error}")

    def mark_processing(self, job_id: str) -> bool:
        now = time.time()
        with self._lock:
            cur = self._conn.execute(
                "UPDATE jobs SET status=?, started_at=?, tries=tries+1 "
                "WHERE id=? AND status=?",
                (JobStatus.PROCESSING.value, now, job_id, JobStatus.PENDING.value),
            )
            self._conn.commit()
            return cur.rowcount > 0

    def mark_ready(self, job_id: str, text: str, confidence: Optional[float],
                   backend: str, solve_ms: int) -> bool:
        now = time.time()
        with self._lock:
            cur = self._conn.execute(
                "UPDATE jobs SET status=?, text=?, confidence=?, backend=?, "
                "finished_at=?, solve_ms=? WHERE id=? AND status IN (?,?)",
                (JobStatus.READY.value, text, confidence, backend, now, solve_ms,
                 job_id, JobStatus.PENDING.value, JobStatus.PROCESSING.value),
            )
            self._conn.commit()
            return cur.rowcount > 0

    def mark_failed(self, job_id: str, code: str, message: str, solve_ms: int = 0) -> bool:
        now = time.time()
        with self._lock:
            cur = self._conn.execute(
                "UPDATE jobs SET status=?, error_code=?, error_text=?, finished_at=?, "
                "solve_ms=? WHERE id=? AND status IN (?,?)",
                (JobStatus.FAILED.value, code, message, now, solve_ms, job_id,
                 JobStatus.PENDING.value, JobStatus.PROCESSING.value),
            )
            self._conn.commit()
            return cur.rowcount > 0

    # ----------------------------------------------------------------- reads
    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            row = self._conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        return _row_to_job(row) if row else None

    def recent(self, limit: int = 20) -> List[Job]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (int(limit),)
            ).fetchall()
        return [_row_to_job(r) for r in rows]

    def queue_depth(self) -> int:
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) AS n FROM jobs WHERE status=?", (JobStatus.PENDING.value,)
            ).fetchone()
            busy = self._conn.execute(
                "SELECT COUNT(*) AS n FROM jobs WHERE status=?", (JobStatus.PROCESSING.value,)
            ).fetchone()
        return int(row["n"]) + int(busy["n"])

    def stats(self) -> Dict[str, object]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT status, COUNT(*) AS n, AVG(solve_ms) AS avg_ms FROM jobs GROUP BY status"
            ).fetchall()
            total = self._conn.execute("SELECT COUNT(*) AS n FROM jobs").fetchone()["n"]
            last_hour = self._conn.execute(
                "SELECT COUNT(*) AS n FROM jobs WHERE created_at > ?", (time.time() - 3600,)
            ).fetchone()["n"]
        by_status = {r["status"]: int(r["n"]) for r in rows}
        avg_ms = next(
            (r["avg_ms"] for r in rows if r["status"] == JobStatus.READY.value), None
        )
        return {
            "total": int(total),
            "byStatus": by_status,
            "solved": by_status.get(JobStatus.READY.value, 0),
            "failed": by_status.get(JobStatus.FAILED.value, 0),
            "pending": by_status.get(JobStatus.PENDING.value, 0),
            "processing": by_status.get(JobStatus.PROCESSING.value, 0),
            "lastHour": int(last_hour),
            "avgSolveMs": round(avg_ms) if avg_ms is not None else None,
        }

    def expire_stale(self, timeout_s: float) -> List[str]:
        """Fail jobs stuck in ``processing`` longer than ``timeout_s``."""
        cutoff = time.time() - timeout_s
        with self._lock:
            rows = self._conn.execute(
                "SELECT id FROM jobs WHERE status=? AND started_at < ?",
                (JobStatus.PROCESSING.value, cutoff),
            ).fetchall()
        return [r["id"] for r in rows]

    def cleanup(self, ttl_s: int) -> int:
        """Delete finished jobs older than the retention window."""
        cutoff = time.time() - ttl_s
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM jobs WHERE finished_at IS NOT NULL AND finished_at < ?",
                (cutoff,),
            )
            self._conn.commit()
            return cur.rowcount

    def close(self) -> None:
        with self._lock:
            self._conn.close()


def _new_id() -> str:
    """Paid services hand out numeric ids; mimic that for client compatibility."""
    return str(random.randint(100_000_000, 999_999_999))


def _row_to_job(row: sqlite3.Row) -> Job:
    return Job(
        id=row["id"],
        method=row["method"],
        charset=row["charset"],
        status=JobStatus(row["status"]),
        text=row["text"],
        confidence=row["confidence"],
        backend=row["backend"],
        error_code=row["error_code"],
        error_text=row["error_text"],
        created_at=row["created_at"],
        started_at=row["started_at"],
        finished_at=row["finished_at"],
        solve_ms=row["solve_ms"],
        tries=row["tries"],
        key_hint=row["key_hint"],
        image_sha=row["image_sha"],
    )
