"""Human-in-the-loop labelling.

This is the *legitimate* reading of "train it with humans": the people who
operate the deployment label captchas they are entitled to process, and the
server turns those labels into a model that gets better at that specific
captcha style. Nobody is shown captchas belonging to somebody else, there is no
crowd and no queue of strangers — the humans are you and your team.

The loop has three stages:

1. **Capture** — while solving, the worker keeps a copy of interesting samples:
   every failure, anything the model was unsure about, plus a random slice of
   ordinary traffic (a training set of only hard cases teaches a model to be
   nervous). Deduplicated by image hash so a human never sees the same image
   twice.
2. **Label** — ``GET /api/labels/pending`` hands one sample to a person, who
   types the characters they see. ``POST /api/labels`` files it as ground truth;
   a human label always beats a model guess.
3. **Train** — ``scripts/train.py`` turns the label set into a glyph model,
   measures it against every other engine on a held-out split, and records the
   winner so ``auto`` starts using it.

Storage lives in the same SQLite file as the job store, so a deployment gains
no new moving parts.
"""

from __future__ import annotations

import hashlib
import os
import sqlite3
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

_SCHEMA = """
CREATE TABLE IF NOT EXISTS label_candidates (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id     TEXT,
    image      BLOB NOT NULL,
    image_sha  TEXT NOT NULL UNIQUE,
    model_text TEXT,
    confidence REAL,
    backend    TEXT,
    reason     TEXT NOT NULL DEFAULT 'uncertain',
    status     TEXT NOT NULL DEFAULT 'pending',
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_candidates_status ON label_candidates (status, created_at);

CREATE TABLE IF NOT EXISTS labels (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    image      BLOB NOT NULL,
    image_sha  TEXT NOT NULL,
    text       TEXT NOT NULL,
    source     TEXT NOT NULL DEFAULT 'human',
    tag        TEXT NOT NULL DEFAULT '',
    confidence REAL,
    backend    TEXT,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_labels_created ON labels (created_at);
"""

MIN_TEXT_LENGTH = 1
MAX_TEXT_LENGTH = 24


@dataclass
class Candidate:
    id: int
    image: bytes
    image_sha: str
    model_text: Optional[str]
    confidence: Optional[float]
    backend: Optional[str]
    reason: str
    created_at: float

    def to_public(self, include_image: bool = False) -> dict:
        payload = {
            "id": self.id,
            "modelText": self.model_text,
            "confidence": self.confidence,
            "backend": self.backend,
            "reason": self.reason,
            "createdAt": self.created_at,
        }
        if include_image:
            import base64

            payload["pngBase64"] = base64.b64encode(self.image).decode("ascii")
        return payload


def image_digest(image: bytes) -> str:
    return hashlib.sha256(image).hexdigest()[:32]


class LabelStore:
    """SQLite-backed store for candidates awaiting a human and their labels."""

    def __init__(self, path: str = "data/ysolver.db", max_candidates: int = 5000):
        self.path = path
        self.max_candidates = max(100, max_candidates)
        if path != ":memory:":
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(path, check_same_thread=False, timeout=30.0)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            if path != ":memory:":
                self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    # ------------------------------------------------------------- capture
    def add_candidate(
        self,
        image: bytes,
        job_id: Optional[str] = None,
        model_text: Optional[str] = None,
        confidence: Optional[float] = None,
        backend: Optional[str] = None,
        reason: str = "uncertain",
    ) -> bool:
        """Queue a sample for labelling. Returns False if it was seen already."""
        digest = image_digest(image)
        with self._lock:
            try:
                self._conn.execute(
                    "INSERT INTO label_candidates "
                    "(job_id, image, image_sha, model_text, confidence, backend, reason, "
                    " status, created_at) VALUES (?,?,?,?,?,?,?,'pending',?)",
                    (job_id, image, digest, model_text, confidence, backend, reason, time.time()),
                )
                self._conn.commit()
            except sqlite3.IntegrityError:
                return False  # already queued or already labelled
            self._trim_candidates()
            return True

    def _trim_candidates(self) -> None:
        """Keep the pending pile bounded; oldest unlabelled samples go first."""
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) AS n FROM label_candidates WHERE status='pending'"
            ).fetchone()
            if row["n"] <= self.max_candidates:
                return
            excess = row["n"] - self.max_candidates
            self._conn.execute(
                "DELETE FROM label_candidates WHERE id IN ("
                "  SELECT id FROM label_candidates WHERE status='pending'"
                "  ORDER BY created_at ASC LIMIT ?)",
                (excess,),
            )
            self._conn.commit()

    # --------------------------------------------------------------- label
    def pending(self, limit: int = 1, include_image: bool = False) -> List[Candidate]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM label_candidates WHERE status='pending' "
                "ORDER BY created_at ASC LIMIT ?",
                (int(limit),),
            ).fetchall()
        return [_to_candidate(row) for row in rows]

    def pending_count(self) -> int:
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) AS n FROM label_candidates WHERE status='pending'"
            ).fetchone()
        return int(row["n"])

    def submit_label(self, candidate_id: int, text: str, tag: str = "") -> dict:
        """File a human label for a candidate (or corrections to a model guess)."""
        cleaned = (text or "").strip()
        if not (MIN_TEXT_LENGTH <= len(cleaned) <= MAX_TEXT_LENGTH):
            raise ValueError(
                f"a label must be {MIN_TEXT_LENGTH}-{MAX_TEXT_LENGTH} characters long"
            )
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM label_candidates WHERE id=?", (int(candidate_id),)
            ).fetchone()
            if row is None:
                raise KeyError(f"candidate {candidate_id} does not exist")
            self._conn.execute(
                "INSERT INTO labels (image, image_sha, text, source, tag, confidence, "
                "backend, created_at) VALUES (?,?,?,'human',?,?,?,?)",
                (
                    row["image"], row["image_sha"], cleaned, tag or "",
                    row["confidence"], row["backend"], time.time(),
                ),
            )
            self._conn.execute(
                "UPDATE label_candidates SET status='labeled' WHERE id=?", (int(candidate_id),)
            )
            self._conn.commit()
        return {"id": candidate_id, "text": cleaned, "tag": tag or ""}

    def skip(self, candidate_id: int) -> bool:
        """Mark a sample unreadable/unusable so it stops being offered."""
        with self._lock:
            cur = self._conn.execute(
                "UPDATE label_candidates SET status='skipped' WHERE id=? AND status='pending'",
                (int(candidate_id),),
            )
            self._conn.commit()
            return cur.rowcount > 0

    def add_label(self, image: bytes, text: str, tag: str = "",
                  source: str = "import") -> None:
        """Import a label directly (bulk dataset load, no candidate step)."""
        cleaned = (text or "").strip()
        if not (MIN_TEXT_LENGTH <= len(cleaned) <= MAX_TEXT_LENGTH):
            raise ValueError("label length out of range")
        with self._lock:
            self._conn.execute(
                "INSERT INTO labels (image, image_sha, text, source, tag, confidence, "
                "backend, created_at) VALUES (?,?,?,?,?,NULL,NULL,?)",
                (image, image_digest(image), cleaned, source, tag or "", time.time()),
            )
            self._conn.commit()

    # ---------------------------------------------------------------- read
    def dataset(self, tag: Optional[str] = None) -> List[Tuple[bytes, str, str]]:
        """Every label as ``(image, text, tag)``."""
        query = "SELECT image, text, tag FROM labels"
        params: Sequence = ()
        if tag:
            query += " WHERE tag=?"
            params = (tag,)
        query += " ORDER BY created_at ASC"
        with self._lock:
            rows = self._conn.execute(query, params).fetchall()
        return [(row["image"], row["text"], row["tag"]) for row in rows]

    def stats(self) -> Dict[str, object]:
        with self._lock:
            labeled = self._conn.execute("SELECT COUNT(*) AS n FROM labels").fetchone()["n"]
            pending = self.pending_count_locked()
            skipped = self._conn.execute(
                "SELECT COUNT(*) AS n FROM label_candidates WHERE status='skipped'"
            ).fetchone()["n"]
            by_tag = self._conn.execute(
                "SELECT tag, COUNT(*) AS n FROM labels GROUP BY tag ORDER BY n DESC LIMIT 10"
            ).fetchall()
            by_reason = self._conn.execute(
                "SELECT reason, COUNT(*) AS n FROM label_candidates "
                "WHERE status='pending' GROUP BY reason"
            ).fetchall()
        return {
            "labeled": int(labeled),
            "pending": int(pending),
            "skipped": int(skipped),
            "byTag": {row["tag"] or "(untagged)": int(row["n"]) for row in by_tag},
            "pendingByReason": {row["reason"]: int(row["n"]) for row in by_reason},
        }

    def pending_count_locked(self) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) AS n FROM label_candidates WHERE status='pending'"
        ).fetchone()
        return int(row["n"])

    def close(self) -> None:
        with self._lock:
            self._conn.close()


def _to_candidate(row: sqlite3.Row) -> Candidate:
    return Candidate(
        id=row["id"],
        image=row["image"],
        image_sha=row["image_sha"],
        model_text=row["model_text"],
        confidence=row["confidence"],
        backend=row["backend"],
        reason=row["reason"],
        created_at=row["created_at"],
    )
