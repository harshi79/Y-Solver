"""In-process worker pool.

Solving is pure CPU work, so a small thread pool is the right shape: jobs are
accepted instantly, the HTTP layer never blocks on OCR, and ``res.php`` polling
sees the same pending -> processing -> ready lifecycle as the paid services.
"""

from __future__ import annotations

import contextlib
import hashlib
import heapq
import logging
import os
import queue
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

from .models import Job, JobStatus
from .protocol import enforce_length
from .solvers import CaptchaSolver, SolverError
from .store import JobStore

log = logging.getLogger("ysolver.worker")


class QueueFull(RuntimeError):
    pass


@dataclass
class WorkItem:
    job_id: str
    image: bytes
    charset: str
    numeric: bool
    min_len: Optional[int] = None
    max_len: Optional[int] = None


#: Inject your own callable in tests / experiments to replace real OCR.
WorkerFn = Callable[[JobStore, WorkItem], None]


def _capture(labels, learn: bool, item: WorkItem, text: Optional[str],
             confidence: Optional[float], backend: Optional[str], reason: str,
             learn_rate: float = 0.0) -> None:
    """Queue a sample for human labelling.

    Failures and low-confidence readings always go in the pile; ordinary
    confident traffic is sampled at ``learn_rate`` so the training set is not
    made up entirely of the weird cases.
    """
    if not learn or labels is None:
        return
    if reason == "sample":
        import random

        if learn_rate <= 0 or random.random() > learn_rate:
            return
    try:
        labels.add_candidate(
            item.image, job_id=item.job_id, model_text=text,
            confidence=confidence, backend=backend, reason=reason,
        )
    except Exception:  # pragma: no cover - labelling must never break solving
        log.exception("could not queue job %s for labelling", item.job_id)


def _notify(callback: Optional[Callable[[str], None]], job_id: str) -> None:
    """Fire a settlement callback; a broken listener must never kill a worker."""
    if callback is None:
        return
    try:
        callback(job_id)
    except Exception:  # pragma: no cover - defensive
        log.exception("settlement callback failed for job %s", job_id)


def process_item(store: JobStore, item: WorkItem, solver: CaptchaSolver,
                 on_settled: Optional[Callable[[str], None]] = None,
                 labels=None, learn: bool = False,
                 learn_confidence: float = 0.85, learn_rate: float = 0.0) -> None:
    """Default worker body: solve one captcha and record the outcome."""
    started = time.perf_counter()
    if not store.mark_processing(item.job_id):
        return  # already finished or expired
    try:
        result = solver.solve(item.image, charset=item.charset, numeric=item.numeric)
    except SolverError as exc:
        store.mark_failed(item.job_id, exc.code, exc.message)
        log.info("job %s failed: %s", item.job_id, exc.code)
        _capture(labels, learn, item, None, None, solver.backend_name, "failed",
                 learn_rate)
        _notify(on_settled, item.job_id)
        return
    except Exception as exc:  # pragma: no cover - unexpected engine crash
        store.mark_failed(item.job_id, "ERROR_INTERNAL", str(exc))
        log.exception("job %s crashed", item.job_id)
        _notify(on_settled, item.job_id)
        return
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    text = enforce_length(result.text, item.min_len, item.max_len)
    store.mark_ready(
        item.job_id, text, result.confidence, result.backend, elapsed_ms
    )
    log.debug("job %s solved as %r in %sms", item.job_id, text, elapsed_ms)
    uncertain = result.confidence is not None and result.confidence < learn_confidence
    _capture(
        labels, learn, item, text, result.confidence, result.backend,
        "uncertain" if uncertain else "sample", learn_rate,
    )
    _notify(on_settled, item.job_id)


class Scheduler(threading.Thread):
    """Holds jobs submitted with a ``delay`` and releases them when due.

    The paid services sell "scheduled captcha" as a premium feature; here it is
    just a heap and a condition variable.
    """

    def __init__(self, pool: "WorkerPool"):
        super().__init__(name="ysolver-scheduler", daemon=True)
        self.pool = pool
        self._heap: list = []
        self._counter = 0
        self._cv = threading.Condition()
        # NB: Thread already owns the name `_stop`, so use a distinct attribute.
        self._stopped = threading.Event()

    def schedule(self, delay_s: float, item: WorkItem) -> None:
        with self._cv:
            self._counter += 1
            heapq.heappush(self._heap, (time.time() + delay_s, self._counter, item))
            self._cv.notify()

    def pending(self) -> int:
        with self._cv:
            return len(self._heap)

    def stop(self) -> None:
        self._stopped.set()
        with self._cv:
            self._cv.notify_all()

    def run(self) -> None:
        while not self._stopped.is_set():
            with self._cv:
                if not self._heap:
                    self._cv.wait(0.5)
                    continue
                due, _, item = self._heap[0]
                wait_for = due - time.time()
                if wait_for > 0:
                    self._cv.wait(min(wait_for, 0.5))
                    continue
                heapq.heappop(self._heap)
            try:
                self.pool.enqueue(item)
            except QueueFull as exc:
                self.pool.store.mark_failed(item.job_id, "ERROR_NO_SLOT_AVAILABLE", str(exc))
            except Exception as exc:  # pragma: no cover
                log.exception("scheduled job %s failed to enqueue", item.job_id)
                self.pool.store.mark_failed(item.job_id, "ERROR_INTERNAL", str(exc))


class WorkerPool:
    """Thread pool plus a settlement callback.

    ``on_settled`` is invoked from the worker thread the moment a job reaches a
    terminal state. The API layer uses it to wake a waiting HTTP request
    immediately instead of polling the store every few milliseconds — which was
    costing more latency than the OCR itself on the one-shot endpoint.
    """

    def __init__(
        self,
        store: JobStore,
        solver: CaptchaSolver,
        workers: int = 4,
        max_queue: int = 500,
        worker_fn: Optional[WorkerFn] = None,
        image_dir: Optional[str] = None,
        on_settled: Optional[Callable[[str], None]] = None,
        labels=None,
        learn: bool = False,
        learn_confidence: float = 0.85,
        learn_rate: float = 0.0,
    ):
        self.store = store
        self.solver = solver
        self.workers = max(1, workers)
        self.max_queue = max(1, max_queue)
        self.worker_fn = worker_fn
        self.image_dir = image_dir
        self.on_settled = on_settled
        self.labels = labels
        self.learn = learn
        self.learn_confidence = learn_confidence
        self.learn_rate = learn_rate
        self._queue: "queue.Queue[Optional[WorkItem]]" = queue.Queue()
        self._threads: list[threading.Thread] = []
        self._stop = threading.Event()
        self._guard = threading.Lock()
        self._in_flight = 0
        self.scheduler = Scheduler(self)

    # ------------------------------------------------------------- lifecycle
    def start(self) -> None:
        if self._threads:
            return
        for index in range(self.workers):
            thread = threading.Thread(
                target=self._loop, name=f"ysolver-worker-{index}", daemon=True
            )
            thread.start()
            self._threads.append(thread)
        self.scheduler.start()
        log.info("started %d worker(s) with backend %s", self.workers, self.solver.backend_name)

    def stop(self, timeout: float = 5.0) -> None:
        self.scheduler.stop()
        self._stop.set()
        for _ in self._threads:
            with contextlib.suppress(queue.Full):  # pragma: no cover
                self._queue.put_nowait(None)
        for thread in self._threads:
            thread.join(timeout=timeout)
        self._threads.clear()

    # ---------------------------------------------------------------- submit
    @property
    def in_flight(self) -> int:
        with self._guard:
            return self._in_flight

    def queue_depth(self) -> int:
        return max(self.in_flight, self.store.queue_depth())

    def submit(self, image: bytes, method: str = "base64", charset: str = "",
               key_hint: str = "", numeric: bool = False,
               min_len: Optional[int] = None, max_len: Optional[int] = None,
               delay: float = 0.0) -> Job:
        digest = hashlib.sha256(image).hexdigest()[:16]
        job = self.store.create(
            method=method, charset=charset, key_hint=key_hint, image_sha=digest
        )
        item = WorkItem(job.id, image, charset, numeric, min_len, max_len)
        if delay and delay > 0:
            self.scheduler.schedule(delay, item)
            return job
        self.enqueue(item)
        return job

    def enqueue(self, item: WorkItem) -> None:
        with self._guard:
            if self._in_flight >= self.max_queue:
                raise QueueFull(f"queue is full ({self.max_queue} awaiting a worker)")
            self._in_flight += 1
        try:
            self._store_image(item)
            self._queue.put_nowait(item)
        except Exception:
            with self._guard:
                self._in_flight -= 1
            raise

    def _store_image(self, item: WorkItem) -> None:
        """Optional debug aid (``YSOLVER_STORE_IMAGES=1``): keep a copy on disk."""
        if not self.image_dir:
            return
        try:
            os.makedirs(self.image_dir, exist_ok=True)
            with open(os.path.join(self.image_dir, f"{item.job_id}.png"), "wb") as handle:
                handle.write(item.image)
        except OSError:  # pragma: no cover - disk full / permissions
            log.warning("could not store image for job %s", item.job_id)

    def scheduled_depth(self) -> int:
        return self.scheduler.pending()

    def drain(self, timeout: float = 10.0) -> bool:
        """Test helper: block until the queue is empty."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.in_flight == 0 and self._queue.empty():
                return True
            time.sleep(0.02)
        return False

    # ----------------------------------------------------------------- loops
    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                item = self._queue.get(timeout=0.25)
            except queue.Empty:
                continue
            if item is None:
                break
            try:
                self._handle(item)
            finally:
                with self._guard:
                    self._in_flight -= 1
                self._queue.task_done()

    def _handle(self, item: WorkItem) -> None:
        if self.worker_fn is not None:
            try:
                self.worker_fn(self.store, item)
            finally:
                _notify(self.on_settled, item.job_id)
            return
        process_item(
            self.store, item, self.solver,
            on_settled=self.on_settled,
            labels=self.labels,
            learn=self.learn,
            learn_confidence=self.learn_confidence,
            learn_rate=self.learn_rate,
        )


class Watchdog(threading.Thread):
    """Fail jobs whose worker died mid-flight and expire old results."""

    def __init__(self, store: JobStore, timeout_s: float, ttl_s: int, interval: float = 30.0):
        super().__init__(name="ysolver-watchdog", daemon=True)
        self.store = store
        self.timeout_s = timeout_s
        self.ttl_s = ttl_s
        self.interval = interval
        # NB: Thread already owns the name `_stop`, so use a distinct attribute.
        self._stopped = threading.Event()

    def stop(self) -> None:
        self._stopped.set()

    def run(self) -> None:
        while not self._stopped.wait(self.interval):
            try:
                for job_id in self.store.expire_stale(self.timeout_s):
                    self.store.mark_failed(
                        job_id, "ERROR_INTERNAL", "solver timed out on this image"
                    )
                removed = self.store.cleanup(self.ttl_s)
                if removed:
                    log.info("expired %d old job(s)", removed)
            except Exception:  # pragma: no cover - watchdog must never die
                log.exception("watchdog tick failed")


def job_is_settled(job: Optional[Job]) -> bool:
    return job is not None and job.status in (JobStatus.READY, JobStatus.FAILED)
