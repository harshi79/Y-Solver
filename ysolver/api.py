"""HTTP surface.

Three faces of the same engine:

* ``/in.php`` + ``/res.php``      — classic form API (drop-in replacement)
* ``/createTask`` + ``/getTaskResult`` — JSON task API (drop-in replacement)
* ``/api/*``                      — small native REST API + dashboard
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from . import __version__
from .config import Settings
from .labels import LabelStore
from .models import ERROR_MESSAGES, JobStatus
from .protocol import (
    ProtocolError,
    classify_method,
    decode_base64_payload,
    describe_unsupported,
    extract_options,
    first,
    mask_key,
    normalize_id,
    parse_task,
    truthy,
)
from .scope import recommendation, scope_payload
from .solvers import CaptchaSolver, SolverError, engine_status, get_engine
from .store import JobStore
from .worker import QueueFull, Watchdog, WorkerPool

log = logging.getLogger("ysolver.api")

KEY_FIELDS = ("key", "clientkey", "client_key", "apikey", "api_key")
STATIC_DIR = Path(__file__).parent / "static"


@dataclass
class RequestData:
    params: Dict[str, Any] = field(default_factory=dict)
    upload: Optional[UploadFile] = None
    raw: Optional[bytes] = None

    def key(self) -> Optional[str]:
        return first(self.params, KEY_FIELDS)


async def read_request(request: Request) -> RequestData:
    """Collect parameters no matter how this client chose to send them."""
    data = RequestData()
    for name, value in request.query_params.items():
        data.params.setdefault(name, value)

    content_type = (request.headers.get("content-type") or "").lower()
    # Read the payload exactly once: Starlette forbids consuming the stream twice.
    body = await request.body()

    if "json" in content_type:
        try:
            import json

            payload = json.loads(body or b"{}")
        except Exception:
            payload = None
        if isinstance(payload, list):
            payload = payload[0] if payload else {}
        if isinstance(payload, dict):
            for name, value in payload.items():
                data.params.setdefault(str(name), value)
    elif "form-data" in content_type or "urlencoded" in content_type:
        try:
            form = await request.form()
        except Exception as exc:
            raise ProtocolError("ERROR_BAD_PARAMETERS", f"could not parse form: {exc}") from exc
        for name, value in form.multi_items():
            # The parser hands back starlette's UploadFile, not fastapi's
            # subclass, so duck-type instead of using isinstance.
            if isinstance(value, UploadFile) or (
                hasattr(value, "read") and hasattr(value, "filename")
            ):
                if data.upload is None:
                    data.upload = value
            else:
                data.params.setdefault(name, str(value))
    elif body:
        data.raw = body

    if data.raw is None and data.upload is None and body and _looks_binary(body):
        # Some clients post a raw image with no usable content type at all.
        data.raw = body
    return data


def _looks_binary(body: bytes) -> bool:
    sample = body[:16]
    signatures = (b"\x89PNG", b"GIF8", b"\xff\xd8", b"BM", b"RIFF", b"II*\x00", b"MM\x00*")
    return any(sample.startswith(sig) for sig in signatures)


async def image_bytes(data: RequestData) -> bytes:
    if data.upload is not None:
        return await data.upload.read()
    if data.raw:
        return data.raw
    payload = first(data.params, ("body", "image", "captcha", "file", "base64"))
    if payload:
        return decode_base64_payload(payload)
    raise ProtocolError("ERROR_BAD_PARAMETERS", "no image supplied (send 'file' or 'body')")


class JobWaiters:
    """Wakes waiting HTTP requests the instant a job settles.

    A worker thread cannot touch an asyncio.Event directly, so it hands the
    notification back to the owning event loop with ``call_soon_threadsafe``.
    Requests still re-read the store afterwards, which also closes the tiny
    race where a job settles between submission and registration.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._waiters: Dict[str, "asyncio.Event"] = {}
        self._loops: Dict[str, "asyncio.AbstractEventLoop"] = {}

    def register(self, job_id: str, loop, event) -> None:
        with self._lock:
            self._waiters[job_id] = event
            self._loops[job_id] = loop

    def release(self, job_id: str) -> None:
        with self._lock:
            self._waiters.pop(job_id, None)
            self._loops.pop(job_id, None)

    def notify(self, job_id: str) -> None:
        """Called from a worker thread when a job reaches a terminal state."""
        with self._lock:
            event = self._waiters.get(job_id)
            loop = self._loops.get(job_id)
        if event is None or loop is None:
            return
        with contextlib.suppress(RuntimeError):  # pragma: no cover - loop closed
            loop.call_soon_threadsafe(event.set)


class _Text:
    """Plain-text responses in the exact shape the form API expects."""

    @staticmethod
    def ok(text: str) -> PlainTextResponse:
        return PlainTextResponse(f"OK|{text}", media_type="text/plain; charset=utf-8")

    @staticmethod
    def code(code: str) -> PlainTextResponse:
        return PlainTextResponse(code, media_type="text/plain; charset=utf-8")


def create_app(settings: Optional[Settings] = None, worker_fn=None) -> FastAPI:
    settings = settings or Settings.from_env()
    store = JobStore(settings.db_path)
    labels = LabelStore(settings.db_path)
    solver = CaptchaSolver(settings)
    image_dir = None
    if settings.store_images:
        image_dir = os.path.join(os.path.dirname(os.path.abspath(settings.db_path)), "images")
    waiters = JobWaiters()
    pool = WorkerPool(
        store,
        solver,
        workers=settings.workers,
        max_queue=settings.max_queue,
        worker_fn=worker_fn,
        image_dir=image_dir,
        on_settled=waiters.notify,
        labels=labels,
        learn=settings.learn,
        learn_confidence=settings.learn_confidence,
        learn_rate=settings.learn_rate,
    )
    watchdog = Watchdog(store, settings.solve_timeout * 2, settings.result_ttl)
    started_at = time.time()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        pool.start()
        watchdog.start()
        log.info(
            "Y-Solver ready — backend=%s db=%s require_key=%s",
            solver.backend_name,
            settings.db_path,
            settings.require_key,
        )
        try:
            yield
        finally:
            watchdog.stop()
            pool.stop()
            store.close()
            labels.close()

    app = FastAPI(
        title="Y-Solver",
        version=__version__,
        summary="Free, self-hosted CAPTCHA solving API",
        description=(
            "Drop-in replacement for the paid solving services' HTTP API. "
            "Runs entirely on your own hardware."
        ),
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.store = store
    app.state.labels = labels
    app.state.solver = solver
    app.state.pool = pool
    app.state.waiters = waiters

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ------------------------------------------------------------------ utils
    def text_error(exc: ProtocolError) -> PlainTextResponse:
        return _Text.code(exc.code)

    def json_error(exc: ProtocolError) -> JSONResponse:
        return JSONResponse(
            {
                "errorId": 1,
                "errorCode": exc.code,
                "errorDescription": exc.message,
                "status": "failed",
            }
        )

    def submit_from(
        image: bytes, params: Dict[str, Any], method: str, key: Optional[str]
    ) -> str:
        options = extract_options(params)
        delay = _parse_delay(first(params, ("delay",)))
        try:
            job = pool.submit(
                image=image,
                method=method or "base64",
                charset=options["charset"],
                key_hint=mask_key(key),
                numeric=options["numeric"],
                min_len=options["min_len"],
                max_len=options["max_len"],
                delay=delay,
            )
        except QueueFull as exc:
            raise ProtocolError("ERROR_NO_SLOT_AVAILABLE", str(exc)) from exc
        except ProtocolError:
            raise
        except Exception as exc:  # pragma: no cover
            log.exception("submit failed")
            raise ProtocolError("ERROR_INTERNAL", str(exc)) from exc
        return job.id

    # ------------------------------------------------------- classic form API
    @app.get("/in.php", include_in_schema=False)
    @app.post("/in.php", include_in_schema=False)
    async def in_php(request: Request) -> PlainTextResponse:
        try:
            data = await read_request(request)
        except ProtocolError as exc:
            return text_error(exc)

        key = data.key()
        if not settings.check_key(key):
            return _Text.code("ERROR_WRONG_USER_KEY")
        if settings.balance <= 0:
            return _Text.code("ERROR_ZERO_BALANCE")

        method = first(data.params, ("method",))
        kind = classify_method(method, None)
        if kind in ("interactive", "unknown") and not _has_payload(data):
            log.info("rejecting unsupported method=%s", method)
            return _Text.code("ERROR_METHOD_NOT_SUPPORTED")

        try:
            image = await image_bytes(data)
            job_id = submit_from(image, data.params, method or "base64", key)
        except ProtocolError as exc:
            return text_error(exc)
        return _Text.ok(job_id)

    @app.get("/res.php", include_in_schema=False)
    @app.post("/res.php", include_in_schema=False)
    async def res_php(request: Request) -> PlainTextResponse:
        try:
            data = await read_request(request)
        except ProtocolError as exc:
            return text_error(exc)

        key = data.key()
        if not settings.check_key(key):
            return _Text.code("ERROR_WRONG_USER_KEY")

        action = (first(data.params, ("action", "method")) or "get").strip().lower()
        if action in ("getbalance", "balance"):
            return _Text.ok(f"${settings.balance:.2f}")
        if action in ("reportbad", "report_bad", "report"):
            return _Text.ok("OK")
        if action not in ("get", ""):
            return _Text.code("ERROR_BAD_ACTION")

        try:
            job_id = normalize_id(first(data.params, ("id", "taskid", "captcha_id")))
        except ProtocolError as exc:
            return text_error(exc)

        job = store.get(job_id)
        if job is None:
            return _Text.code("ERROR_WRONG_CAPTCHA_ID")
        if job.status in (JobStatus.PENDING, JobStatus.PROCESSING):
            return _Text.code("CAPCHA_NOT_READY")  # historical spelling, keep it
        if job.status is JobStatus.READY and job.text:
            return _Text.ok(job.text)
        return _Text.code(job.error_code or "ERROR_CAPTCHA_UNSOLVABLE")

    # ---------------------------------------------------------- JSON task API
    @app.post("/createTask", include_in_schema=False)
    @app.get("/createTask", include_in_schema=False)
    async def create_task(request: Request) -> JSONResponse:
        data = await read_request(request)
        params = data.params
        key = data.key()
        if not settings.check_key(key):
            return json_error(ProtocolError("ERROR_WRONG_USER_KEY"))
        if settings.balance <= 0:
            return json_error(ProtocolError("ERROR_ZERO_BALANCE"))

        task = parse_task(params.get("task"))
        merged: Dict[str, Any] = dict(params)
        merged.update({k: v for k, v in task.items() if k not in merged})
        method = first(task, ("type",)) or first(params, ("method",)) or ""

        payload_b64 = first(task, ("image_base64", "body", "image", "captcha"))
        if payload_b64 and data.raw is None and data.upload is None:
            data.params.setdefault("body", payload_b64)

        kind = classify_method(method, "base64" if payload_b64 else None)
        if kind == "interactive":
            return JSONResponse(
                {
                    "errorId": 1,
                    "errorCode": "ERROR_METHOD_NOT_SUPPORTED",
                    "errorDescription": describe_unsupported(method),
                    "suggestions": recommendation(method),
                }
            )

        try:
            image = await image_bytes(data)
            job_id = submit_from(image, merged, method or "base64", key)
        except ProtocolError as exc:
            return json_error(exc)
        except SolverError as exc:  # pragma: no cover - defensive
            return json_error(ProtocolError(exc.code, exc.message))
        return JSONResponse({"errorId": 0, "taskId": job_id})

    @app.post("/getTaskResult", include_in_schema=False)
    @app.get("/getTaskResult", include_in_schema=False)
    async def get_task_result(request: Request) -> JSONResponse:
        data = await read_request(request)
        key = data.key()
        if not settings.check_key(key):
            return json_error(ProtocolError("ERROR_WRONG_USER_KEY"))
        try:
            job_id = normalize_id(first(data.params, ("taskid", "id", "captcha_id")))
        except ProtocolError as exc:
            return json_error(exc)

        job = store.get(job_id)
        if job is None:
            return json_error(ProtocolError("ERROR_WRONG_CAPTCHA_ID"))
        if job.status in (JobStatus.PENDING, JobStatus.PROCESSING):
            return JSONResponse({"errorId": 0, "status": "processing"})
        if job.status is JobStatus.READY and job.text:
            return JSONResponse(
                {
                    "errorId": 0,
                    "status": "ready",
                    "solution": {
                        "text": job.text,
                        "confidence": job.confidence,
                        "backend": job.backend,
                        "solveMs": job.solve_ms,
                    },
                }
            )
        return json_error(
            ProtocolError(job.error_code or "ERROR_CAPTCHA_UNSOLVABLE", job.error_text or "")
        )

    # ------------------------------------------------------------ native API
    @app.post("/api/solve", tags=["native"])
    @app.get("/api/solve", tags=["native"])
    async def api_solve(request: Request) -> JSONResponse:
        """One-shot convenience endpoint: upload, wait, get the text back."""
        data = await read_request(request)
        key = data.key()
        if not settings.check_key(key):
            return JSONResponse(
                {"status": "error", "error": "ERROR_WRONG_USER_KEY"}, status_code=401
            )
        try:
            image = await image_bytes(data)
            job_id = submit_from(image, data.params, "base64", key)
        except ProtocolError as exc:
            return JSONResponse(
                {"status": "error", "error": exc.code, "message": exc.message},
                status_code=400 if exc.code.startswith("ERROR_BAD") else 200,
            )

        wait = (first(data.params, ("wait",)) or "true").lower() not in ("0", "false", "no")
        timeout = float(first(data.params, ("timeout",)) or settings.solve_timeout)
        if not wait:
            return JSONResponse({"status": "pending", "id": job_id}, status_code=202)

        # Wait on an event the worker sets the moment the job settles, rather
        # than polling: polling added up to 50 ms of pure latency per request.
        deadline = max(0.1, min(timeout, 120.0))
        event = asyncio.Event()
        waiters.register(job_id, asyncio.get_running_loop(), event)
        try:
            job = store.get(job_id)
            if job is not None and job.status in (JobStatus.READY, JobStatus.FAILED):
                pass  # settled before we started waiting
            else:
                try:
                    await asyncio.wait_for(event.wait(), timeout=deadline)
                except asyncio.TimeoutError:
                    return JSONResponse({"status": "pending", "id": job_id}, status_code=202)
                job = store.get(job_id)
        finally:
            waiters.release(job_id)

        if job is None:
            return JSONResponse({"status": "error", "error": "ERROR_INTERNAL"})
        if job.status is JobStatus.READY:
            return JSONResponse({"status": "ready", **job.to_public()})
        if job.status is JobStatus.FAILED:
            return JSONResponse({"status": "failed", **job.to_public()})
        return JSONResponse({"status": "pending", "id": job_id}, status_code=202)

    @app.get("/api/jobs/{job_id}", tags=["native"])
    async def api_job(job_id: str) -> JSONResponse:
        job = store.get(job_id)
        if job is None:
            return JSONResponse({"error": "ERROR_WRONG_CAPTCHA_ID"}, status_code=404)
        return JSONResponse(job.to_public())

    @app.get("/api/jobs", tags=["native"])
    async def api_jobs(limit: int = 25) -> JSONResponse:
        return JSONResponse({"jobs": [j.to_public() for j in store.recent(limit)]})

    @app.get("/api/stats", tags=["native"])
    async def api_stats() -> JSONResponse:
        return JSONResponse(
            {
                "version": __version__,
                "uptimeS": round(time.time() - started_at, 1),
                "backend": solver.backend_name,
                "engines": engine_status(),
                "queue": pool.queue_depth(),
                "workers": settings.workers,
                "requireKey": settings.require_key,
                "balance": settings.balance,
                "retentionS": settings.result_ttl,
                **store.stats(),
            }
        )

    @app.get("/api/sample", tags=["native"])
    async def api_sample(length: int = 5, seed: int | None = None, charset: str = "") -> JSONResponse:
        """A freshly generated test captcha (handy for smoke-testing clients)."""
        import base64

        from .synth import make_captcha, to_png

        if seed is None:
            import random

            seed = random.randint(0, 10**6)
        image, label = make_captcha(
            length=max(3, min(10, length)),
            charset=charset or settings.charset,
            seed=seed,
        )
        return JSONResponse(
            {
                "seed": seed,
                "label": label,  # ground truth: this is a generator, not a server captcha
                "width": image.width,
                "height": image.height,
                "pngBase64": base64.b64encode(to_png(image)).decode("ascii"),
            }
        )

    @app.get("/api/scope", tags=["native"])
    async def api_scope() -> JSONResponse:
        """What this server solves, and what it deliberately refuses."""
        return JSONResponse(scope_payload())

    @app.get("/healthz", tags=["native"])
    async def healthz() -> JSONResponse:
        engines = engine_status()
        return JSONResponse(
            {
                "status": "ok" if any(e["available"] for e in engines) else "degraded",
                "version": __version__,
                "backend": solver.backend_name,
                "queue": pool.queue_depth(),
                "engines": {e["name"]: e["available"] for e in engines},
            }
        )

    # --------------------------------------------- human-in-the-loop labels
    @app.get("/api/labels/pending", tags=["training"])
    async def api_labels_pending(limit: int = 1) -> JSONResponse:
        """Hand out the next sample(s) for a human to type the answer for."""
        limit = max(1, min(50, limit))
        candidates = labels.pending(limit=limit, include_image=True)
        return JSONResponse(
            {
                "pending": labels.pending_count(),
                "items": [candidate.to_public(include_image=True) for candidate in candidates],
            }
        )

    @app.post("/api/labels", tags=["training"])
    async def api_labels_submit(request: Request) -> JSONResponse:
        """File a human label (a correction, or the truth for a failed sample)."""
        data = await read_request(request)
        key = data.key()
        if not settings.check_key(key):
            return JSONResponse(
                {"status": "error", "error": "ERROR_WRONG_USER_KEY"}, status_code=401
            )
        raw_id = first(data.params, ("id", "candidate_id", "candidateid"))
        text = first(data.params, ("text", "label", "solution", "answer"))
        tag = first(data.params, ("tag", "site")) or ""
        if not raw_id:
            return JSONResponse(
                {"status": "error", "error": "ERROR_BAD_PARAMETERS",
                 "message": "candidate 'id' is required"}, status_code=400,
            )
        # "skip" is an explicit flag, never a magic label value: a captcha really
        # can read "skip", "unreadable" or "error", and silently dropping those
        # labels would quietly corrupt the dataset.
        wants_skip = truthy(first(data.params, ("skip", "unreadable", "discard")))
        if wants_skip or text is None:
            if not wants_skip and text is None:
                return JSONResponse(
                    {"status": "error", "error": "ERROR_BAD_PARAMETERS",
                     "message": "send 'text' with the label, or 'skip=1' to mark it unreadable"},
                    status_code=400,
                )
            try:
                ok = labels.skip(int(raw_id))
            except ValueError:
                return JSONResponse(
                    {"status": "error", "error": "ERROR_BAD_PARAMETERS",
                     "message": "id must be a numeric candidate id"}, status_code=400,
                )
            return JSONResponse({"status": "skipped" if ok else "not_found", "id": raw_id})
        try:
            result = labels.submit_label(int(raw_id), text, tag)
        except ValueError as exc:
            return JSONResponse(
                {"status": "error", "error": "ERROR_BAD_PARAMETERS", "message": str(exc)},
                status_code=400,
            )
        except KeyError as exc:
            return JSONResponse(
                {"status": "error", "error": "ERROR_BAD_PARAMETERS", "message": str(exc)},
                status_code=404,
            )
        return JSONResponse({"status": "labeled", **result, "stats": labels.stats()})

    @app.get("/api/labels/stats", tags=["training"])
    async def api_labels_stats() -> JSONResponse:
        learned = get_engine("learned")
        return JSONResponse(
            {
                **labels.stats(),
                "learningEnabled": settings.learn,
                "captureConfidenceBelow": settings.learn_confidence,
                "sampleRate": settings.learn_rate,
                "model": learned.info() if learned.available() else None,
            }
        )

    # ------------------------------------------------------------- dashboard
    if settings.dashboard and STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

        @app.get("/", include_in_schema=False)
        async def dashboard() -> FileResponse:
            return FileResponse(STATIC_DIR / "index.html")

    # ------------------------------------------------------- error fallbacks
    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception):  # pragma: no cover
        log.exception("unhandled error on %s", request.url.path)
        if request.url.path.startswith(("/in.php", "/res.php")):
            return _Text.code("ERROR_INTERNAL")
        return JSONResponse(
            {"errorId": 1, "errorCode": "ERROR_INTERNAL", "errorDescription": str(exc)},
            status_code=500,
        )

    return app


def _parse_delay(raw: Optional[str]) -> float:
    """The classic API accepts ``delay`` in seconds with a 100s minimum."""
    if not raw:
        return 0.0
    try:
        delay = float(str(raw).strip())
    except ValueError:
        return 0.0
    return delay if delay >= 100 else 0.0


def _has_payload(data: RequestData) -> bool:
    if data.upload is not None or data.raw:
        return True
    return bool(first(data.params, ("body", "image", "captcha", "file", "base64")))


__all__ = ["ERROR_MESSAGES", "create_app"]
