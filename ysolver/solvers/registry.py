"""Engine selection: explicit via config, otherwise best-available."""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional

from ..config import DEFAULT_CHARSET, Settings
from ..preprocess import ImageDecodeError, prepare
from .base import Engine, SolverError, SolverResult
from .ddddocr_engine import DdddocrEngine
from .learned_engine import LearnedEngine
from .template_engine import TemplateEngine
from .tesseract_engine import TesseractEngine

log = logging.getLogger("ysolver.solver")

ENGINE_CLASSES = {
    "template": TemplateEngine,
    "tesseract": TesseractEngine,
    "ddddocr": DdddocrEngine,
    "learned": LearnedEngine,
}

#: Best first, used when no measured metrics exist.
#: ddddocr wins on general accuracy, tesseract is decent, template always works.
AUTO_ORDER = ("ddddocr", "tesseract", "template")

#: Every engine we can report on, in dashboard order.
ALL_ENGINES = ("ddddocr", "learned", "tesseract", "template")

_instances: Dict[str, Engine] = {}


def get_engine(name: str) -> Engine:
    if name not in _instances:
        _instances[name] = ENGINE_CLASSES[name]()
    return _instances[name]


def engine_status() -> List[dict]:
    """Availability of every engine, for the dashboard and ``/healthz``."""
    out = []
    for name in ALL_ENGINES:
        try:
            out.append(get_engine(name).info())
        except Exception as exc:  # pragma: no cover - defensive
            out.append({"name": name, "available": False, "description": str(exc),
                        "installHint": ""})
    return out


def resolve_backend(preference: str) -> Engine:
    """Pick an engine: honour an explicit choice, else first available."""
    preference = (preference or "auto").strip().lower()
    if preference != "auto":
        if preference not in ENGINE_CLASSES:
            raise SystemExit(
                f"Unknown YSOLVER_BACKEND={preference!r}. "
                f"Choose one of: auto, {', '.join(AUTO_ORDER)}"
            )
        engine = get_engine(preference)
        if not engine.available():
            raise SystemExit(
                f"Backend {preference!r} is not available here. {engine.install_hint or ''}".strip()
            )
        return engine

    # A trained model records which engine actually won on held-out data, so
    # "auto" follows measurement rather than a hard-coded guess.
    measured = _measured_preference()
    if measured:
        return get_engine(measured)

    for name in AUTO_ORDER:
        engine = get_engine(name)
        if engine.available():
            return engine
    raise SystemExit("No solver backend available (this should not happen: 'template' always works)")


def _measured_preference() -> Optional[str]:
    """The engine that beat the others in the last training run, if any."""
    try:
        learned = get_engine("learned")
        if not learned.available():
            return None
        preferred = learned.preferred_engine()
    except Exception:  # pragma: no cover - a broken model must not break startup
        log.exception("could not read the learned model's metrics")
        return None
    if not preferred or preferred not in ENGINE_CLASSES:
        return None
    try:
        if get_engine(preferred).available():
            log.info("auto-selected %r from measured training metrics", preferred)
            return preferred
    except Exception:  # pragma: no cover
        return None
    return None


def reconcile_case(text: str, charset: str) -> str:
    """Fold the reading into the caller's alphabet case.

    Models trained on mixed sources frequently return ``M7JMW`` for an image
    that says ``m7jmw``. When the requested charset is single-case (and
    therefore the caller *knows* the case), correcting it converts near-misses
    into exact matches. Mixed-case and digit-only charsets are left alone.
    """
    letters = [character for character in charset if character.isalpha()]
    if not letters:
        return text
    if all(character.islower() for character in letters):
        return text.lower()
    if all(character.isupper() for character in letters):
        return text.upper()
    return text


class CaptchaSolver:
    """Turns raw image bytes into text plus bookkeeping."""

    def __init__(self, settings: Optional[Settings] = None, engine: Optional[Engine] = None):
        self.settings = settings or Settings.from_env()
        self.engine = engine or resolve_backend(self.settings.backend)

    @property
    def backend_name(self) -> str:
        return self.engine.name

    def solve(self, image: bytes, charset: Optional[str] = None,
              numeric: bool = False) -> SolverResult:
        if not image:
            raise SolverError("ERROR_BAD_PARAMETERS", "no image supplied")
        if len(image) > self.settings.max_image_bytes:
            raise SolverError("ERROR_BAD_PARAMETERS", "image larger than the configured limit")

        if numeric:
            effective_charset = "0123456789"
        else:
            effective_charset = (charset or "").strip() or self.settings.charset
        if len(effective_charset) < 2:
            effective_charset = self.settings.charset or DEFAULT_CHARSET

        try:
            prepared = prepare(image)
        except ImageDecodeError as exc:
            raise SolverError("ERROR_IMAGE_TYPE_NOT_SUPPORTED", str(exc)) from exc

        started = time.perf_counter()
        result = self.engine.solve(prepared, effective_charset)
        elapsed_ms = int((time.perf_counter() - started) * 1000)

        text = (result.text or "").strip()
        if not text:
            raise SolverError("ERROR_CAPTCHA_UNSOLVABLE")
        text = reconcile_case(text, effective_charset)
        return SolverResult(
            text=text,
            confidence=result.confidence,
            backend=self.engine.name,
            engine_ms=elapsed_ms,
        )
