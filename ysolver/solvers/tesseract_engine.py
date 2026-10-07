"""Tesseract OCR engine (optional).

Needs the ``tesseract-ocr`` binary on PATH plus ``pip install pytesseract``.
"""

from __future__ import annotations

import shutil
import threading
from typing import Optional

from ..preprocess import Prepared, binarize, to_png
from .base import Engine, EngineResult, SolverError


class TesseractEngine(Engine):
    name = "tesseract"
    description = "Local Tesseract OCR binary with a character whitelist."
    install_hint = "apt-get install -y tesseract-ocr && pip install pytesseract"

    def __init__(self) -> None:
        self._pytesseract = None
        self._checked = False
        self._lock = threading.Lock()

    def _module(self):
        if self._checked:
            return self._pytesseract
        with self._lock:
            self._checked = True
            try:
                import pytesseract  # type: ignore

                self._pytesseract = pytesseract
            except Exception:
                self._pytesseract = None
        return self._pytesseract

    def available(self) -> bool:
        return bool(shutil.which("tesseract")) and self._module() is not None

    def solve(self, prepared: Prepared, charset: str) -> EngineResult:
        pytesseract = self._module()
        if pytesseract is None or not shutil.which("tesseract"):
            raise SolverError("ERROR_INTERNAL", "tesseract is not installed")

        if prepared.mask.any():
            from PIL import Image

            ink = Image.fromarray((prepared.mask.astype("uint8") * 255), mode="L")
            image = ink.convert("RGB")
        else:
            image = prepared.cleaned

        base = f"--oem 3 -c tessedit_char_whitelist={charset}"
        last_error: Optional[Exception] = None
        for psm in ("8", "7", "6"):
            try:
                with self._lock:
                    raw = pytesseract.image_to_string(image, config=f"{base} --psm {psm}")
            except Exception as exc:  # pragma: no cover - depends on local binary
                last_error = exc
                break
            text = "".join(ch for ch in raw if ch in charset)
            if text:
                return EngineResult(text=text, confidence=None)

        if last_error is not None:
            raise SolverError("ERROR_INTERNAL", f"tesseract failed: {last_error}")
        raise SolverError("ERROR_CAPTCHA_UNSOLVABLE", "tesseract returned no text")


def render_for_debug(prepared: Prepared) -> bytes:  # pragma: no cover - debugging helper
    """Handy when tuning the pipeline by eye."""
    from PIL import Image

    ink = Image.fromarray((binarize(prepared.cleaned).astype("uint8") * 255), mode="L")
    return to_png(ink.convert("RGB"))
