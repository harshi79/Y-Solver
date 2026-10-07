"""ddddocr engine — the recommended backend for real-world captchas.

`ddddocr <https://github.com/sml2h3/ddddocr>`_ ships a small ONNX model that
runs offline on CPU. Install with ``pip install "ysolver[ddddocr]"``.

Note the deliberate design choice here: the model reads *text-in-image*
captchas only. Y-Solver does not implement token harvesting for interactive
challenges (reCAPTCHA / hCaptcha / Turnstile) — those cannot be read out of an
image at all, they need a human or a real browser session. See docs/scope.md.
"""

from __future__ import annotations

import re
import threading
from typing import Dict, List, Sequence, Tuple

import numpy as np

from ..preprocess import Prepared
from .base import Engine, EngineResult, SolverError


def charset_variants(charset: str) -> List[str]:
    """Every character the caller would accept, including either letter case.

    Case is folded later (see ``solver.registry.reconcile_case``), so both cases
    stay legal here — restricting further would throw away model information.
    """
    out: List[str] = []
    seen = set()
    for character in charset:
        for variant in (character, character.lower(), character.upper()):
            if len(variant) == 1 and variant not in seen:
                seen.add(variant)
                out.append(variant)
    return out


def constrained_ctc(
    probabilities: Sequence, model_charset: str, allowed: Sequence[str]
) -> Tuple[str, float]:
    """Greedy CTC decode restricted to ``allowed`` characters.

    Identical to ddddocr's own decoder (blank = index 0, collapse repeats) except
    that argmax runs over the caller's alphabet as well as blank. That keeps the
    model honest when its top pick is not a character the client can accept —
    the common case being a CJK lookalike such as ``十`` where the captcha drew
    an ``x``.
    """
    if not probabilities or not model_charset:
        return "", 0.0
    try:
        rows = np.asarray(probabilities, dtype=np.float32)
    except (TypeError, ValueError):
        return "", 0.0  # a legacy {char: probability} layout — not decodable here
    if rows.size == 0:
        return "", 0.0
    if rows.ndim == 1:
        rows = rows.reshape(1, -1)
    elif rows.ndim == 3:  # (timesteps, 1, classes)
        rows = rows.reshape(rows.shape[0], -1)

    classes = len(model_charset)
    index_of = {character: index for index, character in enumerate(model_charset)}
    candidates = {0}  # blank must remain choosable, it separates repeated chars
    for character in allowed:
        index = index_of.get(character)
        if index is not None:
            candidates.add(index)
    if len(candidates) < 2:
        return "", 0.0
    indices = np.array(sorted(candidates), dtype=np.int64)

    characters: List[str] = []
    scores: List[float] = []
    previous = -1
    for row in rows:
        flattened = row.reshape(-1)
        if flattened.shape[0] != classes:
            continue
        best = int(indices[int(np.argmax(flattened[indices]))])
        if best != previous and best != 0:
            characters.append(model_charset[best])
            scores.append(float(flattened[best]))
        previous = best
    return "".join(characters), (float(np.mean(scores)) if scores else 0.0)


class DdddocrEngine(Engine):
    name = "ddddocr"
    description = "Offline ONNX model, best accuracy on classic distorted captchas."
    install_hint = 'pip install "ysolver[ddddocr]"'

    def __init__(self) -> None:
        self._ocr = None
        self._lock = threading.RLock()
        self._import_failed = False
        self._variant_cache: Dict[str, List[str]] = {}

    def _instance(self):
        if self._ocr is not None:
            return self._ocr
        with self._lock:
            if self._ocr is None:
                try:
                    import ddddocr  # type: ignore
                except Exception as exc:
                    self._import_failed = True
                    raise SolverError("ERROR_INTERNAL", f"ddddocr unavailable: {exc}") from exc
                try:
                    self._ocr = ddddocr.DdddOcr(show_ad=False)
                except TypeError:  # older releases
                    self._ocr = ddddocr.DdddOcr()
        return self._ocr

    def available(self) -> bool:
        if self._import_failed:
            return False
        try:
            self._instance()
            return True
        except Exception:
            return False

    # NOTE: ddddocr's own ``set_ranges`` is deliberately not used here. It
    # mutates the shared model instance, and its reset path (``set_ranges("")``)
    # leaves *only the CTC blank token* selectable, which silently makes every
    # later classification return an empty string. It also reads a string like
    # "0-9" as the three literal characters '0', '-' and '9' rather than a range.
    # Alphabet restriction is handled by constrained_ctc() below instead, which
    # keeps the shared instance untouched and cannot poison later requests.

    def _allowed(self, charset: str) -> List[str]:
        cached = self._variant_cache.get(charset)
        if cached is None:
            cached = charset_variants(charset)
            self._variant_cache[charset] = cached
        return cached

    def _classification(self, ocr, data: bytes, probability: bool = False):
        try:
            return ocr.classification(data, probability=probability)
        except TypeError:  # pragma: no cover - older ddddocr without the kwarg
            if probability:
                raise
            return ocr.classification(data)

    def solve(self, prepared: Prepared, charset: str) -> EngineResult:
        # ddddocr was trained on raw captcha bitmaps: hand it the least-touched
        # version we have (the client's own bytes when still available), never
        # our binarised mask, and never a needless re-encode.
        data = prepared.bytes_for_engine()
        with self._lock:
            ocr = self._instance()
            try:
                text = self._classification(ocr, data)
            except Exception as exc:
                raise SolverError("ERROR_INTERNAL", f"ddddocr failed: {exc}") from exc

        text = re.sub(r"\s+", "", (text or "").strip())
        if not text:
            raise SolverError("ERROR_CAPTCHA_UNSOLVABLE", "ddddocr returned no text")

        allowed = self._allowed(charset)
        if allowed and not self._fits(text, allowed):
            # The model's top pick is not a character this caller can accept —
            # re-decode with the alphabet constraint applied and keep the honest
            # answer instead of returning a stray glyph.
            text, confidence = self._constrained(ocr, data, charset, text)
            return EngineResult(text=text, confidence=confidence)
        return EngineResult(text=text, confidence=None)

    @staticmethod
    def _fits(text: str, allowed: Sequence[str]) -> bool:
        legal = set(allowed)
        return all(character in legal for character in text)

    def _constrained(self, ocr, data: bytes, charset: str, fallback: str):
        try:
            with self._lock:
                payload = self._classification(ocr, data, probability=True)
        except Exception:  # pragma: no cover - probability path unavailable
            return self._drop_strays(fallback, charset), None
        if not isinstance(payload, dict):
            return self._drop_strays(fallback, charset), None
        text, confidence = constrained_ctc(
            payload.get("probabilities") or [],
            payload.get("charset") or "",
            self._allowed(charset),
        )
        if not text:
            return self._drop_strays(fallback, charset), None
        return text, round(confidence, 4) if confidence else None

    @staticmethod
    def _drop_strays(text: str, charset: str) -> str:
        legal = set(charset_variants(charset))
        cleaned = "".join(character for character in text if character in legal)
        return cleaned or text
