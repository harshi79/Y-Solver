"""The engine that learns from human labels.

Every label a person types is cut into glyphs and added to a **glyph bank**: a
set of normalised 24x24 bitmaps, each tagged with the character it shows. At
solve time the same segmentation runs on the incoming captcha and each glyph is
matched against the bank by cosine similarity.

Why this and not a fine-tuned neural network: it trains in under a second on a
CPU, needs no GPU, no torch and no experiment tracking, and it improves exactly
where human labelling helps most — the *font and rendering* of one specific
captcha. Fifty labels of your target site beat a five-million-image general
model at that specific task, because the general model has never seen that
particular renderer.

The model is a single ``.npz`` file next to the database, so deploying it is
copying a file, and retraining is a script run rather than a pipeline.
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..preprocess import Prepared, decode_image
from .base import Engine, EngineResult, SolverError
from .segmentation import segment
from .template_engine import TEMPLATE_SIZE, filter_debris, glyph_mask, vectorise

DEFAULT_MODEL_PATH = "data/learned.npz"
MIN_GLYPHS = 40          # below this the bank is too thin to be worth trusting
MIN_LABELS = 8


@dataclass
class GlyphBank:
    vectors: np.ndarray                     # (n, TEMPLATE_SIZE**2) float32, unit norm
    chars: List[str]
    meta: Dict[str, object] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.chars)

    def filtered(self, allowed: str) -> "GlyphBank":
        """Restrict the bank to characters the caller's alphabet permits."""
        legal = set(allowed) if allowed else set(self.chars)
        keep = np.array([i for i, c in enumerate(self.chars) if c in legal], dtype=np.int64)
        if keep.size == 0:
            return GlyphBank(self.vectors[:0], [], dict(self.meta))
        return GlyphBank(self.vectors[keep], [self.chars[i] for i in keep], dict(self.meta))


def extract_glyphs(image_bytes: bytes, text: str) -> List[Tuple[np.ndarray, str]]:
    """Cut one labelled sample into ``(vector, character)`` pairs.

    Only samples where we find exactly as many glyphs as the human typed
    characters are usable: a mismatch means the segmentation failed on this
    image, and pairing the wrong glyph with the wrong character would teach the
    model nonsense. Those samples are reported by the trainer instead.
    """
    try:
        image = decode_image(image_bytes)
    except Exception:
        return []
    mask = glyph_mask(image)
    glyphs = filter_debris(segment(mask))
    if len(glyphs) != len(text):
        return []
    pairs: List[Tuple[np.ndarray, str]] = []
    for glyph, character in zip(glyphs, text):
        vector = vectorise(glyph.mask)
        if np.any(vector):
            pairs.append((vector, character))
    return pairs


def build_bank(dataset: Sequence[Tuple[bytes, str, str]]) -> Tuple[GlyphBank, Dict[str, int]]:
    """Turn labelled samples into a glyph bank plus build statistics."""
    vectors: List[np.ndarray] = []
    chars: List[str] = []
    stats = {"samples": len(dataset), "usable": 0, "skipped_mismatch": 0, "glyphs": 0}
    for image, text, _tag in dataset:
        pairs = extract_glyphs(image, text)
        if not pairs:
            stats["skipped_mismatch"] = stats.get("skipped_mismatch", 0) + 1
            continue
        stats["usable"] += 1
        for vector, character in pairs:
            vectors.append(vector)
            chars.append(character)
    stats["glyphs"] = len(chars)
    matrix = (
        np.vstack(vectors).astype(np.float32)
        if vectors
        else np.zeros((0, TEMPLATE_SIZE * TEMPLATE_SIZE), dtype=np.float32)
    )
    return GlyphBank(matrix, chars), stats


def save_bank(path: str, bank: GlyphBank) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    meta = dict(bank.meta)
    meta.setdefault("savedAt", time.time())
    np.savez_compressed(
        path,
        vectors=bank.vectors.astype(np.float32),
        chars=np.array(bank.chars, dtype="U1"),
        meta=np.array([json.dumps(meta)], dtype="U"),
    )


def load_bank(path: str) -> Optional[GlyphBank]:
    if not os.path.exists(path):
        return None
    try:
        with np.load(path, allow_pickle=False) as payload:
            vectors = np.asarray(payload["vectors"], dtype=np.float32)
            chars = [str(c) for c in payload["chars"].tolist()]
            raw_meta = payload["meta"]
            meta = json.loads(str(raw_meta[0])) if raw_meta.size else {}
    except Exception:
        return None
    if vectors.ndim != 2 or vectors.shape[0] != len(chars):
        return None
    return GlyphBank(vectors, chars, meta)


class LearnedEngine(Engine):
    """Nearest-neighbour glyph matching over human-labelled samples."""

    name = "learned"
    description = "Trained on your own human-labelled captchas (scripts/train.py)."
    install_hint = "label captchas in the dashboard, then run: python scripts/train.py"

    def __init__(self, model_path: Optional[str] = None):
        # An explicit path wins; otherwise re-read the environment on every
        # access, because the registry caches engine instances and a stale path
        # silently grades one style's bank against another style's captchas.
        self._explicit_path = model_path
        self.model_path = model_path or os.environ.get("YSOLVER_MODEL", DEFAULT_MODEL_PATH)
        self._bank: Optional[GlyphBank] = None
        self._stamp_value: Optional[tuple] = None
        self._lock = threading.RLock()

    def _resolve_path(self) -> str:
        if self._explicit_path:
            return self._explicit_path
        return os.environ.get("YSOLVER_MODEL", DEFAULT_MODEL_PATH)

    # -------------------------------------------------------------- loading
    def _stamp(self) -> Optional[tuple]:
        """Cheap identity for the model file.

        ``st_mtime`` alone is not enough: two retrains inside the same
        filesystem timestamp tick would look identical and the engine would keep
        serving the old model. Size is free and breaks that tie.
        """
        try:
            info = os.stat(self.model_path)
        except OSError:
            return None
        return (info.st_mtime_ns, info.st_size)

    def bank(self) -> Optional[GlyphBank]:
        """Load the bank, reloading when the file or the configured path changes."""
        resolved = self._resolve_path()
        with self._lock:
            if resolved != self.model_path:
                self.model_path = resolved
                self._bank = None
                self._stamp_value = None

        stamp = self._stamp()
        if stamp is None:
            with self._lock:
                self._bank = None
            return None
        with self._lock:
            if self._bank is None or stamp != self._stamp_value:
                self._bank = load_bank(self.model_path)
                self._stamp_value = stamp
            return self._bank

    def available(self) -> bool:
        bank = self.bank()
        if bank is None:
            return False
        return len(bank) >= MIN_GLYPHS or bool(bank.meta.get("force"))

    def metrics(self) -> Dict[str, object]:
        bank = self.bank()
        return dict(bank.meta.get("metrics", {})) if bank else {}

    def preferred_engine(self) -> Optional[str]:
        """The engine that won the last training run's held-out comparison."""
        bank = self.bank()
        if bank is None:
            return None
        preferred = bank.meta.get("preferred")
        return str(preferred) if preferred else None

    def info(self) -> dict:
        payload = super().info()
        bank = self.bank()
        if bank is not None:
            payload["modelPath"] = self.model_path
            payload["glyphs"] = len(bank)
            payload["metrics"] = self.metrics()
            payload["preferred"] = self.preferred_engine()
        return payload

    # -------------------------------------------------------------- solving
    def solve(self, prepared: Prepared, charset: str) -> EngineResult:
        bank = self.bank()
        if bank is None or len(bank) == 0:
            raise SolverError("ERROR_INTERNAL", "no learned model available")
        subset = bank.filtered(charset)
        if len(subset) == 0:
            raise SolverError("ERROR_CAPTCHA_UNSOLVABLE", "no learned glyphs for this charset")

        mask = glyph_mask(prepared.original)
        glyphs = filter_debris(segment(mask))
        if not glyphs:
            raise SolverError("ERROR_CAPTCHA_UNSOLVABLE", "no glyphs found in image")

        characters: List[str] = []
        scores: List[float] = []
        for glyph in glyphs:
            vector = vectorise(glyph.mask)
            if not np.any(vector):
                continue
            similarities = subset.vectors @ vector
            best = int(np.argmax(similarities))
            characters.append(subset.chars[best])
            scores.append(float(similarities[best]))
        if not characters:
            raise SolverError("ERROR_CAPTCHA_UNSOLVABLE", "no glyphs could be matched")
        return EngineResult(
            text="".join(characters),
            confidence=round(float(np.clip(np.mean(scores), 0.0, 1.0)), 3),
        )
