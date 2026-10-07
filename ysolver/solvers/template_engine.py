"""Template-matching engine — the zero-dependency fallback.

Renders every character of the requested charset in the local fonts at several
rotations, then matches each segmented glyph by cosine similarity. No model
download, no binary, works completely offline.

Tuned defaults (measured on the synthetic corpus, see ``scripts/benchmark.py``):
2x upscale -> Otsu -> morphological opening(3) -> connected-component
segmentation -> 6 local fonts x 5 rotations. Accuracy is good on crisp,
evenly-spaced captchas and degrades on heavily warped ones — install the
``ddddocr`` extra when you need more.
"""

from __future__ import annotations

import glob
import os
import threading
from typing import Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..preprocess import Prepared, binarize, decode_image, otsu_threshold
from .base import Engine, EngineResult, SolverError
from .segmentation import Glyph, binary_opening, segment

TEMPLATE_SIZE = 24
MAX_GLYPHS = 16
GLYPH_SCALE = 2.0
OPENING_SIZE = 3
ROTATIONS: Tuple[float, ...] = (0.0, -20.0, -10.0, 10.0, 20.0)

_PREFERRED_FONTS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    "/usr/share/fonts/truetype/ubuntu/Ubuntu-B.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/Library/Fonts/Arial.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
    "C:/Windows/Fonts/tahoma.ttf",
]

_SEARCH_DIRS = [
    "/usr/share/fonts",
    "/usr/local/share/fonts",
    os.path.expanduser("~/.fonts"),
    os.path.expanduser("~/Library/Fonts"),
]


def discover_fonts(limit: int = 6) -> List[str]:
    """Best-effort list of usable TrueType fonts on this machine."""
    found: List[str] = []
    for path in _PREFERRED_FONTS:
        if os.path.exists(path) and path not in found:
            found.append(path)
        if len(found) >= limit:
            return found
    for directory in _SEARCH_DIRS:
        if len(found) >= limit or not os.path.isdir(directory):
            continue
        for pattern in ("**/*.ttf", "**/*.otf"):
            for path in sorted(glob.glob(os.path.join(directory, pattern), recursive=True)):
                if path not in found:
                    found.append(path)
                if len(found) >= limit:
                    return found
    return found


def _crop_to_ink(mask: np.ndarray) -> np.ndarray:
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if rows.size == 0 or cols.size == 0:
        return mask
    return mask[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]


def _pad_to_square(mask: np.ndarray) -> np.ndarray:
    """Keep the aspect ratio: aspect is signal ('1' vs 'm'), squashing loses it."""
    height, width = mask.shape
    side = max(height, width)
    canvas = np.zeros((side, side), dtype=bool)
    top = (side - height) // 2
    left = (side - width) // 2
    canvas[top : top + height, left : left + width] = mask
    return canvas


def _resize_mask(mask: np.ndarray, size: int = TEMPLATE_SIZE) -> np.ndarray:
    img = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    img = img.resize((size, size), Image.LANCZOS)
    return np.asarray(img) > 110


def vectorise(mask: np.ndarray) -> np.ndarray:
    """Crop -> aspect-preserving square -> mean-centre -> L2-normalise."""
    crop = _crop_to_ink(mask)
    if crop.size == 0:
        return np.zeros(TEMPLATE_SIZE * TEMPLATE_SIZE, dtype=np.float32)
    flat = _resize_mask(_pad_to_square(crop)).astype(np.float32).ravel()
    flat -= flat.mean()
    norm = float(np.linalg.norm(flat))
    return flat / norm if norm > 0 else flat


def glyph_mask(image: Image.Image, scale: float = GLYPH_SCALE,
               opening: int = OPENING_SIZE) -> np.ndarray:
    """Binarise at a working resolution the font templates match well."""
    work = image.convert("L")
    if scale != 1.0:
        work = work.resize(
            (max(1, int(work.width * scale)), max(1, int(work.height * scale))),
            Image.LANCZOS,
        )
    mask = binarize(work)
    if opening and mask.any():
        opened = binary_opening(mask, opening)
        if opened.any() and opened.sum() >= mask.sum() * 0.25:
            mask = opened
    return mask


def filter_debris(glyphs: List[Glyph]) -> List[Glyph]:
    """Drop dots and slivers left behind by anti-bot lines."""
    if len(glyphs) < 3:
        return glyphs
    median_h = sorted(g.h for g in glyphs)[len(glyphs) // 2]
    median_area = sorted(g.area for g in glyphs)[len(glyphs) // 2]
    kept = [
        g for g in glyphs
        if g.h >= 0.35 * median_h or g.area >= 0.08 * median_area
    ]
    return kept or glyphs


class TemplateEngine(Engine):
    name = "template"
    description = "Built-in font template matching (no downloads, works offline)."
    install_hint = ""

    def __init__(self, fonts: Sequence[str] | None = None):
        self._fonts = list(fonts) if fonts else None
        self._cache: Dict[str, Tuple[np.ndarray, List[str]]] = {}
        self._lock = threading.RLock()

    # --------------------------------------------------------------- fonts
    def available(self) -> bool:
        return bool(self.fonts())

    def fonts(self) -> List[str]:
        if self._fonts is None:
            with self._lock:
                if self._fonts is None:
                    self._fonts = discover_fonts()
        return self._fonts

    def _render(self, char: str, font_path: str) -> np.ndarray | None:
        canvas = 96
        image = Image.new("L", (canvas, canvas), 255)
        drawer = ImageDraw.Draw(image)
        try:
            font = ImageFont.truetype(font_path, 64)
        except Exception:
            return None
        try:
            drawer.text((canvas // 2, canvas // 2), char, font=font, fill=0, anchor="mm")
        except Exception:
            drawer.text((12, 12), char, font=font, fill=0)
        mask = np.asarray(image) < 128
        return _crop_to_ink(mask) if mask.any() else None

    def templates(self, charset: str) -> Tuple[np.ndarray, List[str]]:
        with self._lock:
            cached = self._cache.get(charset)
            if cached is not None:
                return cached
            vectors: List[np.ndarray] = []
            labels: List[str] = []
            for char in charset:
                for font_path in self.fonts():
                    glyph = self._render(char, font_path)
                    if glyph is None:
                        continue
                    image = Image.fromarray((glyph.astype(np.uint8) * 255), mode="L")
                    for angle in ROTATIONS:
                        rotated = (
                            image
                            if angle == 0.0
                            else image.rotate(angle, resample=Image.BICUBIC, fillcolor=0)
                        )
                        rotated_mask = np.asarray(rotated) > 110
                        if rotated_mask.any():
                            vectors.append(vectorise(rotated_mask))
                            labels.append(char)
            matrix = (
                np.vstack(vectors).astype(np.float32)
                if vectors
                else np.zeros((0, TEMPLATE_SIZE * TEMPLATE_SIZE), dtype=np.float32)
            )
            self._cache[charset] = (matrix, labels)
            return matrix, labels

    # --------------------------------------------------------------- solving
    def read(self, mask: np.ndarray, charset: str) -> Tuple[str, float]:
        """Match an already-binarised captcha; returns ``(text, confidence)``."""
        glyphs = filter_debris(segment(mask))[:MAX_GLYPHS]
        if not glyphs:
            raise SolverError("ERROR_CAPTCHA_UNSOLVABLE", "no glyphs found in image")
        matrix, labels = self.templates(charset)
        if matrix.shape[0] == 0:
            raise SolverError("ERROR_CAPTCHA_UNSOLVABLE", "no font templates available")

        chars: List[str] = []
        scores: List[float] = []
        for glyph in glyphs:
            vector = vectorise(glyph.mask)
            if not np.any(vector):
                continue
            similarities = matrix @ vector
            chars.append(labels[int(np.argmax(similarities))])
            scores.append(float(similarities.max()))
        if not chars:
            raise SolverError("ERROR_CAPTCHA_UNSOLVABLE", "no glyphs could be matched")
        return "".join(chars), float(np.clip(np.mean(scores), 0.0, 1.0))

    def solve(self, prepared: Prepared, charset: str) -> EngineResult:
        mask = glyph_mask(prepared.original)
        text, confidence = self.read(mask, charset)
        return EngineResult(text=text, confidence=round(confidence, 3))


__all__ = [
    "TemplateEngine",
    "decode_image",
    "discover_fonts",
    "filter_debris",
    "glyph_mask",
    "otsu_threshold",
    "vectorise",
]
