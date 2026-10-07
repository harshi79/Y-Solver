"""Image decoding and cleanup for classic distorted-text captchas.

Everything here is pure Pillow + NumPy, runs on CPU in a few milliseconds and
needs no network or model download.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from PIL import Image, ImageFilter, ImageOps


class ImageDecodeError(ValueError):
    """Raised when the payload is not a decodable image."""


@dataclass
class Prepared:
    """One decoded captcha, plus the *lazy* views engines may ask for.

    ``cleaned`` and ``mask`` cost ~10 ms together (autocontrast, median filter,
    Otsu, and a deskew sweep that alone is ~9 ms) — and engines that read the
    original bitmap, which is most of them, never touch either. They are
    therefore computed on first access and cached, so a solve only pays for the
    work its engine actually needs.

    ``raw`` keeps the bytes exactly as the client sent them: the model engines
    decode the original file themselves, which is faster and lossless compared
    with re-encoding our own PNG copy.
    """

    original: Image.Image              # exactly what the client sent (RGB)
    raw: bytes = b""                   # the untouched upload, when still available
    do_deskew: bool = True
    size: Tuple[int, int] = (0, 0)     # (width, height)
    _cleaned: Optional[Image.Image] = None
    _mask: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        if not self.size:
            self.size = self.original.size

    @property
    def cleaned(self) -> Image.Image:
        """Denoised / upscaled / contrast-normalised view."""
        if self._cleaned is None:
            self._cleaned = clean(self.original)
        return self._cleaned

    @property
    def mask(self) -> np.ndarray:
        """Boolean ink mask (True where a glyph was drawn), deskewed and trimmed."""
        if self._mask is None:
            reference = self.cleaned
            if max(reference.size) < max(self.original.size):
                reference = self.original
            mask = binarize(reference)
            if self.do_deskew and mask.any():
                rotated = deskew(reference, mask)
                if rotated is not reference:
                    reference = rotated
                    mask = binarize(reference)
            self._mask = trim(mask)
        return self._mask

    def ensure(self) -> "Prepared":
        """Force every view (used by the benchmark and the tesseract engine)."""
        _ = self.mask
        return self

    def bytes_for_engine(self) -> bytes:
        """Original bytes when we still have them, else a lossless PNG copy."""
        return self.raw or to_png(self.original)


def prepare(data: bytes, do_deskew: bool = True) -> Prepared:
    """Decode an upload (validating it) without paying for cleanup up front."""
    original = decode_image(data)
    return Prepared(original=original, raw=data, do_deskew=do_deskew)


def decode_image(data: bytes) -> Image.Image:
    if not data:
        raise ImageDecodeError("empty payload")
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except Exception as exc:  # PIL raises a zoo of exceptions
        raise ImageDecodeError(f"could not decode image: {exc}") from exc
    if getattr(img, "is_animated", False):
        img.seek(0)
    return img.convert("RGB")


def to_png(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _ink_is_dark(img: Image.Image) -> bool:
    """Captchas are either dark-on-light or light-on-dark; decide which."""
    gray = np.asarray(img.convert("L"), dtype=np.float32)
    if gray.size == 0:
        return True
    border = np.concatenate(
        [gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]]
    )
    return float(np.mean(border)) >= 127.0


def otsu_threshold(values: np.ndarray) -> int:
    """Classic Otsu between-class variance maximisation.

    When two populations are perfectly separated the objective is flat across
    the whole gap, so return the *middle* of the plateau rather than its edge —
    an edge threshold is fragile against JPEG noise on real captchas.
    """
    hist, _ = np.histogram(values, bins=256, range=(0, 256))
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total == 0:
        return 127
    prob = hist / total
    omega = np.cumsum(prob)
    mu = np.cumsum(prob * np.arange(256))
    mu_t = mu[-1]
    denom = omega * (1.0 - omega)
    denom[denom == 0] = 1e-12
    sigma_b = (mu_t * omega - mu) ** 2 / denom
    best = float(sigma_b.max())
    plateau = np.flatnonzero(sigma_b >= best * (1.0 - 1e-9))
    return int(np.median(plateau))


def binarize(img: Image.Image, invert: Optional[bool] = None) -> np.ndarray:
    """Return a boolean ink mask (True where a glyph was drawn)."""
    gray = np.asarray(img.convert("L"), dtype=np.uint8)
    if gray.size == 0:
        return np.zeros((1, 1), dtype=bool)
    thresh = otsu_threshold(gray.ravel())
    dark = gray <= thresh
    if invert is None:
        invert = not _ink_is_dark(img)
    return (~dark) if invert else dark


def clean(img: Image.Image, target_height: int = 64) -> Image.Image:
    """Upscale small captchas, drop speckle noise, flatten contrast."""
    work = img.convert("L")
    w, h = work.size
    if h < target_height and h > 0:
        scale = min(4.0, target_height / float(h))
        work = work.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
    work = ImageOps.autocontrast(work, cutoff=1)
    work = work.filter(ImageFilter.MedianFilter(size=3))
    return work.convert("RGB")


def trim(mask: np.ndarray, pad: int = 1) -> np.ndarray:
    """Crop away uniform borders so the glyphs fill the frame."""
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if rows.size == 0 or cols.size == 0:
        return mask
    r0, r1 = max(0, rows[0] - pad), min(mask.shape[0], rows[-1] + 1 + pad)
    c0, c1 = max(0, cols[0] - pad), min(mask.shape[1], cols[-1] + 1 + pad)
    return mask[r0:r1, c0:c1]


def _projection_sharpness(mask: np.ndarray) -> float:
    if mask.size == 0:
        return 0.0
    profile = mask.sum(axis=0).astype(np.float64)
    if profile.size < 3:
        return 0.0
    return float(np.sum(np.abs(np.diff(profile))))


def deskew(img: Image.Image, mask: Optional[np.ndarray] = None,
           limit: float = 8.0, step: float = 1.0) -> Image.Image:
    """Rotate the image by the angle that sharpens the vertical ink profile."""
    if mask is None:
        mask = binarize(img)
    best_angle, best_score = 0.0, _projection_sharpness(mask)
    angle = -limit
    while angle <= limit + 1e-6:
        if abs(angle) >= step / 2:
            candidate = img.rotate(angle, resample=Image.BILINEAR, fillcolor=255
                                   if _ink_is_dark(img) else 0)
            score = _projection_sharpness(binarize(candidate, invert=False))
            if score > best_score:
                best_angle, best_score = angle, score
        angle += step
    if best_angle == 0.0:
        return img
    return img.rotate(
        best_angle, resample=Image.BILINEAR,
        fillcolor=255 if _ink_is_dark(img) else 0,
    )
