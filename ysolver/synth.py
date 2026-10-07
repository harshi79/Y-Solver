"""Deterministic synthetic captcha generator.

Used by the test-suite, the benchmark script and the dashboard's "try a sample"
button. It renders classic distorted-text captchas: per-glyph rotation, random
noise strokes and a sine warp. No network, no assets — just DejaVu fonts.
"""

from __future__ import annotations

import io
import math
import random
from typing import Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .config import DEFAULT_CHARSET

FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf",
    "/usr/share/fonts/truetype/ubuntu/Ubuntu-B.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
]


def _available_fonts() -> List[str]:
    import os

    fonts = [path for path in FONT_CANDIDATES if os.path.exists(path)]
    return fonts or [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]


def _ink_mask(image: Image.Image, background: int) -> Image.Image:
    """Mask of the glyph pixels, for whichever polarity this captcha uses.

    Getting this wrong is silent: on a light-on-dark image a "darker than the
    background" mask is empty, so the glyph is pasted with a fully transparent
    mask and the captcha ends up containing nothing but noise lines.
    """
    if background >= 128:  # dark ink on a light background
        return image.point(lambda p: 255 if p < background - 25 else 0)
    return image.point(lambda p: 255 if p > background + 25 else 0)


def _sine_warp(image: Image.Image, amplitude: float, rng: random.Random) -> Image.Image:
    if amplitude <= 0:
        return image
    array = np.asarray(image, dtype=np.uint8)
    height, width = array.shape
    phase = rng.uniform(0, 2 * math.pi)
    columns = np.arange(width)[None, :]
    rows = np.arange(height)[:, None]
    offset = (amplitude * np.sin(2 * math.pi * columns / max(8.0, width / 3.0) + phase)).astype(int)
    shifted = np.clip(rows + offset, 0, height - 1)
    return Image.fromarray(array[shifted, columns], mode="L")


def make_captcha(
    text: Optional[str] = None,
    length: int = 5,
    charset: str = DEFAULT_CHARSET,
    size: Tuple[int, int] = (170, 64),
    seed: int = 0,
    noise: float = 1.0,
    warp: float = 1.0,
    dark_on_light: bool = True,
    rotation: float = 22.0,
) -> Tuple[Image.Image, str]:
    """Render one captcha. Returns ``(image, text)``."""
    rng = random.Random(seed)
    if text is None:
        text = "".join(rng.choice(charset) for _ in range(length))
    fonts = _available_fonts()
    width, height = size
    scale = 3
    big_w, big_h = width * scale, height * scale

    background = rng.randint(230, 255) if dark_on_light else rng.randint(0, 25)
    ink_low, ink_high = (0, 90) if dark_on_light else (175, 255)

    canvas = Image.new("L", (big_w, big_h), background)
    slot = big_w // max(1, len(text))

    for index, char in enumerate(text):
        font = ImageFont.truetype(rng.choice(fonts), int(big_h * rng.uniform(0.52, 0.72)))
        glyph = Image.new("L", (slot, big_h), background)
        drawer = ImageDraw.Draw(glyph)
        bbox = drawer.textbbox((0, 0), char, font=font)
        glyph_w, glyph_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        drawer.text(
            ((slot - glyph_w) // 2 - bbox[0], (big_h - glyph_h) // 2 - bbox[1]),
            char,
            font=font,
            fill=rng.randint(ink_low, ink_high),
        )
        if rotation:
            glyph = glyph.rotate(
                rng.uniform(-rotation, rotation) * warp,
                resample=Image.BICUBIC,
                fillcolor=background,
            )
        canvas.paste(glyph, (index * slot, 0), _ink_mask(glyph, background))

    # Noise strokes and speckles.
    if noise > 0:
        drawer = ImageDraw.Draw(canvas)
        for _ in range(int(2.5 * noise * scale)):
            points = [
                (rng.randint(0, big_w), rng.randint(0, big_h)) for _ in range(rng.randint(2, 4))
            ]
            drawer.line(
                points,
                fill=rng.randint(ink_low, ink_high),
                width=max(1, int(scale * rng.uniform(0.25, 0.9))),
            )
        if rng.random() < 0.6:
            for _ in range(rng.randint(1, 3)):
                cx, cy = rng.randint(0, big_w), rng.randint(0, big_h)
                radius = rng.randint(scale * 4, scale * 18)
                drawer.arc(
                    [cx - radius, cy - radius, cx + radius, cy + radius],
                    start=rng.randint(0, 360),
                    end=rng.randint(0, 360),
                    fill=rng.randint(ink_low, ink_high),
                    width=max(1, scale // 2),
                )
        speckles = np.asarray(canvas, dtype=np.uint8).copy()
        count = int(big_w * big_h * 0.0008 * noise)
        for _ in range(count):
            y, x = rng.randrange(big_h), rng.randrange(big_w)
            speckles[y, x] = rng.randint(ink_low, ink_high)
        canvas = Image.fromarray(speckles, mode="L")

    if len(text) > 1:
        canvas = _sine_warp(canvas, amplitude=big_h * 0.05 * warp, rng=rng)

    image = canvas.resize((width, height), Image.LANCZOS)
    return image.convert("RGB"), text


def to_png(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def batch(
    count: int = 10,
    length: int = 5,
    charset: str = DEFAULT_CHARSET,
    seed: int = 0,
    **kwargs,
) -> Iterable[Tuple[Image.Image, str]]:
    for index in range(count):
        yield make_captcha(
            length=length, charset=charset, seed=seed * 1000 + index, **kwargs
        )


def write_dataset(
    directory: str, count: int = 20, **kwargs
) -> List[Tuple[str, str]]:
    """Write ``captcha_XXXX.png`` files plus a ``labels.tsv`` manifest."""
    import os

    os.makedirs(directory, exist_ok=True)
    rows: List[Tuple[str, str]] = []
    for index, (image, text) in enumerate(batch(count=count, **kwargs)):
        name = f"captcha_{index:04d}.png"
        path = os.path.join(directory, name)
        with open(path, "wb") as handle:
            handle.write(to_png(image))
        rows.append((name, text))
    with open(os.path.join(directory, "labels.tsv"), "w", encoding="utf-8") as handle:
        for name, text in rows:
            handle.write(f"{name}\t{text}\n")
    return rows


if __name__ == "__main__":  # pragma: no cover - convenience for eyeballing output
    import sys

    target = sys.argv[1] if len(sys.argv) > 1 else "data/samples"
    amount = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    for name, text in write_dataset(target, count=amount)[:5]:
        print(f"{target}/{name} -> {text}")
