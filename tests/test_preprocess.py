"""Image decoding, binarisation and morphology."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from ysolver.preprocess import (
    ImageDecodeError,
    binarize,
    clean,
    decode_image,
    deskew,
    otsu_threshold,
    prepare,
    to_png,
    trim,
)
from ysolver.solvers.segmentation import (
    binary_dilate,
    binary_erode,
    binary_opening,
    connected_components,
)
from ysolver.synth import make_captcha


def test_decode_rejects_garbage():
    with pytest.raises(ImageDecodeError):
        decode_image(b"not an image at all")


def test_decode_rejects_empty():
    with pytest.raises(ImageDecodeError):
        decode_image(b"")


def test_decode_accepts_png_and_reports_size():
    image = Image.new("RGB", (40, 20), "white")
    decoded = decode_image(to_png(image))
    assert decoded.size == (40, 20)


def test_otsu_splits_bimodal_histogram():
    values = np.concatenate([np.full(500, 20), np.full(500, 230)]).astype(np.uint8)
    assert 20 < otsu_threshold(values) < 230


def test_binarize_detects_dark_ink():
    image = Image.new("L", (10, 10), 255)
    for x in range(4, 7):
        for y in range(10):
            image.putpixel((x, y), 0)
    mask = binarize(image)
    assert mask.sum() == 30
    assert mask[0, 5]


def test_binarize_inverts_light_on_dark():
    image = Image.new("L", (10, 10), 0)
    for x in range(4, 7):
        for y in range(10):
            image.putpixel((x, y), 255)
    mask = binarize(image)
    assert mask.sum() == 30  # ink stays True regardless of polarity


def test_erode_and_dilate_shrink_and_grow():
    mask = np.zeros((21, 21), dtype=bool)
    mask[5:16, 5:16] = True
    assert binary_erode(mask, 3).sum() == 9 * 9
    assert binary_dilate(mask, 3).sum() == 13 * 13


def test_opening_removes_thin_lines_but_keeps_blobs():
    mask = np.zeros((21, 21), dtype=bool)
    mask[5:16, 5:16] = True       # a fat blob
    mask[0, :] = True             # a 1px line
    opened = binary_opening(mask, 3)
    assert opened[10, 10]
    assert not opened[0, :].any()


def test_connected_components_counts_letters():
    mask = np.zeros((20, 60), dtype=bool)
    mask[5:15, 5:10] = True
    mask[5:15, 20:25] = True
    mask[5:15, 40:45] = True
    glyphs = connected_components(mask)
    assert len(glyphs) == 3
    assert [g.x0 for g in glyphs] == [5, 20, 40]


def test_trim_removes_uniform_border():
    mask = np.zeros((20, 20), dtype=bool)
    mask[8:12, 8:12] = True
    trimmed = trim(mask, pad=0)
    assert trimmed.shape == (4, 4)


def test_clean_upscales_small_images():
    image, _ = make_captcha(seed=2, size=(60, 20))
    cleaned = clean(image, target_height=64)
    assert cleaned.height >= 64
    assert cleaned.mode == "RGB"


def test_deskew_straightens_a_rotated_line():
    image = Image.new("L", (120, 60), 255)
    for x in range(20, 100):  # a 12-degree diagonal bar
        y = int(30 + (x - 60) * 0.21)
        for dy in range(-2, 3):
            image.putpixel((x, max(0, min(59, y + dy))), 0)
    before = np.asarray(image, dtype=np.uint8)
    after = np.asarray(deskew(image, limit=15.0), dtype=np.uint8)
    # After deskewing, ink should concentrate in fewer distinct rows.
    assert (after < 128).any(axis=1).sum() < (before < 128).any(axis=1).sum()


def test_prepare_returns_all_three_views():
    image, _ = make_captcha(seed=3)
    prepared = prepare(to_png(image))
    assert prepared.original.size == (170, 64)
    assert prepared.mask.dtype == bool
    assert prepared.mask.any()
    assert prepared.cleaned.mode == "RGB"


def test_prepare_rejects_garbage():
    with pytest.raises(ImageDecodeError):
        prepare(b"\x00\x01\x02")
