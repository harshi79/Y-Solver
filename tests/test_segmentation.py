"""Glyph segmentation behaviour."""

from __future__ import annotations

import numpy as np

from ysolver.solvers.segmentation import (
    Glyph,
    density,
    drop_noise,
    merge_stacked,
    segment,
    split_wide,
)


def _glyph(x0: int, y0: int, w: int, h: int) -> Glyph:
    return Glyph(mask=np.ones((h, w), dtype=bool), x0=x0, y0=y0)


def test_density_of_solid_block_is_one():
    assert density(_glyph(0, 0, 5, 5)) == 1.0


def test_merge_stacked_reattaches_a_dot():
    stem = _glyph(10, 10, 6, 20)
    dot = _glyph(11, 2, 4, 4)
    merged = merge_stacked([stem, dot])
    assert len(merged) == 1
    assert (merged[0].y0, merged[0].h) == (2, 28)  # covering dot + stem


def test_merge_stacked_keeps_distant_blobs_apart():
    stem = _glyph(10, 40, 6, 20)
    far = _glyph(11, 0, 4, 4)
    assert len(merge_stacked([stem, far])) == 2


def test_split_wide_cuts_a_touching_pair():
    # Two 'O'-like rings fused by a thin bridge.
    mask = np.zeros((20, 40), dtype=bool)
    mask[3:17, 2:6] = True
    mask[3:17, 16:20] = True
    mask[3:17, 34:38] = True
    mask[9:11, 6:34] = True  # thin connector
    glyphs = segment(mask)
    assert len(glyphs) >= 3


def test_drop_noise_removes_thin_diagonal_lines():
    letters = [_glyph(10 * i, 0, 12, 20) for i in range(4)]
    line_mask = np.zeros((15, 60), dtype=bool)
    for column in range(60):  # a 1px diagonal sweep
        line_mask[column // 6, column] = True
    line = Glyph(mask=line_mask, x0=0, y0=30)
    kept = drop_noise([*letters, line])
    assert all(g is not line for g in kept)
    assert len(kept) == 4


def test_segment_orders_glyphs_left_to_right():
    mask = np.zeros((20, 80), dtype=bool)
    for x in (60, 5, 35):
        mask[4:16, x : x + 10] = True
    glyphs = segment(mask)
    assert [g.x0 for g in glyphs] == [5, 35, 60]


def test_segment_on_generated_captcha_finds_expected_glyphs():
    from ysolver.preprocess import decode_image
    from ysolver.solvers.template_engine import glyph_mask
    from ysolver.synth import make_captcha, to_png

    image, text = make_captcha(text="abcde", seed=11, rotation=0, noise=0.0)
    glyphs = segment(glyph_mask(decode_image(to_png(image))))
    assert len(glyphs) == len(text)
