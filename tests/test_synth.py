"""The synthetic generator is test infrastructure, so it gets tested too."""

from __future__ import annotations

import numpy as np

from ysolver.preprocess import decode_image
from ysolver.solvers.segmentation import segment
from ysolver.solvers.template_engine import glyph_mask
from ysolver.synth import make_captcha, to_png


def test_dark_on_light_captcha_contains_the_requested_characters():
    image, text = make_captcha(text="abcde", seed=3, rotation=10, noise=0.3, warp=0.4)
    assert text == "abcde"
    assert len(segment(glyph_mask(image))) == len(text)


def test_light_on_dark_captcha_also_contains_its_characters():
    """Regression: the paste mask used to be polarity-blind.

    A 'darker than background' mask is empty on a light-on-dark image, so the
    glyphs were never pasted and the captcha held nothing but noise strokes.
    """
    image, text = make_captcha(
        text="abcde", seed=3, rotation=10, noise=0.3, warp=0.4, dark_on_light=False
    )
    assert (np.asarray(image.convert("L")) > 128).any(), "expected light glyph pixels"
    glyphs = segment(glyph_mask(image))
    assert len(glyphs) == len(text), f"found {len(glyphs)} glyphs, expected {len(text)}"


def test_both_polarities_are_readable_by_the_template_engine():
    from ysolver.config import Settings
    from ysolver.solvers import CaptchaSolver

    solver = CaptchaSolver(Settings(backend="template"))
    for dark in (True, False):
        image, truth = make_captcha(
            text="mw2jk", seed=9, rotation=8, noise=0.2, warp=0.3, dark_on_light=dark
        )
        assert solver.solve(to_png(image)).text == truth


def test_generated_captchas_are_deterministic_for_a_seed():
    first, text_a = make_captcha(seed=77)
    second, text_b = make_captcha(seed=77)
    assert text_a == text_b
    assert np.array_equal(np.asarray(first), np.asarray(second))


def test_decode_round_trip_through_png():
    image, _ = make_captcha(seed=5)
    assert decode_image(to_png(image)).size == image.size
