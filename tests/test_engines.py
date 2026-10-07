"""Solver-engine level tests, run against generated captchas."""

from __future__ import annotations

import contextlib

import pytest

from ysolver.config import Settings
from ysolver.preprocess import prepare
from ysolver.solvers import CaptchaSolver, SolverError, engine_status, get_engine
from ysolver.solvers.registry import resolve_backend
from ysolver.synth import make_captcha, to_png

CHARSET = "abcdefghjkmnpqrstuvwxyz23456789"


def _settings(**overrides) -> Settings:
    defaults = dict(backend="template", charset=CHARSET)
    defaults.update(overrides)
    return Settings(**defaults)


def test_engine_status_lists_all_three_engines(client=None):
    names = {engine["name"] for engine in engine_status()}
    assert {"template", "tesseract", "ddddocr"} <= names


def test_resolve_backend_honours_explicit_choice():
    assert resolve_backend("template").name == "template"


def test_resolve_backend_auto_never_fails():
    assert resolve_backend("auto").name in {"ddddocr", "tesseract", "template"}


def test_resolve_backend_rejects_unknown_name():
    with pytest.raises(SystemExit):
        resolve_backend("magic")


def test_template_engine_reads_clean_unrotated_text():
    """The zero-dependency engine must nail easy captchas in every run."""
    solver = CaptchaSolver(_settings())
    for seed in range(6):
        image, truth = make_captcha(text="abcde", seed=seed, rotation=0, noise=0.0)
        result = solver.solve(to_png(image))
        assert result.text == truth


def test_template_engine_handles_realistic_difficulty():
    """Sanity floor on the shared corpus: characters, not whole strings."""
    solver = CaptchaSolver(_settings())
    correct = total = 0
    for seed in range(12):
        image, truth = make_captcha(seed=100 + seed)
        try:
            got = solver.solve(to_png(image)).text
        except SolverError:
            continue
        for index in range(min(len(got), len(truth))):
            correct += got[index] == truth[index]
        total += len(truth)
    assert total > 0
    assert correct / total >= 0.6


def test_numeric_mode_restricts_the_alphabet():
    solver = CaptchaSolver(_settings())
    image, truth = make_captcha(text="12345", seed=5, rotation=6, noise=0.3)
    assert solver.solve(to_png(image), numeric=True).text == truth


def test_explicit_charset_is_respected():
    solver = CaptchaSolver(_settings())
    image, truth = make_captcha(text="0123", charset="0123456789", seed=6, rotation=0, noise=0.0)
    result = solver.solve(to_png(image), charset="0123456789")
    assert result.text == truth


def test_garbage_input_raises_typed_error():
    solver = CaptchaSolver(_settings())
    with pytest.raises(SolverError) as excinfo:
        solver.solve(b"definitely not an image")
    assert excinfo.value.code.startswith("ERROR_")


def test_oversized_image_is_refused():
    solver = CaptchaSolver(_settings(max_image_bytes=16))
    image, _ = make_captcha(seed=1)
    with pytest.raises(SolverError) as excinfo:
        solver.solve(to_png(image))
    assert excinfo.value.code == "ERROR_BAD_PARAMETERS"


def test_solver_result_carries_metadata():
    solver = CaptchaSolver(_settings())
    image, _ = make_captcha(text="abcde", seed=2, rotation=0, noise=0.0)
    result = solver.solve(to_png(image))
    assert result.backend == "template"
    assert result.confidence is not None and 0.0 <= result.confidence <= 1.0
    assert result.engine_ms >= 0


@pytest.mark.skipif(
    not any(engine["name"] == "ddddocr" and engine["available"] for engine in engine_status()),
    reason="ddddocr not installed",
)
def test_ddddocr_beats_the_template_engine_on_hard_samples():
    hard = [
        to_png(make_captcha(seed=200 + index, noise=1.6, warp=1.3)[0]) for index in range(8)
    ]
    truths = [make_captcha(seed=200 + index, noise=1.6, warp=1.3)[1] for index in range(8)]
    dddd = CaptchaSolver(_settings(backend="ddddocr"))
    exact = 0
    for data, truth in zip(hard, truths):
        with contextlib.suppress(SolverError):
            exact += dddd.solve(data).text == truth
    assert exact >= 3


def test_prepare_pipeline_used_by_engines_is_stable():
    image, _ = make_captcha(seed=4)
    prepared = prepare(to_png(image))
    assert get_engine("template").available()
    assert prepared.mask.any()


def test_case_is_reconciled_with_the_requested_charset():
    from ysolver.solvers.registry import reconcile_case

    assert reconcile_case("M7JMW", "abcdefghjkmnpqrstuvwxyz23456789") == "m7jmw"
    assert reconcile_case("ab12c", "ABCDEF123") == "AB12C"
    assert reconcile_case("Ab12", "0123456789") == "Ab12"      # digit-only: untouched
    assert reconcile_case("Ab12", "aAbB12") == "Ab12"          # mixed case: untouched
    assert reconcile_case("1234", "0123456789") == "1234"


def test_solver_folds_case_end_to_end():
    solver = CaptchaSolver(_settings(charset="abcdefghjkmnpqrstuvwxyz23456789"))
    # The engine must return text inside the caller's alphabet, so a lowercase
    # image is read back exactly.
    image, truth = make_captcha(text="mw2jk", charset="abcdefghjkmnpqrstuvwxyz23456789",
                                seed=21, rotation=0, noise=0.0)
    assert solver.solve(to_png(image)).text == truth


def test_charset_variants_covers_both_cases():
    from ysolver.solvers.ddddocr_engine import charset_variants

    variants = charset_variants("ab2")
    assert {"a", "A", "b", "B", "2"} <= set(variants)
    assert len(variants) == len(set(variants))


def test_constrained_ctc_prefers_charset_characters():
    """A lookalike top pick must yield to the best in-charset alternative."""
    from ysolver.solvers.ddddocr_engine import constrained_ctc

    # ddddocr's real layout: index 0 is the CTC blank, then the alphabet.
    model_charset = "\x00abc"
    rows = [
        [0.05, 0.90, 0.02, 0.01],   # 'a'
        [0.05, 0.10, 0.15, 0.70],   # 'c'
        [0.60, 0.05, 0.30, 0.05],   # blank -> separator
        [0.05, 0.02, 0.80, 0.13],   # 'b'
    ]
    assert constrained_ctc(rows, model_charset, "abc")[0] == "acb"

    # The caller only accepts {a, b}, so the confident 'c' (index 3) is illegal
    # and the decoder must fall back to the best legal character.
    text, confidence = constrained_ctc(rows, model_charset, "ab")
    assert text == "ab" or text == "aab" or text == "abb"
    assert "c" not in text
    assert 0.0 < confidence <= 1.0


def test_constrained_ctc_accepts_timestep_first_3d_probabilities():
    from ysolver.solvers.ddddocr_engine import constrained_ctc

    rows = [[[0.05, 0.90, 0.02, 0.01]], [[0.05, 0.02, 0.80, 0.13]]]
    assert constrained_ctc(rows, "\x00abc", "abc")[0] == "ab"


def test_constrained_ctc_handles_empty_input():
    from ysolver.solvers.ddddocr_engine import constrained_ctc

    assert constrained_ctc([], "abc", "abc") == ("", 0.0)
    assert constrained_ctc([[1.0, 0.0, 0.0, 0.0]], "abc", "")[0] == ""
    assert constrained_ctc([[{"x": 1.0}]], "abc", "abc") == ("", 0.0)


def _ddddocr_ready() -> bool:
    return any(
        engine["name"] == "ddddocr" and engine["available"] for engine in engine_status()
    )


@pytest.mark.skipif(not _ddddocr_ready(), reason="ddddocr not installed")
def test_alphabet_requests_do_not_poison_the_shared_model():
    """Regression: restricting the alphabet must not break later requests.

    ddddocr's own ``set_ranges("")`` reset leaves only the CTC blank selectable,
    so a digit-only request used to make every following solve return nothing.
    """
    solver = CaptchaSolver(_settings(backend="ddddocr", charset=CHARSET))
    numeric_image, numeric_truth = make_captcha(
        text="28471", charset="0123456789", seed=31, rotation=6, noise=0.4
    )
    assert solver.solve(to_png(numeric_image), numeric=True).text == numeric_truth

    # The very next request uses the full alphabet and must still work.
    for seed in range(3):
        image, truth = make_captcha(text="mw2jk", seed=40 + seed, rotation=0, noise=0.0)
        assert solver.solve(to_png(image)).text == truth


@pytest.mark.skipif(not _ddddocr_ready(), reason="ddddocr not installed")
def test_numeric_mode_never_returns_letters():
    solver = CaptchaSolver(_settings(backend="ddddocr", charset=CHARSET))
    for seed in range(6):
        image, _ = make_captcha(length=5, seed=50 + seed)
        text = solver.solve(to_png(image), numeric=True).text
        assert text.isdigit(), f"numeric mode returned {text!r}"


def test_learned_engine_follows_the_configured_model_path(tmp_path, monkeypatch):
    """Regression: the registry caches engine instances, so a stale model path
    used to grade one style's glyph bank against another style's captchas."""
    import numpy as np

    from ysolver.solvers.learned_engine import GlyphBank, LearnedEngine, save_bank

    first = tmp_path / "a.npz"
    second = tmp_path / "b.npz"
    vectors = np.eye(2, 576, dtype=np.float32)
    save_bank(str(first), GlyphBank(vectors, ["a", "b"], {"tag": "first"}))
    save_bank(str(second), GlyphBank(vectors, ["c", "d"], {"tag": "second"}))

    monkeypatch.setenv("YSOLVER_MODEL", str(first))
    engine = LearnedEngine()
    assert engine.bank().chars == ["a", "b"]

    monkeypatch.setenv("YSOLVER_MODEL", str(second))
    assert engine.bank().chars == ["c", "d"], "engine kept serving the old model"

    # An explicitly constructed path is never overridden by the environment.
    pinned = LearnedEngine(str(first))
    monkeypatch.setenv("YSOLVER_MODEL", str(second))
    assert pinned.bank().chars == ["a", "b"]


def test_learned_engine_reloads_when_the_file_changes(tmp_path, monkeypatch):
    import numpy as np

    from ysolver.solvers.learned_engine import GlyphBank, LearnedEngine, save_bank

    path = tmp_path / "m.npz"
    save_bank(str(path), GlyphBank(np.eye(2, 576, dtype=np.float32), ["a", "b"], {}))
    engine = LearnedEngine(str(path))
    assert len(engine.bank()) == 2
    save_bank(str(path), GlyphBank(np.eye(3, 576, dtype=np.float32), ["a", "b", "c"], {}))
    assert len(engine.bank()) == 3
