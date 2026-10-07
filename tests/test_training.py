"""Training, evaluation and the honesty rules around them."""

from __future__ import annotations

import contextlib

import numpy as np
import pytest

from ysolver.labels import LabelStore
from ysolver.solvers.learned_engine import (
    GlyphBank,
    LearnedEngine,
    build_bank,
    extract_glyphs,
    load_bank,
    save_bank,
)
from ysolver.synth import make_captcha, to_png


def _labelled(seed: int, text: str = "abcde"):
    image, _ = make_captcha(text=text, seed=seed, rotation=4, noise=0.2, warp=0.2)
    return to_png(image), text


def test_extract_glyphs_pairs_each_glyph_with_its_character():
    image, text = _labelled(1)
    pairs = extract_glyphs(image, text)
    assert len(pairs) == len(text)
    for vector, character in pairs:
        assert vector.shape == (576,)
        assert abs(float(np.linalg.norm(vector)) - 1.0) < 1e-4  # unit length
        assert character in text


def test_extract_glyphs_refuses_mislabelled_samples():
    """A wrong character count means segmentation failed — do not learn from it."""
    image, _ = _labelled(2)
    assert extract_glyphs(image, "abcdefgh") == []
    assert extract_glyphs(image, "ab") == []


def test_extract_glyphs_ignores_undecodable_images():
    assert extract_glyphs(b"not an image", "ab") == []


def test_build_bank_reports_its_own_quality():
    dataset = [(*_labelled(seed), "test") for seed in range(10)]
    bank, stats = build_bank(dataset)
    assert stats["samples"] == 10
    assert stats["glyphs"] == len(bank)
    assert stats["glyphs"] > 0
    assert stats["usable"] + stats["skipped_mismatch"] == 10


def test_bank_round_trips_through_disk(tmp_path):
    dataset = [(*_labelled(seed), "") for seed in range(6)]
    bank, _ = build_bank(dataset)
    bank.meta["preferred"] = "learned"
    path = tmp_path / "bank.npz"
    save_bank(str(path), bank)
    reloaded = load_bank(str(path))
    assert reloaded is not None
    assert len(reloaded) == len(bank)
    assert reloaded.chars == bank.chars
    assert reloaded.meta["preferred"] == "learned"
    assert np.allclose(reloaded.vectors, bank.vectors, atol=1e-5)


def test_load_bank_returns_none_for_missing_or_broken_files(tmp_path):
    assert load_bank(str(tmp_path / "nope.npz")) is None
    broken = tmp_path / "broken.npz"
    broken.write_bytes(b"this is not a numpy archive")
    assert load_bank(str(broken)) is None


def test_bank_filtered_by_charset():
    bank = GlyphBank(
        np.eye(3, 576, dtype=np.float32), ["a", "b", "c"], {"tag": "x"}
    )
    subset = bank.filtered("ac")
    assert subset.chars == ["a", "c"]
    assert subset.vectors.shape == (2, 576)
    assert subset.meta["tag"] == "x"
    assert bank.filtered("").chars == ["a", "b", "c"]
    assert len(bank.filtered("zzz")) == 0


def test_learned_engine_needs_a_minimum_number_of_glyphs(tmp_path, monkeypatch):
    from ysolver.solvers.learned_engine import MIN_GLYPHS

    path = tmp_path / "thin.npz"
    save_bank(str(path), GlyphBank(
        np.eye(5, 576, dtype=np.float32), list("abcde"), {"force": False}
    ))
    monkeypatch.setenv("YSOLVER_MODEL", str(path))
    engine = LearnedEngine()
    assert engine.available() is False, "a five-glyph bank must not be trusted"
    assert MIN_GLYPHS > 5


def test_learned_engine_can_be_forced_for_experiments(tmp_path, monkeypatch):
    path = tmp_path / "thin.npz"
    save_bank(str(path), GlyphBank(
        np.eye(5, 576, dtype=np.float32), list("abcde"), {"force": True}
    ))
    monkeypatch.setenv("YSOLVER_MODEL", str(path))
    engine = LearnedEngine()
    assert engine.available() is True


def test_trained_model_reads_the_style_it_was_trained_on(tmp_path, monkeypatch):
    """The whole promise of the loop: label a site, then read that site."""
    from ysolver.config import Settings
    from ysolver.solvers import CaptchaSolver, SolverError

    dataset = [(*_labelled(seed), "site") for seed in range(40)]
    bank, _ = build_bank(dataset)
    assert len(bank) >= 40
    path = tmp_path / "site.npz"
    save_bank(str(path), bank)
    monkeypatch.setenv("YSOLVER_MODEL", str(path))

    solver = CaptchaSolver(Settings(backend="learned", charset="abcdefghjkmnpqrstuvwxyz2"))
    hits = 0
    for seed in range(200, 215):  # never seen during training
        image, truth = _labelled(seed)
        with contextlib.suppress(SolverError):
            hits += solver.solve(image).text == truth
    assert hits >= 12, f"trained model only read {hits}/15 unseen samples"


def test_split_is_deterministic_and_disjoint():
    from scripts.train import split_labels

    dataset = [(*_labelled(seed), "") for seed in range(30)]
    train_a, held_a, real_a = split_labels(dataset, 0.3)
    train_b, held_b, real_b = split_labels(dataset, 0.3)
    assert [x[1] for x in train_a] == [x[1] for x in train_b]
    assert [x[1] for x in held_a] == [x[1] for x in held_b]
    assert len(train_a) + len(held_a) == len(dataset)
    assert len(held_a) >= 3
    assert real_a is True and real_b is True


def test_split_falls_back_to_training_on_everything_when_the_holdout_is_tiny():
    from scripts.train import split_labels

    dataset = [(*_labelled(seed), "") for seed in range(4)]
    train, held, is_real = split_labels(dataset, 0.5)
    assert len(train) == len(held) == len(dataset)
    assert is_real is False, "a self-graded split must be flagged as such"


def test_train_script_scores_engines_and_records_a_winner(tmp_path):
    """End-to-end: labels in, model plus a measured verdict out."""
    import subprocess
    import sys

    db = tmp_path / "labels.db"
    out = tmp_path / "model.npz"
    store = LabelStore(str(db))
    for seed in range(45):
        image, text = _labelled(seed)
        store.add_label(image, text, tag="site")
    store.close()

    result = subprocess.run(
        [
            sys.executable, "scripts/train.py",
            "--db", str(db), "--out", str(out),
            "--holdout", "0.25", "--min-glyphs", "20",
        ],
        capture_output=True, text=True,
        env={"PATH": "/usr/bin:/bin", "YSOLVER_MODEL": str(out), "HOME": "/tmp"},
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "scoreboard" in result.stdout
    assert "best on held-out data" in result.stdout
    assert out.exists()

    bank = load_bank(str(out))
    assert bank is not None and len(bank) > 0
    assert bank.meta["preferred"] in {"ddddocr", "template", "tesseract", "learned"}
    assert bank.meta["selfGraded"] is False, "45 labels is enough for a real holdout"
    assert bank.meta["labels"] == 45
    assert "metrics" in bank.meta


def test_train_script_explains_itself_with_no_labels(tmp_path):
    import subprocess
    import sys

    db = tmp_path / "empty.db"
    store = LabelStore(str(db))
    store.close()
    result = subprocess.run(
        [sys.executable, "scripts/train.py", "--db", str(db)],
        capture_output=True, text=True,
        env={"PATH": "/usr/bin:/bin", "HOME": "/tmp"},
    )
    assert result.returncode != 0
    assert "no human labels yet" in result.stdout + result.stderr


def test_a_thin_label_set_cannot_pick_the_default_engine(tmp_path):
    """Regression: with too few labels the run graded itself and claimed a winner.

    A self-graded scoreboard must still save the model — but it must never be
    allowed to change which engine a deployment prefers.
    """
    import subprocess
    import sys

    db = tmp_path / "thin.db"
    out = tmp_path / "thin.npz"
    store = LabelStore(str(db))
    for seed in range(11):
        image, text = _labelled(seed)
        store.add_label(image, text, tag="site")
    store.close()

    result = subprocess.run(
        [sys.executable, "scripts/train.py",
         "--db", str(db), "--out", str(out), "--holdout", "0.25", "--min-glyphs", "10"],
        capture_output=True, text=True,
        env={"PATH": "/usr/bin:/bin", "YSOLVER_MODEL": str(out), "HOME": "/tmp"},
    )
    assert "WARNING" in result.stdout and "nothing could be held out" in result.stdout
    assert "not adopted" in result.stdout
    assert out.exists(), "the model itself is still saved for manual use"

    bank = load_bank(str(out))
    assert bank is not None
    assert bank.meta["selfGraded"] is True
    assert bank.meta["preferred"] is None, "auto must not follow a self-graded verdict"


def test_auto_ignores_a_self_graded_model(tmp_path, monkeypatch):
    """`auto` keeps its default order when the only model is self-graded."""
    from ysolver.solvers.learned_engine import GlyphBank, save_bank

    path = tmp_path / "self_graded.npz"
    vectors = np.eye(60, 576, dtype=np.float32)
    save_bank(str(path), GlyphBank(vectors, list("abcde" * 12), {
        "selfGraded": True, "preferred": None, "metrics": {"learned": {"exactRate": 1.0}},
    }))
    monkeypatch.setenv("YSOLVER_MODEL", str(path))
    assert LearnedEngine().preferred_engine() is None
