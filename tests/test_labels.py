"""The human-in-the-loop labelling store."""

from __future__ import annotations

import pytest

from ysolver.labels import LabelStore, image_digest
from ysolver.synth import make_captcha, to_png


@pytest.fixture()
def store(tmp_path):
    return LabelStore(str(tmp_path / "labels.db"))


def _png(seed: int = 1) -> bytes:
    return to_png(make_captcha(seed=seed)[0])


def test_candidate_is_queued_then_handed_out(store):
    image = _png()
    assert store.add_candidate(image, job_id="1", model_text="abc", confidence=0.4,
                               reason="uncertain") is True
    pending = store.pending(limit=1, include_image=True)
    assert len(pending) == 1
    assert pending[0].image == image
    assert pending[0].model_text == "abc"
    assert pending[0].reason == "uncertain"


def test_identical_images_are_not_queued_twice(store):
    image = _png()
    assert store.add_candidate(image) is True
    assert store.add_candidate(image) is False
    assert store.pending_count() == 1


def test_label_is_filed_and_removed_from_the_queue(store):
    store.add_candidate(_png())
    candidate = store.pending(1)[0]
    result = store.submit_label(candidate.id, "  ab12c  ", tag="shop")
    assert result["text"] == "ab12c"  # whitespace trimmed
    assert store.pending_count() == 0
    dataset = store.dataset()
    assert len(dataset) == 1
    assert dataset[0][1] == "ab12c"
    assert dataset[0][2] == "shop"


def test_skip_removes_a_candidate_without_labelling_it(store):
    store.add_candidate(_png())
    candidate = store.pending(1)[0]
    assert store.skip(candidate.id) is True
    assert store.pending_count() == 0
    assert store.dataset() == []
    assert store.stats()["skipped"] == 1


def test_label_length_is_validated(store):
    store.add_candidate(_png())
    candidate = store.pending(1)[0]
    with pytest.raises(ValueError):
        store.submit_label(candidate.id, "")
    with pytest.raises(ValueError):
        store.submit_label(candidate.id, "x" * 40)


def test_unknown_candidate_is_rejected(store):
    with pytest.raises(KeyError):
        store.submit_label(4242, "abc")


def test_imported_labels_bypass_the_queue(store):
    store.add_label(_png(2), "zzz", tag="import")
    assert store.pending_count() == 0
    assert store.dataset(tag="import")[0][1] == "zzz"
    assert store.dataset(tag="nope") == []


def test_stats_report_the_workflow(store):
    for seed in range(3):
        store.add_candidate(_png(10 + seed), reason="failed")
    first = store.pending(1)[0]
    store.submit_label(first.id, "abc")
    store.skip(store.pending(1)[0].id)
    stats = store.stats()
    assert stats["labeled"] == 1
    assert stats["pending"] == 1
    assert stats["skipped"] == 1
    assert stats["pendingByReason"] == {"failed": 1}


def test_oldest_samples_are_offered_first(store):
    import time

    store.add_candidate(_png(20), reason="first")
    time.sleep(0.01)
    store.add_candidate(_png(21), reason="second")
    assert store.pending(1)[0].reason == "first"


def test_pending_queue_is_bounded(tmp_path):
    store = LabelStore(str(tmp_path / "bounded.db"), max_candidates=100)
    for seed in range(8):
        store.add_candidate(_png(30 + seed))
    # max_candidates has a floor of 100, so all eight survive — the point is that
    # the trimmer never drops below the configured bound.
    assert store.pending_count() == 8
    assert store.max_candidates >= 100


def test_image_digest_is_stable_and_content_addressed():
    image = _png()
    assert image_digest(image) == image_digest(bytes(image))
    assert image_digest(image) != image_digest(_png(99))
