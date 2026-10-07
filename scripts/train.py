#!/usr/bin/env python
"""Train the learned engine from human labels and measure it honestly.

    python scripts/train.py                   # train on everything labelled so far
    python scripts/train.py --holdout 0.25    # hold out a quarter for the verdict
    python scripts/train.py --tag shop.example.com

The script does three things, in this order:

1. splits the label set into train / holdout (deterministic, by image hash, so
   the same sample never migrates between runs),
2. builds the glyph bank from the *train* half only,
3. runs every available engine over the *holdout* half and prints the scoreboard.

The winner is written into the model file, and ``auto`` starts preferring it —
so a model only takes over when it has actually earned the job. Nothing here
touches the network, and a run on a few hundred labels takes about a second.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import statistics
import sys
import time
from typing import Dict, List, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ysolver.config import DEFAULT_CHARSET, Settings
from ysolver.labels import LabelStore
from ysolver.solvers import CaptchaSolver, SolverError
from ysolver.solvers.learned_engine import (
    MIN_GLYPHS,
    MIN_LABELS,
    build_bank,
    save_bank,
)

EVALABLE = ("ddddocr", "learned", "tesseract", "template")


def split_labels(
    dataset: Sequence[Tuple[bytes, str, str]], holdout: float
) -> Tuple[List, List, bool]:
    """Deterministic split keyed on the image hash.

    Returns ``(train, held_out, is_real)``. ``is_real`` is False when there are
    too few labels to hold anything back — in that case the caller is scoring a
    model on the data it was trained on, which proves nothing and must not be
    allowed to decide which engine the deployment uses.
    """
    if holdout <= 0:
        return list(dataset), list(dataset), False
    train, held = [], []
    for image, text, tag in dataset:
        bucket = int(hashlib.sha256(image).hexdigest()[:8], 16) / 0xFFFFFFFF
        (held if bucket < holdout else train).append((image, text, tag))
    if len(held) < 3 or len(train) < MIN_LABELS:
        return list(dataset), list(dataset), False
    return train, held, True


def evaluate(engine_name: str, samples, charset: str) -> Dict[str, object]:
    """Exact-match score for one engine over a fixed sample list."""
    try:
        solver = CaptchaSolver(Settings(backend=engine_name, charset=charset))
    except (SystemExit, Exception):
        return {"engine": engine_name, "available": False, "exact": None, "n": len(samples)}

    exact = correct = total = errors = 0
    latencies: List[float] = []
    for image, truth, _tag in samples:
        started = time.perf_counter()
        try:
            text = solver.solve(image).text
        except SolverError:
            text, errors = "", errors + 1
        latencies.append((time.perf_counter() - started) * 1000)
        exact += text == truth
        for index in range(min(len(text), len(truth))):
            correct += text[index] == truth[index]
        total += len(truth)
    return {
        "engine": engine_name,
        "available": True,
        "n": len(samples),
        "exact": exact,
        "exactRate": exact / max(1, len(samples)),
        "charRate": correct / max(1, total),
        "medianMs": statistics.median(latencies) if latencies else 0.0,
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=os.environ.get("YSOLVER_DB", "data/ysolver.db"))
    parser.add_argument("--out", default=os.environ.get("YSOLVER_MODEL", "data/learned.npz"))
    parser.add_argument("--holdout", type=float, default=0.25,
                        help="fraction reserved for the verdict (default 0.25)")
    parser.add_argument("--tag", help="only train on labels with this tag")
    parser.add_argument("--charset", default=DEFAULT_CHARSET)
    parser.add_argument("--force", action="store_true",
                        help="save the model even when it loses to another engine")
    parser.add_argument("--min-glyphs", type=int, default=MIN_GLYPHS)
    args = parser.parse_args()

    if not os.path.exists(args.db):
        raise SystemExit(f"no label database at {args.db} — label some captchas first")

    store = LabelStore(args.db)
    dataset = store.dataset(tag=args.tag)
    store.close()
    if not dataset:
        raise SystemExit(
            "no human labels yet.\n"
            "  1. run the server:  ysolver\n"
            "  2. open the dashboard and use the 'Teach it' tab, or POST to /api/labels\n"
            "  3. re-run this script"
        )

    print(f"\nY-Solver training — {len(dataset)} human label(s)"
          + (f", tag={args.tag!r}" if args.tag else ""))
    train, held, is_real_holdout = split_labels(dataset, args.holdout)
    if is_real_holdout:
        print(f"  train {len(train)} · holdout {len(held)}\n")
    else:
        print(f"  train {len(train)} · holdout 0  (not enough labels to hold any back)\n")

    bank, stats = build_bank(train)
    usable = stats.get("usable", len(train))
    print(f"  usable samples   {usable}/{len(train)}"
          if isinstance(usable, int) else f"  usable samples   {usable}")
    if stats.get("skipped_mismatch"):
        print(f"  skipped          {stats['skipped_mismatch']} "
              f"(glyph count did not match the label length)")
    print(f"  glyphs in bank   {len(bank)}")

    if len(bank) < args.min_glyphs:
        print(
            f"\n  Only {len(bank)} glyphs — below the {args.min_glyphs} needed for a "
            f"trustworthy model.\n  Label more captchas in the dashboard and run this again."
        )
        if not args.force:
            raise SystemExit(1)
        print("  --force given, saving anyway.")

    if not is_real_holdout:
        print(
            "  WARNING: fewer than ~12 labels, so nothing could be held out.\n"
            "           The numbers below grade the model on its own training data\n"
            "           and are therefore optimistic by construction. They are fine\n"
            "           for a smoke test, but they will NOT be used to pick the\n"
            "           default engine. Label more captchas for a real verdict.\n"
        )

    # Score every engine on the same held-out samples.
    print("scoreboard (held-out samples)\n" if is_real_holdout
          else "scoreboard (training samples — optimistic, see warning)\n")
    header = f"{'engine':<10} {'exact':>12} {'char acc':>10} {'median':>10}"
    print(header)
    print("-" * len(header))
    scores: Dict[str, dict] = {}

    # The learned engine must be scored from the freshly built bank, not from
    # whatever is currently on disk — otherwise every run would grade its own
    # previous homework.
    save_bank(args.out, bank)
    for name in EVALABLE:
        row = evaluate(name, held, args.charset)
        if not row.get("available", False):
            print(f"{name:<10} {'unavailable':>12}")
            continue
        scores[name] = row
        marker = " *" if name == "learned" else ""
        print(
            f"{name:<10} {row['exact']:>4}/{row['n']:<7} "
            f"{row['charRate'] * 100:>9.1f}% {row['medianMs']:>8.0f}ms{marker}"
        )

    if not scores:
        raise SystemExit("no engine could be scored")

    winner = max(scores.items(), key=lambda item: (item[1]["exactRate"], -item[1]["medianMs"]))
    if is_real_holdout:
        print(f"\nbest on held-out data: {winner[0]} ({winner[1]['exact']}/{winner[1]['n']})")
    else:
        print(f"\nbest on training data: {winner[0]} ({winner[1]['exact']}/{winner[1]['n']}) "
              "— not adopted")

    bank.meta.update(
        {
            "metrics": {name: {k: v for k, v in row.items() if k != "available"}
                        for name, row in scores.items()},
            # Only a genuinely held-out result may decide the deployment default.
            "preferred": winner[0] if is_real_holdout else None,
            "selfGraded": not is_real_holdout,
            "labels": len(dataset),
            "trainedOn": len(train),
            "holdout": len(held),
            "charset": args.charset,
            "tag": args.tag or "",
        }
    )
    save_bank(args.out, bank)
    if is_real_holdout:
        print(f"saved {args.out} — 'auto' will now prefer {winner[0]!r}")
    else:
        print(
            f"saved {args.out} — the model is usable, but 'auto' will keep its default\n"
            f"engine until a held-out run picks a winner. Use YSOLVER_BACKEND=learned\n"
            f"to force it, or label more captchas and re-run."
        )


if __name__ == "__main__":
    main()
