#!/usr/bin/env python
"""Benchmark every available engine on a generated captcha corpus.

    python scripts/benchmark.py --count 60
    python scripts/benchmark.py --engines ddddocr template --length 6

Reports exact-match rate, per-character accuracy and median latency, which is
the honest way to answer "how good is the free version?" on *your* captchas.
Point ``--dir`` at real samples (``labels.tsv`` from ``make_samples.py``) once
you have them.
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from typing import List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ysolver.config import DEFAULT_CHARSET, Settings
from ysolver.solvers import CaptchaSolver, SolverError, engine_status
from ysolver.synth import batch, to_png


def load_corpus(args) -> List[Tuple[bytes, str]]:
    if args.dir:
        manifest = os.path.join(args.dir, "labels.tsv")
        if not os.path.exists(manifest):
            raise SystemExit(f"{manifest} not found — run scripts/make_samples.py first")
        corpus = []
        with open(manifest, encoding="utf-8") as handle:
            for line in handle:
                name, _, label = line.strip().partition("\t")
                if not name:
                    continue
                with open(os.path.join(args.dir, name), "rb") as image:
                    corpus.append((image.read(), label))
        return corpus
    return [
        (to_png(image), text)
        for image, text in batch(
            count=args.count,
            length=args.length,
            charset=args.charset,
            seed=args.seed,
            noise=args.noise,
            warp=args.warp,
        )
    ]


def run_engine(name: str, corpus, charset: str) -> dict:
    solver = CaptchaSolver(Settings(backend=name, charset=charset))
    exact = correct = total = 0
    latencies: List[float] = []
    failures = 0
    for data, truth in corpus:
        started = time.perf_counter()
        try:
            text = solver.solve(data).text
        except SolverError:
            text, failures = "", failures + 1
        latencies.append((time.perf_counter() - started) * 1000)
        exact += text == truth
        for index in range(min(len(text), len(truth))):
            correct += text[index] == truth[index]
        total += len(truth)
    return {
        "engine": name,
        "n": len(corpus),
        "exact": exact,
        "exact_rate": exact / max(1, len(corpus)),
        "char_rate": correct / max(1, total),
        "median_ms": statistics.median(latencies) if latencies else 0.0,
        "p95_ms": sorted(latencies)[int(len(latencies) * 0.95) - 1] if latencies else 0.0,
        "errors": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=40, help="generated samples (default 40)")
    parser.add_argument("--length", type=int, default=5, help="characters per captcha")
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--noise", type=float, default=1.0, help="clutter intensity")
    parser.add_argument("--warp", type=float, default=1.0, help="distortion intensity")
    parser.add_argument("--charset", default=DEFAULT_CHARSET)
    parser.add_argument("--dir", help="use images from this directory instead of generating")
    parser.add_argument("--engines", nargs="*", help="subset of engines to test")
    args = parser.parse_args()

    status = {engine["name"]: engine for engine in engine_status()}
    wanted = args.engines or ["ddddocr", "tesseract", "template"]
    corpus = load_corpus(args)

    print(f"\nY-Solver benchmark — {len(corpus)} captchas, {args.length} chars, "
          f"charset {args.charset!r}\n")
    header = f"{'engine':<10} {'exact':>12} {'char acc':>10} {'median':>10} {'p95':>9} {'errors':>7}"
    print(header)
    print("-" * len(header))

    rows = []
    for name in wanted:
        info = status.get(name)
        if info is None:
            continue
        if not info["available"]:
            print(f"{name:<10} {'unavailable':>12}   {info.get('installHint', '')}")
            continue
        row = run_engine(name, corpus, args.charset)
        rows.append(row)
        print(
            f"{name:<10} {row['exact']:>4}/{row['n']:<7} "
            f"{row['char_rate'] * 100:>9.1f}% {row['median_ms']:>8.0f}ms "
            f"{row['p95_ms']:>7.0f}ms {row['errors']:>7}"
        )

    print()
    if rows:
        best = max(rows, key=lambda row: row["exact_rate"])
        print(f"Best exact-match: {best['engine']} at {best['exact_rate'] * 100:.1f}%")
        print("Cost per solve: $0.00 — it runs on this machine.\n")


if __name__ == "__main__":
    main()
