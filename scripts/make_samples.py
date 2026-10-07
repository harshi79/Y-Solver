#!/usr/bin/env python
"""Write a labelled dataset of generated captchas.

    python scripts/make_samples.py data/samples --count 25

Produces ``captcha_0000.png`` … plus ``labels.tsv`` (used by
``scripts/benchmark.py --dir``). Swap in real captchas later: keep the same
two-column manifest and the benchmark keeps working.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ysolver.config import DEFAULT_CHARSET
from ysolver.synth import write_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", default="data/samples")
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--length", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--noise", type=float, default=1.0)
    parser.add_argument("--warp", type=float, default=1.0)
    parser.add_argument("--charset", default=DEFAULT_CHARSET)
    parser.add_argument(
        "--difficulty", choices=("easy", "normal", "hard"), default="normal",
        help="preset for noise/warp/rotation",
    )
    args = parser.parse_args()

    presets = {
        "easy": dict(noise=0.2, warp=0.4, rotation=8.0),
        "normal": dict(noise=1.0, warp=1.0, rotation=22.0),
        "hard": dict(noise=1.8, warp=1.5, rotation=32.0),
    }
    rows = write_dataset(
        args.directory,
        count=args.count,
        length=args.length,
        charset=args.charset,
        seed=args.seed,
        noise=args.noise if args.noise != 1.0 else presets[args.difficulty]["noise"],
        warp=args.warp if args.warp != 1.0 else presets[args.difficulty]["warp"],
        rotation=presets[args.difficulty]["rotation"],
    )
    print(f"wrote {len(rows)} captchas to {os.path.abspath(args.directory)}")
    print("manifest:", os.path.join(args.directory, "labels.tsv"))
    print("benchmark:", f"python scripts/benchmark.py --dir {args.directory}")


if __name__ == "__main__":
    main()
