# Training it with humans — how it works, and what it actually buys you

"Train it with humans" is the right instinct, and it is built in. But it deserves an
honest write-up, because the numbers are not what the pitch usually implies.

## The loop

```
    solve ──► capture ──► label ──► train ──► measure ──► maybe take over
              (auto)     (human)  (1 second) (holdout)     (only if it wins)
```

1. **Capture** (`YSOLVER_LEARN`, on by default). While solving, the worker keeps a copy
   of: every failure, every reading the model was unsure about
   (`YSOLVER_LEARN_CONFIDENCE`, default 0.85), and a random slice of ordinary traffic
   (`YSOLVER_LEARN_RATE`, default 0.15). Deduplicated by image hash. The pile is capped
   (5,000 pending) so a busy server cannot fill a disk.
2. **Label**. `GET /api/labels/pending` hands a sample and the model's guess to a person;
   `POST /api/labels` files what they typed, or `skip=1` if it is unreadable. The
   dashboard's **Teach it** tab is this workflow with a picture and a text box.
3. **Train**. `python scripts/train.py` cuts each labelled image into glyphs, builds a
   glyph bank, and writes `data/learned.npz`. About a second on a CPU — no GPU, no
   PyTorch, no pipeline.
4. **Measure**. The same run scores *every* engine on samples held out of training.
5. **Take over — only on evidence**. The winner is recorded in the model file, and
   `auto` prefers it. If the learned model loses, `auto` keeps using whatever won; a bad
   model can never silently degrade a working deployment.

If a model does not beat the alternatives, the trainer exits non-zero and tells you
why: too few labels, glyph-count mismatches (which mean segmentation, not labelling, is
the bottleneck), or simply that another engine reads your captchas better.

**Below about twelve labels nothing can be held out**, and the run says so in a warning
before printing a scoreboard that grades the model on its own training data. That
scoreboard is a smoke test, not evidence: the model is still saved (so you can force it
with `YSOLVER_BACKEND=learned`), but `preferred` stays unset and `auto` will not follow
it. A self-graded number must never be allowed to change a deployment's behaviour.

## What it actually buys you — measured

100 human labels, 80 fresh captchas never shown to any engine, generated corpus:

| Captcha style | ddddocr | template | learned (this loop) |
|---|---|---|---|
| Random rotation per glyph, heavy noise (hard) | **98.8 %** | 67.5 % | 57.5 % |
| One site, consistent serif render (rotation ±4°) | **100 %** | 86.2 % | 88.8 % |
| One site, consistent light-on-dark mono | **100 %** | 100 % | 100 % |
| One site, consistent but noisy (rotation ±14°) | **98.8 %** | 78.8 % | 78.8 % |

Read that honestly: **the learned engine beat the built-in font matcher on 2 of 4 styles
and never beat `ddddocr`** on this corpus.

Why: `ddddocr` was trained on millions of captchas that look exactly like the ones my
generator makes — DejaVu-rendered, warped, noisy text. There is no headroom left for a
200-glyph bank to find. The learned model also cannot generalise across rendering the way
a CNN can: with per-glyph rotation varying wildly, the same character looks different
every time, which is precisely what a rotation-invariant network handles and a
nearest-neighbour bank cannot.

Where the loop *does* pay off, and what the numbers above do not cover:

* **A renderer the general model has never seen.** A site with a bespoke font, unusual
  colour treatment, or an odd pipeline is out-of-distribution for ddddocr, and 200
  labelled samples of *that exact renderer* capture it cheaply. I could not reproduce
  that here — my generator only has DejaVu and Liberation to render with, both heavily
  represented in the general model's training data.
* **Knowing which engine to use.** The label set doubles as a benchmark on your traffic.
  The table above is exactly why `auto` now follows measurement instead of a hard-coded
  guess: on the light-on-dark style the humble `template` engine matches `ddddocr` at
  100 %, and someone paying per solve deserves to know that.
* **A floor under regressions.** The bank is a file. Commit it, and a captcha that used
  to be readable stays readable even if a dependency changes.

## When to invest more labels

The trainer tells you, per run:

* `usable samples X/Y` — if this is low, the **segmentation** is failing, not the human.
  More labels will not help; the images need a segmentation tweak for that site.
* `glyphs in bank` — below 40 the model is not used at all (`--min-glyphs`), because a
  thin bank is a confident liar.
* the scoreboard — compare `learned` against the winner. If it is behind by a wide
  margin on consistent-rendering captchas, your labels are probably misaligned: check a
  few in the dashboard, then look at segmentation.

## Honest summary

The loop is real, it trains in a second, and it will not make your deployment worse — it
only takes over when it measurably wins. On ordinary captchas like the ones I can
generate, the general model already wins, and this loop's value is the **measurement and
the engine choice**, plus a site-specific model that pays off on renderers the general
model has never seen.

If you want more than that — a model that generalises across wildly varying styles — the
honest answer is that it needs a trained CNN on thousands of your labels, not 200. That
is a different project, and ddddocr is already that model.
