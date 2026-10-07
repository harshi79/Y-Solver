# Scope, limits and ethics

This document exists because "free captcha solver" is a phrase that attracts both
legitimate automation engineers and people looking for something else. Here is exactly
what Y-Solver does, what it refuses to do, and why.

## What it solves

**Text-in-image captchas.** The classic "type the characters shown" image that the
requesting client already has in hand: a login form, a ticket system, an internal tool, a
bank of scanned faxes, a test harness you own. Concretely:

* distorted alphanumeric images (the main case), including noise strokes, arcs, speckles,
  per-glyph rotation and sine warping;
* numeric-only codes (`numeric=1` restricts the alphabet and measurably improves
  accuracy);
* arbitrary alphabets via `charset=…` — the smaller and more predictable the alphabet,
  the better the accuracy.

## What it does not solve, and why

| Challenge | Why not |
|---|---|
| reCAPTCHA v2/v3, hCaptcha, Cloudflare Turnstile | The image is only the *hint*; the answer is a token produced by an interactive browser session with proprietary scoring. There is nothing to read out of a bitmap. Solving these requires a human (a "click farm") or a browser-automation farm. |
| GeeTest, FunCaptcha/Arkose, DataDome sliders | Same: stateful interactive challenge, not an image. |
| Audio captchas | Not an image; would be a speech model (feasible, just not implemented). |
| "Please solve the math problem" prompts | A text puzzle, not an image — out of scope, and trivially solved by the caller anyway. |

When a client asks for one of these, Y-Solver answers
`ERROR_METHOD_NOT_SUPPORTED` with a message explaining the situation, plus a
`suggestions` array with the practical alternative (usually: use the vendor's official
test keys in development, and let the user's own browser solve the challenge in
production). `GET /api/scope` returns the same capability statement as JSON, so a
pipeline can check what a deployment supports before sending it work.

The alternative — returning some random string with `OK|` — would corrupt your pipeline
and hide the fact that the job was never solvable. Failures are loud on purpose.

## Why the paid solvers are not a scam

They cost money because a human or a purpose-trained model sits behind the API. Where the
work is human labour, the price is the point: it pays somebody. Y-Solver replaces that
with a local model that is *good enough* on plain captchas and honest about the rest.

It also means Y-Solver will not beat a paid service on the hardest images. If you need
99 % on adversarial captchas, keep paying; if you need 80–95 % on ordinary ones for free,
you are in the right place.

## Intended use

* Your own applications and internal tools.
* Accessibility: helping a user who cannot read a distorted image get past a form on a
  site they are entitled to use.
* Load, regression and QA testing of systems you are authorised to test.
* Low-volume automation of services whose terms of service permit it.
* Research and teaching on OCR, segmentation and classical computer vision.

## Not intended use

* Spamming, credential stuffing, scalping, or evading rate limits on third-party
  services.
* Creating accounts in bulk where doing so violates the target's terms.
* Any use that requires defeating an anti-bot measure the site deployed specifically to
  stop automation. If a site asks humans to prove they are human, respect it or find an
  official API.

Captcha solving exists on a legal and ethical spectrum that varies by jurisdiction and by
site. The software is provided under the MIT licence, without warranty: what you point it
at is your responsibility. See `LICENSE`.

## Data handling

* Images are processed in memory only. Nothing is written to disk by default
  (`YSOLVER_STORE_IMAGES=1` changes that).
* The database stores the job id, timings, backend name and the recognised text, and
  results are deleted after `YSOLVER_RESULT_TTL` (default 30 minutes).
* The API key is never stored — only a masked hint (`de***mo`) appears in listings.
* There is no telemetry, no update check and no outbound network call at any point.
  Y-Solver works with the network cable unplugged.

## Development without solving anything

For reCAPTCHA v2 specifically, Google publishes test keys that always pass
verification: site key `6LeIxAcTAAAAAJcZVRqyHh71UMIEGNQ_MXjiZKhI`, secret key
`6LeIxAcTAAAAAGG-vFI1TnRWxMZNFuojJ4WifJWe`. The widget shows a "testing only" banner and
every verification succeeds, so you can exercise a signup form end to end without
automating anything. Use a separate production key, and let real users solve the real
challenge in their own browser.

## Reporting

Security or misuse concerns: open an issue (or a private security advisory) on the
repository. Please include a reproducer if the problem is technical.
