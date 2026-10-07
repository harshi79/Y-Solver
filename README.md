# Y-Solver

**A free, self-hosted CAPTCHA solving API.** Same HTTP contract as the paid solving
services — `in.php` / `res.php` and `createTask` / `getTaskResult` — but it runs on your
own machine. No per-solve fee, no monthly minimum, no quota, no captcha images leaving
your network.

```bash
pip install -e ".[ddddocr]"     # or: pip install -e .   (works with zero model downloads)
ysolver                          # → http://localhost:8000
```

Then point your existing client at `http://localhost:8000` instead of
`https://api.2captcha.com`. That is the whole migration.

```bash
curl -s "http://localhost:8000/in.php" -d "key=demo" -d "method=post" -F "file=@captcha.png"
# OK|248173905

curl -s "http://localhost:8000/res.php?key=demo&action=get&id=248173905"
# OK|a7f3k     (CAPCHA_NOT_READY while it is still queued)
```

Measured on the built-in corpus (170×64 px, 5 characters, noise strokes, per-glyph
rotation, sine warp — reproduce it with `python scripts/make_samples.py` +
`python scripts/benchmark.py --dir …`):

| Engine | Install | Exact · normal | Exact · hard | Per-char · normal | Median |
|---|---|---|---|---|---|
| `ddddocr` (default when installed) | `pip install "ysolver[ddddocr]"` | **100/100** | **45/60** | 100 % | 27 ms |
| `template` (always available) | none — pure Pillow/NumPy | 58/100 | 2/60 | 85 % | 18 ms |
| `tesseract` | `apt install tesseract-ocr` + `pip install pytesseract` | *optional extra* | | | |

"hard" = noise 1.8, warp 1.5, ±32° per-glyph rotation. Two accuracy tricks are built in
and worth knowing about, because they are what turn near-misses into exact matches:
readings are **folded to the requested alphabet's letter case** (`M7JMW` → `m7jmw` when the
charset is lowercase), and when the model's top pick is not a character the caller can
accept, the image is **re-decoded with the alphabet constraint applied** instead of
returning a stray glyph.

That is the honest trade-off: a paid service has a person or a purpose-trained model
behind it and charges per solve. Y-Solver gives you a local model plus a font-matching
fallback for **$0.00 per solve**, and it tells you exactly how well it is doing on *your*
captchas via `scripts/benchmark.py --dir your_samples/`.

---

## Why this exists

Paid captcha-solving APIs are priced per 1 000 solves, bundle minimum top-ups, and require
you to upload the captcha images you scrape to a third party. For the common case —
distorted-text captchas on forms, low volumes, internal tools, test automation, CI —
that is a lot of money and a lot of data exposure for a problem that OCR mostly solves.

Y-Solver is the same API shape with a local engine behind it, so:

* **zero marginal cost** — the 10 000th solve costs exactly as much as the first;
* **no rate limits** — the only limit is your CPU, and you can raise `YSOLVER_WORKERS`;
* **keeps images in-house** — nothing is uploaded anywhere, ever (no telemetry at all);
* **same client code** — swap the base URL; parameters, `OK|text`, `CAPCHA_NOT_READY`,
  `ERROR_*` codes and the JSON task flow all behave as your client expects.

### Honest scope (please read)

Y-Solver reads **text-in-image captchas the client already has access to** — the classic
"type the characters you see" image. It is *not* a reCAPTCHA/hCaptcha/Turnstile bypass:

* Interactive challenges are refused with `ERROR_METHOD_NOT_SUPPORTED` and a message
  explaining why. No silent failures, no fake guesses.
* The intended uses are your own sites, internal tooling, accessibility workflows,
  load/regression testing, and low-volume automation of services whose terms allow it.
* Respect `robots.txt`, terms of service and local law. The authors do not condone
  using this to attack, spam or scrape against a site's wishes.
* `docs/scope.md` has the longer version, including what it would take to support
  interactive challenges (a human or a real browser session — not an OCR API).

---

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .                      # core: Pillow + NumPy + FastAPI, template engine ready
pip install -e ".[ddddocr]"           # recommended accuracy (downloads a small ONNX model once)
pip install -e ".[dev]"               # pytest + httpx + ruff
```

Requires Python 3.9+. No Redis, no Postgres, no Docker, no GPU — SQLite and threads.

### Docker

```bash
docker build -t ysolver .
docker run --rm -p 8000:8000 -e YSOLVER_API_KEYS=my-secret ysolver
```

### Run

```bash
ysolver                                   # console script
python -m ysolver                         # module form
uvicorn ysolver.api:create_app --factory # if you prefer driving uvicorn yourself
```

Open <http://localhost:8000> for the dashboard: live stats, engine availability, a
"generate a sample captcha and solve it" playground, copy-paste client snippets, and the
recent-jobs table.

---

## Configuration

Everything is environment variables (or a `.env` file you source yourself):

| Variable | Default | Meaning |
|---|---|---|
| `YSOLVER_API_KEYS` | `demo` | Comma-separated keys accepted. Change this before exposing the port. |
| `YSOLVER_REQUIRE_KEY` | `1` | `0` accepts any (or no) key — handy on a trusted LAN. |
| `YSOLVER_BACKEND` | `auto` | `auto`, `ddddocr`, `tesseract` or `template`. `auto` picks the best available. |
| `YSOLVER_WORKERS` | `4` | Solver threads. Set to your vCPU count for throughput. |
| `YSOLVER_DB` | `data/ysolver.db` | SQLite file for jobs/results. |
| `YSOLVER_RESULT_TTL` | `1800` | Seconds a finished result stays retrievable. |
| `YSOLVER_MAX_QUEUE` | `500` | Jobs waiting; beyond this `/in.php` answers `ERROR_NO_SLOT_AVAILABLE`. |
| `YSOLVER_SOLVE_TIMEOUT` | `30` | Seconds before a stuck job is failed by the watchdog. |
| `YSOLVER_CHARSET` | `abcdefghjkmnpqrstuvwxyz23456789` | Default alphabet (per-request `charset` overrides). |
| `YSOLVER_BALANCE` | `9999` | Number reported by `action=getbalance`. |
| `YSOLVER_HOST` / `YSOLVER_PORT` | `0.0.0.0` / `8000` | Bind address. |
| `YSOLVER_LOG_LEVEL` | `info` | `debug` shows per-job timings. |
| `YSOLVER_DASHBOARD` | `1` | Serve the dashboard at `/`. |
| `YSOLVER_LEARN` | `1` | Queue interesting captchas for human labelling. |
| `YSOLVER_LEARN_CONFIDENCE` | `0.85` | Readings below this confidence are always queued. |
| `YSOLVER_LEARN_RATE` | `0.15` | Fraction of ordinary confident traffic that is sampled. |
| `YSOLVER_MODEL` | `data/learned.npz` | Where the trained model lives. |

---

## API

### Classic form API (drop-in)

| Call | Behaviour |
|---|---|
| `POST /in.php` with `key`, `method=post` and a file field (`file`, `image` or `captcha`) | `OK\|<id>` |
| `POST /in.php` with `key`, `method=base64`, `body=<base64>` (data URLs and URL-safe base64 OK) | `OK\|<id>` |
| `POST /in.php` with a raw image body | `OK\|<id>` |
| `GET /res.php?key=…&action=get&id=…` | `OK\|<text>`, `CAPCHA_NOT_READY`, or `ERROR_*` |
| `GET /res.php?key=…&action=getbalance` | `OK\|$9999.00` |
| `GET /res.php?key=…&action=reportbad&id=…` | `OK\|OK` (accepted, no-op) |

Honoured parameters: `numeric=1`, `min_len`, `max_len`, `charset`, `delay` (scheduled
jobs, ≥100 s — no premium tier here, it is just a timer), `comment` (kept for
compatibility, ignored), `textinstructions` (accepted, ignored).

`method` values treated as image captchas: `post`, `base64`, `file`, `image`,
`imagetotext`. Everything else that names an interactive challenge is refused with
`ERROR_METHOD_NOT_SUPPORTED`.

### JSON task API (drop-in)

```jsonc
POST /createTask
{"clientKey": "demo",
 "task": {"type": "ImageToTextTask", "body": "<base64>", "numeric": 1}}
→ {"errorId": 0, "taskId": "248173905"}

POST /getTaskResult
{"clientKey": "demo", "taskId": "248173905"}
→ {"errorId": 0, "status": "processing"}
→ {"errorId": 0, "status": "ready",
   "solution": {"text": "a7f3k", "confidence": 0.98, "backend": "ddddocr", "solveMs": 27}}
```

Flat bodies (`{"clientKey": …, "type": "ImageToTextTask", "body": …}`) work too, since
clients in the wild are inconsistent about nesting.

### Native helpers

| Endpoint | Purpose |
|---|---|
| `GET\|POST /api/solve` | One-shot: upload and receive the text in the same call (`wait=false` → `202` + id) |
| `GET /api/sample?length=5` | Generate a test captcha **with its ground-truth label** |
| `GET /api/jobs` · `GET /api/jobs/{id}` | Recent jobs / one job |
| `GET /api/stats` · `GET /healthz` | Counters, queue depth, engine availability |
| `GET /api/labels/pending` · `POST /api/labels` · `GET /api/labels/stats` | Human labelling queue and model status |
| `GET /api/scope` | Machine-readable statement of what this server will and will not solve |
| `GET /docs` | Interactive OpenAPI schema |

### Error codes

Faithful to the paid services: `ERROR_WRONG_USER_KEY`, `ERROR_ZERO_BALANCE`,
`ERROR_NO_SLOT_AVAILABLE`, `ERROR_WRONG_CAPTCHA_ID`, `ERROR_BAD_PARAMETERS`,
`ERROR_WRONG_FILE_EXTENSION`, `ERROR_IMAGE_TYPE_NOT_SUPPORTED`,
`ERROR_CAPTCHA_UNSOLVABLE`, `ERROR_METHOD_NOT_SUPPORTED`, `ERROR_BAD_ACTION`,
`ERROR_INTERNAL`.

---

## Engines

`auto` tries these in order and uses the first that is available:

1. **`ddddocr`** — a compact ONNX model that runs offline on CPU. Recommended; it is the
   accuracy you want on real distorted captchas.
2. **`tesseract`** — your local Tesseract binary with a character whitelist and PSM
   sweeps. Decent on clean captchas, and pleasantly boring to operate.
3. **`template`** — always available, zero downloads: Otsu binarisation, morphological
   opening to erase anti-bot lines, connected-component segmentation (union-find over
   pixel runs), then cosine matching against every character in the charset rendered in
   the machine's fonts at five rotations. ~85 % per-character accuracy on ordinary
   captchas, and it collapses on heavy warping — that is what `ddddocr` is for.

4. **`learned`** — trained from your own human labels (see below). Off until a model
   exists, and only preferred by `auto` when it beat the others on held-out data.

Adding an engine is ~50 lines: subclass `Engine`, implement `available()` and
`solve(prepared, charset)`, and register it in `ysolver/solvers/registry.py`.

## Performance

Measured on a small cloud vCPU, single process, 4 worker threads:

| | ddddocr | template |
|---|---|---|
| Engine only, median | **10.0 ms** | 4 ms |
| Engine only, p95 | 11.1 ms | 6 ms |
| Full HTTP round trip (`/api/solve`), median | **12.0 ms** | 6 ms |
| Throughput, 16 clients · 4 workers | **84 captchas/s** | ~250/s |

Two things bought most of that: preprocessing is **lazy**, so an engine that reads the
original bitmap never pays for the autocontrast/deskew pipeline it does not use (that was
10.1 ms of a 23 ms solve, of which `deskew` alone was 8.9 ms); and `/api/solve` waits on
an event the worker sets instead of polling, which removed a 50 ms quantum per request.
Scaling out is horizontal: run several instances behind a load balancer, or simply
`YSOLVER_WORKERS=$(nproc)`.

## Teaching it your captchas (optional)

Y-Solver can learn the captcha style of one specific site from human labels — your own
team's, on your own traffic. Nothing is uploaded anywhere and there is no crowd.

1. Solve captchas as usual. Failures, unsure readings and a random sample of the rest are
   queued automatically (set `YSOLVER_LEARN=0` to switch this off).
2. Label them in the dashboard's **Teach it** tab — or `GET /api/labels/pending` and
   `POST /api/labels` if you would rather script it.
3. Train and measure:

```bash
python scripts/train.py --holdout 0.25
```

It builds `data/learned.npz`, scores **every** engine on held-out samples, prints a
scoreboard, and only then lets `auto` prefer the learned model — so a model that loses
cannot silently degrade a working deployment. Below ~12 labels nothing can be held out;
the run warns that the scoreboard is self-graded and refuses to change the default engine.

Honest expectations, measured on 100 labels and 80 unseen captchas (full write-up and
method in [`docs/training.md`](docs/training.md)):

| Captcha style | ddddocr | template | learned |
|---|---|---|---|
| Random rotation per glyph, heavy noise | **98.8 %** | 67.5 % | 57.5 % |
| One site, consistent rendering | **100 %** | 86.2 % | 88.8 % |

The learned engine beat the font matcher but did *not* beat `ddddocr` — the general model
was trained on millions of captchas just like my generator produces, so there was no
headroom left. It pays off on renderers the general model has never seen, and its real
value is the measurement: on one style the humble `template` engine equalled `ddddocr` at
100 %, and you deserve to know that before paying anyone per solve.

## Testing

```bash
pip install -e ".[dev]"
pytest                                   # 154 tests, no network, ~7 s
python scripts/benchmark.py --count 60   # accuracy/latency on a generated corpus
python scripts/make_samples.py data/samples --count 25
```

The test-suite covers the wire protocol (all three client styles), the job store, the
worker pool (queue limits, scheduled jobs, watchdog), preprocessing and segmentation, and
runs the real template engine against generated captchas.

## Project layout

```
ysolver/
  api.py            FastAPI app: form API, task API, native API, dashboard mount
  protocol.py       request parsing / compatibility glue (the messy part, isolated)
  worker.py         thread pool, scheduled queue, watchdog
  store.py          SQLite job store
  preprocess.py     decode, Otsu, denoise, deskew
  synth.py          synthetic captcha generator (tests, benchmarks, /api/sample)
  labels.py         human-labelling queue and dataset
  scope.py          what this server will and will not solve, stated once
  solvers/          engines: ddddocr, learned, tesseract, template + segmentation
  static/index.html dashboard (no build step, no CDN)
tests/              154 tests
scripts/            benchmark.py, make_samples.py
examples/           form API, task API and shell clients
docs/               scope.md (ethics + limits), training.md, api-compat.md, deploy.md
```

## FAQ

**Is this really the same as a paid solver?**
For readable text captchas, functionally yes — same endpoints, same responses. On the
hardest captchas (heavy occlusion, cursive handwriting, interactive challenges) a paid
service still wins because it has better models or humans. Y-Solver is honest about that
instead of guessing: unreadable images come back as `ERROR_CAPTCHA_UNSOLVABLE`.

**Why does it refuse reCAPTCHA/hCaptcha?**
Because those are not images — solving them requires a token from an interactive browser
session (a human, or a farm/browser-automation setup). An OCR API that claims to solve
them is either lying, or doing something you should not be comfortable with. See
`docs/scope.md`.

**Can it solve my captcha?**
Test it: generate samples with `scripts/make_samples.py`, drop in a few of your real
images with a `labels.tsv` manifest, run `scripts/benchmark.py --dir …`. If accuracy is
too low, tune `charset`/`numeric` (a smaller alphabet helps a lot) or add a font template
for the target site.

**Can I train it on my own captchas?**
Yes — that is the **Teach it** tab plus `scripts/train.py`, and it trains in about a
second. Read [`docs/training.md`](docs/training.md) first: on ordinary captchas the
general model usually already wins, and the loop's most reliable payoff is telling you
which engine is best for *your* traffic.

**Does it phone home?**
No. There is no telemetry, no licence check, no model download at runtime (only the
`ddddocr` wheel's bundled model, installed by pip). Images are processed in memory and
only the recognised text is stored, unless you set `YSOLVER_STORE_IMAGES=1`.

## License

MIT — see [LICENSE](LICENSE).
