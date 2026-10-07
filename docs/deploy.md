# Deploying Y-Solver

Y-Solver is a single Python process with a SQLite file next to it. Anything that can run
`ysolver` can host it: a container, a systemd unit, a Raspberry Pi, a laptop.

## Docker

```bash
docker build -t ysolver .
docker run -d --name ysolver -p 8000:8000 \
  -e YSOLVER_API_KEYS=change-me \
  -e YSOLVER_WORKERS=4 \
  -v ysolver-data:/data \
  -e YSOLVER_DB=/data/ysolver.db \
  ysolver
```

`docker-compose.yml` in the repository root does the same with the API key read from the
environment.

## systemd

```ini
# /etc/systemd/system/ysolver.service
[Unit]
Description=Y-Solver captcha API
After=network.target

[Service]
WorkingDirectory=/opt/ysolver
Environment=YSOLVER_API_KEYS=change-me
Environment=YSOLVER_WORKERS=4
Environment=YSOLVER_DB=/var/lib/ysolver/jobs.db
ExecStart=/opt/ysolver/.venv/bin/ysolver
Restart=always
User=ysolver

[Install]
WantedBy=multi-user.target
```

## Sizing and throughput

Work is CPU-bound (OCR), so scale with cores:

* `YSOLVER_WORKERS = number of vCPUs` is a good starting point.
* Rough per-captcha cost: `template` ~17 ms, `ddddocr` ~28 ms on a small cloud vCPU.
  Four workers therefore serve roughly 140–230 captchas per second in aggregate.
* For more than one CPU-bound process, run several containers behind a load balancer and
  give each its own database file (or one shared file — SQLite with WAL handles a handful
  of processes, but per-instance files are simpler).
* Memory: ~150 MB resident with `ddddocr` loaded (the ONNX model is ~10 MB), ~60 MB for
  the template engine alone.

## Tuning accuracy

1. **`charset`** is the cheapest win. A 5-character lowercase alphabet beats a 62-character
   one by a wide margin. Use `numeric=1` for digit-only captchas.
2. **`min_len` / `max_len`** let the caller pin the length; the server uses them to pad or
   trim, which turns near-misses into exact matches.
3. **Fonts** matter for the template engine: install the font family your target site uses
   (`apt install fonts-roboto`, or drop TTFs in `~/.fonts`) and the engine picks them up on
   the next start.
4. **`YSOLVER_BACKEND=ddddocr`** when you need the best accuracy and can afford 28 ms.
5. **Benchmark on your own traffic** — this is the only number that matters:

   ```bash
   # collect N real captchas, then label them into labels.tsv ("name<TAB>text" per line)
   python scripts/benchmark.py --dir my_samples --engines ddddocr template
   ```

## Operability

* `GET /healthz` — liveness plus which engines are usable. Wire it to your orchestrator.
* `GET /api/stats` — totals, average solve time, queue depth, last-hour volume.
* `GET /` — the dashboard, useful for a human checking a deployment. Set
  `YSOLVER_DASHBOARD=0` to disable it entirely on a public host.
* Logs go to stdout (`YSOLVER_LOG_LEVEL=debug` adds per-job timings). Nothing is written
  anywhere else.

## Security notes

* Change `YSOLVER_API_KEYS` before exposing the port, and put the service behind TLS if it
  leaves localhost. The keys are simple bearer tokens, not signed credentials.
* The API is unauthenticated when `YSOLVER_REQUIRE_KEY=0` — only do that on a trusted
  network.
* `YSOLVER_STORE_IMAGES` is off by default: images are never written to disk. If you turn
  it on for debugging, remember they are the captchas your clients were solving.
* The server makes no outbound connections, so it is safe on an isolated network; there is
  nothing to egress-allow.
