VENV ?= .venv
PY   ?= $(VENV)/bin/python

.PHONY: help install dev test lint bench samples run docker clean

help:            ## list targets
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | sed 's/:.*##/\t/' | column -t -s "$$(printf '\t')"

install:         ## venv + core deps (template engine only)
	python3 -m venv $(VENV)
	$(PY) -m pip install -q --upgrade pip
	$(PY) -m pip install -e .

dev:             ## venv + dev deps + ddddocr (recommended for real captchas)
	python3 -m venv $(VENV)
	$(PY) -m pip install -q --upgrade pip
	$(PY) -m pip install -e ".[dev,ddddocr]"

test:            ## run the test suite
	$(PY) -m pytest

lint:            ## ruff
	$(PY) -m ruff check ysolver tests scripts

bench:           ## accuracy/latency benchmark on a generated corpus
	$(PY) scripts/benchmark.py --count 60

samples:         ## write 25 labelled sample captchas to data/samples
	$(PY) scripts/make_samples.py data/samples --count 25

run:             ## start the server on :8000
	$(PY) -m ysolver

docker:          ## build the container image
	docker build -t ysolver .

clean:           ## remove caches and local data
	rm -rf $(VENV) data .pytest_cache .ruff_cache **/__pycache__
