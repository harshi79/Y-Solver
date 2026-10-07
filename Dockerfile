# Y-Solver — free, self-hosted CAPTCHA solving API.
#
#   docker build -t ysolver .
#   docker run --rm -p 8000:8000 -e YSOLVER_API_KEYS=change-me ysolver
#
# The ddddocr extra is installed by default here (it is the accuracy you want);
# build with --build-arg WITH_DDDDOCR=0 for a slim image with the built-in
# template engine only.
FROM python:3.11-slim

ARG WITH_DDDDOCR=1

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    YSOLVER_HOST=0.0.0.0 \
    YSOLVER_PORT=8000 \
    YSOLVER_DB=/data/ysolver.db

WORKDIR /app

# Fonts for the template engine, plus what ddddocr needs to import cleanly.
RUN apt-get update \
 && apt-get install -y --no-install-recommends fonts-dejavu-core \
 && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md LICENSE ./
COPY ysolver ./ysolver

RUN pip install --no-cache-dir --upgrade pip \
 && if [ "$WITH_DDDDOCR" = "1" ]; then \
      pip install --no-cache-dir ".[ddddocr]"; \
    else \
      pip install --no-cache-dir .; \
    fi

VOLUME ["/data"]
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request,sys; \
sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/healthz').status == 200 else 1)"

CMD ["ysolver"]
