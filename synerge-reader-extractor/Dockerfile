# Private SynergeReader extraction worker: CPU only, internal network only, no runtime downloads.
# Build context: synerge-reader-extractor/. models/ must first be staged with tools/stage_models.py;
# the build verifies every model file against MODEL_LOCK.json and every wheel against the hash lock.
FROM python:3.11.16-slim-bookworm@sha256:a36c24f9cbdf4fd0f52d67f0823eeac19c2028c637cecc392d97f980d4fec56b

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_INPUT=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    DO_NOT_TRACK=1 \
    HOME=/tmp/home \
    HF_HOME=/tmp/hf \
    XDG_CACHE_HOME=/tmp/cache \
    TORCH_HOME=/tmp/torch \
    OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2 \
    OPENBLAS_NUM_THREADS=2 \
    TESSDATA_PREFIX=/models/tessdata/ \
    EXTRACTOR_MODEL_DIR=/models \
    EXTRACTOR_TESSDATA_DIR=/models/tessdata \
    EXTRACTOR_WORK_DIR=/tmp/extractor-jobs

WORKDIR /app

COPY locks/requirements-linux-x86_64.lock locks/requirements-linux-aarch64.lock /app/locks/
RUN set -eu; \
    arch="$(uname -m)"; \
    case "$arch" in x86_64|aarch64) ;; *) echo "unsupported architecture: $arch" >&2; exit 1 ;; esac; \
    pip install --no-deps --require-hashes --only-binary=:all: \
        --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cpu \
        -r "/app/locks/requirements-linux-${arch}.lock"; \
    pip check

COPY MODEL_LOCK.json /app/MODEL_LOCK.json
COPY extractor_service /app/extractor_service
COPY models /models

RUN set -eu; \
    python -m extractor_service.model_check /models /app/MODEL_LOCK.json; \
    useradd --system --uid 10001 --home-dir /nonexistent --shell /usr/sbin/nologin extractor; \
    chmod -R a+rX,go-w /models /app

USER 10001:10001
EXPOSE 8090
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8090/healthz', timeout=4)"]
CMD ["python", "-m", "uvicorn", "extractor_service.app:build_app_from_env", "--factory", \
     "--host", "0.0.0.0", "--port", "8090", "--workers", "1", "--no-access-log", "--timeout-graceful-shutdown", "15"]
