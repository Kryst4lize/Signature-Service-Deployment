# syntax=docker/dockerfile:1

# Public image by default so the build works from any clone. Inside the NVIDIA
# network, override in .env:
#   API_BASE_IMAGE=10.254.144.152/tessel/python:3.11-slim-bookworm
#   USE_INTERNAL_APT=1
ARG API_BASE_IMAGE=python:3.11-slim-bookworm

# ── Stage 1: dependency builder ───────────────────────────────────────────────
FROM ${API_BASE_IMAGE} AS builder

ARG USE_INTERNAL_APT=0
WORKDIR /build

# Both files are optional. The bracket glob matches zero files without failing
# the build, which is what lets a public-registry build run without the
# internal mirror config that only exists on NVIDIA machines.
COPY local-repository-confi[g] pip.con[f] /opt/netcfg/
RUN if [ "$USE_INTERNAL_APT" = "1" ] && [ -f /opt/netcfg/local-repository-config ]; then \
        cp /opt/netcfg/local-repository-config /etc/apt/sources.list; \
    fi; \
    if [ -f /opt/netcfg/pip.conf ]; then cp /opt/netcfg/pip.conf /etc/pip.conf; fi

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM ${API_BASE_IMAGE} AS runtime

ARG USE_INTERNAL_APT=0
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TZ=Asia/Ho_Chi_Minh

COPY local-repository-confi[g] /opt/netcfg/
RUN if [ "$USE_INTERNAL_APT" = "1" ] && [ -f /opt/netcfg/local-repository-config ]; then \
        cp /opt/netcfg/local-repository-config /etc/apt/sources.list; \
    fi

# poppler-utils backs pdf2image; libpq5 backs asyncpg; curl backs HEALTHCHECK.
RUN ln -snf "/usr/share/zoneinfo/$TZ" /etc/localtime \
    && echo "$TZ" > /etc/timezone \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        libpq5 \
        poppler-utils \
        curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

RUN groupadd -r appuser && useradd -r -g appuser -s /sbin/nologin appuser

WORKDIR /app
COPY --from=builder /install /usr/local
COPY app/ ./app/

RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -sf http://localhost:8080/health || exit 1

# No --reload here. uvicorn's Config.should_reload wins over --workers, so the
# previous `--workers 2 --reload` silently ran a single-process reloader in
# production and restarted the API whenever a mounted file changed.
# For hot reload during development use docker-compose.dev.yml.
CMD ["uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8080", \
     "--workers", "2", \
     "--loop", "uvloop", \
     "--http", "httptools"]
