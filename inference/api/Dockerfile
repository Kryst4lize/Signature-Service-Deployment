# syntax=docker/dockerfile:1

# Public image by default so the build works from any clone. Inside the NVIDIA
# network, override in .env:
#   API_BASE_IMAGE=10.254.144.152/tessel/python:3.11-slim-bookworm
#   USE_INTERNAL_APT=1
ARG API_BASE_IMAGE=python:3.11-slim-bookworm
# Pinned: uv is part of the build toolchain, so it is versioned like everything
# else rather than floating on :latest.
ARG UV_VERSION=0.11.21

# ── Stage 1: dependency builder ───────────────────────────────────────────────
FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv

FROM ${API_BASE_IMAGE} AS builder
COPY --from=uv /uv /usr/local/bin/uv

ARG USE_INTERNAL_APT=0
WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    # Use the base image's interpreter rather than fetching a second one;
    # pyproject.toml requires ==3.11.*, which python:3.11-slim satisfies.
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv

# Both files are optional. The bracket glob matches zero files without failing
# the build, which is what lets a public-registry build run without the
# internal mirror config that only exists on NVIDIA machines.
COPY local-repository-confi[g] pip.con[f] /opt/netcfg/
RUN if [ "$USE_INTERNAL_APT" = "1" ] && [ -f /opt/netcfg/local-repository-config ]; then \
        cp /opt/netcfg/local-repository-config /etc/apt/sources.list; \
    fi; \
    if [ -f /opt/netcfg/pip.conf ]; then cp /opt/netcfg/pip.conf /etc/pip.conf; fi

# Dependencies first, without the project, so editing app/ does not invalidate
# the dependency layer. --frozen fails rather than silently re-resolving if
# uv.lock is out of date with pyproject.toml.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

COPY app/ ./app/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev


# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM ${API_BASE_IMAGE} AS runtime

ARG USE_INTERNAL_APT=0
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TZ=Asia/Ho_Chi_Minh \
    PATH="/opt/venv/bin:$PATH"

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
# The venv lives outside /app so the bind mount in docker-compose.dev.yml
# cannot shadow it.
COPY --from=builder /opt/venv /opt/venv
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
