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

# Strip the base image's system pip/setuptools/wheel.
#
# The service runs entirely from /opt/venv, which uv builds without pip and
# which does not enable system-site-packages — so nothing here is reachable at
# runtime. It is still in the image, and it is still the only part of the image
# with CVEs anyone can actually act on: of the 80 findings the container scan
# reports, exactly two carry a fixed version, and both live here —
# CVE-2026-24049 (wheel) and CVE-2026-23949 (jaraco.context, vendored inside
# setuptools). Every Debian-layer finding is will_not_fix / fix_deferred with no
# patched package to move to.
#
# Removing it also takes a package manager out of a production container, which
# is worth doing on its own.
# distutils-precedence.pth is removed with them on purpose: it outlives the
# packages and then fails to import `_distutils_hack` on every interpreter
# start, printing a ModuleNotFoundError to stderr from `python`, `alembic` and
# anything else that runs in this image.
RUN for pkg in pip setuptools wheel pkg_resources _distutils_hack; do \
        rm -rf /usr/local/lib/python*/site-packages/"$pkg"*; \
    done \
    && rm -f /usr/local/bin/pip /usr/local/bin/pip3 /usr/local/bin/pip3.* \
    && rm -f /usr/local/lib/python*/site-packages/distutils-precedence.pth

# Assert the interpreter starts silently. Deliberately its own RUN with no
# pipe: piping into grep needs `SHELL -o pipefail` to be meaningful (DL4006),
# and `A && B || C` is not if-then-else (SC2015).
RUN startup="$(python -c 'print("ok")' 2>&1)"; \
    if [ "$startup" != "ok" ]; then \
        echo "interpreter startup is not clean: $startup"; \
        exit 1; \
    fi

RUN groupadd -r appuser && useradd -r -g appuser -s /sbin/nologin appuser

WORKDIR /app
# The venv lives outside /app so the bind mount in docker-compose.dev.yml
# cannot shadow it.
COPY --from=builder /opt/venv /opt/venv
COPY app/ ./app/
# The compose `migrate` service runs `alembic upgrade head` from this image.
# Without these it fails with "No config file 'alembic.ini' found" — which the
# test suite does not catch, because it bind-mounts the source tree instead.
COPY alembic.ini ./
COPY migrations/ ./migrations/

RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 8080

# Exec form: no shell per probe. `curl -sf` already exits non-zero on a
# connection failure (7) or an HTTP >= 400 (22), which is all HEALTHCHECK needs.
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD ["curl", "-sf", "http://localhost:8080/health"]

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
