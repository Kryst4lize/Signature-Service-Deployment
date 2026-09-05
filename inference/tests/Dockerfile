# Test runner. Same locked dependency set as the service image plus the dev
# group; no application code is baked in, the tree is bind-mounted at run time.
#
# Build context is inference/, so both api/ and this file are reachable:
#   docker build -f tests/Dockerfile .
ARG API_BASE_IMAGE=python:3.11-slim-bookworm
ARG UV_VERSION=0.11.21

FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv

FROM ${API_BASE_IMAGE}
COPY --from=uv /uv /usr/local/bin/uv

RUN apt-get update \
    && apt-get install -y --no-install-recommends libpq5 poppler-utils \
    && rm -rf /var/lib/apt/lists/*

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

# Same uv.lock as the runtime image, so the tests run against exactly the
# versions that ship — plus the dev group.
WORKDIR /build
COPY api/pyproject.toml api/uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project

# `app` is imported from the bind-mounted source at /w/api.
ENV PYTHONPATH=/w/api
WORKDIR /w
