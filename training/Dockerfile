# syntax=docker/dockerfile:1

# GPU training image.
#
# Default is a public CUDA base so the image builds from any clone. On the
# NVIDIA network, override in .env:
#   TRAINING_BASE_IMAGE=10.254.144.152/tessel/training-service-base:0.0.1
#   USE_INTERNAL_APT=1
# NOTE on the CUDA stack, verified while upgrading torch to 2.6.0:
#   * torch brings its own CUDA via the nvidia-* wheels (now cu124, cudnn
#     9.1.0.70), so it does not use this base image's CUDA at all — the base
#     only has to be new enough for the host driver.
#   * TensorFlow is installed WITHOUT its `and-cuda` extra, so it looks for
#     system CUDA. TF 2.21 wants cudnn >= 9.3.0.75 and this base ships cudnn 8,
#     so TF very likely falls back to CPU here. That predates this upgrade —
#     it was already true with torch 2.3 — and is untested because there is no
#     GPU on hand. If you need GPU-accelerated Keras training, either move to a
#     cudnn 9.3+ base or install `tensorflow[and-cuda]`; note the latter pins
#     cudnn >= 9.3 against torch's ==9.1.0.70 and will not co-resolve.
ARG TRAINING_BASE_IMAGE=nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04
# Pinned: uv is part of the build toolchain, so it is versioned like everything
# else rather than floating on :latest.
ARG UV_VERSION=0.11.21

FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv

FROM ${TRAINING_BASE_IMAGE}
COPY --from=uv /uv /usr/local/bin/uv

ARG USE_INTERNAL_APT=0
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TZ=Asia/Ho_Chi_Minh \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

# Optional internal mirror configs. Bracket globs match zero files without
# failing the build, which is what lets this image build off the NVIDIA network.
COPY local-repository-confi[g] pip.con[f] /opt/netcfg/
RUN if [ "$USE_INTERNAL_APT" = "1" ] && [ -f /opt/netcfg/local-repository-config ]; then \
        cp /opt/netcfg/local-repository-config /etc/apt/sources.list; \
    fi; \
    if [ -f /opt/netcfg/pip.conf ]; then cp /opt/netcfg/pip.conf /etc/pip.conf; fi

# No python here. uv provisions the interpreter that .python-version pins, so
# the image no longer depends on whatever python3.x the CUDA base happens to
# carry — previously this stanza apt-installed python3.10 and symlinked it into
# /usr/local/bin, which silently decided the version the whole pipeline ran on.
#
# libgl1/libglib2.0-0 are needed by opencv even in the headless build;
# git is needed to clone the CycleGAN repo at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        git \
        libglib2.0-0 \
        libgl1 \
        libgomp1 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/training

# Dependencies first, without the project, so source edits do not invalidate
# the (very large) CUDA wheel layer. --frozen fails rather than silently
# re-resolving if uv.lock has drifted from pyproject.toml.
COPY pyproject.toml uv.lock .python-version ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

# Optional offline wheels, e.g. TensorRT. Previously this was
# `COPY ../libs_local/*.whl`, which escapes the build context and made the
# image unbuildable. Put wheels in training/wheels/ instead.
COPY wheel[s] /tmp/wheels/
RUN if ls /tmp/wheels/*.whl >/dev/null 2>&1; then \
        uv pip install --no-cache-dir /tmp/wheels/*.whl; \
    fi; \
    rm -rf /tmp/wheels

COPY src/ ./src/
COPY configs/ ./configs/
COPY assets/ ./assets/
COPY README.md ./

# Editable, so a bind-mounted src/ is picked up without reinstalling — and so
# paths.py still resolves assets/ and configs/ relative to this directory.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

CMD ["sigtrain", "--help"]
