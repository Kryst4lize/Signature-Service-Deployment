# syntax=docker/dockerfile:1

# GPU training image.
#
# Default is a public CUDA base so the image builds from any clone. On the
# NVIDIA network, override in .env:
#   TRAINING_BASE_IMAGE=10.254.144.152/tessel/training-service-base:0.0.1
#   USE_INTERNAL_APT=1
ARG TRAINING_BASE_IMAGE=nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

FROM ${TRAINING_BASE_IMAGE}

ARG USE_INTERNAL_APT=0
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TZ=Asia/Ho_Chi_Minh

# Optional internal mirror configs. Bracket globs match zero files without
# failing the build, which is what lets this image build off the NVIDIA network.
# The previous Dockerfile had an unconditional `COPY pip.conf`, which aborted
# every build because pip.conf is gitignored and never committed.
COPY local-repository-confi[g] pip.con[f] /opt/netcfg/
RUN if [ "$USE_INTERNAL_APT" = "1" ] && [ -f /opt/netcfg/local-repository-config ]; then \
        cp /opt/netcfg/local-repository-config /etc/apt/sources.list; \
    fi; \
    if [ -f /opt/netcfg/pip.conf ]; then cp /opt/netcfg/pip.conf /etc/pip.conf; fi

# libgl1/libglib2.0-0 are needed by opencv even in the headless build;
# git is needed to clone the CycleGAN repo at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 \
        python3-pip \
        python3.10-dev \
        git \
        libglib2.0-0 \
        libgl1 \
        libgomp1 \
    && ln -sf /usr/bin/python3.10 /usr/local/bin/python \
    && ln -sf /usr/bin/python3.10 /usr/local/bin/python3 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/training

# Dependencies first so source edits do not invalidate the (very large) layer.
COPY requirements.txt .
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements.txt

# Optional offline wheels, e.g. TensorRT. Previously this was
# `COPY ../libs_local/*.whl`, which escapes the build context and made the
# image unbuildable. Put wheels in training/wheels/ instead.
COPY wheel[s] /tmp/wheels/
RUN if ls /tmp/wheels/*.whl >/dev/null 2>&1; then \
        python -m pip install --no-cache-dir /tmp/wheels/*.whl; \
    fi; \
    rm -rf /tmp/wheels

COPY pyproject.toml ./
COPY src/ ./src/
COPY configs/ ./configs/
COPY assets/ ./assets/

# Editable so a bind-mounted source tree is picked up without reinstalling.
RUN python -m pip install --no-cache-dir -e .

CMD ["sigtrain", "--help"]
