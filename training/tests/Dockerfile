# Test runner for the stages that need neither TensorFlow nor torch — dataset
# building, augmentation, config, metrics, pair construction and the CLI.
#
# Deliberately not the full training image: that one is ~9 GB of CUDA wheels and
# takes minutes to build, which makes `make test` useless as a feedback loop.
# The dependency set here is therefore a hand-picked subset of the locked one,
# not `uv sync` — the point is to be fast.
ARG UV_VERSION=0.11.21

FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv

FROM python:3.11-slim-bookworm
COPY --from=uv /uv /usr/local/bin/uv

RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 libgl1 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

# Versions match the corresponding entries in uv.lock; `make test-lock-drift`
# checks they have not diverged.
RUN uv venv /opt/venv \
    && uv pip install --no-cache \
        numpy==1.26.4 \
        scipy==1.14.1 \
        scikit-learn==1.5.2 \
        pillow==11.0.0 \
        opencv-python-headless==4.10.0.84 \
        matplotlib==3.9.3 \
        tqdm==4.67.1 \
        PyYAML==6.0.2 \
        onnx==1.17.0 \
        pytest==8.3.4

ENV PYTHONPATH=/w/src
WORKDIR /w
