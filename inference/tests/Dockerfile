# Test runner. Same dependency set as the service image plus the test tools;
# no application code is baked in, the tree is bind-mounted at run time.
#
# Build context is inference/, so both api/requirements.txt and this file are
# reachable:  docker build -f tests/Dockerfile .
ARG API_BASE_IMAGE=python:3.11-slim-bookworm
FROM ${API_BASE_IMAGE}

RUN apt-get update \
    && apt-get install -y --no-install-recommends libpq5 poppler-utils \
    && rm -rf /var/lib/apt/lists/*

COPY api/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt \
    && pip install --no-cache-dir pytest==8.3.4 httpx==0.28.1 anyio==4.7.0

ENV PYTHONPATH=/w/api
WORKDIR /w
