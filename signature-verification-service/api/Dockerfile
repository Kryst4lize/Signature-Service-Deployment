# ── Stage 1: dependency builder ───────────────────────────────────────────────
FROM 10.254.144.152/tessel/python:3.10-slim-buster AS builder

WORKDIR /build

# 1. Copy custom network configurations FIRST so downloads work
COPY local-repository-config /etc/apt/sources.list
COPY pip.conf /etc/pip.conf

# System deps needed only to compile some wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install --prefix=/install --no-cache-dir -r requirements.txt


# ── Stage 2: runtime image ────────────────────────────────────────────────────
FROM 10.254.144.152/tessel/python:3.10-slim-buster AS runtime

# Set Python optimizations and Timezone
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TZ=Asia/Ho_Chi_Minh

# 1. You MUST copy the APT config here too, because Stage 2 also runs apt-get
COPY local-repository-config /etc/apt/sources.list

# Runtime system deps & Timezone setup
RUN ln -snf /usr/share/zoneinfo/"$TZ" /etc/localtime \
    && echo "$TZ" > /etc/timezone \
    && apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    poppler-utils \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

# Non-root user for security (adding -s /sbin/nologin from your base setup)
RUN groupadd -r appuser && useradd -r -g appuser -s /sbin/nologin appuser

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application source
COPY app/ ./app/

RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -sf http://localhost:8080/health || exit 1

CMD ["uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8080", \
     "--workers", "2", \
     "--loop", "uvloop", \
     "--http", "httptools",\
     "--reload"]