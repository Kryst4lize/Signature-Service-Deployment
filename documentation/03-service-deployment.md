# 3 — Service Deployment — Setup & Run

> **Folder:** `signature-verification-service/`
>
> This guide covers how to deploy the production microservice stack:
> FastAPI + Triton Inference Server + PostgreSQL (pgvector) + Nginx frontend.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Project Structure](#2-project-structure)
3. [Step 1 — Place ONNX Model Files](#3-step-1--place-onnx-model-files)
4. [Step 2 — Configure Environment](#4-step-2--configure-environment)
5. [Step 3 — Deploy with Docker Compose](#5-step-3--deploy-with-docker-compose)
6. [Step 4 — Verify Deployment](#6-step-4--verify-deployment)
7. [GPU vs CPU Mode](#7-gpu-vs-cpu-mode)
8. [Service Architecture Details](#8-service-architecture-details)
9. [Frontend (SigVerify Web UI)](#9-frontend-sigverify-web-ui)
10. [Configuration Reference](#10-configuration-reference)

---

## 1. Prerequisites

### Hardware
- **GPU (recommended)**: NVIDIA GPU with ≥6 GB VRAM for Triton inference
- **CPU-only**: Possible but slower — change Triton config (see [GPU vs CPU](#7-gpu-vs-cpu-mode))
- **RAM**: ≥8 GB
- **Disk**: ≥5 GB (for ONNX models + Docker images)

### Software
- **Docker** ≥ 20.10
- **Docker Compose** v2
- **NVIDIA Container Toolkit** (for GPU mode)
- **NVIDIA Driver** ≥ 525.x

---

## 2. Project Structure

```
signature-verification-service/
├── .env                          ← Environment variables
├── docker-compose.yml            ← Full-stack orchestration
├── pip.conf                      ← Python package mirror config
├── local-repository-config       ← APT mirror config (for Docker build)
├── signatures.py                 ← Reference implementation of router logic
│
├── api/                          ← FastAPI application
│   ├── Dockerfile                ← Multi-stage build (builder + runtime)
│   ├── requirements.txt          ← Python dependencies
│   ├── pip.conf
│   ├── local-repository-config
│   └── app/
│       ├── main.py               ← Application entry point
│       ├── core/
│       │   ├── config.py          ← Pydantic settings (env vars → config)
│       │   └── database.py        ← SQLAlchemy async engine + session
│       ├── models/
│       │   └── db.py              ← ORM models (items, models tables)
│       ├── routers/
│       │   └── signatures.py      ← API endpoints (/register, /verify, /signatures)
│       └── services/
│           ├── triton.py          ← Triton client (YOLOv8, CycleGAN, ResNet, VGG)
│           ├── preprocessing.py   ← Image/PDF → tensor conversion
│           └── image_utils.py     ← Tensor ↔ PIL/base64 utilities
│
├── triton/
│   └── model_repository/          ← ONNX models served by Triton
│       ├── yolov8s/
│       │   ├── config.pbtxt
│       │   └── 1/model.onnx       ← ★ Place your ONNX model here
│       ├── latest_net_G_A/
│       │   ├── config.pbtxt
│       │   └── 1/model.onnx
│       ├── latest_net_G_B/         ← ★ CycleGAN denoiser (used in pipeline)
│       │   ├── config.pbtxt
│       │   └── 1/model.onnx
│       ├── resnet50_extractor/
│       │   ├── config.pbtxt
│       │   └── 1/model.onnx
│       └── vgg16_extractor/
│           ├── config.pbtxt
│           └── 1/model.onnx
│
├── postgres/
│   └── init.sql                   ← pgvector schema + seed data
│
├── frontend/
│   └── index.html                 ← SigVerify web UI (single-page app)
│
└── nginx/
    └── nginx.conf                 ← Reverse proxy + static file serving
```

---

## 3. Step 1 — Place ONNX Model Files

Each model must be exported to ONNX format and placed at the correct path:

```
triton/model_repository/<model_name>/1/model.onnx
```

### Required models

| Model | Path | Input Shape | Output Shape | Source |
|-------|------|-------------|--------------|--------|
| YOLOv8s | `yolov8s/1/model.onnx` | `[1,3,640,640]` | `[5,N]` | Ultralytics export |
| CycleGAN G_B | `latest_net_G_B/1/model.onnx` | `[1,3,224,224]` | `[1,3,224,224]` | `export_onnx.py` |
| CycleGAN G_A | `latest_net_G_A/1/model.onnx` | `[1,3,224,224]` | `[1,3,224,224]` | `export_onnx.py` |
| ResNet50 | `resnet50_extractor/1/model.onnx` | `[1,3,224,224]` | `[1,4096]` | `simpleconver.py` |
| VGG16 | `vgg16_extractor/1/model.onnx` | `[1,3,224,224]` | `[1,4096]` | `simpleconver.py` |

> **Note:** TensorRT `.plan` files may also be present alongside `.onnx` files.
> Triton will automatically prefer `.plan` if available (faster inference).

See [04 — Model Conversion Guide](./04-model-conversion.md) for how to produce these files.

---

## 4. Step 2 — Configure Environment

### Create/edit `.env`

```bash
cp .env.example .env
# or edit .env directly:
```

```env
POSTGRES_DB=signature_db
POSTGRES_USER=siguser
POSTGRES_PASSWORD=sigpass
TRITON_HOST=triton
APP_ENV=production
```

### Environment variables reference

| Variable | Default | Description |
|----------|---------|-------------|
| `POSTGRES_DB` | `signature_db` | Database name |
| `POSTGRES_USER` | `siguser` | Database user |
| `POSTGRES_PASSWORD` | `sigpass` | Database password |
| `TRITON_HOST` | `triton` | Triton hostname (Docker service name) |
| `APP_ENV` | `production` | App environment (`production` or `development`) |

---

## 5. Step 3 — Deploy with Docker Compose

### Build and start all services

```bash
cd signature-verification-service/
docker compose up --build -d
```

This starts four containers:

| Container | Image | Purpose |
|-----------|-------|---------|
| `sig_postgres` | `pgvector/pgvector:pg17` | PostgreSQL + pgvector |
| `sig_triton` | `tritonserver:24.01-py3` | NVIDIA Triton Inference Server |
| `sig_api` | Built from `api/Dockerfile` | FastAPI backend |
| `sig_frontend` | `nginx:alpine` | Static frontend + reverse proxy |

### Service startup order

1. **postgres** starts first → healthcheck (`pg_isready`)
2. **triton** starts → healthcheck (`/v2/health/ready`)
3. **api** waits for both postgres + triton to be healthy
4. **frontend** waits for api

### Check logs

```bash
# All services
docker compose logs -f

# Individual services
docker compose logs -f api
docker compose logs -f triton
docker compose logs -f postgres
docker compose logs -f frontend
```

### Stop all services

```bash
docker compose down
```

### Stop and remove volumes (full reset)

```bash
docker compose down -v
```

---

## 6. Step 4 — Verify Deployment

### Health checks

```bash
# API health
curl http://localhost:8080/health
# Expected: {"status":"ok","env":"production"}

# Triton health
curl http://localhost:8010/v2/health/ready
# Expected: 200 OK

# Frontend
curl http://localhost:8110/nginx-health
# Expected: ok
#
# NOTE: The docker-compose.yml frontend healthcheck references port 3000
# internally, but nginx.conf listens on port 80. This may cause the Docker
# healthcheck to report "unhealthy" despite the service working correctly.
# The external endpoint (:8110) works regardless.
```

### Test signature registration

```bash
curl -X POST http://localhost:8080/register-signature \
  -F "username=john_doe" \
  -F "file=@/path/to/signature.png"
```

Expected response:
```json
{
  "id": 1,
  "username": "john_doe",
  "user_created_date": "2024-01-01T00:00:00"
}
```

### Test document verification

```bash
curl -X POST http://localhost:8080/verify-document \
  -F "file=@/path/to/document.pdf"
```

---

## 7. GPU vs CPU Mode

### Current state: GPU mode (default)

The `docker-compose.yml` has GPU reservations enabled for Triton,
and all `config.pbtxt` files use `KIND_GPU`.

### Switching to CPU mode

1. **Comment out GPU reservations** in `docker-compose.yml`:

```yaml
# triton:
#   deploy:
#     resources:
#       reservations:
#         devices:
#           - driver: nvidia
#             count: 1
#             capabilities: [gpu]
```

2. **Change each `config.pbtxt`** — replace `KIND_GPU` with `KIND_CPU`:

```protobuf
instance_group [
  {
    kind: KIND_CPU
    count: 1
  }
]
```

Files to update:
- `triton/model_repository/yolov8s/config.pbtxt`
- `triton/model_repository/latest_net_G_B/config.pbtxt`
- `triton/model_repository/latest_net_G_A/config.pbtxt`
- `triton/model_repository/resnet50_extractor/config.pbtxt`
- `triton/model_repository/vgg16_extractor/config.pbtxt`

---

## 8. Service Architecture Details

### FastAPI Application (`api/`)

**Entry point:** `app/main.py`

- **Lifespan:** Connects to Triton on startup, closes on shutdown
- **CORS:** Configured for frontend origins
- **Workers:** 2 Uvicorn workers with uvloop + httptools

**Configuration:** `app/core/config.py` — Pydantic `BaseSettings` reads from env vars

**Database:** `app/core/database.py` — SQLAlchemy async engine with connection pooling
- Pool size: 10
- Max overflow: 20

**Triton Client:** `app/services/triton.py` — Async HTTP client using `tritonclient`
- `detect_signature()` — YOLOv8 inference → bbox crop
- `denoise()` — CycleGAN inference
- `extract_features()` — Parallel ResNet50 + VGG16 inference

**Preprocessing:** `app/services/preprocessing.py`
- Supports JPG, PNG, WEBP images
- Supports PDF (via `pdf2image` + poppler-utils)
- Converts to `[1, 3, H, W]` float32 tensors

### Triton Inference Server

Serves all four ONNX models via HTTP (:8010) and gRPC (:8011).

**Model configs:**

| Model | Backend | Input Name | Input Dims | Output Name | Output Dims |
|-------|---------|------------|------------|-------------|-------------|
| `yolov8s` | onnxruntime | `images` | `[3,640,640]` | `output0` | `[5,-1]` |
| `latest_net_G_B` | onnxruntime | `input` | `[3,224,224]` | `output` | `[3,224,224]` |
| `latest_net_G_A` | onnxruntime | `input` | `[3,224,224]` | `output` | `[3,224,224]` |
| `resnet50_extractor` | onnxruntime | `input_layer_1` | `[3,224,224]` | `fc1` | `[4096]` |
| `vgg16_extractor` | onnxruntime | `input_layer` | `[3,224,224]` | `fc1` | `[4096]` |

### PostgreSQL + pgvector

- **pgvector extension** enabled for vector similarity search
- **HNSW indexes** on both `resnet50_vector` and `vgg16_vector` columns
- L2 distance operator: `<->` for nearest-neighbour search

---

## 9. Frontend (SigVerify Web UI)

Access at: **http://localhost:8110**

The frontend is a single-page application (`index.html`) with three tabs:

| Tab | Function |
|-----|----------|
| **Register** | Upload a clean signature image + username → calls `/register-signature` |
| **Signatures** | List all registered signatures → calls `/signatures` with pagination |
| **Verify Document** | Upload a PDF/image → calls `/verify-document`, displays results with annotated images |

Features:
- Real-time API health indicator (green dot = connected)
- Configurable API base URL
- Drag-and-drop file upload
- Visual display of: annotated page (red bbox), cropped signature, cleaned signature
- Distance bars showing ResNet50 and VGG16 distances

---

## 10. Configuration Reference

### Docker Compose services

| Service | Container | Ports | Volumes |
|---------|-----------|-------|---------|
| `postgres` | `sig_postgres` | `5432:5432` | `postgres_data:/var/lib/postgresql/data`, `init.sql` |
| `triton` | `sig_triton` | `8010:8000`, `8011:8001`, `8012:8002` | `model_repository:/models:ro` |
| `api` | `sig_api` | `8080:8080` | `app:/app/app:ro` |
| `frontend` | `sig_frontend` | `8110:80` | `index.html`, `nginx.conf` |

### API Dockerfile details

- **Base image:** Python 3.10-slim-buster
- **Multi-stage build:** Builder stage installs pip packages, runtime stage is minimal
- **System deps:** `libpq5` (PostgreSQL client), `poppler-utils` (PDF processing), `curl`
- **Security:** Runs as non-root `appuser`
- **Timezone:** Asia/Ho_Chi_Minh
- **Health check:** `curl -sf http://localhost:8080/health`

### Match threshold tuning

In `api/app/routers/signatures.py`, the match threshold is:

```python
"matched": float(match["avg_distance"]) < 0.5   # lower = stricter
```

- `avg_distance` = average of ResNet50 L2 distance and VGG16 L2 distance
- Lower threshold = stricter matching (fewer false positives)
- Higher threshold = more lenient (fewer false negatives)
- **Start with 0.5** and calibrate against your dataset
