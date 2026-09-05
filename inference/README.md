# Inference Service

Production deployment of the signature verification system: FastAPI + NVIDIA
Triton + PostgreSQL/pgvector + a single-page web UI.

This half of the repository only *serves* models. Everything that produces them
— datasets, training, evaluation, ONNX export — lives in [`../training/`](../training/).

---

## Contents

- [Pipeline](#pipeline)
- [Layout](#layout)
- [Quick start](#quick-start)
- [Model files](#model-files)
- [Configuration](#configuration)
- [API](#api)
- [Tensor conventions](#tensor-conventions)
- [Calibrating the threshold](#calibrating-the-threshold)
- [Tests](#tests)
- [Operational notes](#operational-notes)

---

## Pipeline

```
POST /register-signature          POST /verify-document
  upload (a clean crop)             upload (PDF or image)
        │                                   │
        │                           per page, at full resolution
        │                                   │
        │                           yolov8s ─── detect ──> normalised bbox
        │                                   │
        │                           crop from the ORIGINAL page
        ▼                                   ▼
  latest_net_G_B (CycleGAN denoise, 224x224)
        │
        ├── resnet50_extractor ──> 4096-d ──┐
        └── vgg16_extractor    ──> 4096-d ──┤ L2-normalised
                                            ▼
                              INSERT into items        cosine NN search
```

Registration skips detection: the upload is assumed to already be an isolated
signature.

---

## Layout

```
inference/
├── docker-compose.yml          full stack
├── docker-compose.dev.yml      overlay: bind mount + --reload
├── .env.example                copy to .env
├── Makefile
├── api/
│   ├── Dockerfile
│   ├── pip.conf.example        internal PyPI mirror template
│   ├── local-repository-config internal apt mirror
│   ├── requirements.txt
│   └── app/
│       ├── main.py             app factory, lifespan, CORS, /health
│       ├── config.py           env-driven settings
│       ├── db.py               engine, session, the `items` table
│       ├── images.py           PDF/image decode, tensor <-> PIL, base64
│       ├── triton.py           Triton client + per-model tensor conventions
│       └── routes.py           the four endpoints
├── nginx/nginx.conf            serves the SPA, proxies /api/
├── frontend/index.html         single-file UI
├── postgres/init.sql           schema
├── triton/model_repository/    config.pbtxt per model (weights not committed)
└── tests/
```

---

## Quick start

```bash
cd inference/
cp .env.example .env
# POSTGRES_PASSWORD has no default — compose refuses to start until you set it.
#   openssl rand -base64 24

# Put the ONNX files in place first (see below), then:
docker compose up -d --build

docker compose ps
curl -s localhost:8080/health
```

Open the UI at <http://localhost:8110>. Set its API field to `/api` to go
through the nginx proxy (same-origin, no CORS involved), or to
`http://localhost:8080` to hit the API directly — the latter requires the origin
to be listed in `CORS_ORIGINS`.

Development, with hot reload:

```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml up
```

### Running without a GPU

`docker compose up` fails on a CPU-only host with
`could not select device driver "nvidia" with capabilities: [[gpu]]`.

1. Delete the `deploy.resources` block from the `triton` service.
2. Change `kind: KIND_GPU` to `kind: KIND_CPU` and drop the `gpus: [ 0 ]` line
   in each `triton/model_repository/*/config.pbtxt`.

---

## Model files

Weights are **not** in git — they are hundreds of MB and are build outputs.
Triton refuses to start if a model directory has no version subdirectory, so
place them before the first `docker compose up`:

```
triton/model_repository/
├── yolov8s/1/model.onnx
├── latest_net_G_B/1/model.onnx
├── resnet50_extractor/1/model.onnx
└── vgg16_extractor/1/model.onnx
```

The training pipeline writes this tree directly:

```bash
cd ../training
sigtrain export --output ../inference/triton/model_repository
```

The input and output tensor names in each `config.pbtxt` must match the ONNX
graph exactly — `images`/`output0`, `input`/`output`, `input_layer_1`/`fc1`,
`input_layer`/`fc1`. `sigtrain export` generates configs that already agree.

---

## Configuration

Everything is read from the environment; see [`.env.example`](.env.example).

| Variable | Default | Notes |
|---|---|---|
| `POSTGRES_DB` / `_USER` / `_PASSWORD` | — | Required; compose fails fast if unset |
| `CORS_ORIGINS` | *(empty)* | Comma-separated. Empty ⇒ middleware not installed |
| `MATCH_THRESHOLD` | `0.30` | Mean cosine distance below which `matched` is true |
| `DETECTION_CONFIDENCE` | `0.5` | YOLOv8 objectness floor |
| `MAX_UPLOAD_BYTES` | `20971520` | 20 MiB |
| `MAX_PDF_PAGES` | `20` | Pages beyond this are not rendered |
| `API_BASE_IMAGE` | `python:3.11-slim-bookworm` | Set to the internal registry image on the NVIDIA network |
| `USE_INTERNAL_APT` | `0` | `1` applies `local-repository-config` |
| `TRITON_IMAGE` | `nvcr.io/nvidia/tritonserver:24.01-py3` | |

Model names (`yolov8s`, `latest_net_G_B`, …) are also environment-overridable,
so renaming a directory in the model repository needs no image rebuild.

---

## API

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Liveness |
| `GET` | `/signatures?skip=&limit=` | List registrations, newest first |
| `DELETE` | `/signatures/{id}` | Remove one |
| `POST` | `/register-signature` | Enrol (`username`, `file`) |
| `POST` | `/verify-document` | Verify (`file`) |

```bash
curl -X POST localhost:8080/register-signature \
  -F "username=nguyen_van_a" -F "file=@signature.png"

curl -X POST localhost:8080/verify-document -F "file=@contract.pdf"
```

`/verify-document` returns one entry per page:

```jsonc
{
  "total_pages": 2,
  "results": [
    {
      "page": 1,
      "status": "matched",          // pipeline stage reached, NOT the verdict
      "username": "nguyen_van_a",   // nearest neighbour, whatever the distance
      "avg_distance": 0.118,        // cosine distance in [0, 2]
      "resnet_distance": 0.104,
      "vgg_distance": 0.132,
      "matched": true,              // ← the verdict: avg_distance < MATCH_THRESHOLD
      "bbox": [412, 980, 947, 1183],
      "confidence": 0.93,
      "page_annotated": "<base64 png>",
      "crop_before": "<base64 png>",
      "crop_after": "<base64 png>"
    },
    { "page": 2, "status": "no_signature" }
  ]
}
```

`status` is `matched` whenever the table is non-empty, because the query always
returns a nearest neighbour. **`matched` is the decision** — always read that,
never `status`. Other values: `no_signature`, `no_match_in_db` (empty table).

Errors: `400` undecodable upload, `413` over `MAX_UPLOAD_BYTES`, `422` invalid
`username`, `502` Triton failure.

---

## Tensor conventions

Every tensor crossing a module boundary is `float32 [1, 3, H, W]` in `[0, 1]`,
RGB. Each model wants something different, and each conversion lives in
`triton.py` next to the call that needs it:

| Model | Expects | Conversion |
|---|---|---|
| `yolov8s` | `[0, 1]` RGB | none |
| `latest_net_G_B` | `[-1, 1]` RGB, returns Tanh `[-1, 1]` | `to_cyclegan` / `from_cyclegan` |
| `resnet50_extractor` | Caffe BGR, ≈`[-124, +151]` | `to_caffe` |
| `vgg16_extractor` | Caffe BGR, ≈`[-124, +151]` | `to_caffe` |

Embeddings are L2-normalised before storage, matching what
`sigtrain evaluate` measures. `to_caffe` reproduces Keras
`preprocess_input(mode="caffe")` exactly — the test suite pins this, and it was
verified bit-for-bit against TensorFlow 2.21.

> **Getting these wrong is silent.** Every model still returns a well-formed
> tensor of the right shape; it is just computed on out-of-distribution input.
> Nothing appears in the logs. If you change a conversion, run `tests/test_tensors.py`.

---

## Calibrating the threshold

`MATCH_THRESHOLD` is a mean cosine **distance** (0 = identical, 1 = orthogonal).
`sigtrain evaluate` reports the EER operating point as a cosine **similarity**
`s`; set

```
MATCH_THRESHOLD = 1 - s
```

Lower is stricter. Calibrate on your own enrolled population — the value that
balances FAR and FRR depends on how many people are registered and how similar
their signatures are.

---

## Tests

```bash
make test              # tensor conventions only, no services needed
make test-integration  # + endpoint tests against a throwaway pgvector container
```

Triton is faked (no GPU required). Postgres is real for the endpoint tests,
because the pgvector cast, the cosine ordering and the `VARCHAR(50)` limit have
no meaningful in-memory equivalent.

---

## Operational notes

### Re-register after upgrading past the tensor-contract fix

Embeddings written before that change were computed from input the extractors
were never trained on, and are not comparable with ones written after it.
Mixing them silently degrades matching. After deploying:

```sql
TRUNCATE items RESTART IDENTITY;
```

then re-register every signature.

### There is no ANN index, deliberately

pgvector caps HNSW and IVFFlat at 2000 dimensions; these columns are 4096-d.
The previous `CREATE INDEX ... USING hnsw` did not build a slow index — it
aborted initialisation and took the whole container down with it. Search is an
exact sequential scan, ~80 ms at 5,000 registrations, against a request that
already spends far longer in four GPU inferences. `postgres/init.sql` documents
the escalation options if enrolment outgrows that.

### Threshold semantics changed

Distances are cosine now, not raw L2. A `MATCH_THRESHOLD` carried over from the
old `0.5` L2 constant does not mean the same thing — recalibrate.
