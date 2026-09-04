# Signature Verification System

Detect, clean and verify handwritten signatures in scanned documents.

Given a PDF or image, the system locates the signature with YOLOv8, strips
away stamps, form rules and printed text with a CycleGAN denoiser, embeds the
result with fine-tuned ResNet50 and VGG16, and matches it against registered
signatures using pgvector.

```
document ──> yolov8s ──> latest_net_G_B ──> resnet50 + vgg16 ──> pgvector
              detect         denoise            embed             match
```

---

## Two halves

The repository is split into two self-contained services with one interface
between them.

| | | |
|---|---|---|
| **[`training/`](training/)** | Produces models | Datasets, CycleGAN training, backbone fine-tuning, evaluation, ONNX export |
| **[`inference/`](inference/)** | Serves models | FastAPI + Triton + PostgreSQL/pgvector + web UI |

The interface is a Triton model repository:

```bash
cd training  && sigtrain export        # -> ../inference/triton/model_repository/
cd ../inference && docker compose up -d
```

Nothing else crosses. Neither half imports the other.

---

## Quick start

### Serve (models already exported)

```bash
cd inference/
cp .env.example .env          # POSTGRES_PASSWORD has no default — set one
docker compose up -d --build
curl -s localhost:8080/health
```

UI at <http://localhost:8110>. Full instructions:
[`inference/README.md`](inference/README.md).

### Train

```bash
cd training/
docker compose up -d && docker compose exec training bash

sigtrain setup      # clone the CycleGAN repo, check the data layout
sigtrain all        # every stage, in order
```

Full instructions: [`training/README.md`](training/README.md).

---

## Documentation

Setup and run instructions live next to the code, in the two READMEs above.
[`documentation/`](documentation/) covers what spans both halves:

| # | Document |
|---|---|
| 1 | [Architecture](documentation/01-architecture.md) — design, the four models, data flow, ports |
| 2 | [Pipeline deep dive](documentation/02-pipeline-deep-dive.md) — why the ML is built this way; the preprocessing contract |
| 3 | [API reference](documentation/03-api-reference.md) — endpoints, schemas, error codes |
| 4 | [Database](documentation/04-database.md) — schema, vector search, scaling |
| 5 | [Operations](documentation/05-operations.md) — upgrades, backup, monitoring, capacity |
| 6 | [Troubleshooting](documentation/06-troubleshooting.md) — symptom → cause → fix |

---

## Things worth knowing before you deploy

**There is no authentication.** Both upload endpoints are open and each costs
GPU time. Put a gateway in front of it before exposing it —
[Operations § Exposure](documentation/05-operations.md#exposure).

**Preprocessing is a contract, and violating it fails silently.** Each of the
four models expects a different pixel convention; send the wrong one and the
model still returns a well-formed tensor, just computed on input it never saw
in training. No error, no log line, just poor matching that looks like a
threshold problem —
[Pipeline deep dive](documentation/02-pipeline-deep-dive.md#the-preprocessing-contract).

**Embeddings are only comparable within one preprocessing regime.** Changing
preprocessing or retraining the extractors invalidates every stored vector.
`TRUNCATE items` and re-enrol.

**Model weights are not in git.** They are hundreds of MB of build output.
`sigtrain export` regenerates them into the serving tree.

---

## External dependencies

Neither is vendored:

- [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
  — cloned by `sigtrain setup` into `training/external/`.
- [Kaggle signature-verification-dataset](https://www.kaggle.com/datasets/robinreni/signature-verification-dataset)
  — see [`training/data/README.md`](training/data/README.md).

---

## Repository layout

```
.
├── training/
│   ├── src/signature_training/   the pipeline package (`sigtrain`)
│   ├── configs/default.yaml      one config for every stage
│   ├── assets/                   caption font
│   ├── data/                     datasets (gitignored)
│   ├── artifacts/                checkpoints, models, ONNX, plots (gitignored)
│   └── tests/                    62 tests, no GPU required
│
├── inference/
│   ├── api/app/                  main, config, db, images, triton, routes
│   ├── triton/model_repository/  config.pbtxt per model
│   ├── postgres/init.sql         schema
│   ├── frontend/index.html       web UI
│   ├── nginx/nginx.conf          SPA + /api proxy
│   └── tests/                    33 tests against a real pgvector
│
└── documentation/
```

---

## Tests

```bash
cd training  && make test               # ~1s, no GPU, no TF/torch
cd inference && make test-integration   # spins up a throwaway pgvector
```

---

## License

See [LICENSE](LICENSE).
