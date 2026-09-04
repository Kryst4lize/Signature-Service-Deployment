# Documentation

Cross-cutting reference for the signature verification system.

**Setup and run instructions are not here** — they live next to the code they
describe, so they cannot drift out of sync with it:

- [`../training/README.md`](../training/README.md) — build datasets, train, evaluate, export
- [`../inference/README.md`](../inference/README.md) — deploy and operate the service

This folder covers the things that span both halves.

| # | Document | Covers |
|---|----------|--------|
| 1 | [Architecture](./01-architecture.md) | System design, the four models, data flow, ports |
| 2 | [Pipeline deep dive](./02-pipeline-deep-dive.md) | Why the ML is built this way; the preprocessing contract |
| 3 | [API reference](./03-api-reference.md) | REST endpoints, schemas, error codes |
| 4 | [Database](./04-database.md) | Schema, vector search, why there is no ANN index |
| 5 | [Operations](./05-operations.md) | Upgrades, backup, monitoring, capacity |
| 6 | [Troubleshooting](./06-troubleshooting.md) | Symptom → cause → fix |

## Repository map

```
.
├── training/           produces models
│   ├── src/signature_training/    the pipeline package (`sigtrain`)
│   ├── configs/default.yaml       one config for every stage
│   ├── data/                      datasets (gitignored)
│   ├── artifacts/                 checkpoints, models, ONNX, plots (gitignored)
│   └── tests/
│
├── inference/          serves models
│   ├── api/app/                   FastAPI service (6 modules)
│   ├── triton/model_repository/   config.pbtxt per model
│   ├── postgres/init.sql          schema
│   ├── frontend/index.html        web UI
│   ├── nginx/nginx.conf           SPA + /api proxy
│   └── tests/
│
└── documentation/      this folder
```

The interface between the two halves is a Triton model repository.
`sigtrain export` writes it directly into `inference/triton/model_repository/`.
