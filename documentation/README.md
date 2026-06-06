# Signature Verification System — Documentation Index

> Complete documentation for the **Signature Verification** ecosystem, covering both
> the **ML training pipeline** and the **production inference service**.

---

## Documents

| # | Document | Description |
|---|----------|-------------|
| 1 | [Architecture Overview](./01-architecture-overview.md) | High-level system design, ML pipeline stages, and component relationships |
| 2 | [Training Pipeline — Setup & Run](./02-training-pipeline.md) | How to set up, train, and evaluate models in `trainingfiles/` |
| 3 | [Service Deployment — Setup & Run](./03-service-deployment.md) | How to deploy the FastAPI + Triton + Postgres microservice in `signature-verification-service/` |
| 4 | [Model Conversion Guide](./04-model-conversion.md) | Converting trained `.keras` / `.pth` models to `.onnx` for Triton |
| 5 | [API Reference](./05-api-reference.md) | REST API endpoints, request/response schemas, and examples |
| 6 | [Database Schema](./06-database-schema.md) | PostgreSQL + pgvector tables, indexes, and migration info |
| 7 | [Troubleshooting & FAQ](./07-troubleshooting.md) | Common issues, debugging tips, and environment-specific notes |

---

## Folder Map

```
_srv/
├── trainingfiles/              ← ML Training & Evaluation
│   ├── pyfile/                         ← Training scripts
│   ├── convert_model/                  ← Model format conversion
│   ├── data/                           ← Training datasets
│   ├── model/                          ← Pretrained weights & outputs
│   ├── pytorch-CycleGAN-and-pix2pix/  ← CycleGAN training framework
│   └── docker-compose.yaml            ← GPU training container
│
├── signature-verification-service/     ← Production Microservice
│   ├── api/                            ← FastAPI application
│   ├── triton/model_repository/        ← ONNX models for Triton
│   ├── postgres/                       ← DB schema init
│   ├── frontend/                       ← Web UI (SigVerify)
│   ├── nginx/                          ← Reverse proxy
│   └── docker-compose.yml             ← Full-stack deployment
│
└── documentation/                      ← This folder
```
