# 1 — Architecture Overview

## System Goal

Verify the identity of a handwritten signature found inside scanned documents (PDF or image).
The system answers: **"Whose signature is on this page, and how confident are we?"**

---

## End-to-End ML Pipeline

```
┌──────────────────── TRAINING (trainingfiles/) ────────────────────────────┐
│                                                                           │
│  ① Data Prep          ② CycleGAN Training       ③ Verification Training   │
│  ─────────────        ─────────────────────      ──────────────────────── │
│  clean_signatures     trainA (clean) ──┐         Cleaned signature crops  │
│        │              trainB (noisy) ──┤─→ G_B   ──→ VGG16 fine-tune      │
│        ▼                               │ (denoise)   ResNet50 fine-tune   │
│  + stamp noise         testA / testB   │                │                 │
│  + line noise                          │                ▼                 │
│  + text overlay                        │         Feature Extractors       │
│        │                               │           VGG16  → 4096-d        │
│        ▼                               │           ResNet50 → 4096-d      │
│  CycleGAN dataset                      │                                  │
│                                                                           │
└───────────────────────────────────────────────────────────────────────────┘
                              │
                    convert to ONNX
                              │
                              ▼
┌──────────────── SERVICE (signature-verification-service/) ────────────────┐
│                                                                           │
│  ┌──────────┐   ┌─────────────────────────────────────────┐  ┌────────┐   │
│  │ Frontend │──>│              FastAPI                    │  │Postgres│   │
│  │ (nginx)  │   │                                         │  │pgvector│   │
│  │ :8110    │   │  /register-signature                    │  │ :5432  │   │
│  └──────────┘   │    image → CycleGAN → ResNet+VGG → DB   │  └────────┘   │
│                 │                                         │               │
│                 │  /verify-document                       │               │
│                 │    PDF→pages → YOLOv8 → CycleGAN        │               │
│                 │    → ResNet+VGG → pgvector L2 search    │               │
│                 │                                         │               │
│                 │  :8080                                  │               │
│                 └────────────┬────────────────────────────┘               │
│                              │                                            │
│                  ┌───────────▼──────────┐                                 │
│                  │   Triton Inference   │                                 │
│                  │   Server             │                                 │
│                  │   ┌─────────────┐    │                                 │
│                  │   │ yolov8s     │    │                                 │
│                  │   │ latest_G_B  │    │                                 │
│                  │   │ resnet50    │    │                                 │
│                  │   │ vgg16       │    │                                 │
│                  │   └─────────────┘    │                                 │
│                  │   :8010 (HTTP)       │                                 │
│                  │   :8011 (gRPC)       │                                 │
│                  └──────────────────────┘                                 │
│                                                                           │
└───────────────────────────────────────────────────────────────────────────┘
```

---

## Four Models Explained

| Model | Type | Input | Output | Purpose |
|-------|------|-------|--------|---------|
| **YOLOv8s** | Detector | `[1,3,640,640]` | `[5,N]` bboxes | Locate signature regions inside a document page |
| **CycleGAN (G_B)** | Denoiser | `[1,3,224,224]` | `[1,3,224,224]` | Remove stamp, line, and text noise from signature crops |
| **ResNet50** | Extractor | `[1,3,224,224]` | `[4096]` vector | Produce a 4096-d embedding for identity matching |
| **VGG16** | Extractor | `[1,3,224,224]` | `[4096]` vector | Produce a 4096-d embedding for identity matching |

### Inference Pipeline — Register Signature

```
Upload (clean signature image)
  → Resize to 224×224
  → CycleGAN denoise
  → ResNet50 extract 4096-d vector ─┐
  → VGG16 extract 4096-d vector ────┼─→ INSERT into PostgreSQL (pgvector)
                                     │
                                     └─→ Return { id, username }
```

### Inference Pipeline — Verify Document

```
Upload (PDF or image)
  → Convert to page images (640×640 each)
  FOR each page:
    → YOLOv8 detect → crop signature region
    → Resize crop to 224×224
    → CycleGAN denoise
    → ResNet50 + VGG16 extract vectors
    → pgvector L2 nearest-neighbour search
    → avg_distance = (d_resnet + d_vgg) / 2
    → matched = avg_distance < 0.5
  → Return results per page
```

---

## Technology Stack

| Layer | Technology | Version |
|-------|-----------|---------|
| **Inference Engine** | NVIDIA Triton Inference Server | 24.01-py3 |
| **API** | FastAPI + Uvicorn | FastAPI 0.111 / Uvicorn 0.30 |
| **Database** | PostgreSQL + pgvector extension | PG 17 / pgvector 0.7+ |
| **Frontend** | Static HTML/CSS/JS served by Nginx | Alpine |
| **ML Training** | TensorFlow 2.21 + PyTorch 2.3 | — |
| **Object Detection** | YOLOv8s (Ultralytics) | — |
| **Image Translation** | CycleGAN (junyanz/pytorch-CycleGAN-and-pix2pix) | — |
| **Feature Extraction** | VGG16 + ResNet50 (ImageNet pretrained, fine-tuned) | — |
| **Model Format** | ONNX (via tf2onnx / torch.onnx) | Opset 13-17 |
| **Container Runtime** | Docker Compose | — |

---

## Network Ports (default deployment)

| Port | Service | Protocol |
|------|---------|----------|
| `8080` | FastAPI | HTTP REST |
| `8010` | Triton HTTP | HTTP |
| `8011` | Triton gRPC | gRPC |
| `8012` | Triton Metrics | HTTP |
| `8110` | Frontend (nginx) | HTTP |
| `5432` | PostgreSQL | TCP |
