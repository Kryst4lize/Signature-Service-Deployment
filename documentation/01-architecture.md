# 1 — Architecture

## What the system does

Given a scanned document, find the signature on it, clean it up, and decide
whether it belongs to a registered person.

Four models, in sequence. None of them is a classifier over "genuine vs
forged" — the system answers *whose signature is this*, by comparing embeddings
against an enrolled set.

## Component map

```
                    browser (:8110)
                          │
                     nginx  ──── /api/ proxy ────┐
                          │                      │
                    index.html                   │
                                                 ▼
                                        FastAPI (:8080)
                                            │        │
                        ┌───────────────────┘        └──────────┐
                        ▼                                       ▼
              Triton Inference Server                   PostgreSQL + pgvector
                   (:8010 HTTP)                              (:5432)
                        │                                       │
        ┌───────────────┼───────────────┐              items(id, username,
        ▼               ▼               ▼               resnet50_vector,
    yolov8s      latest_net_G_B   resnet50_extractor     vgg16_vector)
    (detect)      (denoise)       vgg16_extractor
                                  (embed)
```

## The four models

| Triton name | Role | In | Out |
|---|---|---|---|
| `yolov8s` | Locate the signature on a page | `[3,640,640]` `[0,1]` RGB | `[5,N]` — cx, cy, w, h, conf |
| `latest_net_G_B` | Remove stamps, rules, printed text | `[3,224,224]` `[-1,1]` RGB | `[3,224,224]` `[-1,1]` (Tanh) |
| `resnet50_extractor` | Identity embedding | `[3,224,224]` Caffe BGR | `[4096]` |
| `vgg16_extractor` | Identity embedding | `[3,224,224]` Caffe BGR | `[4096]` |

Two extractors rather than one because averaging two independently-trained
embedding spaces is more robust than trusting either. The API stores both
vectors and averages their cosine distances.

`latest_net_G_B` and not `latest_net_G_A`: CycleGAN trains both directions.
With `trainA` = clean and `trainB` = noisy, `G_B` maps noisy → clean, which is
the direction serving needs. `G_A` is a training by-product and is not deployed.

## Request flows

### `POST /register-signature`

```
upload (an already-isolated signature crop)
  └─ resize 224
     └─ latest_net_G_B          denoise
        └─ resnet50 + vgg16     embed, in parallel
           └─ L2-normalise
              └─ INSERT INTO items
```

Detection is skipped: the input is assumed to be a crop already.

### `POST /verify-document`

```
upload (PDF or image)
  └─ render pages at 200 dpi, capped by MAX_PDF_PAGES
     └─ for each page:
        ├─ resize a copy to 640 -> yolov8s -> normalised bbox
        ├─ crop the ORIGINAL page at that bbox      ← full resolution
        │  └─ resize 224
        │     └─ latest_net_G_B -> resnet50 + vgg16 -> L2-normalise
        │        └─ SELECT ... ORDER BY cosine distance LIMIT 1
        └─ matched = avg_distance < MATCH_THRESHOLD
```

The crop is taken from the original page, not from the 640×640 detector input.
Cropping the detector input would bake its downsampling into the embedding, so
a verified crop would be measurably blurrier than the sharp image the same
signature was enrolled from — a systematic mismatch between the two endpoints.

## Ports

| Port | Service | Notes |
|---|---|---|
| 8110 | nginx | Web UI, and `/api/` proxied to the API |
| 8080 | FastAPI | Direct access; governed by `CORS_ORIGINS` |
| 8010 | Triton HTTP | KServe v2 |
| 8011 | Triton gRPC | Not used by the API |
| 8012 | Triton metrics | Prometheus |
| 5432 | PostgreSQL | Bound to 127.0.0.1 |

## Stack

| Layer | Choice | Why |
|---|---|---|
| Serving | Triton 24.01 | One process holds all four models on the GPU; the API stays stateless |
| API | FastAPI + uvicorn | The two extractor calls are issued concurrently with `asyncio.gather` |
| Vectors | PostgreSQL + pgvector | Embeddings live beside the identity rows; no second datastore |
| Training | TensorFlow (extractors), PyTorch (CycleGAN) | Follows the reference implementations |
| Export | ONNX | One format both frameworks reach and Triton serves |

## Boundary between the halves

`training/` produces a Triton model repository. `inference/` consumes one.
Nothing else crosses.

```bash
cd training && sigtrain export        # -> ../inference/triton/model_repository/
cd ../inference && docker compose up -d
```

The generated `config.pbtxt` is derived from the exported ONNX graph, so tensor
names and shapes cannot disagree with the model beside them.

**The one contract not enforced by a file format** is pixel preprocessing —
each model expects a different tensor convention, and sending the wrong one
fails silently. See [Pipeline deep dive](./02-pipeline-deep-dive.md).
