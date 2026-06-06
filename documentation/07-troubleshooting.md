# 7 — Troubleshooting & FAQ

> Common issues, debugging tips, and environment-specific notes.

---

## Table of Contents

1. [Docker & Deployment Issues](#1-docker--deployment-issues)
2. [Triton Inference Server Issues](#2-triton-inference-server-issues)
3. [Model Training Issues](#3-model-training-issues)
4. [Model Conversion Issues](#4-model-conversion-issues)
5. [API & Database Issues](#5-api--database-issues)
6. [Frontend Issues](#6-frontend-issues)
7. [Performance Tuning](#7-performance-tuning)

---

## 1. Docker & Deployment Issues

### Triton container fails to start

**Symptom:** `sig_triton` container exits immediately or healthcheck fails.

**Common causes:**

1. **Missing ONNX files** — Triton won't start if `model.onnx` is missing from any model directory.

   ```bash
   # Check all model dirs have model.onnx
   ls -la triton/model_repository/*/1/
   ```

2. **GPU not available** — If NVIDIA drivers or Container Toolkit aren't installed.

   ```bash
   # Test GPU access
   docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi
   ```

   **Fix:** Install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html), or switch to CPU mode (see [03-service-deployment.md §7](./03-service-deployment.md#7-gpu-vs-cpu-mode)).

3. **config.pbtxt mismatch** — Input/output names or shapes don't match the ONNX graph.

   ```bash
   # Check Triton logs for model loading errors
   docker compose logs triton | grep -i "error\|fail"
   ```

### API container restarts in a loop

**Symptom:** `sig_api` keeps restarting, logs show connection errors.

**Cause:** API starts before Postgres/Triton are healthy.

**Fix:** The `docker-compose.yml` already has `depends_on` with `condition: service_healthy`. If it still fails, increase the `start_period` in the healthcheck config.

### Private registry images can't be pulled

**Symptom:** `Error: pull access denied for 10.254.144.152/tessel/...`

**Cause:** The docker-compose references a private registry (`10.254.144.152`).

**Fix:**
- Ensure you're on the same network as the registry
- Or replace the image references with public equivalents:
  ```yaml
  # Replace:
  image: 10.254.144.152/tessel/pgvector/pgvector:pg17
  # With:
  image: pgvector/pgvector:pg17
  
  # Replace:
  image: 10.254.144.152/tessel/tritonserver:24.01-py3
  # With:
  image: nvcr.io/nvidia/tritonserver:24.01-py3
  
  # Replace:
  image: 10.254.144.152/tessel/nginx:alpine
  # With:
  image: nginx:alpine
  
  # In api/Dockerfile, replace:
  FROM 10.254.144.152/tessel/python:3.10-slim-buster
  # With:
  FROM python:3.10-slim-buster
  ```

---

## 2. Triton Inference Server Issues

### "Model not found" error

**Symptom:** API returns `502` with `Triton CycleGAN error` or similar.

**Cause:** Triton model name in code doesn't match the directory name.

**Check:**
| Code uses | Must exist as directory |
|-----------|----------------------|
| `yolov8s` | `triton/model_repository/yolov8s/` |
| `latest_net_G_B` | `triton/model_repository/latest_net_G_B/` |
| `resnet50_extractor` | `triton/model_repository/resnet50_extractor/` |
| `vgg16_extractor` | `triton/model_repository/vgg16_extractor/` |

### ONNX input/output name mismatch

**Symptom:** Inference returns garbage or errors about missing tensor names.

**Fix:** The Triton `config.pbtxt` input/output names MUST match the ONNX graph:

| Model | config.pbtxt input name | config.pbtxt output name |
|-------|------------------------|-------------------------|
| `yolov8s` | `images` | `output0` |
| `latest_net_G_B` | `input` | `output` |
| `resnet50_extractor` | `input_layer_1` | `fc1` |
| `vgg16_extractor` | `input_layer` | `fc1` |

Verify with:
```python
import onnx
model = onnx.load("model.onnx")
print([inp.name for inp in model.graph.input])
print([out.name for out in model.graph.output])
```

### Triton running but very slow

**Possible causes:**
1. Using CPU mode instead of GPU
2. ONNX model not optimized — try converting to TensorRT `.plan`
3. Model files too large — VGG16 ONNX is ~470 MB

---

## 3. Model Training Issues

### ResNet50 produces terrible verification results

**Symptom:** ResNet50 extractor has very low AUC-ROC or high EER.

**Root cause (90% of cases):** Wrong preprocessing or BatchNorm corruption.

**Checklist:**
- [ ] Using `keras.applications.resnet50.preprocess_input` (Caffe-style), NOT `rescale=1/255`
- [ ] ALL BatchNorm layers are frozen (`layer.trainable = False`) during BOTH phases
- [ ] Phase 2 does NOT unfreeze BatchNorm layers

The code in `train_verification.py` already has this fix. If you modify training code, keep this:

```python
for layer in model.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False  # NEVER unfreeze BN for small datasets
```

### `ValueError: target.shape=(None, 21) output.shape=(None, 64)`

**Cause:** Using `test/` as validation data — test set has different persons than train set.

**Fix:** Validation is split from `train/` (15% by default). The `--test_dir` argument
is NOT used during training. This is by design — the dataset is writer-independent.

### CycleGAN training produces blurry outputs

**Checklist:**
- [ ] Using `--norm instance` (instance normalization, not batch norm)
- [ ] Data is 512×512 (standard CycleGAN resolution)
- [ ] Training for enough epochs (200+ total: 100 constant LR + 100 decaying)
- [ ] trainA and trainB have matching image counts

### Stamp augmentation not working

**Symptom:** trainB images look the same as trainA.

**Cause:** Stamp folder is empty or path is wrong.

**Fix:**
```bash
# Check stamp images exist
ls data/stamp_noise_data/
# Should contain .jpg/.png stamp images
```

---

## 4. Model Conversion Issues

### `inputs_as_nchw` — NHWC vs NCHW confusion

**Symptom:** After conversion, Triton inference produces wrong results.

**Root cause:** Keras models use NHWC (channels-last) internally. Triton sends NCHW
(channels-first). The `inputs_as_nchw` flag in `tf2onnx` adds a transpose node.

**Fix:** Always use `inputs_as_nchw` when converting Keras models:

```python
tf2onnx.convert.from_keras(
    model,
    input_signature=spec,
    opset=13,
    inputs_as_nchw=[input_name]  # ← REQUIRED
)
```

### TensorRT `.plan` file doesn't work on a different GPU

**Cause:** `.plan` files are compiled for a specific GPU architecture.

**Fix:** Rebuild the `.plan` on the target GPU. Use ONNX as the portable format.

### ONNX export fails with "Integer Overflow"

**Cause:** Some TensorFlow/Keras operations produce very large constants.

**Fix:** Use `onnxsim` (ONNX Simplifier) to optimize the graph:

```python
from onnxsim import simplify
import onnx

model = onnx.load("model.onnx")
simplified, check = simplify(model)
onnx.save(simplified, "model_simplified.onnx")
```

---

## 5. API & Database Issues

### pgvector L2 search returns wrong results

**Symptom:** Verification always matches the wrong person or distances are very large.

**Checklist:**
- [ ] Registration used the same preprocessing as verification
- [ ] CycleGAN is working correctly (check `crop_after` images in verify response)
- [ ] Vector dimensions match (both should be 4096-d)
- [ ] Vectors are not all zeros (check Triton model outputs)

### Database connection refused

**Symptom:** `Connection refused` or `pg_isready` fails.

**Fix:**
```bash
# Check postgres is running
docker compose ps postgres
docker compose logs postgres

# Check the connection string
docker exec -it sig_api env | grep POSTGRES
```

### PDF processing fails

**Symptom:** `RuntimeError: pdf2image not available – install poppler-utils`

**Fix:** The Dockerfile already installs `poppler-utils`. If running locally:

```bash
# Ubuntu/Debian
sudo apt-get install poppler-utils

# macOS
brew install poppler

# Windows
# Download from https://github.com/oschwartz10612/poppler-windows/releases
```

---

## 6. Frontend Issues

### Frontend can't connect to API

**Symptom:** API health dot is red, all operations fail.

**Fixes:**
1. Check the API base URL in the frontend header (should be `http://<host>:8080`)
2. Check CORS settings in `api/app/main.py` — add your frontend URL to `allow_origins`
3. Check nginx config — the API is NOT proxied through nginx; the frontend calls API directly

### Images not displaying in verification results

**Symptom:** Blank image boxes in the Verify Document tab.

**Cause:** The API returns base64-encoded PNG images. If Triton inference fails,
these fields will be empty.

**Fix:** Check Triton logs and ensure all models are loaded correctly.

---

## 7. Performance Tuning

### Match threshold calibration

The default threshold is `0.5` (L2 distance). To calibrate:

1. Register signatures from known users
2. Run verification with known documents
3. Collect `avg_distance` values for true matches and non-matches
4. Set threshold to separate the two distributions

```python
# In api/app/routers/signatures.py
"matched": float(match["avg_distance"]) < 0.5   # Adjust this value
```

Lower threshold = stricter (fewer false positives, more false negatives).

### Triton performance

| Setting | Location | Impact |
|---------|----------|--------|
| `max_batch_size` | `config.pbtxt` | Higher = better throughput |
| `instance_count` | `config.pbtxt` | More instances = more parallel requests |
| TensorRT `.plan` | Model directory | 2-5× faster than ONNX |
| Dynamic batching | `config.pbtxt` | Groups requests for efficiency |

### Database performance

| Optimisation | Impact |
|-------------|--------|
| HNSW index `m` parameter | Higher = better recall, slower inserts |
| HNSW `ef_construction` | Higher = better index quality, slower build |
| `ef_search` (query time) | `SET hnsw.ef_search = 100;` — higher = better recall |
| Connection pooling | Already set: pool_size=10, max_overflow=20 |

### API performance

| Setting | Value | Location |
|---------|-------|----------|
| Uvicorn workers | 2 | `api/Dockerfile` CMD |
| Event loop | uvloop | `api/Dockerfile` CMD |
| HTTP parser | httptools | `api/Dockerfile` CMD |
| Auto-reload | enabled | Remove `--reload` for production |

---

## Environment-Specific Notes

### Network (Internal/Air-gapped)

This project was developed for an internal network environment:
- Docker images are pulled from a private registry at `10.254.144.152`
- Python packages use a private PyPI mirror at `10.254.144.164:8081`
- APT packages use an internal Ubuntu mirror

If deploying on a public network, replace all internal URLs with their public equivalents (see [Docker section](#private-registry-images-cant-be-pulled)).

### Timezone

The API container uses `Asia/Ho_Chi_Minh` timezone (UTC+7). This affects:
- `user_created_date` and `user_modified_date` timestamps
- Log timestamps

To change, modify `TZ` environment variable in `api/Dockerfile`.
