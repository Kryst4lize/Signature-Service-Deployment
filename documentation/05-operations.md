# 5 — Operations

Running the service over time. For first-time setup see
[`../inference/README.md`](../inference/README.md).

---

## Exposure

**There is no authentication.** `/register-signature` and `/verify-document`
are open, unauthenticated multipart endpoints that each cost GPU time.

Before putting this anywhere reachable:

- Terminate TLS and authenticate in front of it — the nginx container is the
  natural place, or an existing gateway.
- Keep `CORS_ORIGINS` to the origins you actually serve. It is not a security
  boundary (it constrains browsers, not `curl`), but a permissive value invites
  a browser-driven enrolment from any page a user visits.
- `MAX_UPLOAD_BYTES` and `MAX_PDF_PAGES` are the only backpressure. A 20-page
  PDF is 80 GPU inferences from one unauthenticated request.
- Postgres publishes on `127.0.0.1` only. Keep it that way.

---

## Upgrading

### Re-enrol after a preprocessing or model change

Embeddings are comparable only with others produced by the same preprocessing
and the same extractor weights. Changing either invalidates every stored row.

This applies to the v2.0.0 upgrade specifically: rows written before it were
computed from input the extractors were never trained on.

```bash
docker compose exec postgres psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" \
  -c "TRUNCATE items RESTART IDENTITY;"
# then re-register every signature
```

Nothing detects a mixed table. It degrades matching quietly.

### Replacing models

```bash
cd training && sigtrain export
cd ../inference && docker compose restart triton
```

Triton loads at startup, so the restart is required. To avoid it, run the
server with `--model-control-mode=explicit` and use the model-repository API.

Keep the old ONNX files until the new ones are validated — rolling back is
copying the directory back and restarting.

### Recalibrating the threshold

After retraining:

```bash
cd training && sigtrain evaluate     # prints MATCH_THRESHOLD directly
```

Set it in `inference/.env` and `docker compose up -d api`. It is read at
startup, so a restart is needed.

---

## Monitoring

### Health

```bash
curl -sf localhost:8080/health                        # API process
curl -sf localhost:8010/v2/health/ready               # Triton + all models
curl -sf localhost:8110/nginx-health                  # frontend
docker compose exec postgres pg_isready -U "$POSTGRES_USER"
```

`/health` reports only that the API process is serving. It does not check
Postgres or Triton, so it stays up while dependencies are down — check all four.

### Triton metrics

Prometheus format on `:8012/metrics`. The useful series:

| Metric | Watch for |
|---|---|
| `nv_inference_request_success` / `_failure` | Failures rising = a model is unhealthy |
| `nv_inference_queue_duration_us` | Growing = GPU saturated |
| `nv_inference_compute_infer_duration_us` | Per-model latency |
| `nv_gpu_memory_used_bytes` | Headroom |

### Logs

```bash
docker compose logs -f api
docker compose logs -f triton
docker compose logs --tail=200 postgres
```

The API logs one line per stage per page. `--log-verbose=1` on Triton logs
every tensor and is not appropriate for production; it was removed from the
compose file for that reason.

---

## Capacity

Per page with a detection: one YOLOv8 pass, one CycleGAN pass, and two
extractor passes issued concurrently. Pages are processed sequentially within a
request.

The database side is an exact scan, ~80 ms at 5,000 rows, growing linearly.
See [Database](./04-database.md#if-enrolment-outgrows-that).

Scaling levers, roughly in order:

1. `--workers` on uvicorn (already 2). The API is I/O-bound on Triton, so more
   workers help until the GPU saturates.
2. `count` in each `config.pbtxt` `instance_group` — more concurrent model
   instances per GPU, at the cost of memory.
3. Raise `max_batch_size` and batch pages. Requires client changes; the current
   client sends one tensor at a time.
4. More GPUs — Triton spreads instances across them.

Watch `nv_inference_queue_duration_us` to tell which side is the bottleneck.

---

## Backup

Two things carry state:

| What | Where | Recovery |
|---|---|---|
| Enrolled signatures | `postgres_data` volume | `pg_dump`; see [Database](./04-database.md#backup) |
| Model weights | `triton/model_repository/*/1/model.onnx` | Re-export from `training/` |

The ONNX files are gitignored build outputs. Either back them up or make sure
the checkpoints in `training/artifacts/` are retained — losing both means
retraining.

---

## Common changes

### CPU-only

```bash
# 1. delete the deploy.resources block from the triton service
# 2. regenerate the configs for CPU:
cd training && sigtrain export --set export.instance_kind=KIND_CPU
```

Expect roughly an order of magnitude more latency per page.

### Hot reload for development

```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml up
```

The production image deliberately omits `--reload`: uvicorn's `should_reload`
takes precedence over `--workers`, so shipping it silently turns a two-worker
service into a single-process reloader that restarts on any file change.

### Behind an existing gateway

Point it at `:8080` and drop the nginx container, or keep nginx and point the
gateway at `:8110` to get the SPA and the API on one origin.
