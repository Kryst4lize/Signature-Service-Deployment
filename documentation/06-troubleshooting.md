# 6 — Troubleshooting

Symptom → cause → fix. Failures in this system are unusually quiet, so the
"looks fine but is wrong" section at the end is worth reading before the
crash-with-a-message ones.

---

## Deployment

### `docker compose up` fails immediately with an empty-variable warning

```
WARN[0000] The "POSTGRES_PASSWORD" variable is not set. Defaulting to a blank string.
error: required variable POSTGRES_PASSWORD is missing a value
```

No `.env`, or it has a blank password.

```bash
cd inference && cp .env.example .env
# POSTGRES_PASSWORD has no default, on purpose:
openssl rand -base64 24
```

### `sig_postgres` exits with code 1

```
Database is uninitialized and superuser password is not specified
```

Same cause as above.

### `sig_postgres` exits with code 3, and later `relation "items" does not exist`

Fixed in v2.0.0. `init.sql` used to create an HNSW index on a 4096-d column,
which pgvector rejects (2000-d cap). Init aborted under `ON_ERROR_STOP=1`,
`restart: unless-stopped` brought the container back onto a non-empty `PGDATA`,
and initialisation was then skipped forever — leaving a healthy-looking
database with no tables.

If you are on an older deployment that hit this, the volume never got a schema:

```bash
docker compose down -v      # destroys the volume
docker compose up -d
```

### The frontend container will not start

```
not a directory: Are you trying to mount a directory onto a file?
```

`nginx/nginx.conf` is missing, so Docker created it as a directory. It ships in
v2.0.0; on older checkouts, `.gitignore`'s `*.conf` prevented it from ever being
committed. Delete the stray directory and pull the file.

### `nginx: [emerg] host not found in upstream "api"`

nginx started before the API was resolvable and resolves literal upstreams at
config load. The shipped config avoids this by holding the host in a variable
with an explicit `resolver 127.0.0.11`. If you edited it back to
`proxy_pass http://api:8080`, restore the variable form.

### `failed to compute cache key: "/pip.conf": not found`

The Dockerfile used to `COPY pip.conf` unconditionally, and `pip.conf` is
gitignored. v2.0.0 makes it optional. On the NVIDIA network:

```bash
cd inference/api && cp pip.conf.example pip.conf
# and in inference/.env:  USE_INTERNAL_APT=1
```

### `failed to resolve source metadata for 10.254.144.152/tessel/...`

The internal registry is unreachable from your network. The images now default
to public bases; if you set `API_BASE_IMAGE` or `TRAINING_BASE_IMAGE` to the
internal one, unset it.

### `could not select device driver "nvidia" with capabilities: [[gpu]]`

No GPU, or the NVIDIA container toolkit is not installed.

```bash
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
```

If that fails, install the toolkit, or switch to CPU — see
[Operations](./05-operations.md#cpu-only).

---

## Triton

### Triton exits at startup

```
failed to load 'vgg16_extractor' version 1: Invalid argument: model directory does not contain any version
```

Weights are not in place. Every model directory needs `1/model.onnx`:

```bash
cd training && sigtrain export
```

The repository ships only `config.pbtxt` — the ONNX files are hundreds of MB of
build output and are gitignored.

### `unexpected inference input 'input_layer'` / shape mismatch

The `config.pbtxt` disagrees with the ONNX graph. Check what the model actually
declares:

```bash
python -c "
import onnx, sys
g = onnx.load(sys.argv[1]).graph
for t in list(g.input) + list(g.output):
    print(t.name, [d.dim_value or d.dim_param for d in t.type.tensor_type.shape.dim])
" triton/model_repository/vgg16_extractor/1/model.onnx
```

Regenerating with `sigtrain export` derives the config from the graph, so it
cannot disagree. This normally means the ONNX was produced some other way —
notably by the old `convert_to_onnx_gemini.py`, which omitted `inputs_as_nchw`
and emitted an NHWC input.

Remember `dims` excludes the batch dimension when `max_batch_size >= 1`:
`[3, 224, 224]` describes a `[1, 3, 224, 224]` request.

### `502 Triton inference error: Cannot connect to host triton:8000`

Triton is not up. `docker compose logs triton`, and
`curl localhost:8010/v2/health/ready`.

---

## Training

### `FileNotFoundError: No images under .../data/raw/sign_data`

The dataset is not extracted, or not in the expected layout. Run
`sigtrain setup` — it prints exactly what it expected and what it found.
See [`../training/data/README.md`](../training/data/README.md).

### `FileNotFoundError: Caption font not found`

`assets/times.ttf` is missing from the install. In a container this usually
means the package was installed without package data; reinstall with
`pip install -e .` from `training/`.

This used to fail differently and much worse: the font was loaded from a path
relative to the working directory, the resulting `OSError` was swallowed
per-image, and the run reported a full dataset having written nothing.

### `RuntimeError: Wrote 0 pairs from N source images`

Every image failed. The preceding log lines list the first ten reasons. This
condition used to be reported as success.

### CycleGAN training exits immediately

```
FileNotFoundError: .../data/processed/cyclegan/trainA not found
```

Run `sigtrain data-cyclegan` first. If the repo itself is missing, `sigtrain
setup` clones it.

### `pip install -r requirements.txt` fails on `triton==2.3.0`

Fixed in v2.0.0. That pin was openai/triton, the GPU kernel DSL — unrelated to
Triton Inference Server. It publishes only manylinux x86_64 wheels, so the
install failed outright on macOS, aarch64 and Python 3.12+. torch declares it
already, correctly guarded by platform markers.

### `ModuleNotFoundError: No module named 'onnxslim'`

Fixed in v2.0.0. The converter imported `onnxsim` while requirements pinned
`onnxslim` — different projects. Resolved in favour of onnxslim (pure Python,
wheels for every platform; onnxsim has no aarch64 wheel and needs cmake).
Simplification is also now optional and skipped with a warning if absent.

### CUDA out of memory

```bash
sigtrain train-verification --set verification.batch_size=8
sigtrain train-cyclegan --set cyclegan_train.crop_size=128
```

`nvidia-smi` to confirm nothing else holds the GPU.

---

## Quiet failures

These produce no error. They are the expensive ones.

### Everything runs, but nothing ever matches

Most likely the preprocessing contract is violated somewhere. Each model wants
a different pixel convention and returns a well-formed tensor regardless — see
[Pipeline deep dive](./02-pipeline-deep-dive.md#the-preprocessing-contract).

Check, in order:

1. **Mixed-vintage embeddings.** Rows enrolled before a preprocessing or model
   change are not comparable with rows after it. `TRUNCATE items` and re-enrol.
2. **Threshold units.** `MATCH_THRESHOLD` is a cosine *distance*;
   `sigtrain evaluate` reports a cosine *similarity*. The correct value is
   `1 - eer_threshold`. Inverting it accepts everyone or no one.
3. **Sanity check the distances.** Register a signature, then verify the same
   image. `avg_distance` should be near 0. If it is near 1, the embeddings are
   effectively random and preprocessing is wrong.

```bash
curl -s -X POST localhost:8080/register-signature \
  -F username=selftest -F file=@sig.png
curl -s -X POST localhost:8080/verify-document -F file=@sig.png \
  | python -m json.tool | grep -E "distance|matched"
```

### Denoised crops are half black

`crop_after` looks like the signature with the darker half missing. The
CycleGAN Tanh output (`[-1, 1]`) is being read as `[0, 1]`, so every negative
value clips to 0. Fixed in v2.0.0 by `from_cyclegan`; if you see it again,
something bypassed that conversion.

### Every generated training image has identical noise

Fixed in v2.0.0. The RNG was re-seeded with a fixed literal inside the
per-image noise function, so all images got the same two rules in the same
places — and the denoiser learned to erase two specific rows.

```bash
cd training && make test    # test_noise.py covers this directly
```

### Verified crops look blurrier than enrolled ones

Fixed in v2.0.0: the crop is taken from the original page rather than from the
640×640 detector input.

### The exported denoiser does nothing

The old converters defined their own CycleGAN generator with the wrong
`state_dict` key layout and loaded with `strict=False`, yielding a randomly
initialised network that exports cleanly. `export/cyclegan_onnx.py` uses the
upstream `define_G` and `strict=True`, so a mismatch is now an error.

---

## Diagnostics

```bash
# service
cd inference && docker compose ps && curl -s localhost:8080/health
curl -s localhost:8010/v2/models/stats | python -m json.tool | head -40

# database
docker compose exec postgres psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" \
  -c "SELECT count(*) FROM items;"

# tests
cd inference && make test-integration
cd training  && make test
```
