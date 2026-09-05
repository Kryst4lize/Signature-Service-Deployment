# Training Pipeline

Everything that produces a model: dataset construction, CycleGAN training,
verification fine-tuning, evaluation, and ONNX export into a Triton model
repository.

The output of this half is the input to [`../inference/`](../inference/).
Nothing here runs in production.

---

## Contents

- [The pipeline](#the-pipeline)
- [Layout](#layout)
- [Setup](#setup)
- [Running it](#running-it)
- [Stages](#stages)
- [Configuration](#configuration)
- [Handing models to the service](#handing-models-to-the-service)
- [Tests](#tests)
- [Preprocessing contract](#preprocessing-contract)

---

## The pipeline

```
  Kaggle signature dataset            stamp/seal images
            │                                │
            ├──────────────┬─────────────────┘
            │              │
   data-verification   data-cyclegan
    genuine folders     clean (A) + synthesised noisy (B) pairs
            │              │
            │        train-cyclegan  ──>  latest_net_G_B.pth   (noisy -> clean)
            │
   train-verification  ──>  vgg16_extractor.keras     4096-d
                            resnet50_extractor.keras  4096-d
            │
        evaluate        ──>  EER / AUC / d' / TAR@FAR  +  MATCH_THRESHOLD
            │
         export         ──>  ../inference/triton/model_repository/
                                 yolov8s/1/model.onnx
                                 latest_net_G_B/1/model.onnx
                                 resnet50_extractor/1/model.onnx
                                 vgg16_extractor/1/model.onnx
```

CycleGAN trains both directions. `trainA` is clean and `trainB` is noisy, so
**`G_B` (noisy → clean) is the denoiser that gets deployed**; `G_A` is a
training by-product.

---

## Layout

```
training/
├── configs/default.yaml        one config for every stage
├── src/signature_training/
│   ├── cli.py                  `sigtrain` — the pipeline driver
│   ├── config.py               YAML + --set overrides
│   ├── paths.py                package-relative asset resolution
│   ├── data/
│   │   ├── cyclegan.py         dataset builders
│   │   └── noise/
│   │       ├── document.py     form rules, cell borders, caption text
│   │       └── stamps.py       DPI-aware seal compositing
│   ├── models/backbones.py     VGG16 / ResNet50 + extractor truncation
│   ├── train/
│   │   ├── verification.py     two-phase fine-tune
│   │   └── cyclegan.py         wraps the upstream repo
│   ├── evaluate/
│   │   ├── metrics.py          EER, TAR@FAR, FNMR@FMR, d'
│   │   ├── pairs.py            genuine/impostor pair construction
│   │   ├── plots.py            distribution, ROC, DET, threshold sweep
│   │   └── runner.py
│   └── export/
│       ├── keras_onnx.py       extractors -> ONNX (NCHW)
│       ├── cyclegan_onnx.py    .pth -> ONNX via the upstream define_G
│       ├── yolo_onnx.py        detector -> ONNX
│       ├── triton_config.py    config.pbtxt from the ONNX graph
│       └── repository.py       assembles the whole model repository
├── assets/times.ttf            caption font
├── data/                       datasets (gitignored; see data/README.md)
├── artifacts/                  checkpoints, models, ONNX, plots (gitignored)
├── external/                   cloned CycleGAN repo (gitignored)
└── tests/
```

---

## Setup

### Docker (recommended — GPU)

```bash
cd training/
docker compose build          # public CUDA base by default
docker compose up -d
docker compose exec training bash
sigtrain --help
```

On the NVIDIA network, put this in `training/.env` first — assignments only,
Compose rejects any line without `=`:

```ini
TRAINING_BASE_IMAGE=10.254.144.152/tessel/training-service-base:0.0.1
USE_INTERNAL_APT=1
```

and, optionally, supply the internal PyPI mirror config:

```bash
cp pip.conf.example pip.conf
```

### Local

```bash
cd training/
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
sigtrain --help
```

### Data

See [`data/README.md`](data/README.md). In short: extract the Kaggle
signature-verification dataset to `data/raw/sign_data/{train,test}/` and drop
stamp images in `data/raw/stamps/`. Then:

```bash
sigtrain setup     # clones the CycleGAN repo, reports anything missing
```

---

## Running it

Everything, in order:

```bash
sigtrain all
```

Or one stage at a time (each is independently re-runnable):

```bash
sigtrain data-cyclegan
sigtrain data-verification
sigtrain train-cyclegan
sigtrain train-verification
sigtrain evaluate
sigtrain export
```

Skip what you have already done:

```bash
sigtrain all --skip train-cyclegan data-cyclegan
```

Override any setting without editing the config. `--set` is accepted either
before or after the stage name:

```bash
sigtrain train-verification --set verification.phase1_epochs=5 \
                            --set verification.batch_size=16

sigtrain --set verification.batch_size=16 train-verification
```

---

## Stages

| Stage | Does | Produces |
|---|---|---|
| `setup` | Clones the CycleGAN repo; validates the data layout | `external/pytorch-CycleGAN-and-pix2pix/` |
| `data-cyclegan` | Pads signatures to a square, synthesises form rules, captions and stamps | `data/processed/cyclegan/{train,test}{A,B}/` |
| `data-verification` | Copies genuine-only person folders | `data/processed/verification/{train,test}/` |
| `train-cyclegan` | Runs upstream `train.py` with config-derived arguments | `artifacts/cyclegan/signature/latest_net_G_{A,B}.pth` |
| `train-verification` | Two-phase fine-tune, then truncates at `fc1` | `artifacts/models/*_extractor.keras` |
| `evaluate` | Genuine/impostor pairs over held-out identities | `artifacts/evaluation/{*.png,metrics.json}` |
| `export` | ONNX + `config.pbtxt`, staged into the service | `../inference/triton/model_repository/` |

### Why train a classifier for a verification task

Following [arXiv:2004.12104](https://arxiv.org/abs/2004.12104): the backbones are
fine-tuned as N-way person classifiers over genuine signatures, then truncated
at `fc1`. The penultimate activations are the identity embedding, and
verification becomes a distance in that space — which generalises to people who
were never in the training set.

Forgeries (`_forg`) are excluded everywhere. Feeding them in as that person's
own signature would teach the model to place them *near* the genuine ones.

### Why validation is split out of `train/`

The dataset is writer-independent: `test/` contains different people. Using it
as validation gives the model 64 output neurons and the validation labels 21
classes —

```
ValueError: target.shape=(None, 21)  output.shape=(None, 64)
```

so 15% is held out of `train/` instead (`verification.val_split`).

### Two-phase fine-tuning

1. Backbone frozen, `lr=1e-3` — trains only the new head, so random head
   gradients cannot wreck the pretrained features.
2. Everything unfrozen except BatchNorm, `lr=1e-4`.

ResNet50's 53 BatchNorm layers stay in inference mode throughout. Letting them
update running statistics on a few thousand signature images — a distribution
nothing like ImageNet — corrupts exactly the pretrained features this approach
depends on. VGG16 has no BatchNorm and never hits it.

---

## Configuration

One [`configs/default.yaml`](configs/default.yaml) drives every stage, so they
cannot disagree about where the data is. Precedence:

```
defaults in config.py  <  configs/default.yaml  <  --set section.field=value
```

Relative paths resolve against `training/`, never the shell's working
directory. Unknown keys are an error, not a silent no-op.

Each training run writes the fully resolved config to
`artifacts/models/config.used.yaml`, so a result can be traced to the settings
that produced it.

---

## Handing models to the service

```bash
sigtrain export
# -> ../inference/triton/model_repository/<model>/{config.pbtxt,1/model.onnx}

cd ../inference && docker compose up -d
```

`config.pbtxt` is generated from the exported ONNX graph, so tensor names and
shapes cannot drift from the model sitting next to them.

Serving without a GPU:

```bash
sigtrain export --set export.instance_kind=KIND_CPU
```

The detector is exported from `artifacts/models/yolov8s.pt` if present. YOLO
training itself is not part of this pipeline — use the ultralytics CLI and drop
the resulting `.pt` there.

---

## Tests

```bash
make test
```

No GPU, no TensorFlow, no torch: the suite covers dataset construction,
augmentation determinism, config resolution, metrics, pair building and the CLI.

---

## Preprocessing contract

The extractors are trained with Keras `preprocess_input(mode="caffe")` applied
**outside** the model by `ImageDataGenerator`. The exported ONNX therefore
begins at Conv1 on an already-preprocessed tensor — BGR, ImageNet mean
subtracted, `[0, 255]` scale — and **the serving side must reproduce it**.

It does, in `inference/api/app/triton.py:to_caffe`, which is pinned by a test
verified bit-for-bit against Keras.

If you change preprocessing here, change it there too. Nothing will fail
loudly if you don't: the models still return well-formed tensors of the right
shape, just computed on input they were never trained on.

The same applies to the denoiser: the CycleGAN generator is trained through
`Normalize(0.5, 0.5)` and ends in `Tanh`, so it expects `[-1, 1]` and returns
`[-1, 1]`.
