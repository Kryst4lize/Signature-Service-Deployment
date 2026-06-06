# 2 — Training Pipeline — Setup & Run

> **Folder:** `trainingfiles/`
>
> This guide covers how to prepare data, train all four models
> (YOLOv8, CycleGAN, VGG16, ResNet50), evaluate, and export them to ONNX.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Environment Setup (Docker)](#2-environment-setup-docker)
3. [Environment Setup (Local)](#3-environment-setup-local)
4. [Data Layout](#4-data-layout)
5. [Step 1 — Prepare CycleGAN Dataset](#5-step-1--prepare-cyclegan-dataset)
6. [Step 2 — Train CycleGAN](#6-step-2--train-cyclegan)
7. [Step 3 — Train Verification Models (VGG16 + ResNet50)](#7-step-3--train-verification-models)
8. [Step 4 — Evaluate Models](#8-step-4--evaluate-models)
9. [Step 5 — Convert Models to ONNX](#9-step-5--convert-models-to-onnx)
10. [Script Reference](#10-script-reference)

---

## 1. Prerequisites

### Hardware
- **GPU**: NVIDIA GPU with ≥8 GB VRAM (training uses CUDA)
- **RAM**: ≥16 GB system RAM
- **Disk**: ≥50 GB free (datasets + model checkpoints)

### Software
- **Docker** with NVIDIA Container Toolkit (for Docker workflow)
- **Python 3.10+** (for local workflow)
- **NVIDIA Driver** ≥ 525.x
- **CUDA** 12.x (bundled in Docker base image)

---

## 2. Environment Setup (Docker)

The Docker approach is **recommended** as it bundles all dependencies.

### 2.1 Build the base image (one-time)

```bash
cd trainingfiles/
docker build -t 10.254.144.152/tessel/training-service-base:0.0.1 -f Dockerfile .
```

> **Note:** The Dockerfile references `../libs_local/*.whl` for TensorRT wheels.
> Ensure those wheel files are available in the parent `libs_local/` directory.
> If TensorRT is not needed, comment out the TensorRT `COPY` and `RUN` lines.

### 2.2 Start the training container

```bash
cd trainingfiles/
docker-compose up -d signature-minh
```

This starts a persistent GPU container with:
- The current directory mounted at `/workspace/train`
- 8 GB shared memory (`shm_size: 8gb`)
- All NVIDIA GPUs available

### 2.3 Enter the container

```bash
docker exec -it signature-minh /bin/bash
cd trainingfiles/pyfile/
```

---

## 3. Environment Setup (Local)

If running directly on the host machine:

```bash
cd trainingfiles/

# Create virtual environment
python -m venv .venv
source .venv/bin/activate    # Linux/Mac
# .venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### Key dependencies (from `requirements.txt`)

| Package | Version | Purpose |
|---------|---------|---------|
| `tensorflow` | 2.21.0 | VGG16 & ResNet50 training |
| `torch` | 2.3.0 | CycleGAN training |
| `torchvision` | 0.18.0 | Image transforms |
| `opencv-python` | 4.13.0 | Image processing & augmentation |
| `wandb` | 0.26.1 | Experiment tracking (optional) |
| `scikit-learn` | 1.7.2 | Evaluation metrics (ROC, AUC) |
| `onnx` | latest | ONNX model validation |
| `tf2onnx` | latest | TensorFlow → ONNX conversion |
| `onnxslim` | 0.1.92 | ONNX graph optimisation |

---

## 4. Data Layout

All data lives under `trainingfiles/data/`:

```
data/
├── cyclegan_unprocessed_data/     ← Raw clean signature images (input for Step 1)
│   ├── train/                     ← Per-person folders: 001_org/, 002_org/, ...
│   ├── test/                      ← Test set (different persons)
│   └── times.ttf                  ← Font file for text noise augmentation
│
├── cyclegan_processed_data/       ← Output of Step 1 (CycleGAN training pairs)
│   ├── trainA/                    ← Clean signatures
│   ├── trainB/                    ← Noisy counterparts
│   ├── testA/
│   └── testB/
│
├── stamp_noise_data/              ← Stamp/seal images for augmentation
│
├── verification_unprocessed_data/ ← Cleaned signature crops for Step 3
│   └── full/                      ← Per-person subfolders
│
└── test_pipeline_data/            ← End-to-end test documents
```

### Dataset Source (Kaggle)

The verification training data follows the **Kaggle signature-verification-dataset** layout:

```
train/
    001_org/    ← genuine person 001  (used)
    001_forg/   ← forged  person 001  (skipped automatically)
    002_org/
    002_forg/
    ...
test/           ← different persons (writer-independent split)
```

---

## 5. Step 1 — Prepare CycleGAN Dataset

This step takes **clean signature images** and generates **noisy counterparts**
(adding stamp seals, printed lines, text overlays) to create paired training data for CycleGAN.

### Files involved

| File | Role |
|------|------|
| `pyfile/dataset_preparation.py` | Main script — builds CycleGAN dataset |
| `pyfile/stamp_augmentation.py` | Module — realistic Vietnamese stamp overlay |

### Command

```bash
cd pyfile/

python dataset_preparation.py \
    --src   ../data/cyclegan_unprocessed_data \
    --dst   ../data/cyclegan_processed_data \
    --stamps ../data/stamp_noise_data
```

### Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--src` | (required) | Folder of clean signature images |
| `--dst` | (required) | Output CycleGAN dataset folder |
| `--stamps` | `stamp_noise` | Folder with stamp/seal images |
| `--split` | `0.10` | Fraction of data reserved for test |
| `--size` | `512` | Output image size (square) |
| `--seed` | `42` | Random seed for reproducibility |

### Output structure

```
cyclegan_processed_data/
    trainA/   ← clean signatures (resized to 512×512)
    trainB/   ← noisy counterparts (lines + text + stamp)
    testA/    ← clean (test split)
    testB/    ← noisy (test split)
```

### Noise augmentation details

1. **Horizontal lines** — Simulates printed form lines (1-3 lines across the image)
2. **Vertical lines** — Simulates signature cell borders (0-2 lines near edges)
3. **Vietnamese text** — Random name/label text below the signature
4. **Stamp overlay** — DPI-aware Vietnamese circular stamp with:
   - Red ink tinting
   - Partial visibility (stamp bleeds off crop edge)
   - Opacity/fade jitter

---

## 6. Step 2 — Train CycleGAN

CycleGAN learns to translate noisy signatures → clean signatures (generator G_B→A).

### Repository Setup

The CycleGAN code relies on a third-party repository. You must clone it into the `trainingfiles/` directory before running the training script:

```bash
cd trainingfiles/
git clone https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix.git
cd pytorch-CycleGAN-and-pix2pix/
pip install -r requirements.txt
```

### Command (single-GPU)

```bash
cd pytorch-CycleGAN-and-pix2pix/

python train.py \
    --dataroot ../data/cyclegan_processed_data \
    --name signature \
    --model cycle_gan \
    --norm instance
```

### Command (multi-GPU, resume training from epoch 130)

```bash
torchrun --nproc_per_node=2 train.py \
    --dataroot ../data/cyclegan_processed_data \
    --name signature \
    --model cycle_gan \
    --norm instance \
    --continue_train \
    --epoch 130 \
    --epoch_count 131
```

### Key CycleGAN options

| Option | Default | Description |
|--------|---------|-------------|
| `--dataroot` | (required) | Path to dataset (must contain trainA, trainB) |
| `--name` | experiment_name | Experiment name (checkpoints saved under this name) |
| `--model` | `cycle_gan` | Model type |
| `--norm` | `instance` | Normalization layer (instance norm is standard for CycleGAN) |
| `--n_epochs` | 100 | Number of epochs with initial learning rate |
| `--n_epochs_decay` | 100 | Number of epochs to linearly decay LR to zero |
| `--continue_train` | — | Resume from latest checkpoint |
| `--epoch` | `latest` | Which epoch to resume from |
| `--batch_size` | 1 | Batch size (1 is standard for CycleGAN) |

### Checkpoints

Saved to: `pytorch-CycleGAN-and-pix2pix/checkpoints/signature/`

```
checkpoints/signature/
    latest_net_G_A.pth    ← Generator: clean → noisy
    latest_net_G_B.pth    ← Generator: noisy → clean  ★ (this is what we deploy)
    latest_net_D_A.pth    ← Discriminator A
    latest_net_D_B.pth    ← Discriminator B
```

> **Important:** Only `latest_net_G_B.pth` is used in production.
> It is the **denoiser** — it transforms noisy/stamped signatures into clean ones.

---

## 7. Step 3 — Train Verification Models

Fine-tune VGG16 and ResNet50 as N-class signature classifiers, then strip the
classification head to create feature extractors.

### Command

```bash
cd pyfile/

python train_verification.py \
    --train_dir ../data/verification_unprocessed_data/full \
    --test_dir  ../data/cyclegan_unprocessed_data/test \
    --output    ../model/verification_model \
    --backbone  both
```

### Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--train_dir` | (required) | Folder with per-person subfolders (e.g., `001_org/`) |
| `--test_dir` | `None` | Held-out test folder (not used during training) |
| `--output` | `saved_models` | Directory for model outputs |
| `--backbone` | `both` | Which model to train: `vgg16`, `resnet50`, or `both` |
| `--batch_size` | `32` | Training batch size |
| `--phase1_epochs` | `50` | Max epochs for frozen backbone (warm-up) |
| `--phase2_epochs` | `10` | Max epochs for full fine-tuning |
| `--val_split` | `0.15` | Fraction of train data used for validation |
| `--vgg16_weights` | `../model/vgg16_weights_...h5` | Path to local VGG16 ImageNet weights |
| `--resnet_weights` | `../model/resnet50_weights_...h5` | Path to local ResNet50 ImageNet weights |

### Training Strategy (Two-Phase)

**Phase 1 — Frozen Backbone (Head Only)**
- Freeze all backbone layers
- Train only the new classification head
- Learning rate: `1e-3` (SGD + momentum 0.9)
- Early stopping on `val_loss` (patience=6)
- Up to 50 epochs

**Phase 2 — Full Fine-Tuning**
- Unfreeze all layers EXCEPT BatchNorm (critical for ResNet50!)
- Learning rate: `1e-4`
- Early stopping continues
- Up to 10 epochs

### Feature Extractors (Output)

After training, the classification head is stripped and only the feature extraction
layers are saved:

| Model | Extractor Tap Point | Output Dimension |
|-------|---------------------|------------------|
| VGG16 | FC1 layer | 4096-d |
| ResNet50 | FC1 (after GAP → Dense(4096, relu)) | 4096-d |

### Output files

```
model/verification_model/
    vgg16_finetuned.keras          ← Full classifier (for debugging)
    vgg16_extractor.keras          ← Feature extractor (for deployment) ★
    resnet50_finetuned.keras       ← Full classifier
    resnet50_extractor.keras       ← Feature extractor ★
    vgg16/                         ← Checkpoints per phase
    resnet50/                      ← Checkpoints per phase
```

### Important Notes

> **ResNet50 BatchNorm bug:** ResNet50 has 53 BatchNormalization layers.
> If BN layers are left in training mode during fine-tuning on a small signature
> dataset, they corrupt their running statistics (mean/variance) because signatures
> look nothing like ImageNet. The fix: **always keep BN layers frozen** (`layer.trainable = False`).
> VGG16 has zero BN layers so it doesn't have this problem.

> **Preprocessing:** Uses `keras.applications.resnet50.preprocess_input` (Caffe-style
> BGR mean subtraction), NOT `rescale=1/255`. Using the wrong preprocessing produces
> terrible results.

---

## 8. Step 4 — Evaluate Models

Evaluate feature extractor quality using genuine-pair vs impostor-pair cosine similarity.

### Command

```bash
cd pyfile/

python evaluate_verification.py \
    --test_dir   ../data/cyclegan_unprocessed_data/test \
    --vgg16      ../model/verification_model/vgg16_extractor.keras \
    --resnet50   ../model/verification_model/resnet50_extractor.keras \
    --output_dir ../model/evaluation
```

### Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--test_dir` | (required) | Test folder (only `_org` subfolders are used) |
| `--vgg16` | `../model/verification_model/vgg16_extractor.keras` | Path to VGG16 extractor |
| `--resnet50` | `../model/verification_model/resnet50_extractor.keras` | Path to ResNet50 extractor |
| `--output_dir` | `../model/evaluation` | Directory for plots and reports |
| `--img_size` | `224` | Image input size |

### Evaluation Metrics

| Metric | Good Value | Description |
|--------|------------|-------------|
| **EER** | < 0.10 | Equal Error Rate — threshold where FAR = FRR |
| **AUC-ROC** | > 0.95 | Area Under ROC Curve |
| **d-prime** | > 2.0 | Separation between genuine/impostor distributions |
| **TAR @ FAR 1%** | > 0.90 | True Acceptance Rate at 1% False Acceptance |
| **FNMR @ FMR 1%** | < 0.10 | False Non-Match Rate at 1% False Match |

### Generated Plots

```
model/evaluation/
    vgg16_score_dist.png       ← Genuine vs impostor score histograms
    vgg16_roc.png              ← ROC curve
    vgg16_det.png              ← DET curve (ISO standard)
    vgg16_threshold.png        ← FAR/FRR vs threshold
    resnet50_score_dist.png
    resnet50_roc.png
    resnet50_det.png
    resnet50_threshold.png
    comparison.png             ← Side-by-side VGG16 vs ResNet50
```

---

## 9. Step 5 — Convert Models to ONNX

See the dedicated [Model Conversion Guide (04-model-conversion.md)](./04-model-conversion.md).

---

## 10. Script Reference

### `pyfile/` directory

| Script | Purpose | Run directly? |
|--------|---------|---------------|
| `dataset_preparation.py` | Build CycleGAN clean/noisy pairs | ✅ CLI |
| `stamp_augmentation.py` | Stamp noise augmentation module | ❌ Imported by dataset_preparation |
| `train_verification.py` | Train VGG16 + ResNet50 classifiers & extractors | ✅ CLI |
| `evaluate_verification.py` | Evaluate extractors with biometric metrics | ✅ CLI |
| `verification_dataset_preparation.py` | Filter Kaggle data (remove `_forg` folders) | ✅ CLI |
| `db_utils.py` | PostgreSQL + pgvector CRUD utilities | ❌ Library module |
| `env.py` | Alembic migration environment script | ❌ Used by Alembic |

### `convert_model/` directory

| Script | Purpose |
|--------|---------|
| `convert_to_onnx_gemini.py` | Universal converter: `.keras`/`.h5`/`.pt`/`.pth` → `.onnx` with Triton support |
| `simpleconver.py` | Quick VGG16 + ResNet50 `.keras` → `.onnx` with `inputs_as_nchw` |
| `convert_to_trt.py` | ONNX → TensorRT `.plan` conversion |

### `pytorch-CycleGAN-and-pix2pix/`

| Script | Purpose |
|--------|---------|
| `train.py` | Train CycleGAN (supports DDP multi-GPU) |
| `test.py` | Test/inference with trained CycleGAN |
| `export_onnx.py` | Export CycleGAN G_B generator to ONNX |

---

## Quick Start (Full Pipeline)

```bash
# 1. Enter the Docker training container
docker-compose up -d signature-minh
docker exec -it signature-minh /bin/bash
cd trainingfiles/pyfile/

# 2. Prepare CycleGAN dataset
python dataset_preparation.py \
    --src ../data/cyclegan_unprocessed_data \
    --dst ../data/cyclegan_processed_data \
    --stamps ../data/stamp_noise_data

# 3. Train CycleGAN (switch to CycleGAN directory)
cd ../pytorch-CycleGAN-and-pix2pix
python train.py \
    --dataroot ../data/cyclegan_processed_data \
    --name signature \
    --model cycle_gan \
    --norm instance

# 4. Train VGG16 + ResNet50
cd ../pyfile
python train_verification.py \
    --train_dir ../data/verification_unprocessed_data/full \
    --test_dir  ../data/cyclegan_unprocessed_data/test \
    --output    ../model/verification_model \
    --backbone  both

# 5. Evaluate
python evaluate_verification.py \
    --test_dir   ../data/cyclegan_unprocessed_data/test \
    --output_dir ../model/evaluation

# 6. Export CycleGAN to ONNX
cd ../pytorch-CycleGAN-and-pix2pix
python export_onnx.py

# 7. Convert verification models to ONNX (see 04-model-conversion.md)
cd ../convert_model
python simpleconver.py
```
