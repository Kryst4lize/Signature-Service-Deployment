# 2 — Pipeline deep dive

Why the ML is built the way it is, and the one contract that spans both halves
of the repo.

For *how to run* any of this, see [`../training/README.md`](../training/README.md).

---

## The preprocessing contract

This is the most important section in the documentation. Every model in the
pipeline expects a different pixel convention, and **sending the wrong one
fails silently** — the model returns a well-formed tensor of the correct shape,
computed on input it was never trained on. Nothing appears in the logs, no
request errors, and the only symptom is that matching quality is poor in a way
that looks like a threshold problem.

The API's internal convention is `float32 [1, 3, H, W]`, `[0, 1]`, RGB.
Conversions happen in `inference/api/app/triton.py`, next to the call each one
serves.

| Model | Expects | Because |
|---|---|---|
| `yolov8s` | `[0, 1]` RGB | Ultralytics convention; the export bakes in no normalisation |
| `latest_net_G_B` | `[-1, 1]` RGB in, `[-1, 1]` out | Trained through `Normalize((0.5,)*3, (0.5,)*3)`; the generator ends in `Tanh` |
| `resnet50_extractor` | Caffe BGR, ≈`[-124, +151]` | Trained with `preprocess_input(mode="caffe")` applied *outside* the model |
| `vgg16_extractor` | Caffe BGR, ≈`[-124, +151]` | Same |

### Why the extractors are the subtle one

`ImageDataGenerator(preprocessing_function=preprocess_input)` applies the
transform *before* the model sees anything. No `Rescaling` or `Normalization`
layer is ever added to the graph. So the saved `.keras` model — and therefore
the exported ONNX — begins at `Conv1` on an **already-preprocessed** tensor.

Caffe mode does three things, none of which the graph knows about:

```
RGB -> BGR                                   channel reversal
x * 255                                      [0,1] -> [0,255]
subtract [103.939, 116.779, 123.68]          per-channel, BGR order
```

`inference/api/app/triton.py:to_caffe` reproduces this, and
`inference/tests/test_tensors.py` pins it. It was verified bit-for-bit against
TensorFlow 2.21 — maximum absolute difference `0.000e+00` for both backbones.

### If you change preprocessing

Change it in both halves and update the tests. Training-side preprocessing is
in `training/src/signature_training/train/verification.py`; serving-side is
`to_caffe` / `to_cyclegan` / `from_cyclegan`.

---

## Why a classifier for a verification task

Following [arXiv:2004.12104](https://arxiv.org/abs/2004.12104).

The backbones are fine-tuned as N-way **person classifiers** over genuine
signatures, then truncated at `fc1`. The penultimate activations become the
identity embedding, and verification is a distance in that space.

The alternative — training a binary "same person?" head — needs pairs, scales
quadratically, and produces a model that only answers the question it was
trained on. A classification objective forces the network to separate
identities, and that separation transfers to people who were never in the
training set. Which is the whole requirement: enrolment happens after training.

### Forgeries are excluded everywhere

`_forg` folders are skipped by every stage. Including a forgery in a person's
class would teach the model to place it *near* that person's genuine
signatures — the opposite of what the embedding is for.

They stay in the raw download because they are the natural evaluation set for a
forgery-detection model, if one is ever added. This system does not attempt
forgery detection: it answers *whose signature is this*, not *is this signature
authentic*.

### Validation comes out of `train/`, not `test/`

The dataset is writer-independent — `test/` contains entirely different people.
Using it as validation gives the model 64 output neurons and the validation
labels 21 classes:

```
ValueError: target.shape=(None, 21)  output.shape=(None, 64)
```

So 15% is held out of `train/` (`verification.val_split`). `test/` is reserved
for `sigtrain evaluate`, which is where unseen-identity performance is actually
measured.

---

## Two-phase fine-tuning

| Phase | Trainable | LR | Why |
|---|---|---|---|
| 1 | The new head only | `1e-3` | A randomly initialised head produces large gradients; letting them reach the pretrained backbone destroys the features being transferred |
| 2 | Everything except BatchNorm | `1e-4` | Adapts the features to signatures, gently |

### BatchNorm stays frozen throughout

ResNet50 has 53 BatchNormalization layers. In training mode they update running
mean and variance from the current batch — here, a few thousand near-binary
white-background signature images, a distribution nothing like ImageNet.
Those corrupted statistics are then used at inference, degrading exactly the
pretrained features the transfer depends on.

VGG16 has no BatchNorm and never encounters this, which is why the two
backbones historically behaved so differently under the same recipe.

### The extractor tap

| Backbone | Tap | Dimension |
|---|---|---|
| VGG16 | native `fc1` | 4096 |
| ResNet50 | `conv5_block3_out` → GAP → `Dense(4096, name="fc1")` | 4096 |

`conv5_block3_out`, not `conv5_block3_2_conv`: the latter is mid-block, so the
residual addition and the block's final activation are both discarded.

4096 is not free to change. `inference/postgres/init.sql` declares
`VECTOR(4096)` and each `config.pbtxt` declares `dims: [4096]`. All three move
together.

---

## The CycleGAN denoiser

### Why unpaired translation

Real documents do not come with a clean counterpart. CycleGAN learns a mapping
between two *unpaired* domains using a cycle-consistency loss, so it needs a
pile of clean signatures and a pile of noisy ones — not aligned pairs.

The dataset builder does produce them in pairs (it synthesises B from A), but
the training objective never uses that correspondence.

### The synthesised noise

`training/src/signature_training/data/noise/` models what actually sits on a
Vietnamese signature block:

- **Horizontal rules** — 1–2 printed form lines.
- **Cell borders** — 0, 1 or 2 vertical box edges, placed near the crop edges,
  weighted 50/35/15 to match how often a tight crop includes them.
- **Caption text** — "(Ký, họ tên)" and a printed name below the signature.
- **Stamps** — round seals, DPI-aware. A standard 36–42 mm seal is scaled to
  what it would occupy at a 150–300 dpi scan, greyscale scans are tinted to the
  standard red ink, and placement is left-biased and usually bleeding off the
  frame, because a tight crop rarely contains the whole seal.

Stamps composite with a **multiply** blend rather than alpha-over, so signature
ink stays visible through the seal — which is what makes the task learnable
rather than a reconstruction problem.

### Resolution

The dataset is written at 512×512, but upstream's defaults resize to
`load_size=286` and random-crop to `crop_size=256`, so **256 is the resolution
the generator actually learns at**. Serving runs it at 224. Both are set in
`configs/default.yaml`; the 512 canvas exists to preserve detail before
downsampling, not as a training resolution.

---

## Evaluation

Genuine and impostor pairs are built from held-out identities. No forgeries.

| Metric | Reading |
|---|---|
| **EER** | Where FAR = FRR. One number, no threshold to pick first. < 0.10 good, < 0.05 strong |
| **AUC-ROC** | Whole trade-off space. > 0.95 good |
| **d′** | Distribution separation in pooled σ. > 2 good |
| **TAR @ FAR** | Operational: "at 1% false accepts, how many genuine do we accept?" |
| **FNMR @ FMR** | ISO/IEC 19795 form of the same trade-off |

### Sampling resolution

With *k* impostor pairs the FPR grid steps by 1/*k*. Sampling one pair per
person-couple on a 21-person split gives 210 pairs, so the finest expressible
FAR is 0.48% — and a reported "TAR @ FAR 0.1%" is really TAR at FAR = 0.

`evaluate.impostor_pairs_per_couple` (default 8) raises the resolution, and the
report marks any operating point the sample cannot support rather than printing
a number that looks meaningful.

### Carrying the threshold across

`sigtrain evaluate` reports the EER threshold as a cosine **similarity**.
`inference/.env` wants a cosine **distance**:

```
MATCH_THRESHOLD = 1 - eer_threshold
```

The runner prints the converted value directly, because getting this inversion
wrong produces a system that looks configured and accepts everyone.
