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

`ImageDataGenerator(preprocessing_function=...)` applies the transform *before*
the model sees anything — today via
`models/preprocess.py:extractor_preprocess`, which composes the colour
correction with Keras' `preprocess_input`. No `Rescaling` or `Normalization`
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
`training/src/signature_training/models/preprocess.py`, which is the single path
used by training and evaluation alike; serving-side is `to_caffe` /
`to_cyclegan` / `from_cyclegan` in `inference/api/app/triton.py`.

`preprocess_input(mode="caffe")` **mutates its argument**: it reverses the
channel axis into a numpy view and subtracts the mean in place. Anything that
reuses the array afterwards needs a copy first.

---

## The paper colour cast

The denoised output looks blue. It is measured, and it is not blue being added —
it is **red being removed**.

Kaggle training set, 400 genuine images, paper = luminance > 200:

| region | R | G | B | B − R |
|---|---|---|---|---|
| paper | **243.28** | 251.47 | 251.46 | **+8.18** |
| ink | 102.15 | 89.88 | 95.67 | −6.49 |

Green and blue agree to 0.01. Red is ~8 levels short, and a red deficit reads as
cyan. It is near-constant across the corpus — per-image B−R has sd **0.15**
(range +7.83..+8.94) — so it is one scanner and one fixed offset, not per-image
variation. (None of the sampled files are achromatic, so the old claim that this
dataset is greyscale is also wrong.)

CycleGAN reproduces the colour distribution it is trained on, so the generator
inherits it. Running the real `latest_net_G_B.pth` on synthesised noisy inputs:

| | R | G | B | B − R |
|---|---|---|---|---|
| noisy input paper | 249.63 | 250.79 | 250.80 | +1.17 |
| **denoised paper** | **249.98** | 253.43 | 253.43 | **+3.45** |

The generator *adds* cast: it takes a near-neutral input and returns a
red-deficient one, because that is what its domain A looked like.

### The correction

`training/src/signature_training/data/colour.py` (mirrored into
`inference/api/app/colour.py`) estimates the per-channel paper level and applies
a diagonal gain.

Two things about it are load-bearing, and both were found by measuring rather
than reasoning:

* **The paper estimate is a MEAN over the paper band**, not a high percentile
  and not a median of the brightest pixels. Denoiser output has paper spread
  from ~249 up to a saturated spike at 255; a percentile or a top-quantile
  median lands on the spike, reports ~255 for every channel, and the correction
  becomes a silent no-op. First two implementations did exactly that
  (+3.46 → +3.45, i.e. nothing).
* **Channels are equalised against the brightest one, not driven to 255.**
  Targeting white looks equivalent and is not: paper often already sits within a
  couple of levels of saturation, so the extra gain is swallowed by the clip and
  the cast survives.

Measured after those two fixes:

| case | before | after `whiten` | after `desaturate` |
|---|---|---|---|
| raw dataset paper | +8.18 | **−0.10** | 0.00 |
| real G_B output paper | +3.45 | **+1.75** | 0.00 |

Ink/paper contrast goes *up* slightly (153.2 → 155.8), so strokes are not eroded.

---

## The colour contract

One setting, `colour.mode` in `training/configs/default.yaml`, applied at every
point an image enters a model:

```yaml
colour:
  mode: none        # none | whiten | desaturate
```

| stage | where | what it corrects |
|---|---|---|
| `data-cyclegan` | `data/cyclegan.py:_make_pair` | the clean image, before noise is added, so domains A and B share a paper colour |
| `data-verification` | `data/cyclegan.py:_copy_person` | the images written to disk, which is what the backbones read |

Both are **dataset writers**. Nothing corrects at load time, and
`train-verification` and `evaluate` read pixels that already carry the
correction — each checks the dataset's recorded mode first and refuses a
mismatch.

### Why at build time, and exactly once

Keras offers only a **post-augmentation** hook: `preprocessing_function` runs
after rotation, shift, shear and zoom, so it sees the borders those fill in. The
configured augmentation fills a mean of **10.8%** of the frame with `cval=255`
(p95 17.8%, over 5,000 draws; stable to 0.2 points across independent seeds).
Those neutral pixels land inside the paper band and drag the per-channel means
together:

| | residual cast after `whiten` |
|---|---|
| clean estimate | +0.00 |
| with 10% `cval=255` fill | **+0.83** |

About a tenth of the cast survives. At build time there is no augmentation and
the estimate is clean.

Doing both looks free and is not. **`whiten` is idempotent only while its gain
clamp does not bind.** The clamp is what breaks idempotence, not what guarantees
it: when `estimate_paper(x).max() / .min()` exceeds `max_gain`, the first pass is
truncated and leaves the paper un-neutral, so a second applies the remainder and
the effective limit becomes `max_gain²` = 2.56 — defeating the clamp that exists
to stop a dark photograph being stretched into white.

| paper | imbalance | one pass | two passes |
|---|---|---|---|
| raw scan 243.3 / 251.5 / 251.5 | 1.034 | +0.00 | +0.00 (drift 0.001) |
| denoised 250.0 / 253.4 / 253.4 | 1.014 | +0.00 | +0.00 (drift 0.026) |
| tungsten 248 / 190 / 130 | **1.908** | blue gain 1.600× | blue gain **1.908×**, 41 levels drift |

The signature corpora are nowhere near the bound, so composition happens to be
safe on this data. It is not safe in general — which is why the serving side must
apply it exactly once too.

### The dataset stamp

`data-verification` writes `.colour_mode` into the dataset directory.
`train-verification` and `evaluate` both read it and refuse to run against a
mismatch, because nothing else compares the two: without it, training could read
a corrected corpus with the correction configured off and record the wrong mode
next to the model.

An **unstamped** directory that already holds people is read as `none`, and that
is a fact rather than a guess — before the correction moved into the builder it
was a bare `shutil.copytree`, so every dataset predating the stamp is provably
uncorrected. Assuming otherwise was a real defect: the first `colour.mode:
whiten` would skip every existing folder, stamp the directory `whiten`, and
report success, leaving a +8.18 corpus certified as corrected and refusing the
one command that would rebuild it.

### One knob, not six

`paper_level`, `max_gain`, `strength` and `lift` are deliberately *not* config
keys. They are module constants in `colour.py`, whose two copies are identical
from the `from __future__` line onward — the module docstrings differ, and the
CI diff compares exactly that region. That leaves exactly one string
that can disagree between training and serving instead of six.

### Changing the mode

It **invalidates the trained extractors** — they learn whatever colour
distribution they were fed. In order:

```bash
rm -rf training/data/processed/verification   # the builder refuses to mix modes
sigtrain data-verification --set colour.mode=whiten
sigtrain train-verification --set colour.mode=whiten
sigtrain evaluate          --set colour.mode=whiten   # prints the new threshold
sigtrain export
```

The delete is not optional: `build_verification_split` skips person folders that
already exist, so without it half the corpus would keep the old distribution.
The stamp turns that into an error rather than a silent mixture.

Rebuilding the **CycleGAN** pair set and retraining the denoiser is a separate,
much longer job (200 epochs). It is the root fix — domain A then has neutral
paper and the generator stops emitting a cast at all — and it is why the
residual on the *current* weights (+1.75) cannot be driven to zero from the
outside: post-hoc correction is undoing something the weights already baked in,
against a saturating ceiling.

### The serving side is not wired yet

`sigtrain evaluate` prints the line to paste:

```
MATCH_THRESHOLD=0.3012   # from resnet50
COLOUR_MODE=whiten
```

**But nothing in the service reads `COLOUR_MODE` today.** `settings.colour_mode`
is declared in `inference/api/app/config.py` and no code consumes it; the
correction has to be applied to the `[0, 1]` RGB tensor immediately before
`to_caffe` in `inference/api/app/triton.py`, and that call does not exist. Until
it does, setting `colour.mode` to anything but `none` produces a **train/serve
skew** — the models are trained on normalised paper and served un-normalised
paper.

`PREVIEW_WHITEN=true` (the default) is a different thing and does work: it
neutralises the base64 previews the UI shows, touching what a human sees and
nothing that produces an embedding. That fixes "the output looks blue" at zero
model risk.

`desaturate` is the stronger option: a signature's identity is stroke geometry,
not colour, so discarding chroma removes this cast and every other
scanner-dependent colour difference at once. It is a bigger change — the
backbones are ImageNet-pretrained and do use colour — so it is offered rather
than assumed.

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

4096 is not free to change. The width is pinned in `inference/api/app/db.py`,
in the migration that creates the columns
(`inference/api/migrations/versions/0001_initial_schema.py`), and in each
`config.pbtxt` as `dims: [4096]`. Changing `verification.embedding_dim` means
changing all three — not `inference/postgres/init.sql`, which is now a bare
`CREATE EXTENSION` because the schema moved to Alembic.

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

It prints `COLOUR_MODE=` on the next line. The threshold is only valid for the
colour mode it was measured under, so the two belong together — see
[the colour contract](#the-colour-contract).
