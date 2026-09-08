# 7 — Models & Datasets — Sources and Downloads

> Restored. An earlier revision of this repository deleted this document while
> reorganising `documentation/`, on the claim that its content had moved into
> other files. The dataset descriptions had; **the download links had not**, and
> they are the only pointer to the trained weights, the stamp corpus and the
> internal datasets. Nothing else in the repo records them.

---

## 1. Datasets

### 1.1 Signature Verification Dataset (Kaggle — primary)

| Property | Value |
|---|---|
| Source | <https://www.kaggle.com/datasets/robinreni/signature-verification-dataset> |
| Author | Robin Reni |
| Content | Handwritten genuine + forged signatures, multiple persons |
| Layout | `sign_data/{train,test}/NNN/` genuine, `NNN_forg/` forged |

**Verified 2026-09-08:** downloads without Kaggle credentials from
`https://www.kaggle.com/api/v1/datasets/download/robinreni/signature-verification-dataset`
(~630 MB zip).

**The archive contains the corpus twice**, and that matters. Reading the zip's
local file headers directly: the first entry is
`sign_data/sign_data/test/049/01_049.png`, and a 360 MB sample of the stream
holds 2,151 entries under `sign_data/sign_data/` alongside 276 under
`sign_data/test/`. So extracting gives

```
sign_data/
├── train/, test/            <- the corpus
└── sign_data/
    └── train/, test/        <- the same corpus again
```

which is where the frequently-quoted "4,298 PNGs" comes from: it is ~2,149
images counted twice.

Delete the nested copy after extracting:

```bash
rm -rf training/data/raw/sign_data/sign_data
```

`build_verification_split` is unaffected either way — it reads `src/train` and
`src/test` by name. **`data-cyclegan` is not**: `collect_images` uses `rglob`, so
it would find every signature twice, and the 90/10 shuffle would then place the
same signature in both `train` and `test`. The denoiser would be evaluated on
images it trained on.

Measured colour statistics of the genuine training images (400 sampled,
background = luminance > 200):

| region | R | G | B | B − R |
|---|---|---|---|---|
| paper | 243.28 | 251.47 | 251.46 | **+8.18** |
| ink | 102.15 | 89.88 | 95.67 | −6.49 |

Green and blue are equal to 0.01; the cast is a **red deficit**, not added blue,
and it is near-constant across the corpus (per-image B−R sd 0.15). None of the
sampled files are achromatic, so the older claim that this dataset is greyscale
is wrong. See `documentation/02-pipeline-deep-dive.md` for what the pipeline
does about it.

### 1.2 Alternative Kaggle source

<https://www.kaggle.com/datasets/mallapraveen/signature-matching> — a substitute,
but **not the same layout**. Its ~4,289 PNGs sit under `custom/full/`, not under
`{train,test}/NNN/`, so `paths.raw_signatures` cannot simply be repointed at it:
`build_verification_split` looks for `train/` and `test/` by name and would find
neither. Reshape it first, or use it only for `data-cyclegan`, which globs
recursively.

### 1.3 Stamp noise corpus (internal)

~16 Vietnamese round seals / company stamps, collected internally. Not
redistributable and not on Kaggle.

```
<<< STAMP_CORPUS_URL — placeholder, not yet set >>>
```

> The link is **deliberately unset**, to be filled in later. Nothing in the
> pipeline reads it, so the placeholder breaks nothing.
>
> A copy of the corpus is currently in the download share below, under
> `stamp_noise_data/` (16 files, ~1.4 MB). Treat the link above as the canonical
> pointer once it is set.

Everything needed to proceed without that link:

| | |
|---|---|
| Location | `training/data/raw/stamps/` (`paths.stamps`) |
| Formats | `.png`, `.jpg`, `.jpeg`, `.bmp`, and the uppercase spellings |
| Count | a few dozen is plenty; the pipeline used 16 |
| Content | round/oval company seals, one per file, any resolution |
| Alpha | **discarded**, not used — `_load_stamps` converts BGRA to BGR and derives its own mask from the background |
| Background | white; `_remove_background` is what makes the seal composite cleanly, so a photographed seal on grey paper will not work well |
| Colour | red or greyscale. Achromatic scans are tinted by `_tint_red`: R +96, G and B x0.52, aiming at roughly (B 30, G 20, R 210). Those are the actual operations — not a hue/saturation set point |

`StampAugmentor` scales each seal to a physically plausible diameter relative to
the signature, rotates it, and **multiply**-blends it so the seal sits under the
ink the way real toner does. Substituting your own stamps therefore needs no code
change.

Without any stamps, `sigtrain data-cyclegan` logs

```
StampAugmentor: no stamp images found in '<path>'
```

and continues. The pair set is still built and CycleGAN still trains — the
denoiser simply never learns to remove seals, which is the single largest thing
it is being asked to do.

### 1.4 Test pipeline documents (internal)

Six real PDFs used for end-to-end checks. **Not in the download share** — it was
enumerated exhaustively on 2026-09-08 and contains no PDFs. They are withheld
under a confidentiality agreement, so end-to-end checks against real documents
need your own.

---

## 1.5 Pretrained backbone weights

Fetched automatically by Keras unless a local file is supplied via
`verification.vgg16_weights` / `verification.resnet50_weights`:

| File | Source |
|---|---|
| `vgg16_weights_tf_dim_ordering_tf_kernels.h5` (553 MB) | <https://keras.io/api/applications/vgg/#vgg16-function> |
| `resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5` (95 MB) | <https://keras.io/api/applications/resnet/#resnet50-function> |

Both are also in the share below, which is faster and avoids a ~650 MB download
on every fresh image.

## 1.6 Reference implementation

The verification approach follows
<https://github.com/amaljoseph/EndToEnd_Signature-Detection-Cleaning-Verification_System_using_YOLOv5-and-CycleGAN>
and the paper it is based on, <https://arxiv.org/abs/2004.12104>. Useful when
deciding whether a change is a deviation from the reference or a bug.

## 1.7 TensorRT engines

The share carries five `model.plan` files — yolov8s 54.65 MB, latest_net_G_A and
G_B 79.42 MB each, resnet50_extractor 130.15 MB, vgg16_extractor 515.67 MB — so
the deployment has run on the TensorRT backend as well as onnxruntime.

**The pipeline no longer builds them.** `convert_to_trt.py` (1,886 lines) was
removed during the refactor: its CycleGAN path was broken (wrong `state_dict`
key prefix, so it exported a randomly-initialised generator) and the deployed
`config.pbtxt` files all specify `backend: "onnxruntime"`. Nothing in the repo
consumed a `.plan`.

To rebuild one from an exported ONNX, `trtexec` is the supported path — no
custom converter required:

```bash
trtexec --onnx=artifacts/onnx/vgg16_extractor.onnx \
        --saveEngine=vgg16_extractor.plan \
        --fp16 --shapes=input_layer:1x3x224x224
```

then place it at `<model>/1/model.plan` and change that model's `config.pbtxt`
to `backend: "tensorrt"`. The engine is specific to the GPU and TensorRT version
it was built on, which is the main reason it is a deployment step rather than a
pipeline output.

The pin used previously, for reproducing that environment:
`tensorrt-cu12==10.16.1.11` (it was a commented-out line in the old
`trainingfiles/requirements.txt`; the Dockerfile installed it separately and
warned "Make sure tensorrt is NOT in this file").

---

## 2. Download share

All internally-held artefacts — trained checkpoints, exported ONNX/TensorRT,
the stamp corpus and the test PDFs — live in one OneDrive/SharePoint folder:

<https://mintcash-my.sharepoint.com/:f:/g/personal/lethanhminh0801_mintcash_onmicrosoft_com/IgBY8nP1MqSLTrbDjnGzVo6nAa94By6uqtOJTEqrPOQzcks?e=gJhgXc>

> **Access:** the share is reachable by **anyone with the link**, without a
> Mintcash account. An earlier revision of this document claimed it returned
> 403 and was tenant-restricted; that was wrong, and it was wrong for a
> mechanical reason worth recording.
>
> A bare `curl` does fail. SharePoint's anonymous-share flow needs two things:
> a browser `User-Agent` (otherwise the request is redirected to the login page)
> and a **cookie jar**, because the first request sets the anonymous-session
> cookies that authorise the second. With both, the REST API works unauthenticated:
>
> ```bash
> UA='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 \
> (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36'
> curl -sL -c jar.txt -A "$UA" '<share link>' -o /dev/null      # seed the cookies
> curl -s  -b jar.txt -A "$UA" \
>   -H 'Accept: application/json;odata=nometadata' \
>   "$WEB/_api/web/GetFolderByServerRelativeUrl('$FOLDER')/Files?\$select=Name,Length"
> ```
>
> where `$WEB` is `https://mintcash-my.sharepoint.com/personal/<user>` and
> `$FOLDER` is `/personal/<user>/Documents/Work/Model_SignatureVerification`.
> Verified 2026-09-08 against the live share.

### What is actually in the share

Enumerated over the REST API on 2026-09-08 using the recipe above. Sizes are
**decimal MB** taken from the `Length` field — an earlier revision of this table
quoted MiB but labelled them MB, which is where "43.4 MB" for a 45.53 MB file
came from.

Everything is reachable from the one link above; per-file direct links are not
issued by an anonymous share, so the table gives paths rather than URLs.

#### Root

| File | MB |
|---|---|
| `latest_net_G_A.pth` | 45.53 |
| `latest_net_G_B.pth` | 45.53 |
| `latest_net_D_A.pth` | 11.06 |
| `latest_net_D_B.pth` | 11.06 |
| `vgg16_extractor.keras` | 470.00 |
| `vgg16_finetuned.keras` | 1078.08 |
| `vgg16_phase1_best.keras` | 541.03 |
| `vgg16_phase2_best.keras` | 1078.08 |
| `resnet50_extractor.keras` | 128.57 |
| `resnet50_finetuned.keras` | 260.01 |
| `resnet50_phase1_best.keras` | 166.05 |
| `resnet50_phase2_best.keras` | 260.01 |
| `vgg16_weights_tf_dim_ordering_tf_kernels.h5` | 553.47 |
| `resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5` | 94.77 |
| `yolo_model.onnx` | 44.62 |

The detector is **`yolo_model.onnx`**, not `yolov8s.onnx`. The pipeline's Triton
model directory is named `yolov8s`, so the file has to be renamed to
`yolov8s/1/model.onnx` on the way in — `sigtrain export` does that itself, but a
hand copy will not.

The four `*_phase{1,2}_best.keras` files are `ModelCheckpoint` output from the
two training phases, not deployment artefacts.

#### `model_repository/` — Triton-ready

| Model | `model.onnx` | `model.plan` |
|---|---|---|
| `yolov8s/1/` | 44.62 (named `yolo_model.onnx`) | 54.65 |
| `latest_net_G_A/1/` | 45.64 | 79.42 |
| `latest_net_G_B/1/` | 45.64 (plus `model_org.onnx`, 45.62) | 79.42 |
| `resnet50_extractor/1/` | 127.78 | 130.15 |
| `vgg16_extractor/1/` | 469.93 | 515.67 |

The `.plan` files are TensorRT engines, specific to the GPU and TensorRT version
that built them — see §1.7. `latest_net_G_B/1/model_org.onnx` is undocumented;
treat `model.onnx` as authoritative.

#### `stamp_noise_data/` — the stamp corpus

16 files, ~1.4 MB total: 12 PNG screenshots (`Annotation 2026-05-02 *.png`) and
4 JPGs (`mau-con-dau-tron-cong-ty{1,2,3}.jpg`, `khac-dau-ten-tai-bien-hoa{2,5}.jpg`).
This is the corpus §1.3 describes.

#### Not in the share

No PDFs. §1.4's six test-pipeline documents are **not** here — the root, both
subfolders and all five `model_repository/*/1/` directories were enumerated and
contain none.

### Datasets

| Dataset | Where |
|---|---|
| Signature corpus (primary) | [kaggle.com/datasets/robinreni/signature-verification-dataset](https://www.kaggle.com/datasets/robinreni/signature-verification-dataset) — see §1.1 for the duplicate-tree caveat |
| Signature corpus (substitute) | [kaggle.com/datasets/mallapraveen/signature-matching](https://www.kaggle.com/datasets/mallapraveen/signature-matching) — different layout, see §1.2 |
| Stamp noise images | `stamp_noise_data/` in the share; canonical link is the §1.3 placeholder |
| Test pipeline PDFs | Not in the share; withheld under a confidentiality agreement |

One raw corpus feeds both stages — `paths.raw_signatures` is read by
`data-cyclegan` and `data-verification` alike. There is not one dataset per
stage.

### Pretrained backbone weights

| File | Source |
|------|--------|
| VGG16 ImageNet weights | Auto-downloaded by Keras, or [direct link](https://storage.googleapis.com/tensorflow/keras-applications/vgg16/vgg16_weights_tf_dim_ordering_tf_kernels.h5) |
| ResNet50 ImageNet weights (no-top) | Auto-downloaded by Keras, or [direct link](https://storage.googleapis.com/tensorflow/keras-applications/resnet/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5) |

Both are also in the share (553.47 MB and 94.77 MB), which avoids a ~650 MB
download on every fresh image.

---

## File Checksums

> _To be populated when download links are provided. Run:_
> ```bash
> sha256sum <filename>
> ```
