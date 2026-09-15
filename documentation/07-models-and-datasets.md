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
(630 MB zip, 4,298 PNGs, 128 person folders under `train/`). The layout matches
what `sigtrain data-cyclegan` and `data-verification` expect, so it can be
extracted straight to `training/data/raw/sign_data/`.

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

<https://www.kaggle.com/datasets/mallapraveen/signature-matching> — same shape,
usable as a substitute.

### 1.3 Stamp noise corpus (internal)

~16 Vietnamese round seals / company stamps, collected internally. Not
redistributable and not on Kaggle — get it from the share below and place it at
`training/data/raw/stamps/`. Without it `sigtrain data-cyclegan` warns and the
denoiser never learns to remove seals.

### 1.4 Test pipeline documents (internal)

Six real PDFs used for end-to-end checks. Same share.

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

The share carries five `model.plan` files, so the deployment has run on the
TensorRT backend as well as onnxruntime.

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

> **Access:** as of 2026-09-08 this link resolves to
> `/Documents/Work/Model_SignatureVerification` but returns **403 FORBIDDEN**
> to an unauthenticated fetch — it is tenant-restricted, or the `?e=` token has
> expired. Sign in with a Mintcash account, or re-share as "Anyone with the
> link" if it needs to be reachable from CI or from a machine outside the
> tenant.

### Trained model files

| File | Format | Size | Download Link |
|------|--------|------|---------------|
| `latest_net_G_B.pth` | PyTorch | 43.4 MB | [link](https://mintcash-my.sharepoint.com/:f:/g/personal/lethanhminh0801_mintcash_onmicrosoft_com/IgBY8nP1MqSLTrbDjnGzVo6nAa94By6uqtOJTEqrPOQzcks?e=gJhgXc) |
| `latest_net_G_A.pth` | PyTorch | 43.4 MB | [link](https://mintcash-my.sharepoint.com/:f:/g/personal/lethanhminh0801_mintcash_onmicrosoft_com/IgBY8nP1MqSLTrbDjnGzVo6nAa94By6uqtOJTEqrPOQzcks?e=gJhgXc) |
| `latest_net_D_A.pth` | PyTorch | 10.6 MB | [link](https://mintcash-my.sharepoint.com/:f:/g/personal/lethanhminh0801_mintcash_onmicrosoft_com/IgBY8nP1MqSLTrbDjnGzVo6nAa94By6uqtOJTEqrPOQzcks?e=gJhgXc) |
| `latest_net_D_B.pth` | PyTorch | 10.6 MB | [link](https://mintcash-my.sharepoint.com/:f:/g/personal/lethanhminh0801_mintcash_onmicrosoft_com/IgBY8nP1MqSLTrbDjnGzVo6nAa94By6uqtOJTEqrPOQzcks?e=gJhgXc) |
| `vgg16_extractor.keras` | Keras | 448 MB | [link](https://mintcash-my.sharepoint.com/:f:/g/personal/lethanhminh0801_mintcash_onmicrosoft_com/IgBY8nP1MqSLTrbDjnGzVo6nAa94By6uqtOJTEqrPOQzcks?e=gJhgXc) |
| `vgg16_finetuned.keras` | Keras | 1,028 MB | [link](https://mintcash-my.sharepoint.com/:f:/g/personal/lethanhminh0801_mintcash_onmicrosoft_com/IgBY8nP1MqSLTrbDjnGzVo6nAa94By6uqtOJTEqrPOQzcks?e=gJhgXc) |
| `resnet50_extractor.keras` | Keras | 123 MB | [link](https://mintcash-my.sharepoint.com/:f:/g/personal/lethanhminh0801_mintcash_onmicrosoft_com/IgBY8nP1MqSLTrbDjnGzVo6nAa94By6uqtOJTEqrPOQzcks?e=gJhgXc) |
| `resnet50_finetuned.keras` | Keras | 248 MB | [link](https://mintcash-my.sharepoint.com/:f:/g/personal/lethanhminh0801_mintcash_onmicrosoft_com/IgBY8nP1MqSLTrbDjnGzVo6nAa94By6uqtOJTEqrPOQzcks?e=gJhgXc) |
| `yolov8s.onnx` | ONNX | 42.6 MB | [link](https://mintcash-my.sharepoint.com/:f:/g/personal/lethanhminh0801_mintcash_onmicrosoft_com/IgBY8nP1MqSLTrbDjnGzVo6nAa94By6uqtOJTEqrPOQzcks?e=gJhgXc) |

### Deployed ONNX Models (Triton-ready)

| File | Format | Size | Download Link |
|------|--------|------|---------------|
| `yolov8s/1/model.onnx` | ONNX | 42.5 MB | [link](https://mintcash-my.sharepoint.com/:f:/g/personal/lethanhminh0801_mintcash_onmicrosoft_com/IgBY8nP1MqSLTrbDjnGzVo6nAa94By6uqtOJTEqrPOQzcks?e=gJhgXc) |
| `latest_net_G_B/1/model.onnx` | ONNX | 43.5 MB | [link](https://mintcash-my.sharepoint.com/:f:/g/personal/lethanhminh0801_mintcash_onmicrosoft_com/IgBY8nP1MqSLTrbDjnGzVo6nAa94By6uqtOJTEqrPOQzcks?e=gJhgXc) |
| `latest_net_G_A/1/model.onnx` | ONNX | 43.5 MB | [link](https://mintcash-my.sharepoint.com/:f:/g/personal/lethanhminh0801_mintcash_onmicrosoft_com/IgBY8nP1MqSLTrbDjnGzVo6nAa94By6uqtOJTEqrPOQzcks?e=gJhgXc) |
| `resnet50_extractor/1/model.onnx` | ONNX | 121.9 MB | [link](https://mintcash-my.sharepoint.com/:f:/g/personal/lethanhminh0801_mintcash_onmicrosoft_com/IgBY8nP1MqSLTrbDjnGzVo6nAa94By6uqtOJTEqrPOQzcks?e=gJhgXc) |
| `vgg16_extractor/1/model.onnx` | ONNX | 448.1 MB | [link](https://mintcash-my.sharepoint.com/:f:/g/personal/lethanhminh0801_mintcash_onmicrosoft_com/IgBY8nP1MqSLTrbDjnGzVo6nAa94By6uqtOJTEqrPOQzcks?e=gJhgXc) |

### TensorRT Plans (Optional, GPU-specific)

| File | Format | Size | Download Link |
|------|--------|------|---------------|
| `yolov8s/1/model.plan` | TensorRT | 52.1 MB | [link](https://mintcash-my.sharepoint.com/:f:/g/personal/lethanhminh0801_mintcash_onmicrosoft_com/IgBY8nP1MqSLTrbDjnGzVo6nAa94By6uqtOJTEqrPOQzcks?e=gJhgXc) |
| `latest_net_G_B/1/model.plan` | TensorRT | 75.8 MB | [link](https://mintcash-my.sharepoint.com/:f:/g/personal/lethanhminh0801_mintcash_onmicrosoft_com/IgBY8nP1MqSLTrbDjnGzVo6nAa94By6uqtOJTEqrPOQzcks?e=gJhgXc) |
| `latest_net_G_A/1/model.plan` | TensorRT | 75.8 MB | [link](https://mintcash-my.sharepoint.com/:f:/g/personal/lethanhminh0801_mintcash_onmicrosoft_com/IgBY8nP1MqSLTrbDjnGzVo6nAa94By6uqtOJTEqrPOQzcks?e=gJhgXc) |
| `resnet50_extractor/1/model.plan` | TensorRT | 124.1 MB | [link](https://mintcash-my.sharepoint.com/:f:/g/personal/lethanhminh0801_mintcash_onmicrosoft_com/IgBY8nP1MqSLTrbDjnGzVo6nAa94By6uqtOJTEqrPOQzcks?e=gJhgXc) |
| `vgg16_extractor/1/model.plan` | TensorRT | 491.8 MB | [link](https://mintcash-my.sharepoint.com/:f:/g/personal/lethanhminh0801_mintcash_onmicrosoft_com/IgBY8nP1MqSLTrbDjnGzVo6nAa94By6uqtOJTEqrPOQzcks?e=gJhgXc) |

### Datasets

| Dataset | Source | Download Link |
|---------|--------|---------------|
| Signature Cleaning Dataset | Kaggle | [kaggle.com/datasets/robinreni/signature-verification-dataset](https://www.kaggle.com/datasets/robinreni/signature-verification-dataset) |
| Signature Verification Dataset | Kaggle | [kaggle.com/datasets/mallapraveen/signature-matching](https://www.kaggle.com/datasets/mallapraveen/signature-matching)
| Stamp noise images | Internal collection | [link](https://mintcash-my.sharepoint.com/:f:/g/personal/lethanhminh0801_mintcash_onmicrosoft_com/IgBY8nP1MqSLTrbDjnGzVo6nAa94By6uqtOJTEqrPOQzcks?e=gJhgXc) |
| Test pipeline PDFs | Internal documents | Unavailable due to confidentiality agreement |

### Pretrained Backbone Weights

| File | Source | Download Link |
|------|--------|---------------|
| VGG16 ImageNet weights | Keras Applications | Auto-downloaded by Keras, or [direct link](https://storage.googleapis.com/tensorflow/keras-applications/vgg16/vgg16_weights_tf_dim_ordering_tf_kernels.h5) |
| ResNet50 ImageNet weights (no-top) | Keras Applications | Auto-downloaded by Keras, or [direct link](https://storage.googleapis.com/tensorflow/keras-applications/resnet/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5) |

---

## File Checksums

> _To be populated when download links are provided. Run:_
> ```bash
> sha256sum <filename>
> ```
