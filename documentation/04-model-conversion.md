# 4 — Model Conversion Guide

> How to convert trained models from native formats (`.keras`, `.h5`, `.pth`)
> to ONNX (`.onnx`) for deployment on NVIDIA Triton Inference Server.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Convert CycleGAN (PyTorch → ONNX)](#2-convert-cyclegan-pytorch--onnx)
3. [Convert VGG16 & ResNet50 (Keras → ONNX)](#3-convert-vgg16--resnet50-keras--onnx)
4. [Convert YOLOv8 (Ultralytics → ONNX)](#4-convert-yolov8-ultralytics--onnx)
5. [Universal Converter Script](#5-universal-converter-script)
6. [ONNX → TensorRT (Optional)](#6-onnx--tensorrt-optional)
7. [Verify ONNX Models](#7-verify-onnx-models)
8. [Deploy to Triton](#8-deploy-to-triton)

---

## 1. Overview

### Model format flow

```
Training Output              Conversion              Triton Deployment
───────────────              ──────────              ──────────────────
latest_net_G_B.pth    ──→    model.onnx       ──→    triton/model_repository/latest_net_G_B/1/model.onnx
vgg16_extractor.keras ──→    model.onnx       ──→    triton/model_repository/vgg16_extractor/1/model.onnx
resnet50_extractor.keras ──→ model.onnx       ──→    triton/model_repository/resnet50_extractor/1/model.onnx
yolov8s.pt            ──→    model.onnx       ──→    triton/model_repository/yolov8s/1/model.onnx
```

### Conversion tools used

| Source Format | Tool | Notes |
|---------------|------|-------|
| `.keras` / `.h5` (TensorFlow) | `tf2onnx` | Must use `inputs_as_nchw` for NCHW output |
| `.pth` (PyTorch CycleGAN) | `torch.onnx.export` | Custom export script |
| `.pt` (Ultralytics YOLOv8) | `ultralytics.YOLO.export()` | Built-in exporter |

---

## 2. Convert CycleGAN (PyTorch → ONNX)

### Using the dedicated export script

```bash
cd trainingfiles/

python pyfile/export_cyclegan_onnx.py \
    --checkpoint convert_model/original_model/latest_net_G_B.pth \
    --output convert_model/original_model/latest_net_G_B.onnx
```

This script:
1. Imports the CycleGAN architecture (`define_G`) from the cloned `pytorch-CycleGAN-and-pix2pix` repository
2. Loads weights from `latest_net_G_B.pth`
3. Exports to `latest_net_G_B.onnx` with input/output names matching Triton config

> **Note:** The `pytorch-CycleGAN-and-pix2pix` repository is not included in git. You must manually clone it into the `trainingfiles/` directory for this script to work:
> ```bash
> cd trainingfiles/
> git clone https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix.git
> ```

#### What the script does internally

```python
# 1. Build generator architecture
netG = define_G(input_nc=3, output_nc=3, ngf=64, netG='resnet_9blocks',
                norm='instance', use_dropout=False)

# 2. Load trained weights
state_dict = torch.load("checkpoints/signature/latest_net_G_B.pth")
netG.load_state_dict(state_dict)
netG.eval()

# 3. Export with fixed input shape
dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(netG, dummy_input, "latest_net_G_B.onnx",
                  input_names=['input'], output_names=['output'],
                  opset_version=14, dynamic_axes={'input': {0: 'batch_size'},
                                                   'output': {0: 'batch_size'}})
```

#### Customisation

| Parameter | Default | Notes |
|-----------|---------|-------|
| `CHECKPOINT` | `checkpoints/signature/latest_net_G_B.pth` | Edit in script to change path |
| `OUTPUT_FILE` | `latest_net_G_B.onnx` | Edit in script |
| `image_size` | `224` | Must match Triton config (224×224) |

> **For G_A:** Repeat with `latest_net_G_A.pth` → `latest_net_G_A.onnx`

---

## 3. Convert VGG16 & ResNet50 (Keras → ONNX)

### Quick method: `simpleconver.py`

```bash
cd trainingfiles/convert_model/

python simpleconver.py
```

This converts both models in one run:

```python
# VGG16
convert_and_simplify(
    keras_path="original_model/vgg16_extractor.keras",
    onnx_path="simple/vgg16_extractor.onnx",
    input_name="input_layer",     # Must match Triton config
    output_name="fc1"
)

# ResNet50
convert_and_simplify(
    keras_path="original_model/resnet50_extractor.keras",
    onnx_path="simple/resnet50_extractor.onnx",
    input_name="input_layer_1",   # Must match Triton config
    output_name="fc1"
)
```

#### Critical: `inputs_as_nchw` flag

The Keras models use **NHWC** (channels-last) internally, but Triton expects **NCHW**
(channels-first). The `inputs_as_nchw` parameter in `tf2onnx` adds a transpose node
at the graph input so Triton can send NCHW data:

```python
model_proto, _ = tf2onnx.convert.from_keras(
    model,
    input_signature=(tf.TensorSpec((1, 224, 224, 3), tf.float32, name=input_name),),
    opset=13,
    inputs_as_nchw=[input_name]   # ← This is crucial!
)
```

### Advanced method: `convert_to_onnx_gemini.py`

This universal converter auto-detects model format and optionally generates Triton configs:

```bash
# Basic conversion
python convert_to_onnx_gemini.py \
    --model original_model/vgg16_extractor.keras \
    --output triton_repo/ \
    --triton \
    --model-name vgg16_extractor

# With custom settings
python convert_to_onnx_gemini.py \
    --model original_model/resnet50_extractor.keras \
    --output triton_repo/ \
    --opset 17 \
    --triton \
    --model-name resnet50_extractor \
    --max-batch-size 8
```

#### Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | (required) | Source model file (`.keras`, `.h5`, `.pt`, `.pth`) |
| `--output` | `<model>.onnx` | Output path (file or directory) |
| `--opset` | `17` | ONNX opset version |
| `--arch` | auto-detect | Architecture name (for raw state_dicts) |
| `--triton` | `False` | Generate Triton model repository structure |
| `--model-name` | source filename | Name for Triton model |
| `--max-batch-size` | `8` | Max batch size in Triton config |

---

## 4. Convert YOLOv8 (Ultralytics → ONNX)

YOLOv8 uses the built-in Ultralytics exporter:

```python
from ultralytics import YOLO

model = YOLO("yolov8s.pt")
model.export(format="onnx", imgsz=640, opset=17, dynamic=True, simplify=True)
```

Or via CLI:

```bash
yolo export model=yolov8s.pt format=onnx imgsz=640 opset=17 dynamic=True simplify=True
```

> The exported file is `yolov8s.onnx`. Copy it to `triton/model_repository/yolov8s/1/model.onnx`.

---

## 5. Universal Converter Script

`convert_to_onnx_gemini.py` supports all model types:

```bash
# Keras / TensorFlow
python convert_to_onnx_gemini.py --model model.keras --output model.onnx

# PyTorch (with known architecture)
python convert_to_onnx_gemini.py --model model.pth --arch cyclegan_resnet_9 --output model.onnx

# Ultralytics YOLOv8 (auto-detected)
python convert_to_onnx_gemini.py --model yolov8s.pt --output model.onnx

# With Triton repo generation
python convert_to_onnx_gemini.py --model model.keras --output triton_repo/ --triton --model-name my_model
```

### Supported architectures (for `--arch` flag)

| Architecture | Description |
|-------------|-------------|
| `cyclegan_resnet_9` | CycleGAN generator with 9 ResNet blocks |
| `resnet18` | torchvision ResNet-18 |
| `resnet50` | torchvision ResNet-50 |
| `yolov8` | Handled via Ultralytics (auto-detected) |

---

## 6. ONNX → TensorRT (Optional)

TensorRT `.plan` files provide faster inference on NVIDIA GPUs.
Triton auto-selects `.plan` over `.onnx` if both are present.

```bash
cd trainingfiles/convert_model/

python convert_to_trt.py  # See script for options
```

> **Note:** `.plan` files are GPU-architecture-specific. A `.plan` built on an
> RTX 3090 will NOT work on a T4. Rebuild on the target hardware.

---

## 7. Verify ONNX Models

After conversion, validate the ONNX model:

```python
import onnx

model = onnx.load("model.onnx")
onnx.checker.check_model(model)
print("✓ Model is valid")

# Print input/output shapes
for inp in model.graph.input:
    print(f"Input:  {inp.name}  shape={[d.dim_value for d in inp.type.tensor_type.shape.dim]}")
for out in model.graph.output:
    print(f"Output: {out.name}  shape={[d.dim_value for d in out.type.tensor_type.shape.dim]}")
```

### Expected shapes

| Model | Input | Output |
|-------|-------|--------|
| `yolov8s` | `[1, 3, 640, 640]` | `[1, 5, 8400]` (or similar) |
| `latest_net_G_B` | `[1, 3, 224, 224]` | `[1, 3, 224, 224]` |
| `resnet50_extractor` | `[1, 3, 224, 224]` | `[1, 4096]` |
| `vgg16_extractor` | `[1, 3, 224, 224]` | `[1, 4096]` |

---

## 8. Deploy to Triton

Copy converted ONNX files to the Triton model repository:

```bash
# From trainingfiles/convert_model/
cp simple/vgg16_extractor.onnx \
   ../signature-verification-service/triton/model_repository/vgg16_extractor/1/model.onnx

cp simple/resnet50_extractor.onnx \
   ../signature-verification-service/triton/model_repository/resnet50_extractor/1/model.onnx

# CycleGAN
cp latest_net_G_B.onnx \
   ../signature-verification-service/triton/model_repository/latest_net_G_B/1/model.onnx

# YOLOv8
cp yolov8s.onnx \
   ../signature-verification-service/triton/model_repository/yolov8s/1/model.onnx
```

### Triton `config.pbtxt` — input/output names must match

Ensure the `name` fields in each `config.pbtxt` match the ONNX graph's input/output names:

| Model | Input Name | Output Name |
|-------|-----------|-------------|
| `yolov8s` | `images` | `output0` |
| `latest_net_G_B` | `input` | `output` |
| `resnet50_extractor` | `input_layer_1` | `fc1` |
| `vgg16_extractor` | `input_layer` | `fc1` |

Then restart Triton:

```bash
cd signature-verification-service/
docker compose restart triton
```
