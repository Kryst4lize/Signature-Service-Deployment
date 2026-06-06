"""
train_verification.py
─────────────────────────────────────────────────────────────────────────────
Genuine-only signature classification for representation learning.

Follows: https://github.com/amaljoseph/EndToEnd_Signature-Detection-Cleaning-
         Verification_System_using_YOLOv5-and-CycleGAN
Paper:    https://arxiv.org/abs/2004.12104

What this does
──────────────
  1. Loads genuine signatures only (_forg folders are skipped).
  2. Fine-tunes VGG16 and/or ResNet50 as N-class classifiers
     (N = number of persons in the dataset).
  3. Strips the classification head → saves feature extractors:
       VGG16   FC1               → 4096-d
       ResNet50 conv5_block3_2_conv → 25088-d   (paper §3.2)
  4. At inference: extract a vector per signature image, compare
     with cosine similarity to verify identity.

Dataset layout  (Kaggle signature-verification-dataset)
────────────────────────────────────────────────────────
    train/
        001_org/    ← genuine person 001  (used)
        001_forg/   ← forged  person 001  (skipped automatically)
        002_org/
        002_forg/
        ...
    test/           ← different persons, kept for later verification eval

Why validation comes from train/, NOT test/
────────────────────────────────────────────
The dataset is writer-independent: persons in test/ are completely
different from those in train/.  If test/ is used as validation:

    ValueError: target.shape=(None, 21)  output.shape=(None, 64)

because the model has 64 output neurons (train classes) but test/
encodes 21 different persons.  Fix: split train/ itself 85/15 via
ImageDataGenerator(validation_split=0.15).

Usage
─────
    python train_verification.py \
        --train_dir ../data/sign_data/train \
        --test_dir  ../data/sign_data/test  \
        --output    ../model/verification_model \
        --backbone  both

    # single backbone
    python train_verification.py --backbone vgg16   ...
    python train_verification.py --backbone resnet50 ...
#NOTE: THE LAST TIME I USE RESCALE, IT WORK ON VGG16 but DOES NOT WORK ON RESNET, which create terrible result
"""

import argparse
import os
import shutil
import tempfile
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore")

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    TensorBoard,
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# ADDED: Import the official ImageNet preprocessing function
from tensorflow.keras.applications.resnet50 import preprocess_input as keras_preprocess_input


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — filter forged folders
# ─────────────────────────────────────────────────────────────────────────────

def create_filtered_dataset(src_dir: str, ignore_suffix: str = "_forg") -> str:
    """
    Copy src_dir to a temp directory keeping only genuine person folders.
    Folders whose name ends with ignore_suffix are silently skipped.
    Returns the temp dir path.
    """
    temp_dir = tempfile.mkdtemp()
    for name in sorted(os.listdir(src_dir)):
        src_path = os.path.join(src_dir, name)
        if not os.path.isdir(src_path):
            continue
        if name.endswith(ignore_suffix):
            continue                        # skip _forg silently
        shutil.copytree(src_path, os.path.join(temp_dir, name))
    return temp_dir


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — data generators
# ─────────────────────────────────────────────────────────────────────────────

def make_generators(
    train_dir: str,
    preprocessing_function,
    batch_size: int = 32,
    img_size:   int = 224,
    val_split:  float = 0.15,
):
    """
    Returns (train_gen, val_gen, num_classes).

    Both generators are built from train_dir so they share the exact
    same class mapping.  val_split (default 15 %) is held out for
    validation during training.
    """
    filtered = create_filtered_dataset(train_dir)

    aug = ImageDataGenerator(
        # FIX: Replaced rescale=1.0/255 with official preprocessing.
        # ImageNet weights expect BGR format zero-centered from -125 to 125, not 0 to 1 floats!
        preprocessing_function = preprocessing_function,
        rotation_range   = 8,       # paper: rotation augmentation
        width_shift_range= 0.10,
        height_shift_range=0.10,
        shear_range      = 0.05,
        zoom_range       = 0.08,
        horizontal_flip  = False,   # signatures are NOT symmetric
        fill_mode        = "constant",
        cval             = 1.0,     # white padding
        validation_split = val_split,
    )

    common = dict(
        directory    = filtered,
        target_size  = (img_size, img_size),
        color_mode   = "rgb",
        batch_size   = batch_size,
        class_mode   = "categorical",
    )

    train_gen = aug.flow_from_directory(**common, shuffle=True,  subset="training")
    val_gen   = aug.flow_from_directory(**common, shuffle=False, subset="validation")

    num_classes = train_gen.num_classes
    print(
        f"[data] {num_classes} persons  |  "
        f"train: {train_gen.samples}  val: {val_gen.samples}"
    )
    return train_gen, val_gen, num_classes


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — model builders
# ─────────────────────────────────────────────────────────────────────────────

def build_vgg16(num_classes: int, weights_path: str = None) -> tf.keras.Model:
    """
    VGG16 pretrained on ImageNet, classification head replaced with
    Dense(num_classes, softmax).

    Feature extraction after training:
        FC1 layer → 4096-d vector  (paper §3.2, repo notebook)
    """
    if weights_path and os.path.isfile(weights_path):
        base = VGG16(weights=None, include_top=True)
        base.load_weights(weights_path)
        print(f"[vgg16] Loaded local weights: {weights_path}")
    else:
        base = VGG16(weights="imagenet", include_top=True)
        print("[vgg16] Loaded ImageNet weights from Keras")

    # Replace Dense(1000) with Dense(num_classes)
    model = models.Sequential(name="vgg16_classifier")
    for layer in base.layers[:-1]:
        model.add(layer)
    model.add(layers.Dense(num_classes, activation="softmax", name="predictions"))

    # Freeze backbone for Phase 1
    for layer in model.layers[:-1]:
        layer.trainable = False

    return model


def build_resnet50(num_classes: int, weights_path: str = None) -> tf.keras.Model:
    """
    ResNet50 pretrained on ImageNet, custom classification head.

    Feature extraction after training: GAP → 2048-d → fc1 → 4096-d

    Why not conv5_block3_2_conv (previous approach)
    ────────────────────────────────────────────────
    That layer is mid-block: the skip connection and final activation are
    both discarded, giving incomplete features.  conv5_block3_out is the
    correct tap point — after the residual add and final activation.

    BatchNormalization fix (critical for ResNet50)
    ───────────────────────────────────────────────
    ResNet50 has 53 BN layers.  When frozen layers run in training mode,
    BN updates its running statistics from signature images (totally
    different distribution from ImageNet), corrupting pretrained features.
    Fix: keep ALL BN layers in inference mode throughout training.
    VGG16 has zero BN layers so it never hits this problem.
    """
    if weights_path and os.path.isfile(weights_path):
        base = ResNet50(weights=None, include_top=False, input_shape=(224, 224, 3))
        base.load_weights(weights_path)
        print(f"[resnet50] Loaded local weights: {weights_path}")
    else:
        base = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
        print("[resnet50] Loaded ImageNet weights from Keras")

    # ── Keep ALL BatchNorm layers in inference mode permanently ───────────
    # This prevents BN statistics from being corrupted by the small
    # signature dataset. This is the most important fix for ResNet50.
    for layer in base.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    inp = base.input
    x   = base.get_layer("conv5_block3_out").output  # (7,7,2048) — complete block output
    x   = layers.GlobalAveragePooling2D(name="gap")(x)   # → 2048-d
    x   = layers.Dense(4096, activation="relu", name="fc1")(x)
    x   = layers.Dropout(0.30)(x)                        # reduced from 0.50
    out = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = tf.keras.Model(inputs=inp, outputs=out, name="resnet50_classifier")

    # Freeze backbone for Phase 1 (BN already frozen above)
    for layer in base.layers:
        if not isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    return model


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — two-phase training
# ─────────────────────────────────────────────────────────────────────────────

def _compile(model: tf.keras.Model, lr: float) -> None:
    model.compile(
        loss      = "categorical_crossentropy",
        optimizer = optimizers.SGD(learning_rate=lr, momentum=0.9, decay=lr / 100),
        metrics   = ["accuracy"],
    )


def _callbacks(name: str, ckpt_dir: str) -> list:
    os.makedirs(ckpt_dir, exist_ok=True)
    return [
        EarlyStopping(
            monitor="val_loss", patience=6,
            restore_best_weights=True, verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_accuracy", factor=0.5,
            patience=4, min_lr=1e-6, verbose=1,
        ),
        ModelCheckpoint(
            filepath=os.path.join(ckpt_dir, f"{name}_best.keras"),
            monitor="val_accuracy", save_best_only=True, verbose=1,
        ),
        TensorBoard(log_dir=os.path.join(ckpt_dir, "logs", name)),
    ]


def train_backbone(
    model:         tf.keras.Model,
    name:          str,
    train_gen,
    val_gen,
    output_dir:    str,
    phase1_epochs: int = 30,
    phase2_epochs: int = 5,
) -> tf.keras.Model:
    """
    Phase 1  Backbone frozen  — only the new classification head is trained.
             lr=1e-3, up to phase1_epochs (early stopping applies).

    Phase 2  Full fine-tune   — all layers unfrozen, lower lr.
             lr=1e-4, up to phase2_epochs (early stopping applies).
    """
    print(f"\n{'='*60}\n  Training: {name.upper()}\n{'='*60}")
    ckpt = os.path.join(output_dir, name)

    # ── Phase 1 ───────────────────────────────────────────────────────────
    print("\n[Phase 1] Frozen backbone — training head only …")
    _compile(model, lr=1e-3)
    model.fit(
        train_gen,
        epochs          = phase1_epochs,
        validation_data = val_gen,
        callbacks       = _callbacks(f"{name}_phase1", ckpt),
    )

    # ── Phase 2 ───────────────────────────────────────────────────────────
    print("\n[Phase 2] Full fine-tuning …")
    for layer in model.layers:
        # Keep BatchNorm in inference mode — never unfreeze BN for small datasets
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = True
    _compile(model, lr=1e-4)
    model.fit(
        train_gen,
        epochs          = phase2_epochs,
        validation_data = val_gen,
        callbacks       = _callbacks(f"{name}_phase2", ckpt),
    )

    save_path = os.path.join(output_dir, f"{name}_finetuned.keras")
    model.save(save_path)
    print(f"\n[saved] Full model → {save_path}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 — feature extractor builders  (strip classification head)
# ─────────────────────────────────────────────────────────────────────────────

def make_vgg16_extractor(model: tf.keras.Model) -> tf.keras.Model:
    """Output: FC1 activations — 4096-d."""
    return tf.keras.Model(
        inputs  = model.inputs,                      # .inputs (plural) for Sequential
        outputs = model.get_layer("fc1").output,
        name    = "vgg16_extractor_fc1",
    )


def make_resnet50_extractor(model: tf.keras.Model) -> tf.keras.Model:
    """Output: FC1 activations — 4096-d."""
    return tf.keras.Model(
        inputs  = model.inputs,
        outputs = model.get_layer("fc1").output,
        name    = "resnet50_extractor_fc1",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Inference helper
# ─────────────────────────────────────────────────────────────────────────────

def extract_and_normalise(
    extractor:  tf.keras.Model,
    image_path: str,
    preprocessing_function,
    img_size:   int = 224,
) -> np.ndarray:
    """
    Load one image → run through extractor → L2-normalise.
    Returns a 1-D float32 array ready for cosine similarity.
    """
    img = tf.keras.preprocessing.image.load_img(
        image_path, target_size=(img_size, img_size)
    )
    arr = tf.keras.preprocessing.image.img_to_array(img)
    
    # FIX: Apply Keras preprocessing here too instead of manually dividing by 255.0
    arr = preprocessing_function(arr) 
    
    vec = extractor.predict(np.expand_dims(arr, 0), verbose=0).flatten()
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train VGG16 / ResNet50 signature feature extractors."
    )
    p.add_argument("--train_dir",
                   required=True,
                   help="Folder containing per-person subfolders (e.g. 001_org/)")
    p.add_argument("--test_dir",
                   required=False, default=None,
                   help="Held-out folder (different persons). "
                        "Not used during training — reserved for verification eval.")
    p.add_argument("--output",
                   default="saved_models",
                   help="Directory where models and extractors are saved")
    p.add_argument("--backbone",
                   choices=["vgg16", "resnet50", "both"], default="both")
    p.add_argument("--batch_size",     type=int,   default=32)
    p.add_argument("--phase1_epochs",  type=int,   default=50,
                   help="Max epochs for frozen warm-up (default 30)")
    p.add_argument("--phase2_epochs",  type=int,   default=10,
                   help="Max epochs for full fine-tune (default 5)")
    p.add_argument("--val_split",      type=float, default=0.15,
                   help="Fraction of train_dir used for validation (default 0.15)")
    p.add_argument("--vgg16_weights",
                   default="../model/vgg16_weights_tf_dim_ordering_tf_kernels.h5",
                   help="Path to local VGG16 .h5 weights file (skips download)")
    p.add_argument("--resnet_weights",
                   default="../model/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5",
                   help="Path to local ResNet50 .h5 weights file (skips download)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # ── Generators ────────────────────────────────────────────────────────
    train_gen, val_gen, num_classes = make_generators(
        train_dir              = args.train_dir,
        preprocessing_function = keras_preprocess_input,
        batch_size             = args.batch_size,
        val_split              = args.val_split,
    )

    os.makedirs(args.output, exist_ok=True)
    trained: dict = {}

    # ── Train VGG16 ───────────────────────────────────────────────────────
    if args.backbone in ("vgg16", "both"):
        vgg = build_vgg16(num_classes, weights_path=args.vgg16_weights)
        vgg.summary(line_length=100)
        trained["vgg16"] = train_backbone(
            model         = vgg,
            name          = "vgg16",
            train_gen     = train_gen,
            val_gen       = val_gen,
            output_dir    = args.output,
            phase1_epochs = args.phase1_epochs,
            phase2_epochs = args.phase2_epochs,
        )

    # ── Train ResNet50 ────────────────────────────────────────────────────
    if args.backbone in ("resnet50", "both"):
        resnet = build_resnet50(num_classes, weights_path=args.resnet_weights)
        resnet.summary(line_length=100)
        trained["resnet50"] = train_backbone(
            model         = resnet,
            name          = "resnet50",
            train_gen     = train_gen,
            val_gen       = val_gen,
            output_dir    = args.output,
            phase1_epochs = args.phase1_epochs,
            phase2_epochs = args.phase2_epochs,
        )

    # ── Save extractors ───────────────────────────────────────────────────
    print("\n[extractors] Saving feature extractor sub-models …")

    if "vgg16" in trained:
        ext = make_vgg16_extractor(trained["vgg16"])
        ext.save(os.path.join(args.output, "vgg16_extractor.keras"))
        print(f"  vgg16_extractor    → {ext.output_shape[-1]}-d "
              f"saved to {args.output}/vgg16_extractor.keras")

    if "resnet50" in trained:
        ext = make_resnet50_extractor(trained["resnet50"])
        ext.save(os.path.join(args.output, "resnet50_extractor.keras"))
        print(f"  resnet50_extractor → {ext.output_shape[-1]}-d (fc1 bottleneck)"
              f"saved to {args.output}/resnet50_extractor.keras")

    print("\n[done]")