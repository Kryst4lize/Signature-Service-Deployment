"""Train VGG16 / ResNet50 as person classifiers, then save their fc1 extractors.

Why classification and not a siamese/triplet objective: the reference work
(arXiv:2004.12104) trains an N-way classifier over genuine signatures and reuses
the penultimate activations as an identity embedding. Verification is then a
distance in that space, which generalises to people not in the training set.

Why validation is split out of train/ and not taken from test/: the dataset is
writer-independent, so test/ contains different people. Using it as validation
gives the model 64 output neurons and the validation labels 21 classes, i.e.
`ValueError: target.shape=(None, 21) output.shape=(None, 64)`.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    TensorBoard,
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from ..config import Config
from ..models.backbones import (
    build_resnet50,
    build_vgg16,
    make_extractor,
    set_finetune_trainable,
)

logger = logging.getLogger(__name__)

PREPROCESS = {"vgg16": vgg_preprocess, "resnet50": resnet_preprocess}

# Caffe preprocessing subtracts the ImageNet BGR mean from a [0, 255] image, so
# white paper (255) maps to about +131/+138/+151. Filling augmentation borders
# with 1.0 - as the original code did, commented "white padding" - puts them at
# almost the darkest value the network ever sees, teaching it that rotation
# introduces black wedges.
_WHITE_AFTER_CAFFE = 255.0


def _generators(cfg: Config, backbone: str):
    """(train_gen, val_gen, num_classes), both split from train/ so they share
    a class mapping."""
    train_dir = cfg.paths.resolve("verification_dataset") / "train"
    if not train_dir.is_dir():
        raise FileNotFoundError(f"{train_dir} not found. Run `sigtrain data-verification` first.")

    filtered = _genuine_only(train_dir)
    v = cfg.verification

    aug = ImageDataGenerator(
        preprocessing_function=PREPROCESS[backbone],
        rotation_range=8,
        width_shift_range=0.10,
        height_shift_range=0.10,
        shear_range=0.05,
        zoom_range=0.08,
        horizontal_flip=False,  # signatures are not mirror-symmetric
        fill_mode="constant",
        cval=_WHITE_AFTER_CAFFE,  # see the note above
        validation_split=v.val_split,
    )
    common = {
        "directory": str(filtered),
        "target_size": (v.image_size, v.image_size),
        "color_mode": "rgb",
        "batch_size": v.batch_size,
        "class_mode": "categorical",
        "seed": v.seed,
    }
    train_gen = aug.flow_from_directory(**common, shuffle=True, subset="training")
    val_gen = aug.flow_from_directory(**common, shuffle=False, subset="validation")

    logger.info(
        "%d persons | train %d | val %d",
        train_gen.num_classes,
        train_gen.samples,
        val_gen.samples,
    )
    return train_gen, val_gen, train_gen.num_classes, filtered


def _genuine_only(src: Path) -> Path:
    """Symlink genuine person folders into a temp tree.

    Symlinks, not copies: the original deep-copied the whole dataset on every
    run and never removed the temp directory, so repeated runs filled /tmp with
    duplicates of the corpus.
    """
    tmp = Path(tempfile.mkdtemp(prefix="sigtrain_genuine_"))
    for folder in sorted(p for p in src.iterdir() if p.is_dir()):
        if folder.name.endswith("_forg") or "forg" in folder.name.lower():
            continue
        (tmp / folder.name).symlink_to(folder.resolve(), target_is_directory=True)
    return tmp


def _callbacks(name: str, ckpt_dir: Path) -> list:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    return [
        EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_accuracy", factor=0.5, patience=4, min_lr=1e-6, verbose=1),
        ModelCheckpoint(
            filepath=str(ckpt_dir / f"{name}_best.keras"),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        TensorBoard(log_dir=str(ckpt_dir / "logs" / name)),
    ]


def _compile(model: tf.keras.Model, lr: float) -> None:
    # No `decay=` here. Keras 3 accepts it for backwards compatibility, pops it,
    # and warns "Argument `decay` is no longer supported and will be ignored" —
    # so the original code's intended schedule was silently doing nothing.
    # ReduceLROnPlateau provides the decay instead.
    model.compile(
        loss="categorical_crossentropy",
        optimizer=optimizers.SGD(learning_rate=lr, momentum=0.9),
        metrics=["accuracy"],
    )


def _train_one(cfg: Config, backbone: str, out_dir: Path) -> tf.keras.Model:
    v = cfg.verification
    train_gen, val_gen, num_classes, tmp_dir = _generators(cfg, backbone)

    try:
        builder = build_vgg16 if backbone == "vgg16" else build_resnet50
        weights = v.vgg16_weights if backbone == "vgg16" else v.resnet50_weights
        model = builder(num_classes, embedding_dim=v.embedding_dim, weights_path=weights)

        logger.info("=== %s: phase 1, frozen backbone ===", backbone)
        _compile(model, v.phase1_lr)
        model.fit(
            train_gen,
            epochs=v.phase1_epochs,
            validation_data=val_gen,
            callbacks=_callbacks(f"{backbone}_phase1", out_dir / backbone),
        )

        logger.info("=== %s: phase 2, full fine-tune ===", backbone)
        set_finetune_trainable(model)
        _compile(model, v.phase2_lr)
        model.fit(
            train_gen,
            epochs=v.phase2_epochs,
            validation_data=val_gen,
            callbacks=_callbacks(f"{backbone}_phase2", out_dir / backbone),
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    model.save(out_dir / f"{backbone}_finetuned.keras")
    extractor = make_extractor(model, backbone)
    extractor.save(out_dir / f"{backbone}_extractor.keras")
    logger.info("Saved %s extractor to %s", backbone, out_dir / f"{backbone}_extractor.keras")
    return model


def run(cfg: Config) -> dict[str, str]:
    """Train the configured backbone(s). Returns {backbone: extractor path}."""
    tf.keras.utils.set_random_seed(cfg.verification.seed)

    out_dir = cfg.paths.resolve("models")
    out_dir.mkdir(parents=True, exist_ok=True)

    selected = (
        ["vgg16", "resnet50"]
        if cfg.verification.backbone == "both"
        else [cfg.verification.backbone]
    )

    produced = {}
    for backbone in selected:
        _train_one(cfg, backbone, out_dir)
        produced[backbone] = str(out_dir / f"{backbone}_extractor.keras")

    cfg.dump(out_dir / "config.used.yaml")
    return produced


def embed(extractor: tf.keras.Model, image_path: str, backbone: str, size: int = 224) -> np.ndarray:
    """One image -> L2-normalised embedding, using the same preprocessing the
    model was trained with. The inference service must match this exactly;
    see inference/api/app/triton.py:to_caffe."""
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(size, size))
    arr = tf.keras.preprocessing.image.img_to_array(img)
    arr = PREPROCESS[backbone](np.expand_dims(arr, 0))
    vec = extractor.predict(arr, verbose=0).flatten()
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm > 0 else vec
