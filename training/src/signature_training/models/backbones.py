"""VGG16 and ResNet50 classifiers, and the feature extractors cut out of them.

Both are trained as N-way person classifiers and then truncated at `fc1`, which
is the 4096-d embedding the inference service stores and compares.
Following https://arxiv.org/abs/2004.12104.

`fc1` for both is not a coincidence to preserve: `inference/postgres/init.sql`
declares VECTOR(4096) and each `config.pbtxt` declares `dims: [4096]`. Changing
`verification.embedding_dim` means changing all three.
"""

from __future__ import annotations

import logging
import os

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50, VGG16

logger = logging.getLogger(__name__)


def build_vgg16(num_classes: int, embedding_dim: int = 4096, weights_path: str = "") -> tf.keras.Model:
    """VGG16 with its 1000-way head replaced.

    VGG16's own `fc1` is already 4096-d, so the extractor tap needs no extra
    layer. `embedding_dim` is asserted rather than applied, because changing it
    would mean rebuilding the classifier head and invalidating the pretrained
    fully-connected weights that make this transfer work at all.
    """
    if embedding_dim != 4096:
        raise ValueError(
            f"VGG16's fc1 is fixed at 4096-d; got embedding_dim={embedding_dim}. "
            "Use --backbone resnet50 if you need a different width."
        )

    if weights_path and os.path.isfile(weights_path):
        base = VGG16(weights=None, include_top=True)
        base.load_weights(weights_path)
        logger.info("VGG16: loaded local weights from %s", weights_path)
    else:
        base = VGG16(weights="imagenet", include_top=True)
        logger.info("VGG16: loaded ImageNet weights")

    model = models.Sequential(name="vgg16_classifier")
    for layer in base.layers[:-1]:          # drop predictions (1000-way)
        model.add(layer)
    model.add(layers.Dense(num_classes, activation="softmax", name="predictions"))

    for layer in model.layers[:-1]:         # phase 1: head only
        layer.trainable = False
    return model


def build_resnet50(
    num_classes: int, embedding_dim: int = 4096, weights_path: str = ""
) -> tf.keras.Model:
    """ResNet50 tapped at conv5_block3_out -> GAP -> Dense(embedding_dim) 'fc1'.

    conv5_block3_out, not conv5_block3_2_conv: the latter is mid-block, so the
    residual add and the final activation are both discarded.

    Every BatchNormalization layer is frozen permanently. ResNet50 has 53 of
    them; letting them update running statistics on a few thousand signature
    images — a distribution nothing like ImageNet — corrupts the pretrained
    features this whole approach depends on. VGG16 has no BN and never hits
    this.
    """
    if weights_path and os.path.isfile(weights_path):
        base = ResNet50(weights=None, include_top=False, input_shape=(224, 224, 3))
        base.load_weights(weights_path)
        logger.info("ResNet50: loaded local weights from %s", weights_path)
    else:
        base = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
        logger.info("ResNet50: loaded ImageNet weights")

    # Phase 1 freezes the whole backbone, BatchNorm included.
    for layer in base.layers:
        layer.trainable = False

    x = base.get_layer("conv5_block3_out").output       # (7, 7, 2048)
    x = layers.GlobalAveragePooling2D(name="gap")(x)    # 2048-d
    x = layers.Dense(embedding_dim, activation="relu", name="fc1")(x)
    x = layers.Dropout(0.30)(x)
    out = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    return tf.keras.Model(inputs=base.input, outputs=out, name="resnet50_classifier")


def make_extractor(model: tf.keras.Model, name: str) -> tf.keras.Model:
    """Truncate a trained classifier at `fc1`."""
    extractor = tf.keras.Model(
        inputs=model.inputs,
        outputs=model.get_layer("fc1").output,
        name=f"{name}_extractor_fc1",
    )
    logger.info("%s extractor output: %d-d", name, extractor.output_shape[-1])
    return extractor


def set_finetune_trainable(model: tf.keras.Model) -> None:
    """Phase 2: unfreeze everything except BatchNorm, which stays in inference
    mode for the reason described in build_resnet50."""
    for layer in model.layers:
        layer.trainable = not isinstance(layer, tf.keras.layers.BatchNormalization)
