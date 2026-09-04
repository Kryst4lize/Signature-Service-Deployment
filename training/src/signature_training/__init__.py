"""Signature verification training pipeline.

Stages, in order:

    fetch-data          download/extract the Kaggle dataset and clone CycleGAN
    data-cyclegan       build paired clean/noisy images for the denoiser
    data-verification   filter genuine-only person folders
    train-cyclegan      train the denoiser (wraps the upstream repo)
    train-verification  fine-tune VGG16 + ResNet50, save feature extractors
    evaluate            EER / AUC / d' / TAR@FAR on held-out identities
    export              ONNX + config.pbtxt into a Triton model repository

Driven by `sigtrain`; see cli.py.
"""

__version__ = "2.0.0"
