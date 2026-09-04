# Test runner for the stages that need neither TensorFlow nor torch — dataset
# building, augmentation, config, metrics, pair construction and the CLI.
#
# Deliberately not the full training image: that one is several GB of CUDA
# wheels and takes minutes to build, which makes `make test` useless as a
# feedback loop.
FROM python:3.11-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 libgl1 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
        numpy==1.26.4 \
        scipy==1.14.1 \
        scikit-learn==1.5.2 \
        pillow==11.0.0 \
        opencv-python-headless==4.10.0.84 \
        matplotlib==3.9.3 \
        tqdm==4.67.1 \
        PyYAML==6.0.2 \
        onnx==1.17.0 \
        pytest==8.3.4

ENV PYTHONPATH=/w/src
WORKDIR /w
