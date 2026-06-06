# Signature Verification System

An end-to-end Machine Learning system for detecting, cleaning, and verifying handwritten signatures in scanned documents (PDFs and images). This repository contains both the **model training pipeline** and the **production deployment service**.

## 🌟 Key Features

- **Document Parsing:** Detect signatures directly from full PDF document pages or scanned images.
- **Signature Denoising:** Automatically removes background noise, red official stamps, printed lines, and overlapping text to extract a clean signature.
- **High-Accuracy Matching:** Extracts deep feature embeddings to compare a queried signature against registered genuine signatures.
- **Scalable Serving:** Production-ready inference powered by NVIDIA Triton Inference Server.
- **Vector Search:** Uses PostgreSQL with the `pgvector` extension for fast L2 nearest-neighbor matching.

## 🏗️ Architecture

The system relies on four distinct deep learning models working together in a pipeline:

1. **YOLOv8s**: Object detection model to locate bounding boxes of signatures inside a full document page.
2. **CycleGAN (ResNet-9)**: Image-to-image translation model acting as a denoiser to remove stamps and text from the cropped signature.
3. **ResNet50**: Feature extractor (fine-tuned) generating a 4096-dimensional embedding of the clean signature.
4. **VGG16**: Feature extractor (fine-tuned) generating a 4096-dimensional embedding of the clean signature.

*For more details, see the [Architecture Overview](documentation/01-architecture-overview.md).*

## 📂 Repository Structure

The repository is logically split into three main components:

- [`trainingfiles/`](trainingfiles/): The research and ML training environment. Includes scripts for dataset preparation, CycleGAN training, ResNet/VGG fine-tuning, and PyTorch/Keras to ONNX model conversion.
- [`signature-verification-service/`](signature-verification-service/): The production microservice environment. Contains the FastAPI backend, Docker Compose setup, Triton Inference Server configurations, and pgvector database integration.
- [`documentation/`](documentation/): Comprehensive, step-by-step guides for setting up, training, and deploying the system.

## 📚 Documentation

Detailed documentation is available in the `documentation/` folder. Start here if you are new to the project:

1. **[Architecture Overview](documentation/01-architecture-overview.md)** — High-level system design and ML pipeline stages.
2. **[Training Pipeline — Setup & Run](documentation/02-training-pipeline.md)** — How to train and evaluate models in `trainingfiles/`.
3. **[Service Deployment — Setup & Run](documentation/03-service-deployment.md)** — How to deploy the production FastAPI/Triton microservice.
4. **[Model Conversion Guide](documentation/04-model-conversion.md)** — Converting `.pth`/`.keras` models to `.onnx` for deployment.
5. **[API Reference](documentation/05-api-reference.md)** — REST API endpoints for the FastAPI backend.
6. **[Database Schema](documentation/06-database-schema.md)** — PostgreSQL & pgvector schema designs.
7. **[Triton Configuration](documentation/07-triton-config.md)** — Inference Server model repository setup.
8. **[Models & Datasets](documentation/08-models-and-datasets.md)** — Inventory of datasets, pretrained weights, and model files.

## 🚀 Quick Start

### 1. Training the Models
If you want to train the models from scratch, evaluate them, or export existing PyTorch checkpoints:
```bash
cd trainingfiles/
# See documentation/02-training-pipeline.md for full instructions
```

### 2. Deploying the Service
To run the pre-configured FastAPI service and Triton Inference Server locally:
```bash
cd signature-verification-service/
cp .env.example .env
# Edit .env to set your secure passwords
docker compose up -d
```
*See the [Deployment Guide](documentation/03-service-deployment.md) for full instructions.*

## 🔒 Security & Configuration

All infrastructure components have been designed securely. Default registry IPs and hardcoded passwords have been stripped from the repository. Ensure you properly configure your local `.env` files based on the provided `.env.example` templates before spinning up the services.

## 📄 License
This project is licensed under the terms of the LICENSE file included in the root of the repository.