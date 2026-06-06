# Signature Verification System

An end-to-end Machine Learning system for detecting, cleaning, and verifying handwritten signatures in scanned documents (PDFs and images). This repository contains both the **model training pipeline** and the **production deployment service**.

## 🌟 Key Features

- **Document Parsing:** Detect signatures directly from full PDF document pages or scanned images using YOLOv8.
- **Signature Denoising:** Automatically removes background noise, red official stamps, printed lines, and overlapping text to extract a clean signature using CycleGAN.
- **High-Accuracy Matching:** Extracts deep feature embeddings (using ResNet50 and VGG16) to compare a queried signature against registered genuine signatures.
- **Scalable Serving:** Production-ready inference powered by NVIDIA Triton Inference Server.
- **Vector Search:** Uses PostgreSQL with the `pgvector` extension for fast L2 nearest-neighbor matching.

## 📂 Repository Structure & Setup

The repository is logically split into two main environments:

### 1. `trainingfiles/` (Model Training & Conversion)
This folder contains the research and ML training environment. It includes scripts for dataset preparation, CycleGAN training, ResNet/VGG fine-tuning, and PyTorch/Keras to ONNX model conversion.

> **⚠️ Important Requirement:** The CycleGAN architecture relies on a third-party repository that is **not** included in this git repository. To run any CycleGAN training or export scripts, you must manually clone it into the training folder:
> ```bash
> cd trainingfiles/
> git clone https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix.git
> ```

### 2. `signature-verification-service/` (Production Service)
The production microservice environment. It contains the FastAPI backend, Docker Compose setup, Triton Inference Server configurations, and pgvector database integration.

## 🚀 Quick Start

### 1. Training the Models
If you want to train the models from scratch, evaluate them, or export existing PyTorch checkpoints to ONNX format:
```bash
cd trainingfiles/
# Make sure you clone the CycleGAN repository as mentioned above!
# Then follow the comprehensive guide in documentation/02-training-pipeline.md
```

### 2. Deploying the Service
To run the pre-configured FastAPI service and Triton Inference Server locally:
```bash
cd signature-verification-service/
cp .env.example .env
# Open .env and set your secure passwords for the database
docker compose up -d
```
*See the [Deployment Guide](documentation/03-service-deployment.md) for full instructions.*

## 🔒 Security & Configuration
All infrastructure components have been designed securely. Ensure you properly configure your local `.env` files based on the provided `.env.example` templates before spinning up the services. Never commit your `.env` files or hardcode passwords in the codebase.

## 📚 Documentation Directory

Detailed documentation is available in the `documentation/` folder. Start here to understand the system in depth:

1. **[Architecture Overview](documentation/01-architecture-overview.md)** — High-level system design and ML pipeline stages.
2. **[Training Pipeline — Setup & Run](documentation/02-training-pipeline.md)** — How to train and evaluate models.
3. **[Service Deployment — Setup & Run](documentation/03-service-deployment.md)** — How to deploy the production FastAPI/Triton microservice.
4. **[Model Conversion Guide](documentation/04-model-conversion.md)** — Converting `.pth`/`.keras` models to `.onnx` for deployment.
5. **[API Reference](documentation/05-api-reference.md)** — REST API endpoints for the FastAPI backend.
6. **[Database Schema](documentation/06-database-schema.md)** — PostgreSQL & pgvector schema designs.
7. **[Triton Configuration](documentation/07-triton-config.md)** — Inference Server model repository setup.
8. **[Models & Datasets](documentation/08-models-and-datasets.md)** — Inventory of datasets, pretrained weights, and model files.

## 📄 License
This project is licensed under the terms of the LICENSE file included in the root of the repository.