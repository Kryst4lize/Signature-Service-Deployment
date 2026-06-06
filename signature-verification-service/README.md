# Signature Verification Service

## Project structure

```
signature-verification/
├── docker-compose.yml
├── .env.example
├── postgres/
│   └── init.sql                  # pgvector schema + seed
├── triton/
│   └── model_repository/
│       ├── yolov8/
│       │   ├── config.pbtxt
│       │   └── 1/model.onnx      # ← place your ONNX file here
│       ├── cyclegan/
│       │   ├── config.pbtxt
│       │   └── 1/model.onnx
│       ├── resnet50/
│       │   ├── config.pbtxt
│       │   └── 1/model.onnx
│       └── vgg16/
│           ├── config.pbtxt
│           └── 1/model.onnx
└── api/
    ├── Dockerfile
    ├── requirements.txt
    └── app/
        ├── main.py
        ├── core/
        │   ├── config.py
        │   └── database.py
        ├── models/
        │   └── db.py
        ├── routers/
        │   └── signatures.py
        └── services/
            ├── triton.py
            └── preprocessing.py
```

## 1. Add your ONNX model files

Each model must be exported to ONNX and placed at:

```
triton/model_repository/<model_name>/1/model.onnx
```

Input/output names in each `config.pbtxt` must match your ONNX graph.

## 2. Configure environment

```bash
cp .env.example .env
# edit .env if needed
```

## 3. Deploy

```bash
# Build and start all services
docker compose up --build -d

# Check logs
docker compose logs -f api
docker compose logs -f triton
docker compose logs -f postgres

# Stop
docker compose down
```

## 4. API usage

### Register a signature

```bash
curl -X POST http://localhost:8080/register-signature \
  -F "username=john_doe" \
  -F "file=@/path/to/signature.png"
```

Response:
```json
{ "id": 1, "username": "john_doe", "user_created_date": "2024-01-01T00:00:00" }
```

### Verify a document

```bash
curl -X POST http://localhost:8080/verify-document \
  -F "file=@/path/to/document.pdf"
```

Response:
```json
{
  "total_pages": 2,
  "results": [
    {
      "page": 1,
      "status": "matched",
      "username": "john_doe",
      "avg_distance": 0.123456,
      "resnet_distance": 0.110000,
      "vgg_distance": 0.136912,
      "matched": true
    },
    { "page": 2, "status": "no_signature" }
  ]
}
```

## 5. GPU support

Uncomment the `deploy.resources` block in `docker-compose.yml` under the `triton` service,
and change `KIND_CPU` to `KIND_GPU` in each `config.pbtxt`.

## 6. Tune the match threshold

In `api/app/routers/signatures.py`, adjust the threshold:

```python
"matched": float(match["avg_distance"]) < 0.5   # lower = stricter
```

Typical L2 distance ranges depend on your model's embedding space.
Start with 0.5 and calibrate against your dataset.
