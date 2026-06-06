# 5 — API Reference

> REST API documentation for the Signature Verification Service.
>
> **Base URL:** `http://<host>:8080`

---

## Endpoints Summary

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Service health check |
| `POST` | `/register-signature` | Register a new signature |
| `POST` | `/verify-document` | Verify signatures in a document |
| `GET` | `/signatures` | List registered signatures |
| `DELETE` | `/signatures/{id}` | Delete a registered signature |

---

## `GET /health`

Check if the API is running.

### Response

```json
{
  "status": "ok",
  "env": "production"
}
```

---

## `POST /register-signature`

Upload a clean signature image to register a user's signature.

### Pipeline

```
Upload → Resize 224×224 → CycleGAN denoise → ResNet50 + VGG16 extract → INSERT into DB
```

> **Note:** YOLOv8 detection is **skipped** here — the uploaded file is assumed to be
> an isolated signature crop.

### Request

- **Content-Type:** `multipart/form-data`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `username` | `string` (Form) | ✅ | User identifier |
| `file` | `file` (Upload) | ✅ | Clean signature image (JPG/PNG/WEBP) |

### Example

```bash
curl -X POST http://localhost:8080/register-signature \
  -F "username=nguyen_van_a" \
  -F "file=@signature.png"
```

### Response — `201 Created`

```json
{
  "id": 1,
  "username": "nguyen_van_a",
  "user_created_date": "2024-01-15T10:30:00"
}
```

### Error Responses

| Status | Detail | Cause |
|--------|--------|-------|
| `400` | `Could not decode image: ...` | Invalid/corrupt image file |
| `502` | `Triton CycleGAN error: ...` | Triton Inference Server error |
| `502` | `Triton extractor error: ...` | ResNet50/VGG16 inference failed |

---

## `POST /verify-document`

Upload a document (PDF or image) to detect and verify signatures on each page.

### Pipeline

```
Upload → PDF→pages (or single image) → FOR each page:
  → YOLOv8 detect signature → crop → CycleGAN denoise
  → ResNet50 + VGG16 extract vectors
  → pgvector L2 nearest-neighbour search
  → avg_distance = (d_resnet + d_vgg) / 2
  → matched = avg_distance < 0.5
```

### Request

- **Content-Type:** `multipart/form-data`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file` | `file` (Upload) | ✅ | Document file (PDF/JPG/PNG) |

### Example

```bash
curl -X POST http://localhost:8080/verify-document \
  -F "file=@contract.pdf"
```

### Response — `200 OK`

```json
{
  "total_pages": 2,
  "results": [
    {
      "page": 1,
      "status": "matched",
      "username": "nguyen_van_a",
      "avg_distance": 0.123456,
      "resnet_distance": 0.110000,
      "vgg_distance": 0.136912,
      "matched": true,
      "bbox": [120, 340, 380, 520],
      "page_annotated": "<base64 PNG — full page with red bbox>",
      "crop_before": "<base64 PNG — cropped signature before CycleGAN>",
      "crop_after": "<base64 PNG — cropped signature after CycleGAN>"
    },
    {
      "page": 2,
      "status": "no_signature"
    }
  ]
}
```

### Result status values

| Status | Meaning |
|--------|---------|
| `matched` | Signature detected, extracted, and matched against DB |
| `no_signature` | No signature region detected on this page (YOLOv8 returned nothing) |
| `no_match_in_db` | Signature detected but no matching user found in database |

### Result fields (when `status == "matched"`)

| Field | Type | Description |
|-------|------|-------------|
| `page` | `int` | Page number (1-indexed) |
| `status` | `string` | Always `"matched"` |
| `username` | `string` | Username of the best-matching registered signature |
| `avg_distance` | `float` | Average of ResNet50 + VGG16 L2 distances |
| `resnet_distance` | `float` | ResNet50 embedding L2 distance |
| `vgg_distance` | `float` | VGG16 embedding L2 distance |
| `matched` | `bool` | `true` if `avg_distance < 0.5` |
| `bbox` | `[x1,y1,x2,y2]` | Bounding box of detected signature in page coordinates |
| `page_annotated` | `string` | Base64-encoded PNG — full page with red rectangle around signature |
| `crop_before` | `string` | Base64-encoded PNG — cropped signature before denoising |
| `crop_after` | `string` | Base64-encoded PNG — cropped signature after CycleGAN denoising |

### Distance interpretation

| avg_distance | Interpretation |
|-------------|----------------|
| `< 0.3` | Strong match (high confidence) |
| `0.3 – 0.5` | Moderate match |
| `> 0.5` | Not matched (`matched` = false) |

---

## `GET /signatures`

List all registered signature entries (without vector data).

### Query Parameters

| Parameter | Type | Default | Constraints | Description |
|-----------|------|---------|-------------|-------------|
| `skip` | `int` | `0` | `≥ 0` | Number of entries to skip (pagination offset) |
| `limit` | `int` | `50` | `1 – 200` | Maximum entries to return |

### Example

```bash
curl "http://localhost:8080/signatures?skip=0&limit=50"
```

### Response — `200 OK`

```json
{
  "total": 42,
  "skip": 0,
  "limit": 50,
  "items": [
    {
      "id": 5,
      "username": "nguyen_van_a",
      "user_created_date": "2024-01-15T10:30:00",
      "user_modified_date": null
    },
    {
      "id": 4,
      "username": "tran_thi_b",
      "user_created_date": "2024-01-14T09:00:00",
      "user_modified_date": null
    }
  ]
}
```

> Items are sorted by `user_created_date` descending (newest first).

---

## `DELETE /signatures/{item_id}`

Delete a registered signature by ID.

### Path Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `item_id` | `int` | ID of the signature entry to delete |

### Example

```bash
curl -X DELETE http://localhost:8080/signatures/5
```

### Response — `204 No Content`

Empty response body.

### Error Responses

| Status | Detail | Cause |
|--------|--------|-------|
| `404` | `Signature not found` | No entry with that ID exists |

---

## Interactive API Documentation

FastAPI provides auto-generated interactive docs:

| URL | Format |
|-----|--------|
| `http://localhost:8080/docs` | Swagger UI (interactive) |
| `http://localhost:8080/redoc` | ReDoc (read-only) |
