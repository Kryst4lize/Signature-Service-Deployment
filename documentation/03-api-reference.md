# 3 — API Reference

Base URL: `http://<host>:8080`, or `/api` through the nginx proxy on `:8110`.

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Liveness |
| `GET` | `/signatures` | List registrations |
| `DELETE` | `/signatures/{id}` | Remove one |
| `POST` | `/register-signature` | Enrol a signature |
| `POST` | `/verify-document` | Verify signatures in a document |

Interactive schema: `http://<host>:8080/docs`.

There is **no authentication**. Both upload endpoints are open. Do not expose
this directly to an untrusted network — see
[Operations](./05-operations.md#exposure).

---

## `GET /health`

```json
{ "status": "ok", "env": "production" }
```

Does not check Postgres or Triton — it reports that the process is serving.
For dependency health use Triton's `/v2/health/ready` on `:8010` and
`pg_isready`.

---

## `GET /signatures`

| Param | Type | Default | Bounds |
|---|---|---|---|
| `skip` | int | 0 | ≥ 0 |
| `limit` | int | 50 | 1–200 |

```json
{
  "total": 42,
  "skip": 0,
  "limit": 50,
  "items": [
    {
      "id": 7,
      "username": "nguyen_van_a",
      "user_created_date": "2026-01-15T10:30:00",
      "user_modified_date": null
    }
  ]
}
```

Newest first. Vectors are never returned — two 4096-d arrays per row would
dominate the payload.

---

## `DELETE /signatures/{id}`

`204` on success, `404` if the id does not exist. No body.

---

## `POST /register-signature`

`multipart/form-data`.

| Field | Type | Required | Constraints |
|---|---|---|---|
| `username` | string | yes | 1–50 characters (the column is `VARCHAR(50)`) |
| `file` | file | yes | Image or PDF; ≤ `MAX_UPLOAD_BYTES` |

The upload is assumed to be an **already-isolated signature** — detection is
skipped. For a full page, use `/verify-document`.

```bash
curl -X POST localhost:8080/register-signature \
  -F "username=nguyen_van_a" \
  -F "file=@signature.png"
```

`201`:

```json
{
  "id": 7,
  "username": "nguyen_van_a",
  "user_created_date": "2026-01-15T10:30:00"
}
```

Usernames are not unique. Registering the same person several times adds
several rows, which is the intended way to enrol multiple specimens — the
nearest-neighbour search then matches against whichever is closest.

---

## `POST /verify-document`

`multipart/form-data`, field `file`. PDF or image, ≤ `MAX_UPLOAD_BYTES`.
PDFs are rendered at 200 dpi, up to `MAX_PDF_PAGES`.

```bash
curl -X POST localhost:8080/verify-document -F "file=@contract.pdf"
```

```jsonc
{
  "total_pages": 2,
  "results": [
    {
      "page": 1,
      "status": "matched",
      "username": "nguyen_van_a",
      "avg_distance": 0.118,
      "resnet_distance": 0.104,
      "vgg_distance": 0.132,
      "matched": true,
      "bbox": [412, 980, 947, 1183],
      "confidence": 0.93,
      "page_annotated": "<base64 png>",
      "crop_before":    "<base64 png>",
      "crop_after":     "<base64 png>"
    },
    { "page": 2, "status": "no_signature" }
  ]
}
```

### `status` is not the verdict

| `status` | Means |
|---|---|
| `matched` | The pipeline completed and a nearest neighbour was returned |
| `no_signature` | No detection cleared `DETECTION_CONFIDENCE` |
| `no_match_in_db` | The `items` table is empty |

`status` is `matched` whenever the table is non-empty, because the query always
returns *something*. **`matched` (the boolean) is the decision** —
`avg_distance < MATCH_THRESHOLD`. Read that, not `status`.

`username` is likewise the nearest neighbour regardless of distance. A stranger
signing a page yields the closest enrolled person with `matched: false` and a
large `avg_distance`.

### Distances

Cosine distances between L2-normalised 4096-d embeddings, so bounded to
`[0, 2]`: 0 identical, 1 orthogonal, 2 opposite.

```
avg_distance = (resnet_distance + vgg_distance) / 2
```

> These were raw **L2** distances before v2.0.0, compared against a hardcoded
> `0.5`. A threshold carried over from that era does not mean the same thing.
> Recalibrate with `sigtrain evaluate`.

### Images

`page_annotated` is the page with a red box, downscaled to a 1024 px long edge.
`crop_before` is the detected region at full resolution; `crop_after` is the
224×224 denoised version.

`bbox` is `[x1, y1, x2, y2]` in **original page pixels**, not detector-input
pixels.

---

## Errors

| Status | When |
|---|---|
| `400` | Undecodable or empty upload |
| `404` | `DELETE` on an unknown id |
| `413` | Upload exceeds `MAX_UPLOAD_BYTES` |
| `422` | `username` empty or over 50 characters, or a bad query parameter |
| `502` | Triton unreachable or a model failed |

```json
{ "detail": "File exceeds the 20971520 byte limit" }
```

A `502` on a multi-page document aborts the whole request; pages already
processed are not returned.

---

## Notes for callers

- **Latency scales with pages.** Each page costs up to four GPU inferences.
  A 20-page PDF can take minutes; the nginx proxy allows 600 s.
- **Responses are large.** Three base64 PNGs per page with a detection. Budget
  ~1–3 MB per page.
- **No pagination on verify.** Use `MAX_PDF_PAGES`, or split the document.
- **Re-enrol after a preprocessing change.** Embeddings are only comparable
  with others produced by the same preprocessing.
