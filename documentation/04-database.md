# 4 — Database

PostgreSQL 17 with [pgvector](https://github.com/pgvector/pgvector).
Schema in [`../inference/postgres/init.sql`](../inference/postgres/init.sql),
applied once by the Postgres entrypoint on first boot.

---

## Schema

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS items (
    id                 SERIAL PRIMARY KEY,
    username           VARCHAR(50)  NOT NULL,
    user_created_date  TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP,
    user_modified_date TIMESTAMP,
    resnet50_vector    VECTOR(4096),
    vgg16_vector       VECTOR(4096)
);

CREATE INDEX IF NOT EXISTS idx_items_username ON items (username);
```

One table. One row per enrolled signature.

| Column | Notes |
|---|---|
| `username` | Not unique — multiple specimens per person is the intended enrolment pattern |
| `resnet50_vector` | L2-normalised `fc1` output |
| `vgg16_vector` | L2-normalised `fc1` output |

4096 is fixed by three files that must agree: this schema,
`Vector(4096)` in `inference/api/app/db.py`, and `dims: [ 4096 ]` in each
extractor's `config.pbtxt`.

---

## There is deliberately no ANN index

pgvector caps HNSW and IVFFlat at **2000 dimensions**. These columns are 4096.

The schema used to contain:

```sql
CREATE INDEX idx_resnet50_hnsw ON items USING hnsw (resnet50_vector vector_l2_ops) ...
```

That did not build a slow index. It aborted:

```
ERROR:  column cannot have more than 2000 dimensions for hnsw index
```

and because the Postgres entrypoint runs init scripts under `ON_ERROR_STOP=1`,
the container exited with code 3 before creating anything. `restart:
unless-stopped` then brought it back onto a now-non-empty `PGDATA`, which makes
the entrypoint skip initialisation permanently. The end state was a
running, healthy-looking database **with no `items` table at all** — and every
API call failing on an undefined relation.

Reproduced on the pinned `pgvector/pgvector:pg17` image and removed in v2.0.0.

### What search costs instead

An exact sequential scan. Measured at **~80 ms for 5,000 rows** on that image —
against a request that already spends far longer in four GPU inferences.

### If enrolment outgrows that

In increasing order of effort:

1. **`halfvec(4096)`** — halves storage and speeds the scan. Still not
   indexable; the 2000-d cap applies to `halfvec` HNSW too.
2. **Two-stage retrieval** — add an indexed low-dimensional column (a 512-d PCA
   or random projection fitted on the enrolled set), retrieve a candidate set
   with HNSW, then re-rank those candidates exactly on the 4096-d columns.
   Keeps exact results at the top while making the scan sub-linear.
3. **Re-export with a smaller head** — change
   `verification.embedding_dim`, retrain, and move the schema and the
   `config.pbtxt` dims with it. 2000-d or below becomes directly indexable.

---

## The search query

```sql
SELECT id,
       username,
       (resnet50_vector <=> CAST(:rv AS vector)) AS d_resnet,
       (vgg16_vector    <=> CAST(:vv AS vector)) AS d_vgg,
       (
           (resnet50_vector <=> CAST(:rv AS vector)) +
           (vgg16_vector    <=> CAST(:vv AS vector))
       ) / 2 AS avg_distance
FROM items
WHERE resnet50_vector IS NOT NULL
  AND vgg16_vector    IS NOT NULL
ORDER BY avg_distance ASC
LIMIT 1;
```

`<=>` is cosine distance. On L2-normalised vectors it is monotonically
equivalent to L2, but bounded to `[0, 2]`, which is what makes
`MATCH_THRESHOLD` a number with meaning rather than a magic constant on an
unbounded scale.

Averaging the two backbones is only valid because both are normalised —
otherwise the average is dominated by whichever embedding happens to have the
larger magnitude.

---

## Operations

### Inspect

```bash
docker compose exec postgres psql -U "$POSTGRES_USER" -d "$POSTGRES_DB"

\d items
SELECT count(*) FROM items;
SELECT username, count(*) FROM items GROUP BY username ORDER BY 2 DESC;
SELECT pg_size_pretty(pg_total_relation_size('items'));
```

Each row carries two 4096-d float4 vectors: ~32 KB before overhead, so roughly
**33 MB per 1,000 signatures**.

### Backup

```bash
docker compose exec -T postgres pg_dump -U "$POSTGRES_USER" "$POSTGRES_DB" \
  | gzip > sig-$(date +%F).sql.gz
```

Vectors dump as text, so the file is several times the table size. Restore into
an already-initialised database.

### Reset

```bash
docker compose exec postgres psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" \
  -c "TRUNCATE items RESTART IDENTITY;"
```

Required after any change to preprocessing or to the extractors: embeddings are
only comparable with others produced by the same pipeline.

### Re-running init.sql

The entrypoint runs `/docker-entrypoint-initdb.d/` **only when `PGDATA` is
empty**. Editing `init.sql` on an existing deployment does nothing. To apply it:

```bash
docker compose down -v      # destroys the volume and all enrolments
docker compose up -d
```

Or apply the DDL by hand.

### No migration tool

The schema is one table and is owned by `init.sql`. An Alembic setup used to
live under `trainingfiles/` — `alembic.ini`, an `env.py`, and a `db_utils.py`
modelling a `signature_records` table that contradicted this one. Nothing
imported it and `script_location = migrations` pointed at a directory that did
not exist, so no migration could ever run. It was removed rather than repaired.

If the schema grows enough to need migrations, they belong in `inference/`,
next to the service that owns it.
