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

## Vector indexing

pgvector's HNSW dimension cap is **per type**, not one global 2000. From
`src/hnswutils.c`, `.maxDimensions` is `HNSW_MAX_DIM` for `vector`,
`HNSW_MAX_DIM * 2` for `halfvec`, and `HNSW_MAX_DIM * 32` for `bit`:

| type | HNSW cap | usable at 4096-d? |
|---|---|---|
| `vector` | 2000 | ✗ |
| `halfvec` | **4000** | ✗ — misses by 96 dimensions |
| `bit` | **64000** | **✓** |
| `sparsevec` | 1e9 non-zero | (not applicable — these are dense) |

Verified on `pgvector/pgvector:pg17` (pgvector 0.8.6); `tests/test_migrations.py`
pins all three outcomes.

> An earlier version of this document said the 2000 limit applied to `halfvec`
> as well, and concluded that 4096-d embeddings could not be indexed at all.
> That was wrong on both counts. `halfvec`'s cap is 4000, and `bit` — reachable
> via `binary_quantize()` — is 64000, which is what makes an index possible.

### What is built

Migration `0002` creates, for each embedding column:

```sql
CREATE INDEX idx_items_ann_resnet50_vector ON items
    USING hnsw ((binary_quantize(resnet50_vector)::bit(4096)) bit_hamming_ops);
```

`binary_quantize` keeps one bit per dimension (`x > 0`). Hamming distance over
that bitmap is a coarse filter; the application re-ranks the candidates exactly
with `<=>` on the full vectors, so **the number compared against
`MATCH_THRESHOLD` is never approximate** — only the candidate set is.

### Measured

4,800 synthetic 4096-d ReLU embeddings, 100 probes, in-database:

| strategy | ms/query | recall@1 | index size |
|---|---|---|---|
| exact sequential scan | 18.5 | baseline | — |
| `binary_quantize` + rerank(100) | **0.5** | 97.5–100% | 3.9 MB |
| `subvector(1,2000)` + rerank(100) | 11.3 | 100% | 75 MB |

(Table: 158 MB. The end-to-end figure through SQLAlchemy is higher than the
in-database one — the 81 KB vector literal costs parse time on every call.)

Recall was 100% when ReLU left ~71% of units active and 97.5% at ~54%, so it
degrades as activations sparsify — `binary_quantize` throws away magnitude and
keeps only which units fired.

### Why it is off by default

`ANN_CANDIDATES=0`. Those recall numbers are from **synthetic** data with
well-separated identity centroids; real signature embeddings are harder. In a
verification system a missed true neighbour is a **false rejection** — the
system tells a legitimate signer their signature does not match — and 18.5 ms
is already negligible beside the four GPU inferences the same request pays for.

Turn it on once you have measured recall on your own enrolled set:

```sql
-- for each enrolled signature, does the ANN path find what exact search finds?
WITH probes AS (SELECT id, resnet50_vector AS v FROM items)
SELECT round(100.0 * avg((ann.best = exact.best)::int), 2) AS "recall@1 %"
FROM probes p
CROSS JOIN LATERAL (
    SELECT id AS best FROM items ORDER BY resnet50_vector <=> p.v LIMIT 1
) exact
CROSS JOIN LATERAL (
    SELECT c.id AS best FROM (
        SELECT id, resnet50_vector FROM items
        ORDER BY binary_quantize(resnet50_vector)::bit(4096)
                 <~> binary_quantize(p.v)::bit(4096)
        LIMIT 100
    ) c ORDER BY c.resnet50_vector <=> p.v LIMIT 1
) ann;
```

Then set `ANN_CANDIDATES=100` in `.env` and restart the api. Raising the
candidate count trades latency back for recall.

### If you would rather not approximate at all

`subvector(embedding, 1, 2000)::vector(2000)` indexes the first 2000 dimensions
exactly and gave 100% recall in the same benchmark, at 1.6× rather than 37×.
It is a one-line change to migration `0002` if you prefer that trade.

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
