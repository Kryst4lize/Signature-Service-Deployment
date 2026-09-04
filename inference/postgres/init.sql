-- ── Extensions ────────────────────────────────────────────────────────────────
CREATE EXTENSION IF NOT EXISTS vector;

-- ── Signature store ───────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS items (
    id                 SERIAL PRIMARY KEY,
    username           VARCHAR(50)  NOT NULL,
    user_created_date  TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP,
    user_modified_date TIMESTAMP,
    resnet50_vector    VECTOR(4096),
    vgg16_vector       VECTOR(4096)
);

CREATE INDEX IF NOT EXISTS idx_items_username ON items (username);

-- ── On vector indexing ────────────────────────────────────────────────────────
--
-- There is deliberately NO ANN index on the two vector columns.
--
-- pgvector caps HNSW and IVFFlat at 2000 dimensions; these columns are 4096-d
-- (the fc1 tap of both extractors). The previous
--
--     CREATE INDEX ... USING hnsw (resnet50_vector vector_l2_ops)
--
-- did not build a slow index — it aborted with
--
--     ERROR: column cannot have more than 2000 dimensions for hnsw index
--
-- and because the Postgres entrypoint runs init scripts with ON_ERROR_STOP=1,
-- the whole container exited with code 3 on first boot. `restart: unless-stopped`
-- then restarted it onto a non-empty PGDATA, so initialisation was skipped
-- forever and the database silently had no tables at all.
--
-- Nearest-neighbour search is therefore an exact sequential scan. Measured at
-- ~80 ms for 5,000 registered signatures on the pinned pgvector/pgvector:pg17
-- image, which is well inside the latency budget of a request that already
-- spends far longer on four GPU inferences.
--
-- If enrolment grows past the point where that is acceptable, the options are,
-- in increasing order of effort:
--   1. `halfvec(4096)` — halves storage and speeds up the scan; still not
--      indexable (the 2000-d cap applies to halfvec HNSW as well).
--   2. Store an additional reduced vector (e.g. 512-d PCA or a random
--      projection fitted on the enrolled set) in an indexed column, use it to
--      retrieve a candidate set, then re-rank those candidates exactly on the
--      4096-d columns.
--   3. Re-export the extractors with a smaller embedding head.
