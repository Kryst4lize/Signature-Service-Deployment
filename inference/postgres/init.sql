-- ── Extensions ────────────────────────────────────────────────────────────────
CREATE EXTENSION IF NOT EXISTS vector;

-- ── Signature store ─────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS items (
    id                 SERIAL PRIMARY KEY,
    username           VARCHAR(50)  NOT NULL,
    user_created_date  TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP,
    user_modified_date TIMESTAMP,
    resnet50_vector    VECTOR(4096),
    vgg16_vector       VECTOR(4096)
);

-- ── HNSW indexes for fast L2 similarity search ─────────────────────────────────
CREATE INDEX IF NOT EXISTS idx_resnet50_hnsw
    ON items USING hnsw (resnet50_vector vector_l2_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS idx_vgg16_hnsw
    ON items USING hnsw (vgg16_vector vector_l2_ops)
    WITH (m = 16, ef_construction = 64);

