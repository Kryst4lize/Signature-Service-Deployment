-- ── Extensions ────────────────────────────────────────────────────────────────
CREATE EXTENSION IF NOT EXISTS vector;

-- ── Model registry ─────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS models (
    id                SERIAL PRIMARY KEY,
    name              VARCHAR(100) NOT NULL,
    version           VARCHAR(20)  NOT NULL DEFAULT '1',
    type              VARCHAR(50)  NOT NULL,   -- detector | denoiser | extractor | comparator
    triton_model_name VARCHAR(100) NOT NULL,
    model_path        TEXT        NOT NULL,
    is_active         BOOLEAN     NOT NULL DEFAULT TRUE,
    created_at        TIMESTAMP   NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (triton_model_name, version)
);

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

-- ── Seed model registry ─────────────────────────────────────────────────────────
INSERT INTO models (name, version, type, triton_model_name, model_path) VALUES
    ('YOLOv8',   '1', 'detector',  'yolov8',   '/models/yolov8'),
    ('CycleGAN', '1', 'denoiser',  'cyclegan',  '/models/cyclegan'),
    ('ResNet50', '1', 'extractor', 'resnet50',  '/models/resnet50'),
    ('VGG16',    '1', 'extractor', 'vgg16',     '/models/vgg16')
ON CONFLICT DO NOTHING;
