# 6 — Database Schema

> PostgreSQL + pgvector database schema for the Signature Verification System.

---

## Database Configuration

| Property | Default Value |
|----------|--------------|
| **Database name** | `signature_db` |
| **User** | `siguser` |
| **Password** | `sigpass` |
| **Port** | `5432` |
| **Extension** | pgvector |

---

## Extensions

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

---

## Tables

### `models` — Model Registry

Tracks which ML models are deployed on Triton.

```sql
CREATE TABLE IF NOT EXISTS models (
    id                SERIAL PRIMARY KEY,
    name              VARCHAR(100) NOT NULL,
    version           VARCHAR(20)  NOT NULL DEFAULT '1',
    type              VARCHAR(50)  NOT NULL,      -- detector | denoiser | extractor
    triton_model_name VARCHAR(100) NOT NULL,
    model_path        TEXT         NOT NULL,
    is_active         BOOLEAN      NOT NULL DEFAULT TRUE,
    created_at        TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (triton_model_name, version)
);
```

| Column | Type | Description |
|--------|------|-------------|
| `id` | `SERIAL` | Auto-incrementing primary key |
| `name` | `VARCHAR(100)` | Human-readable model name (e.g., "YOLOv8") |
| `version` | `VARCHAR(20)` | Model version (e.g., "1") |
| `type` | `VARCHAR(50)` | Role: `detector`, `denoiser`, or `extractor` |
| `triton_model_name` | `VARCHAR(100)` | Name used in Triton model repository |
| `model_path` | `TEXT` | Path to model files in Triton |
| `is_active` | `BOOLEAN` | Whether this model is currently in use |
| `created_at` | `TIMESTAMP` | Record creation timestamp |

#### Seed data

```sql
INSERT INTO models (name, version, type, triton_model_name, model_path) VALUES
    ('YOLOv8',   '1', 'detector',  'yolov8',   '/models/yolov8'),
    ('CycleGAN', '1', 'denoiser',  'cyclegan',  '/models/cyclegan'),
    ('ResNet50', '1', 'extractor', 'resnet50',  '/models/resnet50'),
    ('VGG16',    '1', 'extractor', 'vgg16',     '/models/vgg16')
ON CONFLICT DO NOTHING;
```

---

### `items` — Signature Store

Stores registered user signatures with their embedding vectors.

```sql
CREATE TABLE IF NOT EXISTS items (
    id                 SERIAL PRIMARY KEY,
    username           VARCHAR(50)  NOT NULL,
    user_created_date  TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP,
    user_modified_date TIMESTAMP,
    resnet50_vector    VECTOR(4096),
    vgg16_vector       VECTOR(4096)
);
```

| Column | Type | Description |
|--------|------|-------------|
| `id` | `SERIAL` | Auto-incrementing primary key |
| `username` | `VARCHAR(50)` | User identifier |
| `user_created_date` | `TIMESTAMP` | When signature was registered |
| `user_modified_date` | `TIMESTAMP` | Last modification (nullable) |
| `resnet50_vector` | `VECTOR(4096)` | ResNet50 feature embedding |
| `vgg16_vector` | `VECTOR(4096)` | VGG16 feature embedding |

---

## Indexes

### HNSW indexes for fast vector similarity search

```sql
CREATE INDEX IF NOT EXISTS idx_resnet50_hnsw
    ON items USING hnsw (resnet50_vector vector_l2_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS idx_vgg16_hnsw
    ON items USING hnsw (vgg16_vector vector_l2_ops)
    WITH (m = 16, ef_construction = 64);
```

| Parameter | Value | Description |
|-----------|-------|-------------|
| `m` | `16` | Max number of connections per node in HNSW graph |
| `ef_construction` | `64` | Size of dynamic candidate list during construction |
| `vector_l2_ops` | — | Use L2 (Euclidean) distance for search |

---

## Similarity Search Query

The verification endpoint uses this query to find the closest match:

```sql
SELECT id, username,
       (resnet50_vector <-> CAST(:rv AS vector)) AS d_resnet,
       (vgg16_vector    <-> CAST(:vv AS vector)) AS d_vgg,
       (
           (resnet50_vector <-> CAST(:rv AS vector)) +
           (vgg16_vector    <-> CAST(:vv AS vector))
       ) / 2 AS avg_distance
FROM items
ORDER BY avg_distance ASC
LIMIT 1
```

- `<->` is the pgvector **L2 distance** operator
- Results are sorted by the average of both distances
- The top-1 result is returned
- Match threshold: `avg_distance < 0.5`

---

## SQLAlchemy ORM Models

Located in `api/app/models/db.py`:

### `Model` class

```python
class Model(Base):
    __tablename__ = "models"
    id:                Mapped[int]
    name:              Mapped[str]         # VARCHAR(100)
    version:           Mapped[str]         # VARCHAR(20), default='1'
    type:              Mapped[str]         # VARCHAR(50)
    triton_model_name: Mapped[str]         # VARCHAR(100)
    model_path:        Mapped[str]         # TEXT
    is_active:         Mapped[bool]        # default=True
    created_at:        Mapped[datetime]    # server_default=func.now()
```

### `Item` class

```python
class Item(Base):
    __tablename__ = "items"
    id:                 Mapped[int]
    username:           Mapped[str]                  # VARCHAR(50)
    user_created_date:  Mapped[datetime]             # server_default=func.now()
    user_modified_date: Mapped[datetime | None]      # nullable
    resnet50_vector:    Mapped[list | None]           # Vector(4096)
    vgg16_vector:       Mapped[list | None]           # Vector(4096)
```

---

## Training Pipeline Database (db_utils.py)

The training-side code (`trainingfiles/pyfile/db_utils.py`) defines a different
ORM model for more detailed user records:

### `SignatureRecord` class

```python
class SignatureRecord(Base):
    __tablename__ = "signature_records"
    user_id:           uuid.UUID        # UUID primary key
    username:          str               # VARCHAR(128)
    company_name:      str               # VARCHAR(256), nullable
    position:          str               # VARCHAR(128), nullable
    created_date:      datetime          # with timezone
    modified_date:     datetime          # with timezone
    signature_vector:  Vector(4096)      # L2-normalised embedding
```

> This table uses **cosine distance** (`<=>`) instead of L2 for search.
> The `db_utils.py` module provides CRUD functions (`enroll_user`, `search_top_k`,
> `verify_signature`) and is intended for standalone use outside the service.

### Environment variables for db_utils

```env
DB_HOST=localhost
DB_PORT=5432
DB_NAME=signature_db
DB_USER=postgres
DB_PASSWORD=yourpassword
VECTOR_DIM=4096
```

---

## Alembic Migration

Migration config is in `trainingfiles/alembic.ini`:

```ini
[alembic]
script_location = migrations
sqlalchemy.url = postgresql+psycopg2://user:pass@localhost/signature_db
```

The DB URL is overridden from environment variables in `pyfile/env.py`.

### Running migrations

```bash
# Generate a new migration
alembic revision --autogenerate -m "description"

# Apply all migrations
alembic upgrade head

# Rollback one step
alembic downgrade -1
```

---

## Data Volume

The Docker Compose configuration uses a named volume for data persistence:

```yaml
volumes:
  postgres_data:
```

Mounted at: `/var/lib/postgresql/data`

> **To fully reset the database:** `docker compose down -v` (removes the volume)
