"""HNSW indexes over binary-quantized embeddings

Why this shape, and not `USING hnsw (resnet50_vector vector_l2_ops)`:

pgvector's HNSW dimension cap is PER TYPE, not one global 2000. From
src/hnswutils.c, `.maxDimensions` is set to `HNSW_MAX_DIM` (2000) for `vector`,
`HNSW_MAX_DIM * 2` (4000) for `halfvec`, and `HNSW_MAX_DIM * 32` (64000) for
`bit`. The columns here are 4096-d, so:

    vector(4096)   -> refused, "more than 2000 dimensions"
    halfvec(4096)  -> refused, "more than 4000 dimensions"   (close, but no)
    bit(4096)      -> fine, the cap is 64000

`binary_quantize(v)::bit(4096)` keeps one bit per dimension (`x > 0`). Hamming
distance over that is a coarse filter; the application then re-ranks the
candidates exactly with `<=>` on the full vectors, so the score it finally
compares against MATCH_THRESHOLD is never approximate — only the candidate set
is.

Measured on 4,800 synthetic 4096-d ReLU embeddings (pgvector 0.8.6, in-database,
100 probes), against exact sequential scan:

    exact seq scan                18.5 ms   baseline
    binary_quantize + rerank(100)  0.5 ms   37x faster, recall@1 97.5-100%
    subvector(1,2000) + rerank     11.3 ms  1.6x faster, recall@1 100%

Index size at that scale: 3.9 MB for binary_quantize vs 75 MB for subvector,
against a 158 MB table — which is why binary_quantize is the one built here.

Recall was 100% when ReLU left ~71% of units active and 97.5% when it left
~54%, so it degrades as the activations sparsify. That is on SYNTHETIC data
with well-separated identities; real signature embeddings are harder. The
application therefore defaults to exact search (ANN_CANDIDATES=0) and this
index is unused until someone measures recall on their own enrolled set and
turns it on. See documentation/04-database.md.

Revision ID: 0002
Revises: 0001
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

EMBEDDING_DIM = 4096
COLUMNS = ("resnet50_vector", "vgg16_vector")


def upgrade() -> None:
    for column in COLUMNS:
        op.execute(
            f"CREATE INDEX idx_items_ann_{column} ON items "
            f"USING hnsw ((binary_quantize({column})::bit({EMBEDDING_DIM})) "
            f"bit_hamming_ops)"
        )


def downgrade() -> None:
    for column in COLUMNS:
        op.execute(f"DROP INDEX IF EXISTS idx_items_ann_{column}")
