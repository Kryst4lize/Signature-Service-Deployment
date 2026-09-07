"""initial schema: pgvector extension, items table, username index

This reproduces exactly what postgres/init.sql used to create, so an existing
deployment can adopt alembic without a rebuild:

    docker compose run --rm migrate alembic stamp 0001

marks the database as already at this revision without re-running it. A fresh
deployment just runs `alembic upgrade head` and gets the same schema.

Revision ID: 0001
Revises:
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Must agree with Vector(n) in app/db.py and `dims: [ n ]` in each extractor's
# config.pbtxt. All three move together.
EMBEDDING_DIM = 4096


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "items",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("username", sa.String(length=50), nullable=False),
        sa.Column(
            "user_created_date",
            sa.TIMESTAMP(),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.Column("user_modified_date", sa.TIMESTAMP(), nullable=True),
        sa.Column("resnet50_vector", Vector(EMBEDDING_DIM), nullable=True),
        sa.Column("vgg16_vector", Vector(EMBEDDING_DIM), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_items_username", "items", ["username"])


def downgrade() -> None:
    op.drop_index("idx_items_username", table_name="items")
    op.drop_table("items")
    # The extension is deliberately left in place: other schemas in the same
    # database may depend on it, and dropping it would cascade.
