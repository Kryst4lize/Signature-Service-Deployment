"""Migrations, and why the vector index is shaped the way it is.

These run against the real database the Makefile provisions, because every
claim here is about what pgvector will and will not accept — none of it has a
meaningful in-memory equivalent.

asyncpg is used directly rather than through SQLAlchemy: this is DDL and raw
introspection, and asyncpg is the driver the service actually ships.
"""

import asyncio
import os

import numpy as np
import pytest

DB_URL = os.environ.get("TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not DB_URL, reason="TEST_DATABASE_URL is not set")

EMBEDDING_DIM = 4096


def _dsn() -> str:
    return (DB_URL or "").replace("+asyncpg", "")


def run(coro):
    return asyncio.run(coro)


async def _fetch(sql: str, *args):
    import asyncpg

    conn = await asyncpg.connect(_dsn())
    try:
        return await conn.fetch(sql, *args)
    finally:
        await conn.close()


async def _exec(sql: str, *args):
    import asyncpg

    conn = await asyncpg.connect(_dsn())
    try:
        return await conn.execute(sql, *args)
    finally:
        await conn.close()


# ── the schema alembic produced ───────────────────────────────────────────────


def test_alembic_stamped_the_database():
    rows = run(_fetch("SELECT version_num FROM alembic_version"))
    assert [r["version_num"] for r in rows] == ["0002"]


def test_items_table_matches_the_orm():
    rows = run(
        _fetch(
            "SELECT column_name, data_type FROM information_schema.columns "
            "WHERE table_name = 'items'"
        )
    )
    cols = {r["column_name"]: r["data_type"] for r in rows}
    assert cols["id"] == "integer"
    assert cols["username"] == "character varying"
    # pgvector's `vector` reports as USER-DEFINED.
    assert cols["resnet50_vector"] == "USER-DEFINED"
    assert cols["vgg16_vector"] == "USER-DEFINED"


def test_username_index_exists():
    rows = run(_fetch("SELECT indexname FROM pg_indexes WHERE tablename = 'items'"))
    assert "idx_items_username" in {r["indexname"] for r in rows}


def test_ann_indexes_exist_and_are_hnsw_over_bit():
    rows = run(
        _fetch(
            "SELECT indexname, indexdef FROM pg_indexes "
            "WHERE tablename = 'items' AND indexname LIKE 'idx_items_ann_%'"
        )
    )
    defs = {r["indexname"]: r["indexdef"] for r in rows}
    assert set(defs) == {"idx_items_ann_resnet50_vector", "idx_items_ann_vgg16_vector"}
    for definition in defs.values():
        assert "hnsw" in definition
        assert "binary_quantize" in definition
        assert "bit_hamming_ops" in definition


# ── why the index has to be shaped that way ───────────────────────────────────
#
# pgvector's HNSW dimension cap is PER TYPE (src/hnswutils.c sets
# .maxDimensions to HNSW_MAX_DIM for vector, *2 for halfvec, *32 for bit).
# These three tests pin all three outcomes at 4096 dimensions.


async def _index_attempt(ddl: str) -> str | None:
    """Try an index definition on a throwaway table; return the error text."""
    import asyncpg

    conn = await asyncpg.connect(_dsn())
    try:
        await conn.execute("CREATE TABLE IF NOT EXISTS _cap_probe (v vector(4096))")
        try:
            await conn.execute(ddl)
            return None
        except asyncpg.PostgresError as exc:
            return str(exc)
    finally:
        await conn.execute("DROP TABLE IF EXISTS _cap_probe")
        await conn.close()


def test_hnsw_refuses_a_4096_dim_vector_column():
    """Why there is no plain `USING hnsw (col vector_cosine_ops)`.

    This is the exact error the old init.sql hit, which aborted container
    initialisation and left a database with no tables at all.
    """
    err = run(_index_attempt("CREATE INDEX ON _cap_probe USING hnsw (v vector_cosine_ops)"))
    assert err is not None and "more than 2000 dimensions" in err


def test_halfvec_cap_is_4000_not_2000():
    """4096 misses halfvec's cap by 96 dimensions.

    Pinned because the earlier documentation claimed the 2000 limit applied to
    halfvec too. It does not — the limit is 4000 — and getting that wrong sends
    someone down a dead end for the wrong reason.
    """
    err = run(
        _index_attempt(
            "CREATE INDEX ON _cap_probe USING hnsw ((v::halfvec(4096)) halfvec_cosine_ops)"
        )
    )
    assert err is not None and "more than 4000 dimensions" in err


def test_binary_quantize_is_indexable_at_4096():
    """What makes the ANN path possible at all: `bit`'s cap is 64000."""
    err = run(
        _index_attempt(
            "CREATE INDEX ON _cap_probe "
            "USING hnsw ((binary_quantize(v)::bit(4096)) bit_hamming_ops)"
        )
    )
    assert err is None, err


# ── the ANN query returns what the exact query returns ────────────────────────


def _vec(v: np.ndarray) -> str:
    norm = float(np.linalg.norm(v))
    v = v / norm if norm else v
    return "[" + ",".join(f"{x:.6f}" for x in v) + "]"


def _relu_embedding(rng: np.random.Generator) -> np.ndarray:
    """Shaped like the real thing: non-negative, half the units dead."""
    v = np.abs(rng.normal(size=EMBEDDING_DIM)).astype(np.float32)
    v[rng.choice(EMBEDDING_DIM, EMBEDDING_DIM // 2, replace=False)] = 0.0
    return v


async def _ann_vs_exact():
    import asyncpg

    rng = np.random.default_rng(0)
    conn = await asyncpg.connect(_dsn())
    try:
        await conn.execute("DELETE FROM items WHERE username LIKE 'anntest_%'")

        target = _relu_embedding(rng)
        await conn.execute(
            "INSERT INTO items (username, resnet50_vector, vgg16_vector) "
            "VALUES ($1, $2::vector, $2::vector)",
            "anntest_target",
            _vec(target),
        )
        for i in range(40):
            await conn.execute(
                "INSERT INTO items (username, resnet50_vector, vgg16_vector) "
                "VALUES ($1, $2::vector, $2::vector)",
                f"anntest_{i}",
                _vec(_relu_embedding(rng)),
            )

        probe = _vec(target + rng.normal(scale=0.01, size=EMBEDDING_DIM).astype(np.float32))

        exact = await conn.fetchrow(
            "SELECT username, (resnet50_vector <=> $1::vector) AS d FROM items "
            "WHERE username LIKE 'anntest_%' ORDER BY d LIMIT 1",
            probe,
        )
        ann = await conn.fetchrow(
            "SELECT username, (resnet50_vector <=> $1::vector) AS d FROM ("
            "  SELECT username, resnet50_vector FROM items"
            "  WHERE username LIKE 'anntest_%'"
            "  ORDER BY binary_quantize(resnet50_vector)::bit(4096)"
            "           <~> binary_quantize($1::vector)::bit(4096)"
            "  LIMIT 20"
            ") c ORDER BY d LIMIT 1",
            probe,
        )
        return dict(exact), dict(ann)
    finally:
        await conn.execute("DELETE FROM items WHERE username LIKE 'anntest_%'")
        await conn.close()


def test_ann_rerank_agrees_with_exact_search():
    """The candidate set is approximate; the score that reaches the threshold
    is not. Both paths must pick the same row and report the same distance."""
    exact, ann = run(_ann_vs_exact())
    assert exact["username"] == "anntest_target"
    assert ann["username"] == exact["username"]
    assert ann["d"] == pytest.approx(exact["d"], abs=1e-9)
