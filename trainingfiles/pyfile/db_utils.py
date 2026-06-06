"""
db_utils.py
─────────────────────────────────────────────────────────────────────────────
PostgreSQL + pgvector utility layer.

All database interaction goes here — keep application code clean.

Prerequisites
─────────────
    # Postgres extension (run once as superuser):
    CREATE EXTENSION IF NOT EXISTS vector;

    # Python packages:
    pip install sqlalchemy psycopg2-binary pgvector alembic python-dotenv

Environment variables (put in .env or export directly):
    DB_HOST      = localhost
    DB_PORT      = 5432
    DB_NAME      = signature_db
    DB_USER      = postgres
    DB_PASSWORD  = yourpassword
    VECTOR_DIM   = 4096          # 4096 for VGG16-FC1 | 2048 for ResNet50 / SigNet
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone
from typing import List, Optional

import numpy as np
from dotenv import load_dotenv
from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Column,
    DateTime,
    String,
    create_engine,
    text,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# Config helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_env(key: str, default: str | None = None) -> str:
    val = os.getenv(key, default)
    if val is None:
        raise EnvironmentError(f"Required environment variable '{key}' is not set.")
    return val


def build_db_url() -> str:
    host     = _get_env("DB_HOST")
    port     = _get_env("DB_PORT", "5432")
    name     = _get_env("DB_NAME")
    user     = _get_env("DB_USER")
    password = _get_env("DB_PASSWORD")
    return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{name}"


VECTOR_DIM: int = int(os.getenv("VECTOR_DIM", "4096"))


# ─────────────────────────────────────────────────────────────────────────────
# ORM model
# ─────────────────────────────────────────────────────────────────────────────

class Base(DeclarativeBase):
    pass


class SignatureRecord(Base):
    """
    One row = one enrolled user with their representative signature vector.

    The vector column uses pgvector's <=> (cosine) or <-> (L2) operators
    for ANN search directly in SQL.
    """
    __tablename__ = "signature_records"

    user_id: uuid.UUID = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        nullable=False,
        comment="Immutable surrogate key",
    )
    username: str = Column(
        String(128),
        nullable=False,
        index=True,
        comment="Login name / employee ID",
    )
    company_name: str = Column(
        String(256),
        nullable=True,
        comment="Employer or organisation",
    )
    position: str = Column(
        String(128),
        nullable=True,
        comment="Job title",
    )
    created_date: datetime = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    modified_date: datetime = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )
    signature_vector = Column(
        Vector(VECTOR_DIM),
        nullable=False,
        comment=f"L2-normalised embedding ({VECTOR_DIM}-d)",
    )

    def __repr__(self) -> str:
        return (
            f"<SignatureRecord user_id={self.user_id} "
            f"username={self.username!r} "
            f"company={self.company_name!r}>"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Engine / session factory
# ─────────────────────────────────────────────────────────────────────────────

def make_engine(echo: bool = False):
    """Return a SQLAlchemy Engine.  Call once at application startup."""
    return create_engine(build_db_url(), echo=echo, pool_pre_ping=True)


def make_session_factory(engine=None):
    """Return a bound sessionmaker."""
    if engine is None:
        engine = make_engine()
    return sessionmaker(bind=engine, autoflush=False, autocommit=False)


# ─────────────────────────────────────────────────────────────────────────────
# CRUD helpers
# ─────────────────────────────────────────────────────────────────────────────

def _normalise(vec: np.ndarray) -> List[float]:
    """L2-normalise and convert to plain Python list for pgvector."""
    arr = np.asarray(vec, dtype=np.float32).flatten()
    norm = np.linalg.norm(arr)
    if norm > 0:
        arr = arr / norm
    return arr.tolist()


def enroll_user(
    session: Session,
    username: str,
    signature_vector: np.ndarray,
    company_name: str | None = None,
    position: str | None = None,
) -> SignatureRecord:
    """
    Insert a new user with their signature embedding.

    The vector is L2-normalised before storage so cosine similarity
    queries become simple inner-product queries (<#> or dot-product).
    """
    record = SignatureRecord(
        user_id=uuid.uuid4(),
        username=username,
        company_name=company_name,
        position=position,
        signature_vector=_normalise(signature_vector),
    )
    session.add(record)
    session.commit()
    session.refresh(record)
    return record


def update_vector(
    session: Session,
    user_id: uuid.UUID,
    new_vector: np.ndarray,
) -> Optional[SignatureRecord]:
    """Replace the stored embedding (e.g. after re-enrolment)."""
    record = session.get(SignatureRecord, user_id)
    if record is None:
        return None
    record.signature_vector = _normalise(new_vector)
    record.modified_date = datetime.now(timezone.utc)
    session.commit()
    session.refresh(record)
    return record


def delete_user(session: Session, user_id: uuid.UUID) -> bool:
    """Remove a user record.  Returns True if deleted, False if not found."""
    record = session.get(SignatureRecord, user_id)
    if record is None:
        return False
    session.delete(record)
    session.commit()
    return True


def get_user(session: Session, user_id: uuid.UUID) -> Optional[SignatureRecord]:
    return session.get(SignatureRecord, user_id)


def get_user_by_username(session: Session, username: str) -> Optional[SignatureRecord]:
    return (
        session.query(SignatureRecord)
        .filter(SignatureRecord.username == username)
        .first()
    )


# ─────────────────────────────────────────────────────────────────────────────
# Vector search (cosine similarity via pgvector <=> operator)
# ─────────────────────────────────────────────────────────────────────────────

def search_top_k(
    session: Session,
    query_vector: np.ndarray,
    top_k: int = 5,
) -> List[dict]:
    """
    Return the `top_k` most similar users to `query_vector`.

    Uses the cosine-distance operator `<=>` which requires an IVFFlat or
    HNSW index on the vector column for fast retrieval at scale.
    The index is created by Alembic migration (see migrations/).

    Returns
    -------
    List of dicts:
        {user_id, username, company_name, position, cosine_similarity}
    sorted by similarity descending.
    """
    norm_vec = _normalise(query_vector)

    # pgvector cosine distance: 1 - cosine_similarity
    rows = session.execute(
        text(
            """
            SELECT
                user_id,
                username,
                company_name,
                position,
                1 - (signature_vector <=> CAST(:vec AS vector)) AS cosine_similarity
            FROM signature_records
            ORDER BY signature_vector <=> CAST(:vec AS vector)
            LIMIT :k
            """
        ),
        {"vec": str(norm_vec), "k": top_k},
    ).fetchall()

    return [
        {
            "user_id":          str(r.user_id),
            "username":         r.username,
            "company_name":     r.company_name,
            "position":         r.position,
            "cosine_similarity": float(r.cosine_similarity),
        }
        for r in rows
    ]


def verify_signature(
    session: Session,
    query_vector: np.ndarray,
    threshold: float = 0.90,
) -> Optional[dict]:
    """
    Verify a query signature against all enrolled users.

    Returns the best-matching user dict (with cosine_similarity) if the
    top-1 similarity exceeds `threshold`, else None.

    Tune `threshold` on your held-out set; 0.90 is a conservative start.
    """
    results = search_top_k(session, query_vector, top_k=1)
    if results and results[0]["cosine_similarity"] >= threshold:
        return results[0]
    return None
