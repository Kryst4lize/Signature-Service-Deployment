"""Endpoint tests.

Triton is faked (no GPU in CI); Postgres is real, because the parts most worth
testing here — the pgvector cast, the cosine ordering, the VARCHAR(50) limit —
have no meaningful in-memory equivalent.

Set TEST_DATABASE_URL to run these, e.g.

    docker run -d --name sigtest_pg -p 55432:5432 \
        -e POSTGRES_PASSWORD=test -e POSTGRES_DB=sig -e POSTGRES_USER=sig \
        -v "$PWD/postgres/init.sql:/docker-entrypoint-initdb.d/init.sql:ro" \
        pgvector/pgvector:pg17
    TEST_DATABASE_URL=postgresql+asyncpg://sig:test@localhost:55432/sig pytest
"""

import io
import os

import numpy as np
import pytest
from PIL import Image

DB_URL = os.environ.get("TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not DB_URL, reason="TEST_DATABASE_URL is not set")


@pytest.fixture
def client(monkeypatch):
    from fastapi.testclient import TestClient
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    import app.db as db_mod
    from app.main import app
    from app.triton import TritonService, get_triton_service, l2_normalise

    engine = create_async_engine(DB_URL)
    session_factory = async_sessionmaker(bind=engine, expire_on_commit=False)
    monkeypatch.setattr(db_mod, "AsyncSessionLocal", session_factory)

    rng = np.random.default_rng(0)

    class FakeTriton(TritonService):
        """Deterministic stand-in: the embedding is a function of mean pixel
        value, so two similar images embed close together."""

        def __init__(self):
            self.detect_result = ([0.25, 0.25, 0.75, 0.75], 0.93)

        async def connect(self):
            return None

        async def close(self):
            return None

        async def detect_signature(self, page):
            return self.detect_result

        async def denoise(self, crop):
            return np.clip(crop, 0.0, 1.0)

        async def extract_features(self, clean):
            seed = int(abs(float(clean.mean())) * 1e6) % (2**31)
            local = np.random.default_rng(seed)
            base = local.random(4096).astype(np.float32)
            return l2_normalise(base), l2_normalise(base + 0.01 * rng.random(4096))

    fake = FakeTriton()
    app.dependency_overrides[get_triton_service] = lambda: fake

    with TestClient(app) as c:
        c.fake_triton = fake
        yield c

    app.dependency_overrides.clear()


def _png(color=(10, 10, 10), size=(320, 160)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


# ── health ────────────────────────────────────────────────────────────────────


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


# ── registration ──────────────────────────────────────────────────────────────


def test_register_then_list_and_delete(client):
    r = client.post(
        "/register-signature",
        data={"username": "alice"},
        files={"file": ("sig.png", _png(), "image/png")},
    )
    assert r.status_code == 201, r.text
    created = r.json()
    assert created["username"] == "alice"
    assert isinstance(created["id"], int)

    listed = client.get("/signatures").json()
    assert any(i["id"] == created["id"] for i in listed["items"])
    assert listed["total"] >= 1

    assert client.delete(f"/signatures/{created['id']}").status_code == 204
    assert client.delete(f"/signatures/{created['id']}").status_code == 404


def test_register_rejects_username_over_the_column_width(client):
    """VARCHAR(50). Previously this reached db.flush() and surfaced as a bare
    500 after two GPU inferences had already been paid for."""
    r = client.post(
        "/register-signature",
        data={"username": "x" * 51},
        files={"file": ("sig.png", _png(), "image/png")},
    )
    assert r.status_code == 422


def test_register_rejects_empty_username(client):
    r = client.post(
        "/register-signature",
        data={"username": ""},
        files={"file": ("sig.png", _png(), "image/png")},
    )
    assert r.status_code == 422


def test_register_rejects_a_non_image(client):
    r = client.post(
        "/register-signature",
        data={"username": "bob"},
        files={"file": ("notes.txt", b"this is not an image", "text/plain")},
    )
    assert r.status_code == 400


def test_register_rejects_an_empty_upload(client):
    r = client.post(
        "/register-signature",
        data={"username": "bob"},
        files={"file": ("empty.png", b"", "image/png")},
    )
    assert r.status_code == 400


def test_upload_over_the_size_cap_is_rejected(client, monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "max_upload_bytes", 1024)
    r = client.post(
        "/register-signature",
        data={"username": "bob"},
        files={"file": ("big.png", _png(size=(2000, 2000)), "image/png")},
    )
    assert r.status_code == 413


# ── verification ──────────────────────────────────────────────────────────────


def test_verify_reports_no_signature_when_detection_returns_nothing(client):
    client.fake_triton.detect_result = None
    r = client.post("/verify-document", files={"file": ("doc.png", _png(), "image/png")})
    assert r.status_code == 200
    body = r.json()
    assert body["total_pages"] == 1
    assert body["results"][0]["status"] == "no_signature"


def test_verify_matches_the_signature_it_was_registered_from(client):
    img = _png(color=(33, 44, 55))
    reg = client.post(
        "/register-signature",
        data={"username": "carol"},
        files={"file": ("sig.png", img, "image/png")},
    )
    assert reg.status_code == 201

    # Detection covers the whole page, so the crop embeds like the enrolled image.
    client.fake_triton.detect_result = ([0.0, 0.0, 1.0, 1.0], 0.99)
    r = client.post("/verify-document", files={"file": ("doc.png", img, "image/png")})
    assert r.status_code == 200

    page = r.json()["results"][0]
    assert page["status"] == "matched"
    assert page["username"] == "carol"
    assert page["matched"] is True
    # Cosine distance, so bounded to [0, 2] - never the unbounded raw L2 the
    # 0.5 threshold used to be compared against.
    assert 0.0 <= page["avg_distance"] <= 2.0
    assert {"bbox", "confidence", "page_annotated", "crop_before", "crop_after"} <= page.keys()

    client.delete(f"/signatures/{reg.json()['id']}")


def test_verify_returns_distances_within_the_cosine_range(client):
    reg = client.post(
        "/register-signature",
        data={"username": "dave"},
        files={"file": ("sig.png", _png(color=(1, 2, 3)), "image/png")},
    )
    client.fake_triton.detect_result = ([0.1, 0.1, 0.9, 0.9], 0.8)
    r = client.post(
        "/verify-document",
        files={"file": ("doc.png", _png(color=(200, 200, 200)), "image/png")},
    )
    page = r.json()["results"][0]
    for key in ("avg_distance", "resnet_distance", "vgg_distance"):
        assert 0.0 <= page[key] <= 2.0

    client.delete(f"/signatures/{reg.json()['id']}")


def test_verify_crops_from_the_full_resolution_page(client):
    """The crop preview must carry more detail than a 640x640 page view could.

    A 2000px-wide page cropped to its middle half is 1000px in the original;
    cropping the 640 detector input instead would cap it at 320.
    """
    reg = client.post(
        "/register-signature",
        data={"username": "erin"},
        files={"file": ("sig.png", _png(), "image/png")},
    )
    client.fake_triton.detect_result = ([0.25, 0.25, 0.75, 0.75], 0.9)
    r = client.post(
        "/verify-document",
        files={"file": ("big.png", _png(size=(2000, 1400)), "image/png")},
    )
    page = r.json()["results"][0]
    x1, y1, x2, y2 = page["bbox"]
    assert (x2 - x1) == 1000  # half of 2000, not half of 640
    assert (y2 - y1) == 700

    client.delete(f"/signatures/{reg.json()['id']}")
