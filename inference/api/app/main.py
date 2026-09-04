import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.routes import router as sig_router
from app.triton import get_triton_service

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
@asynccontextmanager
async def lifespan(app: FastAPI):
    triton = get_triton_service()
    await triton.connect()
    print(f"[startup] Triton connected @ {settings.triton_http_url}")
    yield
    await triton.close()
    print("[shutdown] Triton client closed")


app = FastAPI(
    title="Signature Verification API",
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS: allow the frontend origin to call the API ───────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://10.207.178.239:8110",
        "http://10.207.178.239:3000",   # frontend
        "http://localhost:3000",         # local dev
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(sig_router, tags=["signatures"])


@app.get("/health", tags=["ops"])
async def health():
    return {"status": "ok", "env": settings.app_env}
