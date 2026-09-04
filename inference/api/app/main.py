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
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    triton = get_triton_service()
    await triton.connect()
    logger.info("Triton client connected to %s", settings.triton_http_url)
    yield
    await triton.close()
    logger.info("Triton client closed")


app = FastAPI(
    title="Signature Verification API",
    version="2.0.0",
    lifespan=lifespan,
)

if settings.cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

app.include_router(sig_router, tags=["signatures"])


@app.get("/health", tags=["ops"])
async def health():
    return {"status": "ok", "env": settings.app_env}
