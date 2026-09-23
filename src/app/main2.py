"""
OpenAI-Compatible Whisper Intent Classification API - v2.

Standalone app for the 15-intent model (model2 architecture: deeper
classification head + masked-mean pooling over real audio frames only). Kept
as a fully separate app/process from main.py (v1, 17/20-intent) rather than
mounted into it, so testing v2 can never affect the v1 service already
running - separate process, separate model in GPU memory. Uses the same
settings.PORT as v1 (run one at a time, or override PORT via env/.env if you
need both up together).

Expected Hugging Face repository files (config.json's "model_module" must be
"model2" - see services/intent_service_v2.py._check_model_module):
    config.json
    model.bin

The API endpoint is:
    POST /intent/v2/audio/transcriptions

Run directly (`python -m app.main2`) or via uvicorn: `uvicorn app.main2:app`.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

import anyio
import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.v2.routes import router as v2_router
from .config import logger, settings
from .services.intent_service_v2 import service_v2


def _configure_decode_thread_pool():
    """Audio decode/preprocessing (CPU-bound: ffmpeg + mel spectrogram) runs
    in Starlette's default thread pool, one call per request - size it
    explicitly if configured instead of relying on anyio's default."""
    if settings.DECODE_THREAD_POOL_SIZE is None:
        return
    limiter = anyio.to_thread.current_default_thread_limiter()
    limiter.total_tokens = settings.DECODE_THREAD_POOL_SIZE
    logger.info("Decode thread pool size set to %d", settings.DECODE_THREAD_POOL_SIZE)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting FastAPI application (v2)")
    _configure_decode_thread_pool()

    try:
        service_v2.load()
    except Exception:
        logger.exception("=" * 60)
        logger.exception("MODEL STARTUP FAILED (v2)")
        logger.exception("The API will start but the model will remain unavailable.")
        logger.exception("=" * 60)
        service_v2.model = None
        service_v2.model_loaded = False

    yield

    logger.info("Shutting down model service (v2)")
    await service_v2.shutdown()
    service_v2.model = None
    service_v2.model_loaded = False

    if service_v2.device == "cuda":
        try:
            torch.cuda.empty_cache()
        except Exception:
            logger.exception("Failed to empty CUDA cache")


app = FastAPI(
    title="OpenAI-Compatible Whisper Intent Classification API (v2)",
    version="2.0.0",
    lifespan=lifespan,
)
# Open CORS so a browser-hosted test client (different origin) can call this
# API directly - fine for local/dev testing; tighten allow_origins before
# any real deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(v2_router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main2:app", host=settings.HOST, port=settings.PORT, workers=1)
