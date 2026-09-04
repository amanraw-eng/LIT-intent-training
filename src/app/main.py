"""
OpenAI-Compatible Whisper Intent Classification API.

Supports models with arbitrary numbers of intents/classes, e.g.
17-intent and 20-intent models.

Expected Hugging Face repository files:
    config.json
    model.bin

Expected config.json structure:
{
    "intent_to_idx": {"INTENT_A": 0, "INTENT_B": 1, ...},
    "idx_to_intent": {"0": "INTENT_A", "1": "INTENT_B", ...}
}

The API endpoint is:
    POST /intent/v1/audio/transcriptions

Run directly (`python -m app.main`) or via uvicorn: `uvicorn app.main:app`.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI

from .api.v1.routes import router as v1_router
from .core.config import settings
from .core.logging import logger
from .services.intent_service import service


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting FastAPI application")

    try:
        service.load()
    except Exception:
        logger.exception("=" * 60)
        logger.exception("MODEL STARTUP FAILED")
        logger.exception("The API will start but the model will remain unavailable.")
        logger.exception("=" * 60)
        # We do NOT silently hide startup errors - but we let FastAPI start
        # so /health can return useful diagnostics instead of killing the server.
        service.model = None
        service.model_loaded = False

    yield

    logger.info("Shutting down model service")
    service.model = None
    service.model_loaded = False

    if service.device == "cuda":
        try:
            torch.cuda.empty_cache()
        except Exception:
            logger.exception("Failed to empty CUDA cache")


app = FastAPI(
    title="OpenAI-Compatible Whisper Intent Classification API",
    version="1.0.0",
    lifespan=lifespan,
)
app.include_router(v1_router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host=settings.HOST, port=settings.PORT, workers=1)
