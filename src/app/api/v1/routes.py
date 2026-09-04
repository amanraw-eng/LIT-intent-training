"""v1 routes: health check + the OpenAI-compatible audio transcription
(intent classification) endpoint.

Route paths and behavior are unchanged from the original
intent-training/api.py - only the module layout changed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from ...config import MAX_TOP_K, MAX_UPLOAD_BYTES, MODEL_TYPE, logger, settings
from ...services.intent_service import service
from .schemas import (
    HealthResponse,
    IntentScore,
    TranscriptionResponse,
    TranscriptionResponseFormat,
    TranscriptionResponseVerbose,
    TranscriptionSegment,
    TranscriptionUsageAudio,
)

router = APIRouter()


# =====================================================================
# Health
# =====================================================================

@router.get("/health", response_model=HealthResponse)
async def health():
    if service.model is None:
        raise HTTPException(
            status_code=503,
            detail={
                "status": "not_ready",
                "device": service.device,
                "model_type": MODEL_TYPE,
                "num_classes": service.num_classes,
                "model_output_classes": service.model_output_classes,
            },
        )

    return HealthResponse(
        status="ok",
        device=service.device,
        model_type=MODEL_TYPE,
        num_classes=service.num_classes,
    )


# =====================================================================
# Helpers for the transcription route
# =====================================================================

async def _read_upload(file: UploadFile) -> bytes:
    try:
        payload = await file.read(MAX_UPLOAD_BYTES + 1)
    except Exception as exc:
        logger.exception("Failed to read uploaded file")
        raise HTTPException(status_code=400, detail=f"Failed to read uploaded audio: {exc}") from exc
    finally:
        try:
            await file.close()
        except Exception:
            pass

    if not payload:
        raise HTTPException(status_code=422, detail="Audio upload is empty")

    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Audio upload exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit",
        )

    return payload


def _resolve_top_k(top_k: int | None) -> int:
    k_value = top_k if top_k is not None else MAX_TOP_K

    try:
        k_value = int(k_value)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid top_k value: {top_k}") from exc

    if k_value < 1:
        raise HTTPException(status_code=422, detail="top_k must be >= 1")

    return k_value


def _build_inference_error_detail(exc: Exception, file: UploadFile, payload: bytes, k_value: int) -> dict:
    logger.exception("=" * 56)
    logger.exception("INFERENCE REQUEST FAILED")
    logger.exception("File: %s", file.filename)
    logger.exception("File size: %d bytes", len(payload))
    logger.exception("Model classes from config: %d", service.num_classes)
    logger.exception("Model output classes: %s", service.model_output_classes)
    logger.exception("top_k: %d", k_value)
    logger.exception("Device: %s", service.device)
    logger.exception("Exception: %s", exc)
    logger.exception("=" * 56)

    if settings.DEBUG_ERRORS:
        return {
            "error": "Inference failed",
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "model_type": MODEL_TYPE,
            "num_classes": service.num_classes,
            "model_output_classes": service.model_output_classes,
            "top_k": k_value,
        }

    return {"error": "Inference failed", "message": str(exc)}


# =====================================================================
# OpenAI Audio Transcriptions Route
# =====================================================================

@router.post(
    "/intent/v1/audio/transcriptions",
    response_model=TranscriptionResponse | TranscriptionResponseVerbose | str,
)
async def create_transcription(
    file: Annotated[UploadFile, File(description="The audio file object to classify.")],
    model: Annotated[str | None, Form()] = None,
    language: Annotated[str | None, Form()] = None,
    prompt: Annotated[str | None, Form()] = None,
    response_format: Annotated[TranscriptionResponseFormat, Form()] = "json",
    temperature: Annotated[float | None, Form()] = 0.0,
    top_k: Annotated[int | None, Form()] = None,
):
    """OpenAI-compatible audio endpoint. The actual output is the predicted intent."""

    if service.model is None:
        logger.error("Inference request received while model is not ready")
        raise HTTPException(
            status_code=503,
            detail={
                "error": "Model is not ready",
                "num_classes": service.num_classes,
                "model_type": MODEL_TYPE,
            },
        )

    payload = await _read_upload(file)
    suffix = Path(file.filename or "upload.wav").suffix or ".wav"
    k_value = _resolve_top_k(top_k)

    try:
        async with service.inference_lock:
            ranked_intents, duration = await run_in_threadpool(
                service.predict_file, payload, suffix, k_value
            )

    except ValueError as exc:
        logger.warning("Validation error during inference: %s", exc)
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    except Exception as exc:
        detail = _build_inference_error_detail(exc, file, payload, k_value)
        raise HTTPException(status_code=500, detail=detail) from exc

    if not ranked_intents:
        logger.error("Model returned zero intent predictions")
        raise HTTPException(status_code=500, detail="Model returned no intent predictions")

    top_intent = ranked_intents[0]["intent"]
    top_confidence = ranked_intents[0]["confidence"]

    logger.info("REQUEST SUCCESS | intent=%s | confidence=%.6f", top_intent, top_confidence)

    # -- Plain text --------------------------------------------------
    if response_format == "text":
        return top_intent

    # -- Verbose JSON --------------------------------------------------
    if response_format == "verbose_json":
        return TranscriptionResponseVerbose(
            task="transcribe",
            language=language or "en",
            duration=round(duration, 3),
            text=top_intent,
            segments=[
                TranscriptionSegment(id=0, start=0.0, end=round(duration, 3), text=top_intent)
            ],
            top_intent=top_intent,
            confidence=top_confidence,
            intents=[IntentScore(**item) for item in ranked_intents],
            device=service.device,
        )

    # -- Standard JSON (includes top_intent + confidence) --------
    return TranscriptionResponse(
        text=top_intent,
        top_intent=top_intent,
        confidence=top_confidence,
        usage=TranscriptionUsageAudio(type="duration", seconds=round(duration, 3)),
    )
