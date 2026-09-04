"""OpenAI-compatible request/response models for the v1 audio-transcription
(intent classification) endpoint. Moved as-is from intent-training/api.py."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

TranscriptionResponseFormat = Literal["json", "verbose_json", "text"]


class IntentScore(BaseModel):
    intent: str
    confidence: float = Field(ge=0.0, le=1.0)


class TranscriptionUsageAudio(BaseModel):
    type: Literal["duration"] = "duration"
    seconds: float


class TranscriptionSegment(BaseModel):
    id: int
    start: float
    end: float
    text: str
    avg_logprob: float = 0.0
    compression_ratio: float = 1.0
    no_speech_prob: float = 0.0
    temperature: float = 0.0
    tokens: list[int] = Field(default_factory=list)
    seek: int = 0


class TranscriptionResponse(BaseModel):
    """Standard (non-verbose) JSON response - includes intent + confidence."""
    text: str
    top_intent: str
    confidence: float = Field(ge=0.0, le=1.0)
    usage: TranscriptionUsageAudio | None = None


class TranscriptionResponseVerbose(BaseModel):
    task: str = "transcribe"
    language: str = "en"
    duration: float
    text: str
    segments: list[TranscriptionSegment] = Field(default_factory=list)
    top_intent: str
    confidence: float
    intents: list[IntentScore]
    device: str


class HealthResponse(BaseModel):
    status: str
    device: str
    model_type: str
    num_classes: int
