"""Inference service: loads the intent model and runs inference against
uploaded audio. Logic is unchanged from the original intent-training/api.py
IntentService - only the config/logging imports moved.
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from contextlib import nullcontext

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from whisper.audio import N_SAMPLES, load_audio, log_mel_spectrogram, pad_or_trim

from ..core.config import settings
from ..core.logging import logger
from .whisper_model import WhisperIntentClassification


class IntentService:
    """Loads the intent model and runs inference against uploaded audio."""

    def __init__(self):
        self.device = "cuda" if settings.DEVICE == "cuda" and torch.cuda.is_available() else "cpu"

        self.model = None
        self.idx_to_intent: dict[int, str] = {}
        self.intent_to_idx: dict[str, int] = {}
        self.num_classes: int = 0
        self.model_output_classes: int | None = None
        self.model_loaded: bool = False
        self.inference_lock = asyncio.Lock()

    # -------------------------------------------------------------------
    # Model Loading
    # -------------------------------------------------------------------

    def load(self):
        """
        Load config.json and model.bin from Hugging Face.

        Validates the class mapping so a 17-intent model can't silently
        be used with a 20-intent mapping or vice versa.
        """
        logger.info("=" * 80)
        logger.info("Starting intent model loading")
        logger.info("Device: %s", self.device)
        logger.info("MODEL_TYPE: %s", settings.MODEL_TYPE)
        logger.info("HF_MODEL_REPO: %s", settings.HF_MODEL_REPO)
        logger.info("=" * 80)

        hf_repo = settings.HF_MODEL_REPO
        if not hf_repo:
            raise RuntimeError("HF_MODEL_REPO is not configured")

        hf_token = settings.HF_TOKEN
        model_type = settings.MODEL_TYPE
        if not model_type:
            raise RuntimeError("MODEL_TYPE is not configured")

        config_path = self._download_hf_file(hf_repo, "config.json", hf_token)
        model_path = self._download_hf_file(hf_repo, "model.bin", hf_token)

        model_config = self._read_config(config_path)
        self._load_intent_mapping(model_config)
        self._log_intent_mapping()

        model = self._build_model(model_type)
        state_dict = self._load_checkpoint(model_path)
        self._load_state_dict(model, state_dict, model_type)

        self._finalize_model(model, hf_repo, model_type)

    @staticmethod
    def _download_hf_file(hf_repo: str, filename: str, hf_token: str | None) -> str:
        try:
            logger.info("Downloading %s from Hugging Face: %s", filename, hf_repo)
            path = hf_hub_download(repo_id=hf_repo, filename=filename, token=hf_token)
            logger.info("%s downloaded: %s", filename, path)
            return path
        except Exception as exc:
            logger.exception("Failed to download %s from Hugging Face", filename)
            raise RuntimeError(
                f"Failed to download {filename} from HF repo '{hf_repo}': {exc}"
            ) from exc

    @staticmethod
    def _read_config(config_path: str) -> dict:
        try:
            with open(config_path, "r", encoding="utf-8") as handle:
                model_config = json.load(handle)
        except Exception as exc:
            logger.exception("Failed to parse config.json")
            raise RuntimeError(f"Unable to parse config.json: {exc}") from exc

        if not isinstance(model_config, dict):
            raise RuntimeError("config.json must contain a JSON object")
        if "intent_to_idx" not in model_config:
            raise RuntimeError("config.json is missing 'intent_to_idx'")
        if "idx_to_intent" not in model_config:
            raise RuntimeError("config.json is missing 'idx_to_intent'")

        return model_config

    def _load_intent_mapping(self, model_config: dict):
        raw_intent_to_idx = model_config["intent_to_idx"]
        raw_idx_to_intent = model_config["idx_to_intent"]

        if not isinstance(raw_intent_to_idx, dict):
            raise RuntimeError("'intent_to_idx' must be a JSON object")
        if not isinstance(raw_idx_to_intent, dict):
            raise RuntimeError("'idx_to_intent' must be a JSON object")

        try:
            self.intent_to_idx = {str(k): int(v) for k, v in raw_intent_to_idx.items()}
        except Exception as exc:
            raise RuntimeError(
                f"Invalid 'intent_to_idx' in config.json. Could not convert class indices to integers: {exc}"
            ) from exc

        try:
            # JSON always stores object keys as strings, so convert explicitly.
            self.idx_to_intent = {int(k): str(v) for k, v in raw_idx_to_intent.items()}
        except Exception as exc:
            raise RuntimeError(
                f"Invalid 'idx_to_intent' in config.json. Could not convert indices to integers: {exc}"
            ) from exc

        self.num_classes = len(self.intent_to_idx)
        if self.num_classes <= 0:
            raise RuntimeError("No intents found in intent_to_idx")

        expected_indices = list(range(self.num_classes))
        intent_indices = sorted(self.intent_to_idx.values())
        reverse_indices = sorted(self.idx_to_intent.keys())

        if intent_indices != expected_indices:
            raise RuntimeError(
                f"intent_to_idx contains invalid/non-contiguous indices. "
                f"Expected {expected_indices}, got {intent_indices}"
            )
        if reverse_indices != expected_indices:
            raise RuntimeError(
                f"idx_to_intent contains invalid/non-contiguous indices. "
                f"Expected {expected_indices}, got {reverse_indices}"
            )

        for intent, index in self.intent_to_idx.items():
            reverse_intent = self.idx_to_intent.get(index)
            if reverse_intent != intent:
                raise RuntimeError(
                    f"Intent mapping mismatch: intent_to_idx['{intent}'] = {index}, "
                    f"but idx_to_intent['{index}'] = '{reverse_intent}'"
                )

    def _log_intent_mapping(self):
        logger.info("Loaded intent mapping with %d classes", self.num_classes)
        for index in range(self.num_classes):
            logger.info("  class[%d] -> %s", index, self.idx_to_intent[index])

    def _build_model(self, model_type: str):
        logger.info("Creating WhisperIntentClassification with n_class=%d", self.num_classes)
        try:
            return WhisperIntentClassification(model_type, n_class=self.num_classes)
        except Exception as exc:
            logger.exception("Failed to instantiate WhisperIntentClassification")
            raise RuntimeError(f"Failed to create WhisperIntentClassification: {exc}") from exc

    def _load_checkpoint(self, model_path: str) -> dict:
        logger.info("Loading checkpoint: %s", model_path)
        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        except Exception as exc:
            logger.exception("Failed to load model.bin")
            raise RuntimeError(f"Unable to load model checkpoint: {exc}") from exc

        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
            logger.info("Checkpoint contains 'state_dict'")
        else:
            state_dict = checkpoint
            logger.info("Checkpoint itself is being used as state_dict")

        if not isinstance(state_dict, dict):
            raise RuntimeError("Model checkpoint does not contain a valid state_dict")

        keys = list(state_dict.keys())
        has_model_prefix = any(str(k).startswith("model.") for k in keys)

        if has_model_prefix:
            logger.info("Detected 'model.' prefix in checkpoint keys; stripping it")
            state_dict = {
                k[len("model."):]: v for k, v in state_dict.items() if k.startswith("model.")
            }
        else:
            logger.info("No 'model.' prefix detected in checkpoint")

        return state_dict

    def _load_state_dict(self, model, state_dict: dict, model_type: str):
        try:
            incompatible = model.load_state_dict(state_dict, strict=True)
            logger.info("Model state_dict loaded successfully")

            if incompatible.missing_keys:
                logger.error("Missing keys: %s", incompatible.missing_keys)
            if incompatible.unexpected_keys:
                logger.error("Unexpected keys: %s", incompatible.unexpected_keys)

        except Exception as exc:
            logger.exception("FAILED to load model state_dict")
            raise RuntimeError(
                f"Model architecture/checkpoint mismatch. "
                f"MODEL_TYPE='{model_type}', n_class={self.num_classes}. Original error: {exc}"
            ) from exc

    def _finalize_model(self, model, hf_repo: str, model_type: str):
        try:
            model.to(self.device)
            model.eval()
        except Exception as exc:
            logger.exception("Failed to move model to device")
            raise RuntimeError(f"Failed to move model to {self.device}: {exc}") from exc

        if self.device == "cuda":
            torch.backends.cudnn.benchmark = True
            logger.info("CUDA enabled: %s", torch.cuda.get_device_name(0))

        self.model = model
        self.model_loaded = True

        logger.info("=" * 80)
        logger.info("MODEL LOADED SUCCESSFULLY")
        logger.info("Model repo: %s", hf_repo)
        logger.info("Model type: %s", model_type)
        logger.info("Number of intents: %d", self.num_classes)
        logger.info("Device: %s", self.device)
        logger.info("=" * 80)

    # -------------------------------------------------------------------
    # Audio Prediction
    # -------------------------------------------------------------------

    def predict_file(self, payload: bytes, suffix: str, top_k: int):
        """Decode audio and classify intent. Returns (ranked_intents, duration)."""
        if self.model is None:
            raise RuntimeError("Model is not loaded")

        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, received {top_k}")

        if top_k > self.num_classes:
            logger.warning(
                "Requested top_k=%d but model has only %d classes. Clamping top_k to %d.",
                top_k, self.num_classes, self.num_classes,
            )
            top_k = self.num_classes

        audio = self._decode_audio(payload, suffix)
        duration = self._validate_duration(audio)
        mel = self._preprocess_audio(audio)
        ranked_intents = self._run_inference(mel, top_k)

        logger.info(
            "Prediction: %s | duration=%.3fs | top_k=%d",
            ranked_intents[0] if ranked_intents else None, duration, top_k,
        )

        return ranked_intents, duration

    @staticmethod
    def _decode_audio(payload: bytes, suffix: str) -> np.ndarray:
        file_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
                handle.write(payload)
                file_path = handle.name

            logger.debug("Temporary audio file created: %s", file_path)
            return load_audio(file_path, sr=settings.TARGET_SAMPLE_RATE)

        except Exception as exc:
            logger.exception("Audio decoding failed")
            raise ValueError("Unable to decode the uploaded audio.") from exc

        finally:
            if file_path and os.path.exists(file_path):
                try:
                    os.unlink(file_path)
                except Exception:
                    logger.warning("Failed to delete temporary file: %s", file_path)

    @staticmethod
    def _validate_duration(audio: np.ndarray) -> float:
        duration = len(audio) / settings.TARGET_SAMPLE_RATE
        logger.debug("Audio duration: %.3f seconds", duration)

        if not (settings.MIN_AUDIO_DURATION_SECONDS <= duration <= settings.MAX_AUDIO_DURATION_SECONDS):
            raise ValueError(
                f"Audio duration must be between {settings.MIN_AUDIO_DURATION_SECONDS:g} and "
                f"{settings.MAX_AUDIO_DURATION_SECONDS:g} seconds (received {duration:.3f} seconds)."
            )
        return duration

    def _preprocess_audio(self, audio: np.ndarray) -> torch.Tensor:
        try:
            samples = pad_or_trim(np.asarray(audio, dtype=np.float32), N_SAMPLES)
            return log_mel_spectrogram(samples).unsqueeze(0).to(self.device)
        except Exception as exc:
            logger.exception("Audio preprocessing failed")
            raise RuntimeError(f"Audio preprocessing failed: {exc}") from exc

    def _run_inference(self, mel: torch.Tensor, top_k: int) -> list[dict]:
        autocast_context = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if self.device == "cuda" else nullcontext()
        )

        try:
            with torch.inference_mode(), autocast_context:
                logits = self.model(mel)

                if logits.ndim != 2:
                    raise RuntimeError(
                        f"Unexpected model output shape: {tuple(logits.shape)}. "
                        "Expected [batch, num_classes]."
                    )
                if logits.shape[0] != 1:
                    raise RuntimeError(f"Unexpected batch dimension in model output: {logits.shape[0]}")

                output_classes = int(logits.shape[1])
                self.model_output_classes = output_classes

                if output_classes != self.num_classes:
                    raise RuntimeError(
                        f"MODEL CLASS COUNT MISMATCH: config.json contains {self.num_classes} "
                        f"intents, but the model outputs {output_classes} classes. "
                        "Make sure config.json and model.bin belong to the same model."
                    )

                probabilities = torch.softmax(logits, dim=1)[0]
                scores, indices = torch.topk(probabilities, k=top_k)

        except Exception as exc:
            logger.exception("MODEL INFERENCE FAILED")
            raise RuntimeError(f"Model inference failed: {exc}") from exc

        return self._to_ranked_intents(scores, indices)

    def _to_ranked_intents(self, scores: torch.Tensor, indices: torch.Tensor) -> list[dict]:
        ranked_intents = []
        for score, index in zip(scores.detach().cpu().tolist(), indices.detach().cpu().tolist()):
            index = int(index)
            if index not in self.idx_to_intent:
                raise RuntimeError(
                    f"Model returned class index {index}, but that index does not exist in idx_to_intent."
                )
            ranked_intents.append({"intent": self.idx_to_intent[index], "confidence": float(score)})
        return ranked_intents


# Module-level singleton: one model in memory for the process, shared by the
# lifespan startup hook (main.py) and the route handlers (api/v1/routes.py).
service = IntentService()
