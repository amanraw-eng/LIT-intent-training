# Intent Classification API

OpenAI-compatible Whisper intent-classification service. Self-contained -
no dependency on anything else in this repository.

## Requirements

- Python 3.11 or 3.12
- [uv](https://docs.astral.sh/uv/)
  ``` curl -LsSf https://astral.sh/uv/install.sh | sh ```
- `ffmpeg` on `PATH` (used by Whisper to decode uploaded audio)
- A CUDA GPU for production use (falls back to CPU automatically if
  `DEVICE == "cuda"` but none is available - see
  [app/config/constants.py](app/config/constants.py))

## Setup

```bash
cd src
uv sync
cp .env.example ../.env
```

Edit `.env` and set at least `HF_TOKEN` if the model repo in
`HF_MODEL_REPO` is private. `HF_MODEL_REPO` defaults to
`amn-raw/whisper-small-intent17-classifier`; override it in `.env` to
serve a different model. The repo must contain `config.json` (with
`intent_to_idx` / `idx_to_intent`) and `model.bin`.

## v1 vs v2

Two model architectures, served as two fully separate apps/processes -
running one never affects the other, and each holds its own model in GPU
memory:

| | v1 (`app/main.py`) | v2 (`app/main2.py`) |
|---|---|---|
| Architecture | single Linear head, plain mean pooling (`services/whisper_model.py`) | deeper head (Linear-LayerNorm-ReLU x2) + masked-mean pooling over real audio only (`services/whisper_model_v2.py`) |
| Model repo setting | `HF_MODEL_REPO` | `HF_MODEL_REPO_V2` |
| Default repo | `amn-raw/whisper-small-intent17-classifier` | `kapturecx/intents15-V6` |
| Endpoint | `/intent/v1/audio/transcriptions` | `/intent/v2/audio/transcriptions` |
| Health | `/health` | `/intent/v2/health` |

To switch which checkpoint v2 serves later, just change `HF_MODEL_REPO_V2`
in `.env` and restart - no code change needed, as long as the new
checkpoint was pushed with the same `model2` architecture. v2 checks
`config.json`'s `model_module` field on startup and refuses to load a
checkpoint that isn't `"model2"`, so pointing it at a mismatched repo fails
loudly at startup instead of silently mispredicting.

## Run

v1:

```bash
uv run uvicorn app.main:app --host 0.0.0.0 --port 4000 --workers 4
```

or

```bash
uv run python -m app.main
```

v2 (same commands, different module - defaults to the same port, so run one
at a time unless you override `PORT` for one of them):

```bash
uv run uvicorn app.main2:app --host 0.0.0.0 --port 4000 --workers 4
```

or

```bash
uv run python -m app.main2
```

On startup the service downloads the model from Hugging Face and loads it
onto `DEVICE` (falling back to CPU if CUDA isn't available). If loading fails, the server still starts so
`/health` (v1) or `/intent/v2/health` (v2) can report why, instead of
crashing outright.

## Configuration

- [app/config/settings.py](app/config/settings.py) - values that differ per
  deployment (`HF_MODEL_REPO`, `HF_MODEL_REPO_V2`, `HF_TOKEN`, `HOST`,
  `PORT`, `LOG_LEVEL`, `DEBUG_ERRORS`). Read from `.env` / the environment;
  env vars win.
- [app/config/constants.py](app/config/constants.py) - fixed, code-level
  defaults not meant to vary per deployment (model type, audio duration
  limits, upload size cap, max `top_k`). Shared by both v1 and v2.

## Endpoints

v1:

- `GET /health` - `200` with device/model info once the model is loaded,
  `503` with diagnostics otherwise.
- `POST /intent/v1/audio/transcriptions` - OpenAI audio-transcriptions-
  compatible; the returned "transcription" is the predicted intent.

```bash
curl http://localhost:8000/health

curl http://localhost:8000/intent/v1/audio/transcriptions \
  -F file=@sample.wav \
  -F response_format=verbose_json \
  -F top_k=3
```

v2 (same request/response shape, different path):

```bash
curl http://localhost:8000/intent/v2/health

curl http://localhost:8000/intent/v2/audio/transcriptions \
  -F file=@sample.wav \
  -F response_format=verbose_json \
  -F top_k=3
```

`response_format` is `json` (default), `verbose_json`, or `text`.

## Project layout

```
app/
  main.py           v1 FastAPI app, lifespan (model load/unload), uvicorn entrypoint
  main2.py          v2 FastAPI app - separate process, separate model, same shape as main.py
  config/           settings.py (env-driven) + constants.py (fixed) + logging.py
  api/v1/           v1 route handlers (routes.py) and response models (schemas.py)
  api/v2/           v2 route handlers - reuses api/v1/schemas.py, same response shape
  services/         IntentService + WhisperIntentClassification (v1),
                     IntentServiceV2 + WhisperIntentClassification (v2, masked pooling)
```
