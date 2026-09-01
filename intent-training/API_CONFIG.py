"""Runtime configuration for the FastAPI intent-classification service.

Model and audio values mirror UI_CONFIG.py so the UI and API produce the same
predictions for the same file.
"""

import os
from dotenv import load_dotenv
load_dotenv('/home/jovyan/aman_ws/stt/LIT-intent-training/intent-training/.env')

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

# Copied from UI_CONFIG.py
MODEL_PATH = os.path.join(
    THIS_DIR, "models", "v5/checkpoints", "whisper-small-S2I-hi-v5-epochepoch=09.ckpt"
)
MODEL_TYPE = "small"
INTENT_MAP_PATH = os.path.join(THIS_DIR, "models", "intent_to_idx.json")
DEVICE = "cuda"
TARGET_SAMPLE_RATE = 16000
MIN_AUDIO_DURATION_SECONDS = 0.5
MAX_AUDIO_DURATION_SECONDS = 10.0

# API settings
HOST = "0.0.0.0"
PORT = 8000
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
DEFAULT_TOP_K = 3
MAX_TOP_K = 20
HF_MODEL_REPO = os.environ.get("HF_MODEL_REPO", "")
HF_TOKEN = os.environ.get("HF_TOKEN", "")

print(f"HF_MODEL: {HF_MODEL_REPO}")