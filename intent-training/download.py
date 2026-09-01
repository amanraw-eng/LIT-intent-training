
# download_model.py
from huggingface_hub import hf_hub_download
import API_CONFIG as cfg

hf_hub_download(repo_id=cfg.HF_MODEL_REPO, filename="config.json", token=cfg.HF_TOKEN)
hf_hub_download(repo_id=cfg.HF_MODEL_REPO, filename="model.bin", token=cfg.HF_TOKEN)