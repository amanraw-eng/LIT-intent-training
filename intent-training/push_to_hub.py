import argparse
import json
import os
from pathlib import Path

import torch
from dotenv import load_dotenv
from huggingface_hub import HfApi, create_repo

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(THIS_DIR, ".env"))


def export_checkpoint(ckpt_path, export_dir, model_type, intent_to_idx):
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = checkpoint["state_dict"]
    # LightningModel wraps WhisperIntentClassification as `self.model`
    state_dict = {
        key[len("model."):]: value
        for key, value in state_dict.items()
        if key.startswith("model.")
    }

    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    torch.save(state_dict, export_dir / "model.bin")

    idx_to_intent = {v: k for k, v in intent_to_idx.items()}
    config = {
        "model_type": model_type,
        "n_class": len(intent_to_idx),
        "intent_to_idx": intent_to_idx,
        "idx_to_intent": idx_to_intent,
    }
    with open(export_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    readme_path = export_dir / "README.md"
    if not readme_path.exists():
        intents = "\n".join(f"- {intent}" for intent in intent_to_idx)
        readme_path.write_text(
            f"# Whisper Intent Classifier ({model_type})\n\n"
            f"Whisper `{model_type}` encoder + linear classification head, "
            f"fine-tuned on call-transcript audio for {len(intent_to_idx)}-way intent "
            "classification.\n\n"
            "## Files\n\n"
            "- `model.bin` - state dict for `WhisperIntentClassification`\n"
            "- `config.json` - `model_type`, `n_class`, and the `intent_to_idx` / "
            "`idx_to_intent` label mapping\n\n"
            "## Usage\n\n"
            "See `infer.py` in the training repo for a ready-to-use inference script.\n\n"
            f"## Intents\n\n{intents}\n",
            encoding="utf-8",
        )
    return export_dir


def push_to_hub(export_dir, repo_id, private=True):
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN not set. Add it to .env in this directory.")

    create_repo(repo_id, token=token, private=private, exist_ok=True)
    api = HfApi(token=token)
    api.upload_folder(
        folder_path=str(export_dir),
        repo_id=repo_id,
        commit_message="Upload whisper intent classifier",
    )
    print(f"Pushed model to https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", required=True, help="Path to the Lightning .ckpt checkpoint")
    parser.add_argument("--repo_id", required=True, help="e.g. kapturecx/whisper-small-intent-classifier")
    parser.add_argument("--model_type", default="small")
    parser.add_argument(
        "--intent_map_path",
        default=os.path.join(THIS_DIR, "models", "intent_to_idx.json"),
        help="Label mapping saved by trainer.py at the start of training",
    )
    parser.add_argument("--export_dir", default=os.path.join(THIS_DIR, "hf_export"))
    parser.add_argument("--public", action="store_true", help="Push as a public repo (default: private)")
    args = parser.parse_args()

    with open(args.intent_map_path, "r", encoding="utf-8") as f:
        intent_to_idx = json.load(f)

    export_dir = export_checkpoint(args.ckpt_path, args.export_dir, args.model_type, intent_to_idx)
    push_to_hub(export_dir, args.repo_id, private=not args.public)
