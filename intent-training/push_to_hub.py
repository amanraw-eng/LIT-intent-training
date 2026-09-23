import argparse
import json
import os
import sys
from pathlib import Path

import torch
from huggingface_hub import HfApi, create_repo

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Secrets (HF_TOKEN included) live in the project-root .env, not this
# directory's - a bare load_dotenv(THIS_DIR/".env") silently loads nothing
# since that file doesn't exist, leaving HF_TOKEN unset with no clear error
# until push_to_hub() runs.
from shared_config import load_environment

load_environment(legacy_env=Path(THIS_DIR) / ".env")


def export_checkpoint(ckpt_path, export_dir, model_type, intent_to_idx, model_module="model"):
    """model_module must be the module (in this directory) whose
    WhisperIntentClassification produced ckpt_path - "model" (single Linear
    head) or "model2" (Linear-LayerNorm-ReLU-Dropout-Linear-LayerNorm-ReLU-
    Linear head, masked-mean pooling). Loading with the wrong one doesn't
    warn - it crashes on a state_dict shape mismatch, or worse, silently
    evaluates with the wrong pooling if the shapes happen to overlap. This
    gets recorded in config.json so a downstream loader (or a future
    infer.py) can pick the right class instead of assuming model.py's."""
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
        "model_module": model_module,
        "n_class": len(intent_to_idx),
        "intent_to_idx": intent_to_idx,
        "idx_to_intent": idx_to_intent,
    }
    with open(export_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    if model_module == "model2":
        head_desc = (
            "Linear(768,256) -> LayerNorm -> ReLU -> Dropout -> Linear(256,128) -> "
            "LayerNorm -> ReLU -> Linear(128,n_class)"
        )
        pooling_desc = (
            "**Masked** mean pooling over encoder timesteps - this model was trained with "
            "`valid_lengths` (see `dataset2.py`/`model2.py` in the training repo). Pooling "
            "over the full padded window instead (no `valid_lengths`) doesn't error, it just "
            "silently evaluates ~4x worse (measured) since most of a 30s window is padding "
            "silence for typical short utterances."
        )
    else:
        head_desc = "Linear(feature_dim, n_class)"
        pooling_desc = "Plain mean pooling over all encoder timesteps (`torch.mean(x, dim=1)`)."

    readme_path = export_dir / "README.md"
    if not readme_path.exists():
        intents = "\n".join(f"- {intent}" for intent in intent_to_idx)
        readme_path.write_text(
            f"# Whisper Intent Classifier ({model_type}, {model_module})\n\n"
            f"Whisper `{model_type}` encoder + classification head, "
            f"fine-tuned on call-transcript audio for {len(intent_to_idx)}-way intent "
            "classification.\n\n"
            "## Architecture\n\n"
            f"- Encoder: `whisper.load_model('{model_type}').encoder`, fully trainable\n"
            f"- Head (`{model_module}.py`): {head_desc}\n"
            f"- Pooling: {pooling_desc}\n\n"
            "## Files\n\n"
            "- `model.bin` - state dict for `WhisperIntentClassification`\n"
            f"- `config.json` - `model_type`, `model_module` (load weights with "
            f"`{model_module}.py`'s class, not the other one), `n_class`, and the "
            "`intent_to_idx` / `idx_to_intent` label mapping\n\n"
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
    parser.add_argument(
        "--model-module",
        default="model",
        choices=["model", "model2"],
        help=(
            "Which module (in this directory) produced ckpt_path's architecture - "
            "'model' (single Linear head, e.g. trainer.py/train4.py checkpoints) or "
            "'model2' (deeper head + masked pooling, e.g. train15.py checkpoints). "
            "Default 'model' matches this script's original/legacy behavior - "
            "get this wrong and a downstream loader will crash or silently mispredict."
        ),
    )
    parser.add_argument("--export_dir", default=os.path.join(THIS_DIR, "hf_export"))
    parser.add_argument("--public", action="store_true", help="Push as a public repo (default: private)")
    args = parser.parse_args()

    with open(args.intent_map_path, "r", encoding="utf-8") as f:
        intent_to_idx = json.load(f)

    export_dir = export_checkpoint(
        args.ckpt_path, args.export_dir, args.model_type, intent_to_idx, model_module=args.model_module,
    )
    push_to_hub(export_dir, args.repo_id, private=not args.public)
