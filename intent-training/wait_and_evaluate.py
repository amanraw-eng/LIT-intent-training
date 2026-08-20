"""Wait for a training checkpoint, then evaluate its best model and log to MLflow.

Example:
    /mnt/HDD8TB/aman_ws/stt/.venv/bin/python3 wait_and_evaluate.py \
        --experiment-version v5 --delay-hours 10
"""

import argparse
import glob
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone

import torch

from experiment_config import EXPERIMENT_VERSION_ENV, get_experiment_paths


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_PATH = os.path.join(THIS_DIR, "wait_and_evaluate.log")


def log(message):
    line = f"[{datetime.now(timezone.utc).isoformat()}] {message}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def epoch_from_filename(path):
    match = re.search(r"epoch(\d+)", os.path.basename(path))
    return int(match.group(1)) if match else -1


def find_best_checkpoint(checkpoint_dir):
    """Read Lightning callback state to locate the checkpoint with best val/acc."""
    checkpoint_paths = glob.glob(os.path.join(checkpoint_dir, "*.ckpt"))
    if not checkpoint_paths:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")

    checkpoint_paths.sort(key=epoch_from_filename, reverse=True)
    for checkpoint_path in checkpoint_paths:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        for callback_state in checkpoint.get("callbacks", {}).values():
            if not isinstance(callback_state, dict):
                continue
            best_path = callback_state.get("best_model_path")
            if not best_path:
                continue
            if not os.path.isabs(best_path):
                best_path = os.path.join(THIS_DIR, best_path)
            if os.path.exists(best_path):
                score = callback_state.get("best_model_score")
                return best_path, float(score) if score is not None else None

    fallback_path = checkpoint_paths[0]
    log(
        "Could not read ModelCheckpoint best_model_path; "
        f"falling back to latest checkpoint: {fallback_path}"
    )
    return fallback_path, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment-version",
        help=f"Experiment version (defaults to {EXPERIMENT_VERSION_ENV} or v5)",
    )
    parser.add_argument("--delay-hours", type=float, default=10.0)
    args = parser.parse_args()

    if args.delay_hours < 0:
        raise ValueError("--delay-hours must be non-negative")

    paths = get_experiment_paths(args.experiment_version)
    delay_seconds = args.delay_hours * 60 * 60
    log(
        f"Waiting {args.delay_hours:g} hours before checking "
        f"{paths.checkpoint_dir} for the best {paths.version} checkpoint."
    )
    time.sleep(delay_seconds)

    best_checkpoint, best_score = find_best_checkpoint(paths.checkpoint_dir)
    score_text = "unknown" if best_score is None else f"{best_score:.6f}"
    log(f"Best checkpoint: {best_checkpoint} (val/acc={score_text})")

    command = [
        sys.executable,
        "evaluate.py",
        "--experiment-version",
        paths.version,
        "--split",
        "eval",
        "--ckpt_name",
        best_checkpoint,
    ]
    log(f"Running: {' '.join(command)}")
    subprocess.run(command, cwd=THIS_DIR, check=True)
    log(
        "Evaluation complete. Metrics and artifacts were logged to MLflow "
        f"experiment '{paths.mlflow_experiment_name}'."
    )


if __name__ == "__main__":
    main()
