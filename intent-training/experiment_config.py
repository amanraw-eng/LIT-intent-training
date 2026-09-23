"""Shared naming and output paths for intent-training experiments."""

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared_config import load_environment, section


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
load_environment(legacy_env=Path(THIS_DIR) / ".env")
EXPERIMENT_VERSION_ENV = "INTENT_EXPERIMENT_VERSION"
DEFAULT_EXPERIMENT_VERSION = section("training")["experiment_version"]


@dataclass(frozen=True)
class ExperimentPaths:
    version: str
    mlflow_experiment_name: str
    run_name: str
    checkpoint_prefix: str
    checkpoint_dir: str
    intent_map_path: str
    eval_output_dir: str
    manifest_path: str


def get_experiment_paths(version=None):
    """Return names and isolated artifact paths for one experiment version."""
    version = version or os.getenv(EXPERIMENT_VERSION_ENV, DEFAULT_EXPERIMENT_VERSION)
    version = version.strip()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", version):
        raise ValueError(
            f"Invalid experiment version {version!r}. Use letters, numbers, '.', '_' or '-'."
        )

    model_dir = os.path.join(THIS_DIR, "models", version)
    return ExperimentPaths(
        version=version,
        mlflow_experiment_name=f"audio-intent-classification-{version}",
        run_name=f"audio-intent-classification-{version}",
        checkpoint_prefix=f"whisper-small-S2I-hi-{version}",
        checkpoint_dir=os.path.join(model_dir, "checkpoints"),
        intent_map_path=os.path.join(model_dir, "intent_to_idx.json"),
        eval_output_dir=os.path.join(THIS_DIR, "eval_results", version),
        manifest_path=os.path.join(model_dir, "experiment_manifest.json"),
    )


def load_manifest(experiment_paths: ExperimentPaths) -> dict | None:
    """Read the intent-set/model-architecture manifest a train script saved
    for this experiment version, or None if it never wrote one (older runs,
    or scripts that don't call save_manifest)."""
    if not os.path.exists(experiment_paths.manifest_path):
        return None
    with open(experiment_paths.manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_manifest(experiment_paths: ExperimentPaths, *, intent_set: str, model_module: str, n_class: int, whisper_size: str) -> None:
    """Record which intent-set config and model architecture produced this
    experiment's checkpoints, so eval scripts can load the matching
    architecture automatically instead of guessing/hardcoding it."""
    os.makedirs(os.path.dirname(experiment_paths.manifest_path), exist_ok=True)
    with open(experiment_paths.manifest_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "intent_set": intent_set,
                "model_module": model_module,
                "n_class": n_class,
                "whisper_size": whisper_size,
            },
            f,
            indent=2,
        )
