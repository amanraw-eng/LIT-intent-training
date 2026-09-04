"""Shared naming and output paths for intent-training experiments."""

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
    )
