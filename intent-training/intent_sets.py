"""Registry of intent-set configurations.

Training and evaluation scripts in this directory serve two distinct intent
taxonomies (15-intent and 17-intent), each with its own dataset repo, split
names, and model architecture (model.py vs model2.py). Previously those
choices were hardcoded/duplicated per script, which is how train15.py (model2,
15-intent) and evaluate.py/eval3.py (hardcoded import of model.py, 17-intent)
drifted apart: evaluating a 15-intent checkpoint through the 17-intent eval
path loads the wrong architecture and crashes (mismatched state_dict shapes).

Look up a config by name instead of hardcoding these values in a new script:

    from intent_sets import get_intent_set
    cfg = get_intent_set("15")
    ModelClass = cfg.load_model_class()
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass


@dataclass(frozen=True)
class IntentSetConfig:
    name: str  # "15" or "17" - also the value stored in experiment manifests
    n_class: int
    dataset_repo: str
    train_split: str
    val_split: str
    test_split: str
    model_module: str  # module in this directory exposing WhisperIntentClassification
    taxonomy_path: str  # intents15.json / intent17.json, relative to repo root
    cap_per_class: int = 30_000
    # model2's WhisperIntentClassification.forward accepts valid_lengths and
    # masked-mean-pools over just the real (non-padding) audio frames - it was
    # TRAINED that way (dataset2.py's HFIntentDataset). model.py's forward
    # takes no such argument and always does a plain, unmasked mean. Using the
    # wrong dataset/collate pairing for a checkpoint's model_module silently
    # produces valid-looking-but-wrong predictions (~4x accuracy drop
    # measured), not a crash - eval code must pick dataset.py vs dataset2.py
    # to match this flag, not just the right model_cls.
    masked_pooling: bool = False

    def load_model_class(self):
        module = importlib.import_module(self.model_module)
        return module.WhisperIntentClassification


INTENT_SETS = {
    "15": IntentSetConfig(
        name="15",
        n_class=15,
        dataset_repo="kapturecx/S2I-10-v1",
        train_split="train",
        val_split="validation",
        test_split="test",
        model_module="model2",
        taxonomy_path="../intents15.json",
        masked_pooling=True,
    ),
    "17": IntentSetConfig(
        name="17",
        n_class=17,
        dataset_repo="kapturecx/call-transcript-intent-data-v2",
        train_split="train",
        val_split="validation",
        test_split="eval",
        model_module="model",
        taxonomy_path="../intent17.json",
    ),
}


def get_intent_set(name: str) -> IntentSetConfig:
    try:
        return INTENT_SETS[str(name)]
    except KeyError:
        raise KeyError(
            f"Unknown intent set {name!r}; choices: {sorted(INTENT_SETS)}"
        ) from None
