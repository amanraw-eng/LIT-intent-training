from model import WhisperIntentClassification
from dataset import DEFAULT_REPO_ID, HFIntentDataset, cap_dataset_per_class, collate_mel_fn, load_hf_split

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger

import argparse
import os
import json
import mlflow
from collections import Counter
from dotenv import load_dotenv
from experiment_config import EXPERIMENT_VERSION_ENV, get_experiment_paths

# load MLFLOW_TRACKING_URI / HF_TOKEN (and any other secrets) from .env in this directory
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

# SEED
SEED = 100
pl.seed_everything(SEED)
torch.manual_seed(SEED)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

REPO_ID = DEFAULT_REPO_ID
CAP_PER_CLASS = 7000
class LightningModel(pl.LightningModule):
    def __init__(self, n_class, class_weights=None):
        super().__init__()
        # tiny/small model
        self.model = WhisperIntentClassification("small", n_class=n_class)
        self.register_buffer(
            "class_weights",
            class_weights if class_weights is not None else torch.ones(n_class),
        )

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-5, weight_decay=1e-2)
        return [optimizer]

    def loss_fn(self, prediction, targets):
        return nn.CrossEntropyLoss(weight=self.class_weights)(prediction, targets)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.view(-1)
        logits = self(x)
        probs = F.softmax(logits, dim=1)
        loss = self.loss_fn(logits, y)

        winners = logits.argmax(dim=1)
        corrects = (winners == y)
        acc = corrects.sum().float()/float(logits.size(0))

        self.log('train/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train/acc', acc, on_step=False, on_epoch=True, prog_bar=True)

        return {
            'loss':loss,
            'acc':acc
            }

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y.view(-1)

        logits = self(x)
        loss = self.loss_fn(logits, y)

        winners = logits.argmax(dim=1)
        corrects = (winners == y)
        acc = corrects.sum().float() / float( logits.size(0))

        self.log('val/loss' , loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/acc',acc, on_step=False, on_epoch=True, prog_bar=True)

        return {'val_loss':loss,
                'val_acc':acc,
                }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment-version",
        help=f"Experiment version (defaults to {EXPERIMENT_VERSION_ENV} or v5)",
    )
    args = parser.parse_args()
    experiment_paths = get_experiment_paths(args.experiment_version)
    print(f"Experiment version: {experiment_paths.version}")
    print(f"Checkpoint directory: {experiment_paths.checkpoint_dir}")
    print(f"Intent map: {experiment_paths.intent_map_path}")

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    print(tracking_uri)
    user_name = os.environ.get('MLFLOW_USER_NAME')
    if not tracking_uri:
        raise RuntimeError("MLFLOW_TRACKING_URI not set in .env")
    mlflow.set_tracking_uri(tracking_uri)

    if token := os.environ.get("MLFLOW_TRACKING_TOKEN"):
        os.environ["MLFLOW_TRACKING_PASSWORD"] = token

    experiment = mlflow.set_experiment(experiment_paths.mlflow_experiment_name)
    print(f"Experiment '{experiment_paths.mlflow_experiment_name}' -> id={experiment.experiment_id}")

    # v4: same 20-intent taxonomy as v3, pulled from the HF Hub, but the train
    # split is capped at CAP_PER_CLASS rows per class - no single class (e.g.
    # GENERAL_AFFIRMATIVE_ACKNOWLEDGEMENT at 38k+) can dominate the gradient
    # signal. validation is left uncapped so it still reflects real skew.
    train_hf_full = load_hf_split(REPO_ID, "train")
    val_hf = load_hf_split(REPO_ID, "validation")

    train_hf = cap_dataset_per_class(train_hf_full, CAP_PER_CLASS, seed=SEED)
    print(f"train: {len(train_hf_full)} rows -> capped to {len(train_hf)} rows "
          f"(max {CAP_PER_CLASS}/class), validation: {len(val_hf)} rows")

    # label mapping is derived from the FULL train set (not the capped subset)
    # so it stays identical to v3 regardless of capping, and frozen for this
    # run so checkpoints stay valid (push_to_hub.py / infer.py read this file)
    intents = sorted(set(train_hf_full["intent"]))
    intent_to_idx = {intent: idx for idx, intent in enumerate(intents)}
    n_class = len(intent_to_idx)
    print(f"n_class: {n_class}")

    os.makedirs(os.path.dirname(experiment_paths.intent_map_path), exist_ok=True)
    with open(experiment_paths.intent_map_path, "w", encoding="utf-8") as f:
        json.dump(intent_to_idx, f, indent=2, ensure_ascii=False)

    train_dataset = HFIntentDataset(train_hf, intent_to_idx=intent_to_idx)
    val_dataset = HFIntentDataset(val_hf, intent_to_idx=intent_to_idx)

    # class-weighted loss on top of the cap, computed from the POST-cap
    # distribution - capping already removes the most extreme imbalance,
    # weighting corrects whatever skew remains (e.g. DO_NOT_CALL still only
    # has ~427 rows even after capping the majority classes down to 7000).
    train_labels = [intent_to_idx[intent] for intent in train_hf["intent"]]
    class_counts = Counter(train_labels)
    class_weights = torch.tensor(
        [len(train_labels) / (n_class * class_counts[i]) for i in range(n_class)],
        dtype=torch.float32,
    )
    print("post-cap class counts:", {intents[i]: class_counts[i] for i in range(n_class)})
    print("class weights:", {intents[i]: round(w, 3) for i, w in enumerate(class_weights.tolist())})

    # dataloaders
    trainloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=8,
            shuffle=True,
            num_workers=4,
            collate_fn = collate_mel_fn,
        )

    valloader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=8,
            num_workers=4,
            collate_fn = collate_mel_fn,
        )

    model = LightningModel(n_class=n_class, class_weights=class_weights)

    logger = MLFlowLogger(
        experiment_name=experiment_paths.mlflow_experiment_name,
        run_name=experiment_paths.run_name,
        tracking_uri=tracking_uri,
    )

    model_checkpoint_callback = ModelCheckpoint(
            dirpath=experiment_paths.checkpoint_dir,
            monitor='val/acc',
            mode='max',
            save_top_k=3,
            verbose=1,
            filename=experiment_paths.checkpoint_prefix + "-epoch{epoch:02d}")

    early_stopping_callback = EarlyStopping(
            monitor='val/acc',
            mode='max',
            patience=5,
            verbose=True)

    trainer = Trainer(
            fast_dev_run=False, # true for dev run
            accelerator="gpu",
            devices=1,
            max_epochs=20,
            enable_checkpointing=True,
            callbacks=[
                model_checkpoint_callback,
                early_stopping_callback,
            ],
            logger=logger,
            )

    trainer.fit(model, train_dataloaders=trainloader, val_dataloaders=valloader)
