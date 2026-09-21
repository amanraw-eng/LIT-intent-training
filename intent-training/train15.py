from model2 import WhisperIntentClassification
from dataset import cap_dataset_per_class, load_hf_split
from dataset2 import DURATION_CAP_S, HFIntentDataset, collate_mel_fn

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
from mlflow.tracking import MlflowClient
from sklearn.metrics import classification_report
from experiment_config import get_experiment_paths, section
from evaluate import (
    load_model as load_eval_model,
    save_class_distribution,
    plot_confusion_matrix,
    plot_per_class_accuracy,
    log_evaluation_to_mlflow,
)

# Shared defaults setup
_TRAINING = section("training")
SEED = _TRAINING["seed"]
pl.seed_everything(SEED)
torch.manual_seed(SEED)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = _TRAINING["gpu_device"]

# Experiment configuration
REPO_ID = "kapturecx/S2I-10-v1"
# This dataset's splits are named train/val/test, not train/validation/eval
# like shared_settings.py's default dataset_repo.
TRAIN_SPLIT = "train"
VAL_SPLIT = "validation"
TEST_SPLIT = "test"

DEFAULT_EXPERIMENT_VERSION = "S2I-15"

CAP_PER_CLASS = 30_000
BATCH_SIZE = 16
NUM_WORKERS = 4
MAX_EPOCHS = 10
PATIENCE = 5
SAVE_TOP_K = 4

FREEZE_ENCODER_EPOCHS = 3
ENCODER_LR = 1e-6
HEAD_LR = 5e-4


class MlflowStopCallback(pl.Callback):
    """Allows remote job cancellation from the MLflow UI."""

    def __init__(self, tracking_uri, run_id):
        self.client = MlflowClient(tracking_uri=tracking_uri)
        self.run_id = run_id

    def on_validation_end(self, trainer, pl_module):
        try:
            status = self.client.get_run(self.run_id).info.status
        except Exception as e:
            print(f"MlflowStopCallback: could not check run status ({e}) - continuing training")
            return
        if status != "RUNNING":
            print(f"MLflow run status is {status!r} (not RUNNING) - stopping training gracefully")
            trainer.should_stop = True


class LightningModel(pl.LightningModule):
    def __init__(
        self,
        n_class,
        encoder_lr=ENCODER_LR,
        head_lr=HEAD_LR,
        freeze_encoder_epochs=FREEZE_ENCODER_EPOCHS,
        class_weights=None,
    ):
        super().__init__()
        # tiny/small model
        self.model = WhisperIntentClassification(_TRAINING["whisper_size"], n_class=n_class)
        self.encoder_lr = encoder_lr
        self.head_lr = head_lr
        self.freeze_encoder_epochs = freeze_encoder_epochs

        self.register_buffer(
            "class_weights",
            class_weights if class_weights is not None else torch.ones(n_class),
        )
        self.loss_fn = nn.CrossEntropyLoss(weight=self.class_weights)

    def forward(self, x, valid_lengths=None):
        return self.model(x, valid_lengths=valid_lengths)

    def on_train_epoch_start(self):
        # Unfreeze encoder via learning rate modulation in param_groups
        current_encoder_lr = 0.0 if self.current_epoch < self.freeze_encoder_epochs else self.encoder_lr
        opt = self.optimizers()

        if isinstance(opt, torch.optim.Optimizer):
            opt.param_groups[0]["lr"] = current_encoder_lr

        if self.current_epoch == self.freeze_encoder_epochs:
            print(f"--- Epoch {self.current_epoch}: Unfreezing encoder (lr={self.encoder_lr}) ---")

    def configure_optimizers(self):
        init_encoder_lr = 0.0 if self.current_epoch < self.freeze_encoder_epochs else self.encoder_lr

        optimizer = torch.optim.AdamW(
            [
                {"params": self.model.encoder.parameters(), "lr": init_encoder_lr},        # Group 0
                {"params": self.model.intent_classifier.parameters(), "lr": self.head_lr}, # Group 1
            ],
            weight_decay=_TRAINING["weight_decay"],
        )
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y, valid_lengths = batch
        y = y.view(-1)
        logits = self(x, valid_lengths=valid_lengths)
        probs = F.softmax(logits, dim=1)
        loss = self.loss_fn(logits, y)

        winners = logits.argmax(dim=1)
        acc = (winners == y).float().mean()

        self.log('train/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train/acc', acc, on_step=False, on_epoch=True, prog_bar=True)

        return {'loss': loss, 'acc': acc}

    def validation_step(self, batch, batch_idx):
        x, y, valid_lengths = batch
        y = y.view(-1)

        logits = self(x, valid_lengths=valid_lengths)
        loss = self.loss_fn(logits, y)

        winners = logits.argmax(dim=1)
        acc = (winners == y).float().mean()

        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/acc', acc, on_step=False, on_epoch=True, prog_bar=True)

        return {'val_loss': loss, 'val_acc': acc}


@torch.no_grad()
def run_masked_inference(model, dataloader, device):
    """Executes evaluation while preserving length masking."""
    model.eval()
    all_preds = []
    all_labels = []

    for batch in dataloader:
        if len(batch) == 3:
            x, y, valid_lengths = batch
            x, y, valid_lengths = x.to(device), y.to(device), valid_lengths.to(device)
            logits = model(x, valid_lengths=valid_lengths)
        else:
            x, y = batch
            x, y = x.to(device), y.to(device)
            logits = model(x)

        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

    return all_labels, all_preds


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment-version",
        default=DEFAULT_EXPERIMENT_VERSION,
        help=f"Experiment version (defaults to {DEFAULT_EXPERIMENT_VERSION!r})",
    )
    args = parser.parse_args()
    experiment_paths = get_experiment_paths(args.experiment_version)
    print(f"Experiment version: {experiment_paths.version}")
    print(f"Dataset repo: {REPO_ID} (splits: {TRAIN_SPLIT}/{VAL_SPLIT}/{TEST_SPLIT})")

    os.makedirs(experiment_paths.checkpoint_dir, exist_ok=True)
    os.makedirs(os.path.dirname(experiment_paths.intent_map_path), exist_ok=True)
    os.makedirs(experiment_paths.eval_output_dir, exist_ok=True)

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        raise RuntimeError("MLFLOW_TRACKING_URI not set in environment.")
    mlflow.set_tracking_uri(tracking_uri)

    if token := os.environ.get("MLFLOW_TRACKING_TOKEN"):
        os.environ["MLFLOW_TRACKING_PASSWORD"] = token

    experiment = mlflow.set_experiment(experiment_paths.mlflow_experiment_name)

    train_hf_full = load_hf_split(REPO_ID, TRAIN_SPLIT)
    val_hf = load_hf_split(REPO_ID, VAL_SPLIT)

    train_hf = cap_dataset_per_class(train_hf_full, CAP_PER_CLASS, seed=SEED)

    intents = sorted(set(train_hf_full["intent"]))
    intent_to_idx = {intent: idx for idx, intent in enumerate(intents)}
    n_class = len(intent_to_idx)

    with open(experiment_paths.intent_map_path, "w", encoding="utf-8") as f:
        json.dump(intent_to_idx, f, indent=2, ensure_ascii=False)

    train_dataset = HFIntentDataset(train_hf, intent_to_idx=intent_to_idx, duration_cap_s=DURATION_CAP_S)
    val_dataset = HFIntentDataset(val_hf, intent_to_idx=intent_to_idx, duration_cap_s=DURATION_CAP_S)

    train_labels = [intent_to_idx[intent] for intent in train_hf["intent"]]
    class_counts = Counter(train_labels)
    class_weights = torch.tensor(
        [len(train_labels) / (n_class * class_counts[i]) for i in range(n_class)],
        dtype=torch.float32,
    )

    trainloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=collate_mel_fn,
    )

    valloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        collate_fn=collate_mel_fn,
    )

    model = LightningModel(n_class=n_class, class_weights=class_weights)

    logger = MLFlowLogger(
        experiment_name=experiment_paths.mlflow_experiment_name,
        run_name=experiment_paths.run_name,
        tracking_uri=tracking_uri,
    )
    mlflow_run_id = logger.run_id

    model_checkpoint_callback = ModelCheckpoint(
        dirpath=experiment_paths.checkpoint_dir,
        monitor='val/acc',
        mode='max',
        save_top_k=SAVE_TOP_K,
        verbose=1,
        filename=experiment_paths.checkpoint_prefix + "-epoch{epoch:02d}",
    )

    early_stopping_callback = EarlyStopping(
        monitor='val/acc',
        mode='max',
        patience=PATIENCE,
        verbose=True,
    )

    mlflow_stop_callback = MlflowStopCallback(tracking_uri, mlflow_run_id)

    trainer = Trainer(
        fast_dev_run=False,
        accelerator="gpu",
        devices=1,
        max_epochs=MAX_EPOCHS,
        enable_checkpointing=True,
        callbacks=[
            model_checkpoint_callback,
            early_stopping_callback,
            mlflow_stop_callback,
        ],
        logger=logger,
    )

    trainer.fit(model, train_dataloaders=trainloader, val_dataloaders=valloader)

    # Final Evaluation Loop
    best_ckpt_path = model_checkpoint_callback.best_model_path
    best_score = model_checkpoint_callback.best_model_score

    if not best_ckpt_path:
        print("No best checkpoint recorded - skipping test evaluation.")
    else:
        test_hf = load_hf_split(REPO_ID, TEST_SPLIT)
        idx_to_intent = {v: k for k, v in intent_to_idx.items()}

        test_dataset = HFIntentDataset(test_hf, intent_to_idx=intent_to_idx, duration_cap_s=DURATION_CAP_S)
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            collate_fn=collate_mel_fn,
        )

        eval_output_dir = experiment_paths.eval_output_dir
        test_labels_by_idx = [intent_to_idx[intent] for intent in test_hf["intent"]]
        save_class_distribution(
            test_labels_by_idx, idx_to_intent,
            os.path.join(eval_output_dir, "test_class_distribution.txt"),
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        eval_model = load_eval_model(best_ckpt_path, _TRAINING["whisper_size"], n_class, device)
        labels, preds = run_masked_inference(eval_model, test_loader, device)

        present_ids = sorted(set(labels) | set(preds))
        target_names = [idx_to_intent[i] for i in present_ids]

        report_text = classification_report(
            labels, preds, labels=present_ids, target_names=target_names, digits=3, zero_division=0,
        )
        with open(os.path.join(eval_output_dir, "classification_report.txt"), "w", encoding="utf-8") as f:
            f.write(report_text)
        print(report_text)

        report_dict = classification_report(
            labels, preds, labels=present_ids, target_names=target_names, digits=3, zero_division=0, output_dict=True,
        )
        with open(os.path.join(eval_output_dir, "classification_report.json"), "w", encoding="utf-8") as f:
            json.dump(report_dict, f, indent=2)

        plot_confusion_matrix(labels, preds, idx_to_intent, os.path.join(eval_output_dir, "confusion_matrix.png"), normalize=False)
        plot_confusion_matrix(labels, preds, idx_to_intent, os.path.join(eval_output_dir, "confusion_matrix_normalized.png"), normalize=True)
        plot_per_class_accuracy(labels, preds, idx_to_intent, os.path.join(eval_output_dir, "per_class_accuracy.png"))

        log_evaluation_to_mlflow(
            experiment_paths=experiment_paths,
            args=argparse.Namespace(repo_id=REPO_ID, split=TEST_SPLIT, model_type=_TRAINING["whisper_size"]),
            ckpt_path=best_ckpt_path,
            intent_map_path=experiment_paths.intent_map_path,
            output_dir=eval_output_dir,
            report_dict=report_dict,
            eval_rows=len(labels),
        )