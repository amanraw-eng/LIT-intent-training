from model import WhisperIntentClassification
from dataset import HFIntentDataset, cap_dataset_per_class, collate_mel_fn, load_hf_split

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
    run_inference,
    save_class_distribution,
    plot_confusion_matrix,
    plot_per_class_accuracy,
    log_evaluation_to_mlflow,
)

# RUN SETTINGS (edit shared_settings.py [training] for the shared defaults below)
_TRAINING = section("training")
SEED = _TRAINING["seed"]
pl.seed_everything(SEED)
torch.manual_seed(SEED)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = _TRAINING["gpu_device"]

# ============================================================
# SCRIPT-SPECIFIC RUN OVERRIDES (edit here for this training run)
# ============================================================
# This 15-intent dataset lives in its own repo, separate from
# shared_settings.py's [training].dataset_repo (which other experiment
# versions - v4, v6-eval2, etc. - still point at) - overridden locally here
# rather than changed globally so this run doesn't affect those.
REPO_ID = "kapturecx/S2I-10-v1"
# This dataset's splits are named train/val/test, not train/validation/eval
# like shared_settings.py's default dataset_repo.
TRAIN_SPLIT = "train"
VAL_SPLIT = "validation"
TEST_SPLIT = "test"  # not used for training - held out for a final evaluate.py pass

DEFAULT_EXPERIMENT_VERSION = "S2I-15"

CAP_PER_CLASS = 30_000
BATCH_SIZE = 8
NUM_WORKERS = 4
MAX_EPOCHS = 10
PATIENCE = 5
SAVE_TOP_K = 4


class MlflowStopCallback(pl.Callback):
    """Lets you stop this run remotely from MLflow: if the run's status ever
    reads as anything other than RUNNING (e.g. you mark it stopped/killed
    from the MLflow UI or call
    MlflowClient().set_terminated(run_id, status="KILLED") yourself),
    training stops gracefully at the next validation epoch - already-saved
    checkpoints are unaffected. Checked once per validation epoch; a failed
    check (e.g. a transient network blip) is logged and ignored rather than
    treated as a stop signal or allowed to crash training."""

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
    def __init__(self, n_class, class_weights=None):
        super().__init__()
        # tiny/small model
        self.model = WhisperIntentClassification(_TRAINING["whisper_size"], n_class=n_class)
        self.register_buffer(
            "class_weights",
            class_weights if class_weights is not None else torch.ones(n_class),
        )

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=_TRAINING["learning_rate"], weight_decay=_TRAINING["weight_decay"])
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
        default=DEFAULT_EXPERIMENT_VERSION,
        help=f"Experiment version (defaults to {DEFAULT_EXPERIMENT_VERSION!r} for this script, "
             "not shared_settings.py's global default - pass this to run a second variant, "
             "e.g. --experiment-version S2I-15-run2)",
    )
    args = parser.parse_args()
    experiment_paths = get_experiment_paths(args.experiment_version)
    print(f"Experiment version: {experiment_paths.version}")
    print(f"Dataset repo: {REPO_ID} (splits: {TRAIN_SPLIT}/{VAL_SPLIT}/{TEST_SPLIT})")
    print(f"Checkpoint directory: {experiment_paths.checkpoint_dir}")
    print(f"Intent map: {experiment_paths.intent_map_path}")
    print(f"Eval output directory: {experiment_paths.eval_output_dir}")

    # isolated per-version dirs (models/<version>/, eval_results/<version>/) -
    # created up front so nothing later has to assume they already exist
    os.makedirs(experiment_paths.checkpoint_dir, exist_ok=True)
    os.makedirs(os.path.dirname(experiment_paths.intent_map_path), exist_ok=True)
    os.makedirs(experiment_paths.eval_output_dir, exist_ok=True)

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

    # S2I-15: new 15-intent taxonomy from its own HF Hub repo (REPO_ID above),
    # with train/val/test splits already prepared upstream. The train split
    # is capped at CAP_PER_CLASS rows per class so no single class can
    # dominate the gradient signal; validation is left uncapped so it still
    # reflects real skew.
    train_hf_full = load_hf_split(REPO_ID, TRAIN_SPLIT)
    val_hf = load_hf_split(REPO_ID, VAL_SPLIT)

    train_hf = cap_dataset_per_class(train_hf_full, CAP_PER_CLASS, seed=SEED)
    print(f"train: {len(train_hf_full)} rows -> capped to {len(train_hf)} rows "
          f"(max {CAP_PER_CLASS}/class), validation: {len(val_hf)} rows")

    # label mapping is derived from the FULL train set (not the capped subset),
    # and frozen for this run so checkpoints stay valid regardless of how the
    # Hub dataset evolves later (push_to_hub.py / infer.py read this file)
    intents = sorted(set(train_hf_full["intent"]))
    intent_to_idx = {intent: idx for idx, intent in enumerate(intents)}
    n_class = len(intent_to_idx)
    print(f"n_class: {n_class}")

    with open(experiment_paths.intent_map_path, "w", encoding="utf-8") as f:
        json.dump(intent_to_idx, f, indent=2, ensure_ascii=False)

    train_dataset = HFIntentDataset(train_hf, intent_to_idx=intent_to_idx)
    val_dataset = HFIntentDataset(val_hf, intent_to_idx=intent_to_idx)

    # class-weighted loss on top of the cap, computed from the POST-cap
    # distribution - capping already removes the most extreme imbalance,
    # weighting corrects whatever skew remains among the rest.
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
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            collate_fn = collate_mel_fn,
        )

    valloader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            collate_fn = collate_mel_fn,
        )

    model = LightningModel(n_class=n_class, class_weights=class_weights)

    logger = MLFlowLogger(
        experiment_name=experiment_paths.mlflow_experiment_name,
        run_name=experiment_paths.run_name,
        tracking_uri=tracking_uri,
    )
    # accessing .run_id forces MLFlowLogger to create the run now (not lazily
    # on first log call), so the stop-callback has a real run_id to poll
    # from the very first validation epoch
    mlflow_run_id = logger.run_id
    print(f"MLflow run_id: {mlflow_run_id} (set this run's status away from "
          "RUNNING in MLflow at any time to stop training gracefully)")

    model_checkpoint_callback = ModelCheckpoint(
            dirpath=experiment_paths.checkpoint_dir,
            monitor='val/acc',
            mode='max',
            save_top_k=SAVE_TOP_K,
            verbose=1,
            filename=experiment_paths.checkpoint_prefix + "-epoch{epoch:02d}")

    early_stopping_callback = EarlyStopping(
            monitor='val/acc',
            mode='max',
            patience=PATIENCE,
            verbose=True)

    mlflow_stop_callback = MlflowStopCallback(tracking_uri, mlflow_run_id)

    trainer = Trainer(
            fast_dev_run=False, # true for dev run
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

    # ------------------------------------------------------------------
    # final evaluation: best checkpoint (by val/acc, tracked live by
    # model_checkpoint_callback - not re-derived from disk) on the held-out
    # TEST_SPLIT, reusing evaluate.py's reporting/plotting so results land in
    # the same eval_output_dir layout and get logged to MLflow the same way
    # a standalone evaluate.py run would.
    # ------------------------------------------------------------------
    best_ckpt_path = model_checkpoint_callback.best_model_path
    best_score = model_checkpoint_callback.best_model_score
    print(f"Best checkpoint: {best_ckpt_path} (val/acc={best_score})")

    if not best_ckpt_path:
        print("No best checkpoint recorded (training may have stopped before "
              "any validation epoch) - skipping test-set evaluation")
    else:
        test_hf = load_hf_split(REPO_ID, TEST_SPLIT)
        print(f"test: {len(test_hf)} rows")

        unseen_intents = set(test_hf["intent"]) - set(intent_to_idx)
        if unseen_intents:
            raise ValueError(f"Intents in test data missing from intent_map: {unseen_intents}")

        idx_to_intent = {v: k for k, v in intent_to_idx.items()}
        test_dataset = HFIntentDataset(test_hf, intent_to_idx=intent_to_idx)
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
        labels, preds = run_inference(eval_model, test_loader, device)

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
        print(f"Saved + logged best-checkpoint test-set evaluation -> {eval_output_dir}")
