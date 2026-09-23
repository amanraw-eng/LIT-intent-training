from model2 import WhisperIntentClassification
from dataset import cap_dataset_per_class, load_hf_split
from dataset2 import DURATION_CAP_S, HFIntentDataset, collate_mel_fn
from intent_sets import get_intent_set

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
from experiment_config import get_experiment_paths, save_manifest, section
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

# H200 (and any Ampere+/Hopper GPU) gets a free matmul speedup from TF32 for
# whatever stays in fp32 under the bf16-mixed autocast below (norms, softmax,
# etc.) - no accuracy-relevant effect for this workload.
torch.set_float32_matmul_precision("high")

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = _TRAINING["gpu_device"]

# Experiment configuration - dataset repo, split names, and the per-class cap
# all come from the "15" entry in intent_sets.py rather than being hardcoded
# here, so this script can't silently drift from the config eval scripts read
# back (that drift is what caused train15.py's own model2 checkpoints to
# crash when evaluate.py tried to reload them through model.py's
# architecture - see intent_sets.py's docstring).
INTENT_SET = get_intent_set("15")
REPO_ID = INTENT_SET.dataset_repo
TRAIN_SPLIT = INTENT_SET.train_split
VAL_SPLIT = INTENT_SET.val_split
TEST_SPLIT = INTENT_SET.test_split
CAP_PER_CLASS = INTENT_SET.cap_per_class

DEFAULT_EXPERIMENT_VERSION = "S2I-15-V8"

BATCH_SIZE = 32
NUM_WORKERS = 8
MAX_EPOCHS = 15
PATIENCE = 5
SAVE_TOP_K = 4

FREEZE_ENCODER_EPOCHS = 8
ENCODER_LR = 1e-6
HEAD_LR = 6e-4


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
    parser.add_argument(
        "--warm-start-ckpt",
        default=None,
        help=(
            "Path to a previous run's .ckpt (must be this same model2 architecture). "
            "Loads its encoder+classifier weights before training starts, then trains "
            "for MAX_EPOCHS from epoch 0 with a fresh optimizer/scheduler/early-stopping "
            "state - use this to reuse an already-adapted frozen-encoder phase instead of "
            "re-running it while only sweeping post-unfreeze hyperparameters "
            "(ENCODER_LR, FREEZE_ENCODER_EPOCHS, dropout, ...). Pass --freeze-encoder-epochs 0 "
            "alongside this if the checkpoint's encoder is already unfrozen/adapted."
        ),
    )
    parser.add_argument(
        "--freeze-encoder-epochs",
        type=int,
        default=FREEZE_ENCODER_EPOCHS,
        help=f"Epochs to keep the encoder frozen before unfreezing (default {FREEZE_ENCODER_EPOCHS})",
    )
    parser.add_argument(
        "--unfreeze-blocks",
        type=int,
        default=None,
        help=(
            "Only let the LAST N of whisper-small's 12 encoder transformer blocks "
            "(plus the final ln_post) ever become trainable - conv1/conv2 and the "
            "earlier blocks stay permanently frozen (requires_grad=False, so backward "
            "skips them entirely - also faster). The --freeze-encoder-epochs schedule "
            "still applies on top, gating when those last-N blocks' lr turns on. "
            "Default (unset) unfreezes the whole encoder, as before."
        ),
    )
    parser.add_argument(
        "--unclear-weight",
        type=float,
        default=None,
        help=(
            "Override UNCLEAR_INPUT's loss weight after the normal inverse-frequency "
            "computation (which gives it 0.543 - the lowest of all 15 classes, tied "
            "with BACKCHANNEL_OR_NOISE/INFORMATION_REQUEST since all three hit the "
            "per-class cap - see intent_sets.py/CAP_PER_CLASS). UNCLEAR_INPUT recall "
            "is currently only 38%% despite those other two capped classes doing fine, "
            "so this targets just the one class rather than flooring all weights. "
            "Default (unset) leaves the computed weight as-is."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"Train/val/test batch size (default {BATCH_SIZE})",
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

    save_manifest(
        experiment_paths,
        intent_set=INTENT_SET.name,
        model_module=INTENT_SET.model_module,
        n_class=n_class,
        whisper_size=_TRAINING["whisper_size"],
    )

    train_dataset = HFIntentDataset(train_hf, intent_to_idx=intent_to_idx, duration_cap_s=DURATION_CAP_S)
    val_dataset = HFIntentDataset(val_hf, intent_to_idx=intent_to_idx, duration_cap_s=DURATION_CAP_S)

    train_labels = [intent_to_idx[intent] for intent in train_hf["intent"]]
    class_counts = Counter(train_labels)
    class_weights = torch.tensor(
        [len(train_labels) / (n_class * class_counts[i]) for i in range(n_class)],
        dtype=torch.float32,
    )

    if args.unclear_weight is not None:
        unclear_idx = intent_to_idx["UNCLEAR_INPUT"]
        print(f"Overriding UNCLEAR_INPUT class weight: {class_weights[unclear_idx]:.3f} -> {args.unclear_weight:.3f}")
        class_weights[unclear_idx] = args.unclear_weight

    trainloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=collate_mel_fn,
        pin_memory=True,
        persistent_workers=True,
    )

    valloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=NUM_WORKERS,
        collate_fn=collate_mel_fn,
        pin_memory=True,
        persistent_workers=True,
    )

    model = LightningModel(
        n_class=n_class,
        class_weights=class_weights,
        freeze_encoder_epochs=args.freeze_encoder_epochs,
    )

    if args.warm_start_ckpt:
        print(f"Warm-starting weights from {args.warm_start_ckpt}")
        warm_start_checkpoint = torch.load(args.warm_start_ckpt, map_location="cpu")
        warm_start_state_dict = warm_start_checkpoint.get("state_dict", warm_start_checkpoint)
        warm_start_state_dict = {
            key[len("model."):]: value
            for key, value in warm_start_state_dict.items()
            if key.startswith("model.")
        }
        model.model.load_state_dict(warm_start_state_dict)

    if args.unfreeze_blocks is not None:
        encoder = model.model.encoder
        trainable_blocks = set(encoder.blocks[-args.unfreeze_blocks:]) if args.unfreeze_blocks > 0 else set()
        for name, param in encoder.named_parameters():
            # named_parameters() gives dotted paths like "blocks.7.attn.query.weight"
            # or "conv1.weight" - only params under a trainable block, or ln_post
            # (cheap, sits right before pooling), stay adaptable; everything else
            # (conv1/conv2 + the earlier blocks) is frozen for good, not just for
            # the --freeze-encoder-epochs warmup window.
            block_idx = int(name.split(".")[1]) if name.startswith("blocks.") else None
            is_trainable = name.startswith("ln_post") or (
                block_idx is not None and encoder.blocks[block_idx] in trainable_blocks
            )
            param.requires_grad = is_trainable
        n_trainable = sum(p.requires_grad for p in encoder.parameters())
        n_total = sum(1 for _ in encoder.parameters())
        print(f"--unfreeze-blocks {args.unfreeze_blocks}: {n_trainable}/{n_total} encoder "
              f"param tensors left trainable (last {args.unfreeze_blocks} blocks + ln_post)")

    logger = MLFlowLogger(
        experiment_name=experiment_paths.mlflow_experiment_name,
        run_name=experiment_paths.run_name,
        tracking_uri=tracking_uri,
    )
    mlflow_run_id = logger.run_id

    # val/acc plateaus noisily instead of declining cleanly once the model
    # starts overfitting (it can even tick up on noise while val/loss is
    # already rising), so it never triggered EarlyStopping's patience and
    # ModelCheckpoint kept picking a later, worse-generalizing epoch. val/loss
    # showed a clean U-shape across every run we inspected - bottoming out
    # a couple epochs after unfreezing, then rising monotonically - so it's a
    # more reliable signal for both "which checkpoint is best" and "when to
    # stop".
    model_checkpoint_callback = ModelCheckpoint(
        dirpath=experiment_paths.checkpoint_dir,
        monitor='val/loss',
        mode='min',
        save_top_k=SAVE_TOP_K,
        verbose=1,
        filename=experiment_paths.checkpoint_prefix + "-epoch{epoch:02d}",
    )

    early_stopping_callback = EarlyStopping(
        monitor='val/loss',
        mode='min',
        patience=PATIENCE,
        verbose=True,
    )

    mlflow_stop_callback = MlflowStopCallback(tracking_uri, mlflow_run_id)

    trainer = Trainer(
        fast_dev_run=False,
        accelerator="gpu",
        devices=1,
        precision="bf16-mixed",
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
            batch_size=args.batch_size,
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
        # model_cls=WhisperIntentClassification (model2's, imported above) -
        # evaluate.load_model defaults to model.py's architecture, which does
        # NOT match this script's checkpoints and crashes on load_state_dict.
        eval_model = load_eval_model(best_ckpt_path, _TRAINING["whisper_size"], n_class, device, model_cls=WhisperIntentClassification)
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
