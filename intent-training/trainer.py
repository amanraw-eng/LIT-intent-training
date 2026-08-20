from model import WhisperIntentClassification
from dataset import DEFAULT_REPO_ID, HFIntentDataset, collate_mel_fn, load_hf_split

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

import os
import json
from collections import Counter
from dotenv import load_dotenv

# load WANDB_API_KEY / HF_TOKEN (and any other secrets) from .env in this directory
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

# SEED
SEED = 100
pl.seed_everything(SEED)
torch.manual_seed(SEED)

# update the wandb online/offline model and CUDA device
os.environ['WANDB_MODE'] = 'online'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

REPO_ID = DEFAULT_REPO_ID


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

    # v3: 20-intent taxonomy, pulled directly from the HF Hub (train/validation/eval splits)
    train_hf = load_hf_split(REPO_ID, "train")
    val_hf = load_hf_split(REPO_ID, "validation")
    print(f"train: {len(train_hf)} rows, validation: {len(val_hf)} rows")

    # label mapping is derived from train and frozen for this run so
    # checkpoints stay valid regardless of how the Hub dataset evolves later
    # (push_to_hub.py / infer.py read this file too)
    intents = sorted(set(train_hf["intent"]))
    intent_to_idx = {intent: idx for idx, intent in enumerate(intents)}
    n_class = len(intent_to_idx)
    print(f"n_class: {n_class}")

    os.makedirs("models", exist_ok=True)
    with open("models/intent_to_idx.json", "w", encoding="utf-8") as f:
        json.dump(intent_to_idx, f, indent=2, ensure_ascii=False)

    train_dataset = HFIntentDataset(train_hf, intent_to_idx=intent_to_idx)
    val_dataset = HFIntentDataset(val_hf, intent_to_idx=intent_to_idx)

    # class-weighted loss (inverse frequency, "balanced" formula) instead of
    # oversampling - oversampling duplicates the same rare-class waveforms
    # over and over, which memorizes those specific clips instead of
    # generalizing, and still left the dominant class as a catch-all guess
    # under uncertainty. Weighting the loss penalizes wrong guesses on rare
    # classes more, without needing to duplicate any audio.
    train_labels = [intent_to_idx[intent] for intent in train_hf["intent"]]
    class_counts = Counter(train_labels)
    class_weights = torch.tensor(
        [len(train_labels) / (n_class * class_counts[i]) for i in range(n_class)],
        dtype=torch.float32,
    )
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

    # update the logger to Wandb or Tensorboard
    run_name = "whisper-small-S2I-hi-v3"
    logger = WandbLogger(
        name=run_name,
        project='S2I-whisper'
    )

    model_checkpoint_callback = ModelCheckpoint(
            dirpath='models/checkpoints',
            monitor='val/acc',
            mode='max',
            save_top_k=3,
            verbose=1,
            filename=run_name + "-epoch{epoch:02d}")

    # v1/v2 ran a fixed 50 epochs and overfit hard (97% train / 55% val) well
    # before that - stop automatically once val/acc stops improving instead.
    early_stopping_callback = EarlyStopping(
            monitor='val/acc',
            mode='max',
            patience=5,
            verbose=True)

    trainer = Trainer(
            fast_dev_run=False, # true for dev run
            accelerator="gpu",
            devices=1,
            max_epochs=3,
            enable_checkpointing=True,
            callbacks=[
                model_checkpoint_callback,
                early_stopping_callback,
            ],
            logger=logger,
            )

    trainer.fit(model, train_dataloaders=trainloader, val_dataloaders=valloader)
