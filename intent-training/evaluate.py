import argparse
import json
import os
from collections import Counter

import matplotlib
import mlflow
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import DEFAULT_REPO_ID, HFIntentDataset, collate_mel_fn, load_hf_split
from model import WhisperIntentClassification
from experiment_config import EXPERIMENT_VERSION_ENV, get_experiment_paths

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(THIS_DIR, ".env"))
except ImportError:
    pass


def log_evaluation_to_mlflow(experiment_paths, args, ckpt_path, intent_map_path, output_dir, report_dict, eval_rows):
    """Log one completed evaluation and its local reports to MLflow."""
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        raise RuntimeError("MLFLOW_TRACKING_URI not set in .env")

    mlflow.set_tracking_uri(tracking_uri)
    if token := os.environ.get("MLFLOW_TRACKING_TOKEN"):
        os.environ["MLFLOW_TRACKING_PASSWORD"] = token

    mlflow.set_experiment(experiment_paths.mlflow_experiment_name)
    checkpoint_stem = os.path.splitext(os.path.basename(ckpt_path))[0]
    run_name = f"{experiment_paths.run_name}-evaluation-{args.split}-{checkpoint_stem}"
    metrics = {
        "eval_accuracy": float(report_dict["accuracy"]),
        "eval_macro_precision": float(report_dict["macro avg"]["precision"]),
        "eval_macro_recall": float(report_dict["macro avg"]["recall"]),
        "eval_macro_f1": float(report_dict["macro avg"]["f1-score"]),
        "eval_weighted_precision": float(report_dict["weighted avg"]["precision"]),
        "eval_weighted_recall": float(report_dict["weighted avg"]["recall"]),
        "eval_weighted_f1": float(report_dict["weighted avg"]["f1-score"]),
    }

    with mlflow.start_run(run_name=run_name):
        mlflow.set_tags({
            "run_type": "evaluation",
            "experiment_version": experiment_paths.version,
            "checkpoint_name": os.path.basename(ckpt_path),
        })
        mlflow.log_params({
            "repo_id": args.repo_id,
            "split": args.split,
            "model_type": args.model_type,
            "checkpoint_path": os.path.abspath(ckpt_path),
            "intent_map_path": os.path.abspath(intent_map_path),
            "eval_rows": eval_rows,
        })
        mlflow.log_metrics(metrics)
        mlflow.log_artifacts(output_dir, artifact_path="evaluation")

    print(f"Logged evaluation to MLflow run '{run_name}'")


# ---------------------------------------------------------------------------
# model loading
# ---------------------------------------------------------------------------

def load_model(ckpt_path, model_type, n_class, device):
    model = WhisperIntentClassification(model_type, n_class=n_class)
    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    # LightningModel wraps WhisperIntentClassification as `self.model` and also
    # registers a `class_weights` buffer for the loss - only the `model.`-
    # prefixed keys belong to this module, everything else must be dropped
    # rather than passed through unstripped (breaks strict load_state_dict).
    state_dict = {
        key[len("model."):]: value
        for key, value in state_dict.items()
        if key.startswith("model.")
    }
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def run_inference(model, loader, device):
    all_preds, all_labels = [], []
    for mels, labels in tqdm(loader, desc="Evaluating", unit="batch"):
        mels = mels.to(device)
        logits = model(mels)
        preds = logits.argmax(dim=1).cpu()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.tolist())
    return all_labels, all_preds


# ---------------------------------------------------------------------------
# reporting: class distribution, confusion matrix images, per-class accuracy
# ---------------------------------------------------------------------------

def save_class_distribution(labels, idx_to_intent, output_path):
    counts = Counter(labels)
    total = len(labels)
    rows = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"{'count':>7}  {'pct':>6}  intent\n")
        for idx, count in rows:
            f.write(f"{count:7d}  {100 * count / total:5.1f}%  {idx_to_intent[idx]}\n")
        f.write(f"\ntotal: {total}, classes_present: {len(counts)}\n")


def plot_confusion_matrix(labels, preds, idx_to_intent, output_path, normalize=False):
    class_ids = sorted(idx_to_intent.keys())
    class_names = [idx_to_intent[i] for i in class_ids]
    cm = confusion_matrix(labels, preds, labels=class_ids)
    if normalize:
        with np.errstate(all="ignore"):
            cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        cm = np.nan_to_num(cm)

    n = len(class_names)
    fig, ax = plt.subplots(figsize=(max(10, n * 0.5), max(8, n * 0.45)))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f" if normalize else "d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        cbar=True,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix" + (" (normalized)" if normalize else ""))
    plt.setp(ax.get_xticklabels(), rotation=90)
    plt.setp(ax.get_yticklabels(), rotation=0)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_per_class_accuracy(labels, preds, idx_to_intent, output_path):
    class_ids = sorted(idx_to_intent.keys())
    labels_arr = np.array(labels)
    preds_arr = np.array(preds)

    accs, supports = [], []
    for cid in class_ids:
        mask = labels_arr == cid
        support = int(mask.sum())
        supports.append(support)
        accs.append(float((preds_arr[mask] == cid).mean()) if support else 0.0)

    order = np.argsort(accs)
    class_names = [f"{idx_to_intent[class_ids[i]]} (n={supports[i]})" for i in order]
    accs_sorted = [accs[i] for i in order]

    fig, ax = plt.subplots(figsize=(10, max(6, len(class_ids) * 0.4)))
    ax.barh(class_names, accs_sorted, color="#4C72B0")
    ax.set_xlabel("Per-class accuracy")
    ax.set_xlim(0, 1)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment-version",
        help=f"Experiment version (defaults to {EXPERIMENT_VERSION_ENV} or v5)",
    )
    parser.add_argument("--repo_id", default=DEFAULT_REPO_ID)
    parser.add_argument("--split", choices=["validation", "eval"], default="eval",
                         help="Named split on the Hub to evaluate on")

    parser.add_argument("--ckpt_name", required=True, help="Checkpoint filename for this experiment, or a full path")
    parser.add_argument("--model_type", default="small")
    parser.add_argument("--intent_map_path", help="Defaults to this experiment's saved intent map")
    parser.add_argument("--output_dir", help="Defaults to this experiment's evaluation directory")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()
    experiment_paths = get_experiment_paths(args.experiment_version)
    intent_map_path = args.intent_map_path or experiment_paths.intent_map_path
    output_dir = args.output_dir or experiment_paths.eval_output_dir

    os.makedirs(output_dir, exist_ok=True)

    ckpt_path = args.ckpt_name
    if not os.path.isabs(ckpt_path) and not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(experiment_paths.checkpoint_dir, args.ckpt_name)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    with open(intent_map_path, "r", encoding="utf-8") as f:
        intent_to_idx = json.load(f)
    idx_to_intent = {int(v): k for k, v in intent_to_idx.items()}

    hf_split = load_hf_split(args.repo_id, args.split)
    print(f"Loaded {len(hf_split)} rows from {args.repo_id} (split={args.split})")

    unseen_intents = set(hf_split["intent"]) - set(intent_to_idx)
    if unseen_intents:
        raise ValueError(f"Intents in eval data missing from intent_map: {unseen_intents}")

    eval_dataset = HFIntentDataset(hf_split, intent_to_idx=intent_to_idx)
    eval_loader = DataLoader(
        eval_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_mel_fn,
    )

    eval_labels_by_idx = [intent_to_idx[intent] for intent in hf_split["intent"]]
    dist_path = os.path.join(output_dir, "eval_class_distribution.txt")
    save_class_distribution(eval_labels_by_idx, idx_to_intent, dist_path)
    print(f"Saved eval class distribution -> {dist_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(ckpt_path, args.model_type, len(intent_to_idx), device)
    labels, preds = run_inference(model, eval_loader, device)

    present_ids = sorted(set(labels) | set(preds))
    target_names = [idx_to_intent[i] for i in present_ids]

    report_text = classification_report(
        labels, preds, labels=present_ids, target_names=target_names, digits=3, zero_division=0,
    )
    with open(os.path.join(output_dir, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(report_text)
    print(report_text)

    report_dict = classification_report(
        labels, preds, labels=present_ids, target_names=target_names, digits=3, zero_division=0, output_dict=True,
    )
    with open(os.path.join(output_dir, "classification_report.json"), "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2)

    plot_confusion_matrix(labels, preds, idx_to_intent, os.path.join(output_dir, "confusion_matrix.png"), normalize=False)
    plot_confusion_matrix(labels, preds, idx_to_intent, os.path.join(output_dir, "confusion_matrix_normalized.png"), normalize=True)
    plot_per_class_accuracy(labels, preds, idx_to_intent, os.path.join(output_dir, "per_class_accuracy.png"))

    log_evaluation_to_mlflow(
        experiment_paths=experiment_paths,
        args=args,
        ckpt_path=ckpt_path,
        intent_map_path=intent_map_path,
        output_dir=output_dir,
        report_dict=report_dict,
        eval_rows=len(labels),
    )
    print(f"Saved classification report + confusion matrix images -> {output_dir}")


if __name__ == "__main__":
    main()
