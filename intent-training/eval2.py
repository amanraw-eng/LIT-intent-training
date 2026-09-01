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

AFFIRMATIVE_CLASS = "AFFIRMATIVE_ACKNOWLEDGEMENT"
UNCLEAR_CLASS = "UNCLEAR_INPUT"


def log_evaluation_to_mlflow(
    experiment_paths, args, ckpt_path, intent_map_path, output_dir, 
    report_dict, report_dict_adj, eval_rows, aff_metrics, unclear_metrics
):
    """Log evaluation results, specialized metrics, and artifact reports to MLflow."""
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
        # Overall standard metrics
        "eval_accuracy": float(report_dict["accuracy"]),
        "eval_macro_f1": float(report_dict["macro avg"]["f1-score"]),
        "eval_weighted_f1": float(report_dict["weighted avg"]["f1-score"]),
        # Adjusted metrics (excluding UNCLEAR_INPUT misses)
        "eval_adjusted_accuracy": float(report_dict_adj["accuracy"]),
        "eval_adjusted_macro_f1": float(report_dict_adj["macro avg"]["f1-score"]),
        "eval_adjusted_weighted_f1": float(report_dict_adj["weighted avg"]["f1-score"]),
        # Key Critical Metrics
        "affirmative_precision": float(aff_metrics["precision"]),
        "affirmative_false_positives": float(aff_metrics["false_positives"]),
        "unclear_recall": float(unclear_metrics["recall"]),
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


def load_model(ckpt_path, model_type, n_class, device):
    model = WhisperIntentClassification(model_type, n_class=n_class)
    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
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
def run_inference(model, loader, device, hf_split):
    """Run inference while extracting predictions, true labels, and audio IDs."""
    all_preds, all_labels, all_audio_ids = [], [], []

    # Detect the actual ID column present in the HuggingFace dataset split
    possible_id_cols = ["chunk_path"]
    id_col = next((c for c in possible_id_cols if c in hf_split.column_names), None)

    global_idx = 0

    for batch in tqdm(loader, desc="Evaluating", unit="batch"):
        # 1. Extract inputs and labels from batch structure
        if isinstance(batch, dict):
            mels = batch["mel"].to(device)
            labels = batch["label"]
            batch_audio_ids = batch.get("audio_id", None)
        else:
            mels, labels = batch[0].to(device), batch[1]
            batch_audio_ids = batch[2] if len(batch) > 2 else None

        batch_size = len(labels)

        # 2. Extract model predictions
        logits = model(mels)
        preds = logits.argmax(dim=1).cpu()

        all_preds.extend(preds.tolist())
        all_labels.extend(labels.tolist())

        # 3. Robust Audio ID Extraction:
        # Check batch first -> Fallback to HuggingFace dataset column -> Fallback to sample indices
        if batch_audio_ids is not None and any(x is not None for x in batch_audio_ids):
            if isinstance(batch_audio_ids, torch.Tensor):
                all_audio_ids.extend(batch_audio_ids.tolist())
            elif isinstance(batch_audio_ids, (list, tuple)):
                all_audio_ids.extend(list(batch_audio_ids))
            else:
                all_audio_ids.extend([batch_audio_ids])
        elif id_col is not None:
            ids_slice = hf_split[id_col][global_idx : global_idx + batch_size]
            all_audio_ids.extend([str(x) for x in ids_slice])
        else:
            all_audio_ids.extend([f"sample_{i}" for i in range(global_idx, global_idx + batch_size)])

        global_idx += batch_size

    return all_labels, all_preds, all_audio_ids


def analyze_unclear_fallbacks(labels, preds, idx_to_intent, output_path):
    """
    Computes true UNCLEAR_INPUT fallback misses and outputs an adjusted accuracy report.
    Only UNCLEAR_INPUT is considered a safe fallback (ignored execution triggers default behavior).
    """
    unclear_idx = next((k for k, v in idx_to_intent.items() if v == UNCLEAR_CLASS), None)

    class_stats = {}
    for cid in idx_to_intent.keys():
        class_stats[cid] = {"total": 0, "correct": 0, "went_to_unclear": 0}

    for true_id, pred_id in zip(labels, preds):
        class_stats[true_id]["total"] += 1
        if true_id == pred_id:
            class_stats[true_id]["correct"] += 1
        elif unclear_idx is not None and pred_id == unclear_idx:
            class_stats[true_id]["went_to_unclear"] += 1

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"{'Intent':<40} | {'Total':>6} | {'Correct':>7} | {'To Unclear':>10} | {'Raw Acc':>8} | {'Adj Acc':>8}\n")
        f.write("-" * 92 + "\n")

        for cid, stats in sorted(class_stats.items(), key=lambda x: idx_to_intent[x[0]]):
            intent_name = idx_to_intent[cid]
            tot = stats["total"]
            if tot == 0:
                continue
            corr = stats["correct"]
            to_unclear = stats["went_to_unclear"]
            raw_acc = (corr / tot) * 100

            # Adjusted total removes items misclassified as UNCLEAR_INPUT
            adj_tot = tot - to_unclear if intent_name != UNCLEAR_CLASS else tot
            adj_acc = (corr / adj_tot * 100) if adj_tot > 0 else 0.0

            f.write(f"{intent_name:<40} | {tot:6d} | {corr:7d} | {to_unclear:10d} | {raw_acc:7.1f}% | {adj_acc:7.1f}%\n")


def analyze_affirmative_false_positives(labels, preds, idx_to_intent, output_path):
    """
    Analyzes False Positives for GENERAL_AFFIRMATIVE_ACKNOWLEDGEMENT.
    High FP count here indicates severe errors where non-affirmative intents are misclassified as affirmative.
    """
    aff_idx = next((k for k, v in idx_to_intent.items() if v == AFFIRMATIVE_CLASS), None)
    if aff_idx is None:
        return {"precision": 0.0, "false_positives": 0, "total_predictions": 0}

    fps_by_true_class = Counter()
    total_aff_preds = 0
    total_aff_trues = 0
    correct_aff_preds = 0

    for true_id, pred_id in zip(labels, preds):
        if true_id == aff_idx:
            total_aff_trues += 1
        if pred_id == aff_idx:
            total_aff_preds += 1
            if true_id == aff_idx:
                correct_aff_preds += 1
            else:
                fps_by_true_class[true_id] += 1

    precision = (correct_aff_preds / total_aff_preds) if total_aff_preds > 0 else 0.0
    recall = (correct_aff_preds / total_aff_trues) if total_aff_trues > 0 else 0.0
    total_fps = sum(fps_by_true_class.values())

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"=== {AFFIRMATIVE_CLASS} FALSE POSITIVE RISK ANALYSIS ===\n\n")
        f.write(f"Total Predicted as Affirmative : {total_aff_preds}\n")
        f.write(f"Correctly Predicted            : {correct_aff_preds}\n")
        f.write(f"Total Dangerous False Positives: {total_fps}\n")
        f.write(f"Affirmative Precision         : {precision * 100:.2f}%\n")
        f.write(f"Affirmative Recall            : {recall * 100:.2f}%\n\n")
        f.write("Breakdown of True Intents incorrectly misclassified as Affirmative:\n")
        f.write(f"{'True Intent':<40} | {'FP Count':>8} | {'% of Total FPs':>14}\n")
        f.write("-" * 68 + "\n")

        for true_id, count in fps_by_true_class.most_common():
            pct = (count / total_fps * 100) if total_fps > 0 else 0.0
            f.write(f"{idx_to_intent[true_id]:<40} | {count:8d} | {pct:13.1f}%\n")

    return {
        "precision": precision,
        "recall": recall,
        "false_positives": total_fps,
        "total_predictions": total_aff_preds,
    }


def analyze_unclear_recall(labels, preds, idx_to_intent):
    """Extract precision/recall for UNCLEAR_INPUT."""
    unclear_idx = next((k for k, v in idx_to_intent.items() if v == UNCLEAR_CLASS), None)
    if unclear_idx is None:
        return {"precision": 0.0, "recall": 0.0}

    tp, fp, fn = 0, 0, 0
    for true_id, pred_id in zip(labels, preds):
        if true_id == unclear_idx and pred_id == unclear_idx:
            tp += 1
        elif true_id != unclear_idx and pred_id == unclear_idx:
            fp += 1
        elif true_id == unclear_idx and pred_id != unclear_idx:
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return {"precision": precision, "recall": recall}


def save_class_distribution(labels, idx_to_intent, output_path):
    counts = Counter(labels)
    total = len(labels)
    rows = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"{'count':>7}  {'pct':>6}  intent\n")
        for idx, count in rows:
            f.write(f"{count:7d}  {100 * count / total:5.1f}%  {idx_to_intent[idx]}\n")
        f.write(f"\ntotal: {total}, classes_present: {len(counts)}\n")


def write_confusion_to_md(class_ids, class_names, out_dir, cm, title_suffix, labels=None, preds=None, audio_ids=None):
    os.makedirs(out_dir, exist_ok=True)
    suffix = title_suffix.lower().replace(" ", "_").replace("(", "").replace(")", "").strip("_")
    filename_suffix = f"_{suffix}" if suffix else ""
    
    md_confusion_matrix_path = f'{out_dir}/matrix{filename_suffix}.md'
    json_errors_path = f'{out_dir}/highest_confusion{filename_suffix}.json'

    # 1. Build Markdown Table
    headers = ["Actual / Predicted"] + class_names
    header_row = "| " + " | ".join(headers) + " |\n"
    separator_row = "| " + " | ".join(["---"] * len(headers)) + " |\n"

    table_rows = []
    matrix_list = cm.tolist()

    for i, row in enumerate(matrix_list):
        true_class_label = class_names[i]
        row_strings = [str(val) for val in row]
        table_rows.append(f"| {true_class_label} | " + " | ".join(row_strings) + " |\n")

    md_matrix = header_row + separator_row + "".join(table_rows)

    with open(md_confusion_matrix_path, 'w', encoding="utf-8") as f:
        f.write(md_matrix)

    # 2. Extract misclassified audio IDs per confusion pair
    pair_audio_map = {}
    if labels is not None and preds is not None and audio_ids is not None:
        idx_to_name = dict(zip(class_ids, class_names))
        for true_id, pred_id, audio_id in zip(labels, preds, audio_ids):
            if true_id != pred_id:
                pair_key = (idx_to_name[true_id], idx_to_name[pred_id])
                pair_audio_map.setdefault(pair_key, []).append(audio_id)

    # 3. Save sorted errors with wrong audio IDs to JSON
    confusion_pairs = []

    for i, row in enumerate(matrix_list):
        actual_class = class_names[i]
        total_samples = sum(row)
        
        if total_samples == 0:
            continue
            
        for j, count in enumerate(row):
            predicted_class = class_names[j]
            
            if i != j and count > 0:
                confusion_percent = (count / total_samples) * 100
                wrong_audios = pair_audio_map.get((actual_class, predicted_class), [])
                
                confusion_pairs.append({
                    "pair": f"{actual_class} -> {predicted_class}",
                    "confusion_percent": round(confusion_percent, 2),
                    "actual_class": actual_class,
                    "predicted_class": predicted_class,
                    "sample_count": count,
                    "audio_ids": wrong_audios
                })

    confusion_pairs.sort(key=lambda x: x["confusion_percent"], reverse=True)

    with open(json_errors_path, 'w', encoding="utf-8") as f:
        json.dump(confusion_pairs, f, indent=4)


def plot_confusion_matrix(labels, preds, idx_to_intent, output_path, audio_ids=None, normalize=False, title_suffix=""):
    class_ids = sorted(idx_to_intent.keys())
    class_names = [idx_to_intent[i] for i in class_ids]
    cm = confusion_matrix(labels, preds, labels=class_ids)
    out_dir = os.path.dirname(output_path)

    write_confusion_to_md(
        class_ids=class_ids, 
        class_names=class_names, 
        out_dir=out_dir, 
        cm=cm, 
        title_suffix=title_suffix,
        labels=labels,
        preds=preds,
        audio_ids=audio_ids
    )

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
    ax.set_title(f"Confusion Matrix{title_suffix}" + (" (normalized)" if normalize else ""))
    plt.setp(ax.get_xticklabels(), rotation=90)
    plt.setp(ax.get_yticklabels(), rotation=0)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_per_class_accuracy(labels, preds, idx_to_intent, output_path, title="Per-class accuracy"):
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
    ax.set_xlabel(title)
    ax.set_xlim(0, 1)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(ckpt_path, args.model_type, len(intent_to_idx), device)
    
    # Run inference and retain audio IDs via direct fallback onto hf_split
    labels, preds, audio_ids = run_inference(model, eval_loader, device, hf_split)

    present_ids = sorted(set(labels) | set(preds))
    target_names = [idx_to_intent[i] for i in present_ids]

    # --- Standard Classification Reports ---
    report_text = classification_report(
        labels, preds, labels=present_ids, target_names=target_names, digits=3, zero_division=0,
    )
    with open(os.path.join(output_dir, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(report_text)

    report_dict = classification_report(
        labels, preds, labels=present_ids, target_names=target_names, digits=3, zero_division=0, output_dict=True,
    )
    with open(os.path.join(output_dir, "classification_report.json"), "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2)

    plot_confusion_matrix(labels, preds, idx_to_intent, os.path.join(output_dir, "confusion_matrix.png"), audio_ids=audio_ids, normalize=False)
    plot_confusion_matrix(labels, preds, idx_to_intent, os.path.join(output_dir, "confusion_matrix_normalized.png"), audio_ids=audio_ids, normalize=True)
    plot_per_class_accuracy(labels, preds, idx_to_intent, os.path.join(output_dir, "per_class_accuracy.png"))

    # --- Specific Metric Analyses ---
    aff_metrics = analyze_affirmative_false_positives(
        labels, preds, idx_to_intent, os.path.join(output_dir, "affirmative_false_positives.txt")
    )
    unclear_metrics = analyze_unclear_recall(labels, preds, idx_to_intent)
    analyze_unclear_fallbacks(labels, preds, idx_to_intent, os.path.join(output_dir, "unclear_fallback_breakdown.txt"))

    # --- Adjusted Metrics (Removing ONLY UNCLEAR_INPUT Misclassifications) ---
    unclear_idx = next((k for k, v in idx_to_intent.items() if v == UNCLEAR_CLASS), None)
    
    adj_labels, adj_preds, adj_audio_ids = [], [], []
    for l, p, a_id in zip(labels, preds, audio_ids):
        if idx_to_intent[l] != UNCLEAR_CLASS and p == unclear_idx:
            continue
        adj_labels.append(l)
        adj_preds.append(p)
        adj_audio_ids.append(a_id)

    adj_present_ids = sorted(set(adj_labels) | set(adj_preds))
    adj_target_names = [idx_to_intent[i] for i in adj_present_ids]

    report_text_adj = classification_report(
        adj_labels, adj_preds, labels=adj_present_ids, target_names=adj_target_names, digits=3, zero_division=0,
    )
    with open(os.path.join(output_dir, "classification_report_adjusted.txt"), "w", encoding="utf-8") as f:
        f.write(report_text_adj)

    report_dict_adj = classification_report(
        adj_labels, adj_preds, labels=adj_present_ids, target_names=adj_target_names, digits=3, zero_division=0, output_dict=True,
    )
    with open(os.path.join(output_dir, "classification_report_adjusted.json"), "w", encoding="utf-8") as f:
        json.dump(report_dict_adj, f, indent=2)

    plot_confusion_matrix(adj_labels, adj_preds, idx_to_intent, os.path.join(output_dir, "confusion_matrix_adjusted.png"), audio_ids=adj_audio_ids, normalize=False, title_suffix="Adjusted")
    plot_confusion_matrix(adj_labels, adj_preds, idx_to_intent, os.path.join(output_dir, "confusion_matrix_adjusted_normalized.png"), audio_ids=adj_audio_ids, normalize=True, title_suffix="Adjusted")
    plot_per_class_accuracy(adj_labels, adj_preds, idx_to_intent, os.path.join(output_dir, "per_class_accuracy_adjusted.png"), title="Adjusted per-class accuracy")

    log_evaluation_to_mlflow(
        experiment_paths=experiment_paths,
        args=args,
        ckpt_path=ckpt_path,
        intent_map_path=intent_map_path,
        output_dir=output_dir,
        report_dict=report_dict,
        report_dict_adj=report_dict_adj,
        eval_rows=len(labels),
        aff_metrics=aff_metrics,
        unclear_metrics=unclear_metrics,
    )
    print(f"Evaluation complete. Reports generated in '{output_dir}'.")


if __name__ == "__main__":
    main()