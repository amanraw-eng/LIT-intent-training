import argparse
import json
import os
from collections import Counter

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import seaborn as sns
import torch

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
)

from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import (
    DEFAULT_REPO_ID,
    HFIntentDataset,
    collate_mel_fn,
    load_hf_split,
)

from model import WhisperIntentClassification

from experiment_config import (
    EXPERIMENT_VERSION_ENV,
    get_experiment_paths,
)


# ============================================================
# CONSTANTS
# ============================================================

THIS_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

try:
    from dotenv import load_dotenv

    load_dotenv(
        os.path.join(
            THIS_DIR,
            ".env",
        )
    )
except ImportError:
    pass


AFFIRMATIVE_CLASS = "AFFIRMATIVE_ACKNOWLEDGEMENT"


UNCLEAR_CLASS = "UNCLEAR_INPUT"


BACKCHANNEL_CLASS = "BACKCHANNEL_OR_NOISE"



# ============================================================
# DEFAULT PRODUCTION THRESHOLDS
# ============================================================

# These are intentionally conservative because a wrong bypass
# is more dangerous than falling back to UNCLEAR_INPUT.

DEFAULT_BACKCHANNEL_THRESHOLD = 0.95
DEFAULT_BYPASS_THRESHOLD = 0.95

# UNCLEAR itself is the safe fallback and is therefore not
# confidence-gated by the production threshold.
DEFAULT_UNCLEAR_THRESHOLD = 0.0


# ============================================================
# MODEL
# ============================================================

def load_model(
    ckpt_path,
    model_type,
    n_class,
    device,
):
    model = WhisperIntentClassification(
        model_type,
        n_class=n_class,
    )

    checkpoint = torch.load(
        ckpt_path,
        map_location=device,
    )

    state_dict = (
        checkpoint["state_dict"]
        if "state_dict" in checkpoint
        else checkpoint
    )

    state_dict = {
        key[len("model."):]: value
        for key, value in state_dict.items()
        if key.startswith("model.")
    }

    model.load_state_dict(
        state_dict
    )

    model.to(device)
    model.eval()

    return model


# ============================================================
# INFERENCE
# ============================================================

@torch.no_grad()
def run_inference(
    model,
    loader,
    device,
    hf_split,
):
    """
    Returns:
        labels
        raw_preds
        confidences
        audio_ids
    """

    all_labels = []
    all_preds = []
    all_confidences = []
    all_audio_ids = []

    # Prefer the actual unique row ID.
    possible_id_cols = [
        "id",
        "chunk_path",
        "oid",
        "recording_url",
    ]

    id_col = next(
        (
            c
            for c in possible_id_cols
            if c in hf_split.column_names
        ),
        None,
    )

    global_idx = 0

    for batch in tqdm(
        loader,
        desc="Evaluating",
        unit="batch",
    ):

        if isinstance(
            batch,
            dict,
        ):

            mels = batch[
                "mel"
            ].to(device)

            labels = batch[
                "label"
            ]

            batch_audio_ids = batch.get(
                "audio_id"
            )

        else:

            mels = batch[
                0
            ].to(device)

            labels = batch[
                1
            ]

            batch_audio_ids = (
                batch[2]
                if len(batch) > 2
                else None
            )

        batch_size = len(
            labels
        )

        # ----------------------------------------------------
        # Logits
        # ----------------------------------------------------

        logits = model(
            mels
        )

        probabilities = torch.softmax(
            logits,
            dim=1,
        )

        confidence, preds = (
            probabilities.max(
                dim=1
            )
        )

        preds = (
            preds
            .cpu()
            .tolist()
        )

        confidence = (
            confidence
            .cpu()
            .tolist()
        )

        labels_list = (
            labels
            .tolist()
        )

        all_preds.extend(
            preds
        )

        all_labels.extend(
            labels_list
        )

        all_confidences.extend(
            confidence
        )

        # ----------------------------------------------------
        # Audio IDs
        # ----------------------------------------------------

        if (
            batch_audio_ids is not None
            and any(
                x is not None
                for x in batch_audio_ids
            )
        ):

            if isinstance(
                batch_audio_ids,
                torch.Tensor,
            ):
                all_audio_ids.extend(
                    batch_audio_ids.tolist()
                )

            elif isinstance(
                batch_audio_ids,
                (list, tuple),
            ):
                all_audio_ids.extend(
                    list(batch_audio_ids)
                )

            else:
                all_audio_ids.extend(
                    [batch_audio_ids]
                )

        elif id_col is not None:

            ids_slice = hf_split[
                id_col
            ][
                global_idx:
                global_idx + batch_size
            ]

            all_audio_ids.extend(
                [
                    str(x)
                    for x in ids_slice
                ]
            )

        else:

            all_audio_ids.extend(
                [
                    f"sample_{i}"
                    for i in range(
                        global_idx,
                        global_idx + batch_size,
                    )
                ]
            )

        global_idx += batch_size

    return (
        all_labels,
        all_preds,
        all_confidences,
        all_audio_ids,
    )


# ============================================================
# THRESHOLDING
# ============================================================

def build_thresholds(
    idx_to_intent,
    backchannel_threshold,
    bypass_threshold,
    unclear_threshold=0.0,
):
    """
    Create per-class confidence thresholds.

    BACKCHANNEL gets the highest threshold.
    Other bypass intents get the normal high threshold.
    UNCLEAR is not gated.
    """

    thresholds = {}

    for idx, intent in idx_to_intent.items():

        if intent == UNCLEAR_CLASS:

            thresholds[idx] = (
                unclear_threshold
            )

        elif intent == BACKCHANNEL_CLASS:

            thresholds[idx] = (
                backchannel_threshold
            )

        else:

            thresholds[idx] = (
                bypass_threshold
            )

    return thresholds


def apply_confidence_threshold(
    preds,
    confidences,
    idx_to_intent,
    thresholds,
):
    """
    Any predicted bypass intent below its required confidence
    is converted to UNCLEAR_INPUT.

    This is exactly the production behavior we want:
    uncertain bypass -> safe fallback.
    """

    unclear_idx = next(
        idx
        for idx, intent
        in idx_to_intent.items()
        if intent == UNCLEAR_CLASS
    )

    thresholded_preds = []

    for pred, confidence in zip(
        preds,
        confidences,
    ):

        threshold = thresholds.get(
            pred,
            1.0,
        )

        if confidence >= threshold:
            thresholded_preds.append(
                pred
            )
        else:
            thresholded_preds.append(
                unclear_idx
            )

    return thresholded_preds


# ============================================================
# PRODUCTION METRICS
# ============================================================

def calculate_per_class_metrics(
    labels,
    preds,
    idx_to_intent,
):
    results = {}

    for idx, intent in idx_to_intent.items():

        tp = 0
        fp = 0
        fn = 0

        for true_id, pred_id in zip(
            labels,
            preds,
        ):

            if pred_id == idx:

                if true_id == idx:
                    tp += 1
                else:
                    fp += 1

            elif true_id == idx:

                fn += 1

        precision = (
            tp / (tp + fp)
            if tp + fp
            else 0.0
        )

        recall = (
            tp / (tp + fn)
            if tp + fn
            else 0.0
        )

        results[intent] = {
            "precision": precision,
            "recall": recall,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
        }

    return results


def calculate_unclear_metrics(
    labels,
    preds,
    idx_to_intent,
):
    unclear_idx = next(
        idx
        for idx, intent
        in idx_to_intent.items()
        if intent == UNCLEAR_CLASS
    )

    tp = 0
    fp = 0
    fn = 0

    for true_id, pred_id in zip(
        labels,
        preds,
    ):

        if (
            true_id == unclear_idx
            and pred_id == unclear_idx
        ):
            tp += 1

        elif (
            true_id != unclear_idx
            and pred_id == unclear_idx
        ):
            fp += 1

        elif (
            true_id == unclear_idx
            and pred_id != unclear_idx
        ):
            fn += 1

    precision = (
        tp / (tp + fp)
        if tp + fp
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn
        else 0.0
    )

    return {
        "precision": precision,
        "recall": recall,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
    }


def calculate_bypass_precision(
    labels,
    preds,
    idx_to_intent,
):
    """
    Among all non-UNCLEAR predictions, how many are correct?

    This is a key production metric because every such prediction
    attempts to bypass the normal pipeline.
    """

    unclear_idx = next(
        idx
        for idx, intent
        in idx_to_intent.items()
        if intent == UNCLEAR_CLASS
    )

    total_bypass_predictions = 0
    correct_bypass_predictions = 0

    for true_id, pred_id in zip(
        labels,
        preds,
    ):

        if pred_id == unclear_idx:
            continue

        total_bypass_predictions += 1

        if true_id == pred_id:
            correct_bypass_predictions += 1

    precision = (
        correct_bypass_predictions
        / total_bypass_predictions
        if total_bypass_predictions
        else 0.0
    )

    return {
        "precision": precision,
        "total_predictions": total_bypass_predictions,
        "correct_predictions": correct_bypass_predictions,
    }


# ============================================================
# BACKCHANNEL SAFETY METRICS
# ============================================================

def analyze_backchannel(
    labels,
    preds,
    idx_to_intent,
    audio_ids,
    output_path,
):
    backchannel_idx = next(
        (
            idx
            for idx, intent
            in idx_to_intent.items()
            if intent == BACKCHANNEL_CLASS
        ),
        None,
    )

    if backchannel_idx is None:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "false_positives": 0,
            "false_positive_ids": [],
        }

    tp = 0
    fp = 0
    fn = 0

    fp_by_true_class = Counter()
    fp_audio_ids = []

    for true_id, pred_id, audio_id in zip(
        labels,
        preds,
        audio_ids,
    ):

        if pred_id == backchannel_idx:

            if true_id == backchannel_idx:
                tp += 1
            else:
                fp += 1
                fp_by_true_class[
                    idx_to_intent[true_id]
                ] += 1
                fp_audio_ids.append(
                    audio_id
                )

        elif true_id == backchannel_idx:

            fn += 1

    precision = (
        tp / (tp + fp)
        if tp + fp
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn
        else 0.0
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "=== BACKCHANNEL_OR_NOISE SAFETY ===\n\n"
        )

        f.write(
            f"Precision              : "
            f"{precision * 100:.3f}%\n"
        )

        f.write(
            f"Recall                 : "
            f"{recall * 100:.3f}%\n"
        )

        f.write(
            f"True positives         : {tp}\n"
        )

        f.write(
            f"False positives        : {fp}\n"
        )

        f.write(
            f"False negatives        : {fn}\n\n"
        )

        f.write(
            "False positives by actual class:\n"
        )

        for name, count in (
            fp_by_true_class.most_common()
        ):

            f.write(
                f"{name}: {count}\n"
            )

        f.write(
            "\nFalse-positive audio IDs:\n"
        )

        for audio_id in fp_audio_ids:

            f.write(
                f"{audio_id}\n"
            )

    return {
        "precision": precision,
        "recall": recall,
        "false_positives": fp,
        "true_positives": tp,
        "false_negatives": fn,
        "false_positive_ids": fp_audio_ids,
    }


# ============================================================
# ADJUSTED METRICS
# ============================================================

def build_adjusted_predictions(
    labels,
    preds,
    audio_ids,
    idx_to_intent,
):
    """
    Remove ONLY:

        true != UNCLEAR
        predicted == UNCLEAR

    These are safe-fallback misses and are excluded from the
    adjusted view.

    True UNCLEAR samples are retained.
    All other mistakes are retained.
    """

    unclear_idx = next(
        idx
        for idx, intent
        in idx_to_intent.items()
        if intent == UNCLEAR_CLASS
    )

    adj_labels = []
    adj_preds = []
    adj_audio_ids = []

    removed = 0

    for true_id, pred_id, audio_id in zip(
        labels,
        preds,
        audio_ids,
    ):

        if (
            true_id != unclear_idx
            and pred_id == unclear_idx
        ):

            removed += 1
            continue

        adj_labels.append(
            true_id
        )

        adj_preds.append(
            pred_id
        )

        adj_audio_ids.append(
            audio_id
        )

    return (
        adj_labels,
        adj_preds,
        adj_audio_ids,
        removed,
    )


# ============================================================
# SAVE DISTRIBUTION
# ============================================================

def save_class_distribution(
    labels,
    idx_to_intent,
    output_path,
):
    counts = Counter(
        labels
    )

    total = len(
        labels
    )

    rows = sorted(
        counts.items(),
        key=lambda kv: kv[1],
        reverse=True,
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            f"{'count':>8} "
            f"{'pct':>7} "
            f"intent\n"
        )

        for idx, count in rows:

            f.write(
                f"{count:8d} "
                f"{100 * count / total:6.2f}% "
                f"{idx_to_intent[idx]}\n"
            )

        f.write(
            f"\ntotal={total}, "
            f"classes_present={len(counts)}\n"
        )


# ============================================================
# CONFUSION MATRIX ARTIFACTS
# ============================================================

def write_confusion_to_md(
    class_ids,
    class_names,
    out_dir,
    cm,
    title_suffix,
    labels=None,
    preds=None,
    audio_ids=None,
):
    os.makedirs(
        out_dir,
        exist_ok=True,
    )

    suffix = (
        title_suffix
        .lower()
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .strip("_")
    )

    filename_suffix = (
        f"_{suffix}"
        if suffix
        else ""
    )

    md_path = os.path.join(
        out_dir,
        f"matrix{filename_suffix}.md",
    )

    json_path = os.path.join(
        out_dir,
        f"highest_confusion{filename_suffix}.json",
    )

    headers = [
        "Actual / Predicted"
    ] + class_names

    header_row = (
        "| "
        + " | ".join(headers)
        + " |\n"
    )

    separator_row = (
        "| "
        + " | ".join(
            ["---"] * len(headers)
        )
        + " |\n"
    )

    table_rows = []

    matrix_list = cm.tolist()

    for i, row in enumerate(
        matrix_list
    ):

        table_rows.append(
            "| "
            + class_names[i]
            + " | "
            + " | ".join(
                str(x)
                for x in row
            )
            + " |\n"
        )

    with open(
        md_path,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            header_row
        )

        f.write(
            separator_row
        )

        f.writelines(
            table_rows
        )

    pair_audio_map = {}

    if (
        labels is not None
        and preds is not None
        and audio_ids is not None
    ):

        idx_to_name = dict(
            zip(
                class_ids,
                class_names,
            )
        )

        for true_id, pred_id, audio_id in zip(
            labels,
            preds,
            audio_ids,
        ):

            if true_id == pred_id:
                continue

            pair = (
                idx_to_name[true_id],
                idx_to_name[pred_id],
            )

            pair_audio_map.setdefault(
                pair,
                [],
            ).append(
                audio_id
            )

    confusion_pairs = []

    for i, row in enumerate(
        matrix_list
    ):

        actual_class = class_names[i]

        total_samples = sum(
            row
        )

        if total_samples == 0:
            continue

        for j, count in enumerate(
            row
        ):

            if i == j or count == 0:
                continue

            predicted_class = (
                class_names[j]
            )

            confusion_pairs.append(
                {
                    "pair": (
                        f"{actual_class} -> "
                        f"{predicted_class}"
                    ),
                    "confusion_percent": round(
                        count
                        / total_samples
                        * 100,
                        2,
                    ),
                    "actual_class": actual_class,
                    "predicted_class": predicted_class,
                    "sample_count": count,
                    "audio_ids": pair_audio_map.get(
                        (
                            actual_class,
                            predicted_class,
                        ),
                        [],
                    ),
                }
            )

    confusion_pairs.sort(
        key=lambda x: (
            x["confusion_percent"],
            x["sample_count"],
        ),
        reverse=True,
    )

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            confusion_pairs,
            f,
            indent=4,
            ensure_ascii=False,
        )


def plot_confusion_matrix(
    labels,
    preds,
    idx_to_intent,
    output_path,
    audio_ids=None,
    normalize=False,
    title_suffix="",
):
    class_ids = sorted(
        idx_to_intent.keys()
    )

    class_names = [
        idx_to_intent[i]
        for i in class_ids
    ]

    cm = confusion_matrix(
        labels,
        preds,
        labels=class_ids,
    )

    out_dir = os.path.dirname(
        output_path
    )

    write_confusion_to_md(
        class_ids=class_ids,
        class_names=class_names,
        out_dir=out_dir,
        cm=cm,
        title_suffix=title_suffix,
        labels=labels,
        preds=preds,
        audio_ids=audio_ids,
    )

    if normalize:

        with np.errstate(
            all="ignore"
        ):

            cm_plot = (
                cm.astype(float)
                / cm.sum(
                    axis=1,
                    keepdims=True,
                )
            )

        cm_plot = np.nan_to_num(
            cm_plot
        )

    else:

        cm_plot = cm

    n = len(
        class_names
    )

    fig, ax = plt.subplots(
        figsize=(
            max(
                10,
                n * 0.5,
            ),
            max(
                8,
                n * 0.45,
            ),
        )
    )

    sns.heatmap(
        cm_plot,
        annot=True,
        fmt=(
            ".2f"
            if normalize
            else "d"
        ),
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        cbar=True,
    )

    ax.set_xlabel(
        "Predicted"
    )

    ax.set_ylabel(
        "True"
    )

    ax.set_title(
        f"Confusion Matrix"
        f"{title_suffix}"
        + (
            " (normalized)"
            if normalize
            else ""
        )
    )

    plt.setp(
        ax.get_xticklabels(),
        rotation=90,
    )

    plt.setp(
        ax.get_yticklabels(),
        rotation=0,
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=150,
    )

    plt.close(fig)


# ============================================================
# PER CLASS ACCURACY
# ============================================================

def plot_per_class_accuracy(
    labels,
    preds,
    idx_to_intent,
    output_path,
    title="Per-class accuracy",
):
    class_ids = sorted(
        idx_to_intent.keys()
    )

    labels_arr = np.array(
        labels
    )

    preds_arr = np.array(
        preds
    )

    accs = []
    supports = []

    for cid in class_ids:

        mask = (
            labels_arr == cid
        )

        support = int(
            mask.sum()
        )

        supports.append(
            support
        )

        accs.append(
            float(
                (
                    preds_arr[mask]
                    == cid
                ).mean()
            )
            if support
            else 0.0
        )

    order = np.argsort(
        accs
    )

    class_names = [
        (
            f"{idx_to_intent[class_ids[i]]}"
            f" (n={supports[i]})"
        )
        for i in order
    ]

    accs_sorted = [
        accs[i]
        for i in order
    ]

    fig, ax = plt.subplots(
        figsize=(
            10,
            max(
                6,
                len(class_ids) * 0.4,
            ),
        )
    )

    ax.barh(
        class_names,
        accs_sorted,
        color="#4C72B0",
    )

    ax.set_xlabel(
        title
    )

    ax.set_xlim(
        0,
        1,
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=150,
    )

    plt.close(fig)


# ============================================================
# REPORT WRITER
# ============================================================

def write_production_report(
    labels,
    preds,
    idx_to_intent,
    output_path,
    thresholds=None,
):
    metrics = calculate_per_class_metrics(
        labels,
        preds,
        idx_to_intent,
    )

    unclear = calculate_unclear_metrics(
        labels,
        preds,
        idx_to_intent,
    )

    bypass = calculate_bypass_precision(
        labels,
        preds,
        idx_to_intent,
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "=== PRODUCTION METRICS ===\n\n"
        )

        f.write(
            "Priority:\n"
            "1. BACKCHANNEL precision\n"
            "2. Other bypass precision\n"
            "3. UNCLEAR recall\n"
            "4. Recall/F1\n\n"
        )

        if thresholds is not None:

            f.write(
                "Confidence thresholds:\n"
            )

            for idx in sorted(
                thresholds
            ):

                f.write(
                    f"{idx_to_intent[idx]}: "
                    f"{thresholds[idx]:.3f}\n"
                )

            f.write("\n")

        f.write(
            f"Overall bypass precision: "
            f"{bypass['precision'] * 100:.3f}%\n"
        )

        f.write(
            f"UNCLEAR recall: "
            f"{unclear['recall'] * 100:.3f}%\n\n"
        )

        f.write(
            f"{'Intent':<40} | "
            f"{'Precision':>10} | "
            f"{'Recall':>10} | "
            f"{'FP':>8} | "
            f"{'TP':>8} | "
            f"{'FN':>8}\n"
        )

        f.write(
            "-" * 100
            + "\n"
        )

        # Highest-risk classes first.
        ordered = []

        for intent_name in (
            BACKCHANNEL_CLASS,
            UNCLEAR_CLASS,
        ):

            if intent_name in metrics:
                ordered.append(
                    intent_name
                )

        for intent_name in sorted(
            metrics
        ):

            if intent_name not in ordered:
                ordered.append(
                    intent_name
                )

        for intent_name in ordered:

            m = metrics[
                intent_name
            ]

            f.write(
                f"{intent_name:<40} | "
                f"{m['precision'] * 100:9.3f}% | "
                f"{m['recall'] * 100:9.3f}% | "
                f"{m['false_positives']:8d} | "
                f"{m['true_positives']:8d} | "
                f"{m['false_negatives']:8d}\n"
            )

    return metrics


# ============================================================
# MLflow
# ============================================================

def log_evaluation_to_mlflow(
    experiment_paths,
    args,
    ckpt_path,
    intent_map_path,
    output_dir,
    raw_report,
    adjusted_report,
    production_metrics,
    backchannel_metrics,
    unclear_metrics,
    bypass_metrics,
    gated_production_metrics,
    gated_backchannel_metrics,
    gated_unclear_metrics,
    gated_bypass_metrics,
    eval_rows,
):
    tracking_uri = os.environ.get(
        "MLFLOW_TRACKING_URI"
    )

    if not tracking_uri:
        raise RuntimeError(
            "MLFLOW_TRACKING_URI not set in .env"
        )

    mlflow.set_tracking_uri(
        tracking_uri
    )

    if token := os.environ.get(
        "MLFLOW_TRACKING_TOKEN"
    ):
        os.environ[
            "MLFLOW_TRACKING_PASSWORD"
        ] = token

    mlflow.set_experiment(
        experiment_paths.mlflow_experiment_name
    )

    checkpoint_stem = (
        os.path.splitext(
            os.path.basename(
                ckpt_path
            )
        )[0]
    )

    run_name = (
        f"{experiment_paths.run_name}"
        f"-evaluation2-"
        f"{args.split}-"
        f"{checkpoint_stem}"
    )

    metrics = {
        # ----------------------------------------------------
        # Raw
        # ----------------------------------------------------

        "eval_accuracy": float(
            raw_report["accuracy"]
        ),

        "eval_macro_f1": float(
            raw_report[
                "macro avg"
            ]["f1-score"]
        ),

        "eval_weighted_f1": float(
            raw_report[
                "weighted avg"
            ]["f1-score"]
        ),

        # ----------------------------------------------------
        # Adjusted
        # ----------------------------------------------------

        "eval_adjusted_accuracy": float(
            adjusted_report["accuracy"]
        ),

        "eval_adjusted_macro_f1": float(
            adjusted_report[
                "macro avg"
            ]["f1-score"]
        ),

        "eval_adjusted_weighted_f1": float(
            adjusted_report[
                "weighted avg"
            ]["f1-score"]
        ),

        # ----------------------------------------------------
        # Raw production
        # ----------------------------------------------------

        "bypass_precision": float(
            bypass_metrics["precision"]
        ),

        "unclear_recall": float(
            unclear_metrics["recall"]
        ),

        "backchannel_precision": float(
            backchannel_metrics["precision"]
        ),

        "backchannel_false_positives": float(
            backchannel_metrics[
                "false_positives"
            ]
        ),

        # ----------------------------------------------------
        # Thresholded production
        # ----------------------------------------------------

        "gated_bypass_precision": float(
            gated_bypass_metrics[
                "precision"
            ]
        ),

        "gated_unclear_recall": float(
            gated_unclear_metrics[
                "recall"
            ]
        ),

        "gated_backchannel_precision": float(
            gated_backchannel_metrics[
                "precision"
            ]
        ),

        "gated_backchannel_false_positives": float(
            gated_backchannel_metrics[
                "false_positives"
            ]
        ),
    }

    # Log per-intent gated precision.
    for intent_name, item in (
        gated_production_metrics.items()
    ):

        metrics[
            f"gated_{intent_name.lower()}_precision"
        ] = float(
            item["precision"]
        )

        metrics[
            f"gated_{intent_name.lower()}_false_positives"
        ] = float(
            item["false_positives"]
        )

    with mlflow.start_run(
        run_name=run_name
    ):

        mlflow.set_tags(
            {
                "run_type": "evaluation2",
                "experiment_version": (
                    experiment_paths.version
                ),
                "checkpoint_name": (
                    os.path.basename(
                        ckpt_path
                    )
                ),
            }
        )

        mlflow.log_params(
            {
                "repo_id": args.repo_id,
                "split": args.split,
                "model_type": args.model_type,
                "checkpoint_path": os.path.abspath(
                    ckpt_path
                ),
                "intent_map_path": os.path.abspath(
                    intent_map_path
                ),
                "eval_rows": eval_rows,
                "backchannel_threshold": (
                    args.backchannel_threshold
                ),
                "bypass_threshold": (
                    args.bypass_threshold
                ),
            }
        )

        mlflow.log_metrics(
            metrics
        )

        mlflow.log_artifacts(
            output_dir,
            artifact_path="evaluation2",
        )

    print(
        f"Logged evaluation to MLflow run "
        f"'{run_name}'"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--experiment-version",
        help=(
            f"Experiment version "
            f"(defaults to "
            f"{EXPERIMENT_VERSION_ENV} "
            f"or v5)"
        ),
    )

    parser.add_argument(
        "--repo_id",
        default=DEFAULT_REPO_ID,
    )

    parser.add_argument(
        "--split",
        choices=[
            "validation",
            "eval",
        ],
        default="eval",
    )

    parser.add_argument(
        "--ckpt_name",
        required=True,
    )

    parser.add_argument(
        "--model_type",
        default="small",
    )

    parser.add_argument(
        "--intent_map_path",
    )

    parser.add_argument(
        "--output_dir",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
    )

    # --------------------------------------------------------
    # Production thresholds
    # --------------------------------------------------------

    parser.add_argument(
        "--backchannel-threshold",
        type=float,
        default=DEFAULT_BACKCHANNEL_THRESHOLD,
        help=(
            "Minimum confidence to allow "
            "BACKCHANNEL_OR_NOISE. "
            "Default=0.95"
        ),
    )

    parser.add_argument(
        "--bypass-threshold",
        type=float,
        default=DEFAULT_BYPASS_THRESHOLD,
        help=(
            "Minimum confidence for other "
            "bypass intents. "
            "Default=0.95"
        ),
    )

    args = parser.parse_args()

    # ========================================================
    # EXPERIMENT PATHS
    # ========================================================

    experiment_paths = get_experiment_paths(
        args.experiment_version
    )

    intent_map_path = (
        args.intent_map_path
        or experiment_paths.intent_map_path
    )

    output_dir = (
        args.output_dir
        or experiment_paths.eval_output_dir
    )

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    # ========================================================
    # CHECKPOINT
    # ========================================================

    ckpt_path = args.ckpt_name

    if (
        not os.path.isabs(
            ckpt_path
        )
        and not os.path.exists(
            ckpt_path
        )
    ):

        ckpt_path = os.path.join(
            experiment_paths.checkpoint_dir,
            args.ckpt_name,
        )

    if not os.path.exists(
        ckpt_path
    ):
        raise FileNotFoundError(
            f"Checkpoint not found: "
            f"{ckpt_path}"
        )

    # ========================================================
    # INTENT MAP
    # ========================================================

    with open(
        intent_map_path,
        "r",
        encoding="utf-8",
    ) as f:

        intent_to_idx = json.load(
            f
        )

    idx_to_intent = {
        int(v): k
        for k, v in intent_to_idx.items()
    }

    print(
        f"Classes: {len(idx_to_intent)}"
    )

    for idx in sorted(
        idx_to_intent
    ):

        print(
            f"  {idx:2d} -> "
            f"{idx_to_intent[idx]}"
        )

    if len(idx_to_intent) != 17:

        print(
            f"[warning] expected 17 classes, "
            f"found {len(idx_to_intent)}"
        )

    # ========================================================
    # THRESHOLDS
    # ========================================================

    thresholds = build_thresholds(
        idx_to_intent,
        backchannel_threshold=(
            args.backchannel_threshold
        ),
        bypass_threshold=(
            args.bypass_threshold
        ),
        unclear_threshold=0.0,
    )

    print(
        "\nProduction thresholds:"
    )

    for idx in sorted(
        thresholds
    ):

        print(
            f"  {idx_to_intent[idx]:40s} "
            f"{thresholds[idx]:.2f}"
        )

    # ========================================================
    # DATA
    # ========================================================

    hf_split = load_hf_split(
        args.repo_id,
        args.split,
    )

    print(
        f"\nLoaded {len(hf_split)} rows "
        f"from {args.repo_id} "
        f"(split={args.split})"
    )

    unseen_intents = (
        set(
            hf_split[
                "intent"
            ]
        )
        - set(
            intent_to_idx
        )
    )

    if unseen_intents:

        raise ValueError(
            "Intents in eval data missing "
            f"from intent_map: "
            f"{unseen_intents}"
        )

    eval_dataset = HFIntentDataset(
        hf_split,
        intent_to_idx=intent_to_idx,
    )

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_mel_fn,
    )

    eval_labels_by_idx = [
        intent_to_idx[intent]
        for intent in hf_split[
            "intent"
        ]
    ]

    save_class_distribution(
        eval_labels_by_idx,
        idx_to_intent,
        os.path.join(
            output_dir,
            "eval_class_distribution.txt",
        ),
    )

    # ========================================================
    # MODEL
    # ========================================================

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device: {device}"
    )

    model = load_model(
        ckpt_path,
        args.model_type,
        len(intent_to_idx),
        device,
    )

    # ========================================================
    # INFERENCE
    # ========================================================

    (
        labels,
        preds,
        confidences,
        audio_ids,
    ) = run_inference(
        model,
        eval_loader,
        device,
        hf_split,
    )

    print(
        f"Collected {len(labels):,} predictions."
    )

    # ========================================================
    # SAVE RAW CONFIDENCE DATA
    # ========================================================

    confidence_records = []

    for (
        true_id,
        pred_id,
        confidence,
        audio_id,
    ) in zip(
        labels,
        preds,
        confidences,
        audio_ids,
    ):

        confidence_records.append(
            {
                "audio_id": audio_id,
                "true_intent": (
                    idx_to_intent[
                        true_id
                    ]
                ),
                "predicted_intent": (
                    idx_to_intent[
                        pred_id
                    ]
                ),
                "confidence": float(
                    confidence
                ),
            }
        )

    with open(
        os.path.join(
            output_dir,
            "prediction_confidences.json",
        ),
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            confidence_records,
            f,
            indent=2,
            ensure_ascii=False,
        )

    # ========================================================
    # RAW REPORT
    # ========================================================

    present_ids = sorted(
        set(labels)
        | set(preds)
    )

    target_names = [
        idx_to_intent[i]
        for i in present_ids
    ]

    report_text = classification_report(
        labels,
        preds,
        labels=present_ids,
        target_names=target_names,
        digits=3,
        zero_division=0,
    )

    with open(
        os.path.join(
            output_dir,
            "classification_report.txt",
        ),
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            report_text
        )

    raw_report = classification_report(
        labels,
        preds,
        labels=present_ids,
        target_names=target_names,
        digits=3,
        zero_division=0,
        output_dict=True,
    )

    with open(
        os.path.join(
            output_dir,
            "classification_report.json",
        ),
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            raw_report,
            f,
            indent=2,
        )

    plot_confusion_matrix(
        labels,
        preds,
        idx_to_intent,
        os.path.join(
            output_dir,
            "confusion_matrix.png",
        ),
        audio_ids=audio_ids,
    )

    plot_confusion_matrix(
        labels,
        preds,
        idx_to_intent,
        os.path.join(
            output_dir,
            "confusion_matrix_normalized.png",
        ),
        audio_ids=audio_ids,
        normalize=True,
    )

    plot_per_class_accuracy(
        labels,
        preds,
        idx_to_intent,
        os.path.join(
            output_dir,
            "per_class_accuracy.png",
        ),
    )

    # ========================================================
    # RAW PRODUCTION METRICS
    # ========================================================

    production_metrics = (
        calculate_per_class_metrics(
            labels,
            preds,
            idx_to_intent,
        )
    )

    unclear_metrics = (
        calculate_unclear_metrics(
            labels,
            preds,
            idx_to_intent,
        )
    )

    bypass_metrics = (
        calculate_bypass_precision(
            labels,
            preds,
            idx_to_intent,
        )
    )

    backchannel_metrics = (
        analyze_backchannel(
            labels,
            preds,
            idx_to_intent,
            audio_ids,
            os.path.join(
                output_dir,
                "backchannel_safety.txt",
            ),
        )
    )

    write_production_report(
        labels,
        preds,
        idx_to_intent,
        os.path.join(
            output_dir,
            "production_metrics.txt",
        ),
    )

    # ========================================================
    # RAW ADJUSTED METRICS
    # ========================================================

    (
        adj_labels,
        adj_preds,
        adj_audio_ids,
        adjusted_removed,
    ) = build_adjusted_predictions(
        labels,
        preds,
        audio_ids,
        idx_to_intent,
    )

    print(
        f"\n[adjusted] removed "
        f"{adjusted_removed:,} "
        f"non-UNCLEAR -> UNCLEAR cases."
    )

    adj_present_ids = sorted(
        set(adj_labels)
        | set(adj_preds)
    )

    adj_target_names = [
        idx_to_intent[i]
        for i in adj_present_ids
    ]

    adjusted_report_text = (
        classification_report(
            adj_labels,
            adj_preds,
            labels=adj_present_ids,
            target_names=adj_target_names,
            digits=3,
            zero_division=0,
        )
    )

    with open(
        os.path.join(
            output_dir,
            "classification_report_adjusted.txt",
        ),
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            adjusted_report_text
        )

    adjusted_report = (
        classification_report(
            adj_labels,
            adj_preds,
            labels=adj_present_ids,
            target_names=adj_target_names,
            digits=3,
            zero_division=0,
            output_dict=True,
        )
    )

    with open(
        os.path.join(
            output_dir,
            "classification_report_adjusted.json",
        ),
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            adjusted_report,
            f,
            indent=2,
        )

    plot_confusion_matrix(
        adj_labels,
        adj_preds,
        idx_to_intent,
        os.path.join(
            output_dir,
            "confusion_matrix_adjusted.png",
        ),
        audio_ids=adj_audio_ids,
        title_suffix="Adjusted",
    )

    plot_confusion_matrix(
        adj_labels,
        adj_preds,
        idx_to_intent,
        os.path.join(
            output_dir,
            "confusion_matrix_adjusted_normalized.png",
        ),
        audio_ids=adj_audio_ids,
        normalize=True,
        title_suffix="Adjusted",
    )

    plot_per_class_accuracy(
        adj_labels,
        adj_preds,
        idx_to_intent,
        os.path.join(
            output_dir,
            "per_class_accuracy_adjusted.png",
        ),
        title=(
            "Adjusted per-class accuracy"
        ),
    )

    # ========================================================
    # CONFIDENCE-GATED PREDICTIONS
    # ========================================================

    gated_preds = (
        apply_confidence_threshold(
            preds,
            confidences,
            idx_to_intent,
            thresholds,
        )
    )

    changed_to_unclear = sum(
        (
            raw != gated
            and idx_to_intent[raw]
            != UNCLEAR_CLASS
        )
        for raw, gated in zip(
            preds,
            gated_preds,
        )
    )

    print(
        "\n[threshold] converted "
        f"{changed_to_unclear:,} "
        "low-confidence bypass predictions "
        "to UNCLEAR_INPUT."
    )

    # ========================================================
    # GATED REPORT
    # ========================================================

    gated_present_ids = sorted(
        set(labels)
        | set(gated_preds)
    )

    gated_target_names = [
        idx_to_intent[i]
        for i in gated_present_ids
    ]

    gated_report_text = (
        classification_report(
            labels,
            gated_preds,
            labels=gated_present_ids,
            target_names=gated_target_names,
            digits=3,
            zero_division=0,
        )
    )

    with open(
        os.path.join(
            output_dir,
            "classification_report_thresholded.txt",
        ),
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            gated_report_text
        )

    gated_report = (
        classification_report(
            labels,
            gated_preds,
            labels=gated_present_ids,
            target_names=gated_target_names,
            digits=3,
            zero_division=0,
            output_dict=True,
        )
    )

    with open(
        os.path.join(
            output_dir,
            "classification_report_thresholded.json",
        ),
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            gated_report,
            f,
            indent=2,
        )

    plot_confusion_matrix(
        labels,
        gated_preds,
        idx_to_intent,
        os.path.join(
            output_dir,
            "confusion_matrix_thresholded.png",
        ),
        audio_ids=audio_ids,
        title_suffix="Thresholded",
    )

    plot_confusion_matrix(
        labels,
        gated_preds,
        idx_to_intent,
        os.path.join(
            output_dir,
            "confusion_matrix_thresholded_normalized.png",
        ),
        audio_ids=audio_ids,
        normalize=True,
        title_suffix="Thresholded",
    )

    # ========================================================
    # GATED PRODUCTION METRICS
    # ========================================================

    gated_production_metrics = (
        calculate_per_class_metrics(
            labels,
            gated_preds,
            idx_to_intent,
        )
    )

    gated_unclear_metrics = (
        calculate_unclear_metrics(
            labels,
            gated_preds,
            idx_to_intent,
        )
    )

    gated_bypass_metrics = (
        calculate_bypass_precision(
            labels,
            gated_preds,
            idx_to_intent,
        )
    )

    gated_backchannel_metrics = (
        analyze_backchannel(
            labels,
            gated_preds,
            idx_to_intent,
            audio_ids,
            os.path.join(
                output_dir,
                "backchannel_safety_thresholded.txt",
            ),
        )
    )

    write_production_report(
        labels,
        gated_preds,
        idx_to_intent,
        os.path.join(
            output_dir,
            "production_metrics_thresholded.txt",
        ),
        thresholds=thresholds,
    )

    # ========================================================
    # GATED ADJUSTED METRICS
    # ========================================================

    (
        gated_adj_labels,
        gated_adj_preds,
        gated_adj_audio_ids,
        gated_adjusted_removed,
    ) = build_adjusted_predictions(
        labels,
        gated_preds,
        audio_ids,
        idx_to_intent,
    )

    print(
        f"[thresholded adjusted] removed "
        f"{gated_adjusted_removed:,} "
        f"non-UNCLEAR -> UNCLEAR cases."
    )

    gated_adj_present_ids = sorted(
        set(gated_adj_labels)
        | set(gated_adj_preds)
    )

    gated_adj_target_names = [
        idx_to_intent[i]
        for i in gated_adj_present_ids
    ]

    gated_adjusted_report_text = (
        classification_report(
            gated_adj_labels,
            gated_adj_preds,
            labels=gated_adj_present_ids,
            target_names=gated_adj_target_names,
            digits=3,
            zero_division=0,
        )
    )

    with open(
        os.path.join(
            output_dir,
            "classification_report_thresholded_adjusted.txt",
        ),
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            gated_adjusted_report_text
        )

    gated_adjusted_report = (
        classification_report(
            gated_adj_labels,
            gated_adj_preds,
            labels=gated_adj_present_ids,
            target_names=gated_adj_target_names,
            digits=3,
            zero_division=0,
            output_dict=True,
        )
    )

    with open(
        os.path.join(
            output_dir,
            "classification_report_thresholded_adjusted.json",
        ),
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            gated_adjusted_report,
            f,
            indent=2,
        )

    plot_confusion_matrix(
        gated_adj_labels,
        gated_adj_preds,
        idx_to_intent,
        os.path.join(
            output_dir,
            "confusion_matrix_thresholded_adjusted.png",
        ),
        audio_ids=gated_adj_audio_ids,
        title_suffix="Thresholded Adjusted",
    )

    plot_confusion_matrix(
        gated_adj_labels,
        gated_adj_preds,
        idx_to_intent,
        os.path.join(
            output_dir,
            "confusion_matrix_thresholded_adjusted_normalized.png",
        ),
        audio_ids=gated_adj_audio_ids,
        normalize=True,
        title_suffix="Thresholded Adjusted",
    )

    # ========================================================
    # CONSOLE SUMMARY
    # ========================================================

    raw_backchannel = production_metrics[
        BACKCHANNEL_CLASS
    ]

    gated_backchannel = (
        gated_production_metrics[
            BACKCHANNEL_CLASS
        ]
    )

    raw_unclear = production_metrics[
        UNCLEAR_CLASS
    ]

    gated_unclear = (
        gated_production_metrics[
            UNCLEAR_CLASS
        ]
    )

    print(
        "\n"
        "============================================================"
    )

    print(
        "RAW PRODUCTION METRICS"
    )

    print(
        "============================================================"
    )

    print(
        f"Bypass precision       : "
        f"{bypass_metrics['precision'] * 100:.3f}%"
    )

    print(
        f"Backchannel precision  : "
        f"{raw_backchannel['precision'] * 100:.3f}%"
    )

    print(
        f"Backchannel FP         : "
        f"{raw_backchannel['false_positives']}"
    )

    print(
        f"UNCLEAR recall         : "
        f"{raw_unclear['recall'] * 100:.3f}%"
    )

    print(
        "\n"
        "============================================================"
    )

    print(
        "THRESHOLDED PRODUCTION METRICS"
    )

    print(
        "============================================================"
    )

    print(
        f"Bypass precision       : "
        f"{gated_bypass_metrics['precision'] * 100:.3f}%"
    )

    print(
        f"Backchannel precision  : "
        f"{gated_backchannel['precision'] * 100:.3f}%"
    )

    print(
        f"Backchannel FP         : "
        f"{gated_backchannel['false_positives']}"
    )

    print(
        f"UNCLEAR recall         : "
        f"{gated_unclear['recall'] * 100:.3f}%"
    )

    print(
        "\n"
        "============================================================"
    )

    print(
        "ADJUSTED METRICS"
    )

    print(
        "============================================================"
    )

    print(
        f"Raw adjusted accuracy  : "
        f"{adjusted_report['accuracy'] * 100:.3f}%"
    )

    print(
        f"Raw adjusted macro F1  : "
        f"{adjusted_report['macro avg']['f1-score']:.3f}"
    )

    print(
        f"Gated adjusted accuracy: "
        f"{gated_adjusted_report['accuracy'] * 100:.3f}%"
    )

    print(
        f"Gated adjusted macro F1: "
        f"{gated_adjusted_report['macro avg']['f1-score']:.3f}"
    )

    print(
        "\n"
        f"Evaluation complete. "
        f"Reports: {output_dir}"
    )

    # ========================================================
    # MLFLOW
    # ========================================================

    log_evaluation_to_mlflow(
        experiment_paths=experiment_paths,
        args=args,
        ckpt_path=ckpt_path,
        intent_map_path=intent_map_path,
        output_dir=output_dir,
        raw_report=raw_report,
        adjusted_report=adjusted_report,
        production_metrics=production_metrics,
        backchannel_metrics=backchannel_metrics,
        unclear_metrics=unclear_metrics,
        bypass_metrics=bypass_metrics,
        gated_production_metrics=(
            gated_production_metrics
        ),
        gated_backchannel_metrics=(
            gated_backchannel_metrics
        ),
        gated_unclear_metrics=(
            gated_unclear_metrics
        ),
        gated_bypass_metrics=(
            gated_bypass_metrics
        ),
        eval_rows=len(labels),
    )


if __name__ == "__main__":
    main()