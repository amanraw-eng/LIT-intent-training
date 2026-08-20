"""At each mark in CHECK_SCHEDULE (elapsed time since this script started),
check the current best checkpoint - if it's different from the one pushed at
the last check, push it to the HF Hub, evaluate it on the `eval` split
(confusion matrices + all), and log everything to a dedicated wandb run.
Sleeps between marks. Each mark gets its own HF repo / eval folder / wandb
run name suffix so nothing overwrites a previous check's results.

Edit CHECK_SCHEDULE below to add more marks (e.g. "8h", "12h30") - they're
absolute elapsed time since script start, not gaps between checks.

Run in tmux (or nohup it) so it survives your terminal closing, e.g.:

    cd /mnt/HDD8TB/aman_ws/stt/intent-training
    /mnt/HDD8TB/aman_ws/stt/.venv/bin/python wait_push_and_eval.py

Progress is also written to wait_push_and_eval.log (same content, plus
timestamps) so you can tail it from another pane.
"""

import glob
import json
import os
import re
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone

import torch
from dotenv import load_dotenv

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(THIS_DIR, ".env"))

os.environ.setdefault("WANDB_MODE", "online")
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

# absolute elapsed time since script start - add more marks any time before running
CHECK_SCHEDULE = ["1h30", "3h10", "5h30"]

CHECKPOINT_DIR = os.path.join(THIS_DIR, "models", "checkpoints")
INTENT_MAP_PATH = os.path.join(THIS_DIR, "models", "intent_to_idx.json")
EVAL_OUTPUT_ROOT = os.path.join(THIS_DIR, "eval_results")
MODEL_TYPE = "small"

# change this if you want the model pushed somewhere else - each check mark
# gets pushed to f"{HF_MODEL_REPO_ID}-{label}", e.g. "...-v3-1h30"
HF_MODEL_REPO_ID = "amn-raw/whisper-small-intent-classifier-v3"

TRAIN_PROJECT = "S2I-whisper"
TRAIN_RUN_NAME = "whisper-small-S2I-hi-v3"

LOG_PATH = os.path.join(THIS_DIR, "wait_push_and_eval.log")
PYTHON = sys.executable


def log(msg):
    line = f"[{datetime.now(timezone.utc).isoformat()}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def parse_duration(spec):
    """'1h30' / '4h30m' / '2h' / '45m' / '90s' -> seconds. A bare trailing
    number after an 'h' part (or with no unit at all) is treated as minutes."""
    original = spec
    spec = spec.strip().lower().replace(" ", "")
    if not spec:
        raise ValueError(f"Cannot parse duration: {original!r}")

    total_seconds = 0
    m = re.match(r"(\d+)h", spec)
    if m:
        total_seconds += int(m.group(1)) * 3600
        spec = spec[m.end():]

    if spec:
        m = re.match(r"(\d+)m?$", spec)
        if m:
            total_seconds += int(m.group(1)) * 60
            spec = ""
        else:
            m = re.match(r"(\d+)s$", spec)
            if m:
                total_seconds += int(m.group(1))
                spec = ""

    if spec:
        raise ValueError(
            f"Cannot parse duration: {original!r} (expected forms like '1h30', '4h30m', '90m', '2h', '45s')"
        )
    return total_seconds


def with_retries(fn, label, attempts=3, delay_s=30):
    last_exc = None
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as e:
            last_exc = e
            log(f"{label} failed (attempt {attempt}/{attempts}): {type(e).__name__}: {e}")
            if attempt < attempts:
                time.sleep(delay_s)
    raise last_exc


def _epoch_from_filename(path):
    m = re.search(r"epoch(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else -1


def find_best_checkpoint():
    """The checkpoint with the highest epoch number carries the most
    up-to-date ModelCheckpoint callback state (best_model_path/
    best_model_score), regardless of which of the retained save_top_k=3
    files is actually best. Sorting by the epoch number embedded in the
    filename (not mtime) - checkpoints written back-to-back in a fast
    validation loop can land on the same filesystem timestamp."""
    ckpts = glob.glob(os.path.join(CHECKPOINT_DIR, "*.ckpt"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found in {CHECKPOINT_DIR}")
    ckpts.sort(key=_epoch_from_filename)

    latest = ckpts[-1]
    checkpoint = torch.load(latest, map_location="cpu")
    for cb_state in checkpoint.get("callbacks", {}).values():
        if not isinstance(cb_state, dict):
            continue
        best_path = cb_state.get("best_model_path")
        if not best_path:
            continue
        if not os.path.isabs(best_path):
            best_path = os.path.join(THIS_DIR, best_path)
        if os.path.exists(best_path):
            score = cb_state.get("best_model_score")
            return best_path, (float(score) if score is not None else None)

    log(f"Could not resolve best_model_path from checkpoint callback state - "
        f"falling back to the most recently modified checkpoint: {latest}")
    return latest, None


def run_subprocess(args, label):
    log(f"Running: {' '.join(args)}")
    result = subprocess.run(args, cwd=THIS_DIR, capture_output=True, text=True)
    if result.stdout:
        log(f"{label} stdout:\n{result.stdout}")
    if result.stderr:
        log(f"{label} stderr:\n{result.stderr}")
    if result.returncode != 0:
        raise RuntimeError(f"{label} exited with code {result.returncode}")
    return result


def push_model(best_ckpt_path, repo_id):
    def _push():
        run_subprocess(
            [
                PYTHON, "push_to_hub.py",
                "--ckpt_path", best_ckpt_path,
                "--repo_id", repo_id,
                "--model_type", MODEL_TYPE,
                "--intent_map_path", INTENT_MAP_PATH,
            ],
            "push_to_hub.py",
        )

    with_retries(_push, "push_to_hub")
    log(f"Pushed checkpoint to https://huggingface.co/{repo_id}")


def run_eval(best_ckpt_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    def _eval():
        run_subprocess(
            [
                PYTHON, "evaluate.py",
                "--split", "eval",
                "--ckpt_name", best_ckpt_path,
                "--model_type", MODEL_TYPE,
                "--intent_map_path", INTENT_MAP_PATH,
                "--output_dir", output_dir,
            ],
            "evaluate.py",
        )

    with_retries(_eval, "evaluate")
    log(f"Saved eval results (classification report + confusion matrices) to {output_dir}")


def log_to_wandb(best_ckpt_path, best_score, output_dir, run_name, label):
    import wandb

    with open(os.path.join(output_dir, "classification_report.json"), "r", encoding="utf-8") as f:
        report = json.load(f)

    def _log():
        run = wandb.init(
            project=TRAIN_PROJECT,
            name=run_name,
            group=TRAIN_RUN_NAME,
            job_type="evaluation",
            config={
                "checkpoint": os.path.basename(best_ckpt_path),
                "best_val_acc": best_score,
                "eval_split": "eval",
                "check_mark": label,
            },
        )
        run.log({
            "eval/accuracy": report["accuracy"],
            "eval/macro_precision": report["macro avg"]["precision"],
            "eval/macro_recall": report["macro avg"]["recall"],
            "eval/macro_f1": report["macro avg"]["f1-score"],
            "eval/weighted_f1": report["weighted avg"]["f1-score"],
            "eval/confusion_matrix": wandb.Image(os.path.join(output_dir, "confusion_matrix.png")),
            "eval/confusion_matrix_normalized": wandb.Image(os.path.join(output_dir, "confusion_matrix_normalized.png")),
            "eval/per_class_accuracy": wandb.Image(os.path.join(output_dir, "per_class_accuracy.png")),
        })

        table = wandb.Table(columns=["intent", "precision", "recall", "f1-score", "support"])
        for name, row in report.items():
            if name in ("accuracy", "macro avg", "weighted avg"):
                continue
            table.add_data(name, row["precision"], row["recall"], row["f1-score"], row["support"])
        run.log({"eval/per_class_metrics": table})

        run.finish()

    with_retries(_log, "wandb logging")
    log(f"Logged evaluation results to wandb -> project={TRAIN_PROJECT}, run={run_name}")


def log_failure_to_wandb(label, exc):
    """Best-effort: surface the failure in wandb too, so it's visible remotely
    even when a check fails. Never let this itself raise."""
    try:
        import wandb
        run = wandb.init(project=TRAIN_PROJECT, name=f"{TRAIN_RUN_NAME}-evaluation-{label}-FAILED", job_type="evaluation")
        run.log({"eval/error": f"{type(exc).__name__}: {exc}"})
        run.finish()
    except Exception:
        pass


def process_checkpoint(best_ckpt_path, best_score, label):
    repo_id = f"{HF_MODEL_REPO_ID}-{label}"
    output_dir = os.path.join(EVAL_OUTPUT_ROOT, f"v3_eval_{label}")
    run_name = f"{TRAIN_RUN_NAME}-evaluation-{label}"

    push_model(best_ckpt_path, repo_id)
    run_eval(best_ckpt_path, output_dir)
    log_to_wandb(best_ckpt_path, best_score, output_dir, run_name, label)


def main():
    schedule = [(label, parse_duration(label)) for label in CHECK_SCHEDULE]
    schedule.sort(key=lambda item: item[1])
    log(f"Schedule (elapsed seconds since start): {schedule}")

    start_time = time.monotonic()
    last_best_path = None

    for label, target_seconds in schedule:
        remaining = target_seconds - (time.monotonic() - start_time)
        if remaining > 0:
            log(f"Sleeping {remaining:.0f}s until check '{label}' ({target_seconds}s mark)...")
            time.sleep(remaining)
        else:
            log(f"Check '{label}' ({target_seconds}s mark) already passed - checking immediately")

        log(f"=== Check '{label}' ===")
        try:
            best_ckpt_path, best_score = find_best_checkpoint()
        except FileNotFoundError as e:
            log(f"No checkpoints yet at check '{label}': {e}. Skipping this check.")
            continue

        if best_ckpt_path == last_best_path:
            log(f"Best checkpoint unchanged since last check ({best_ckpt_path}) - skipping push+eval for '{label}'")
            continue

        log(f"New best checkpoint at '{label}': {best_ckpt_path} (val/acc={best_score})")
        last_best_path = best_ckpt_path

        try:
            process_checkpoint(best_ckpt_path, best_score, label)
        except Exception:
            log(f"FAILED to process checkpoint at check '{label}':\n" + traceback.format_exc())
            log_failure_to_wandb(label, sys.exc_info()[1])
            # keep going - a failure at one mark shouldn't cancel later marks
            continue

    log("All scheduled checks complete. DONE")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log("FATAL ERROR:\n" + traceback.format_exc())
        log_failure_to_wandb("fatal", e)
        raise
