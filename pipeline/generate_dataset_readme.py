# pipeline/generate_dataset_readme.py

from collections import Counter

from datasets import load_dataset


REPO_ID = "kapturecx/call-transcript-intent-data-v2"
OUTPUT = "README.md"

SPLITS = ("train", "validation", "eval")


def format_hours(seconds: float) -> str:
    return f"{seconds / 3600:.2f} h"


def main():
    print(f"Loading: {REPO_ID}")

    ds = load_dataset(REPO_ID)

    split_stats = {}
    class_counts = Counter()
    total_seconds = 0.0
    total_rows = 0

    for split in SPLITS:
        if split not in ds:
            continue

        data = ds[split]

        n = len(data)

        # duration_s is expected to be present.
        durations = data["duration_s"]
        split_seconds = sum(
            float(x or 0)
            for x in durations
        )

        intents = data["intent"]

        split_classes = Counter(intents)

        split_stats[split] = {
            "rows": n,
            "seconds": split_seconds,
            "classes": split_classes,
        }

        class_counts.update(intents)

        total_rows += n
        total_seconds += split_seconds

    lines = []

    lines.append(
        "# Call Transcript Intent Dataset"
    )
    lines.append("")
    lines.append(
        "Multimodal Hindi/Hinglish customer utterance dataset "
        "for loan/EMI/payment call intent classification."
    )
    lines.append("")

    # --------------------------------------------------------
    # Dataset summary
    # --------------------------------------------------------

    lines.append("## Dataset Summary")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---:|")
    lines.append(f"| Total examples | {total_rows:,} |")
    lines.append(
        f"| Total audio duration | {format_hours(total_seconds)} |"
    )
    lines.append(
        f"| Number of intents | {len(class_counts)} |"
    )
    lines.append("")

    # --------------------------------------------------------
    # Split statistics
    # --------------------------------------------------------

    lines.append("## Split Statistics")
    lines.append("")
    lines.append(
        "| Split | Examples | Duration | Hours |"
    )
    lines.append("|---|---:|---:|---:|")

    for split in SPLITS:
        if split not in split_stats:
            continue

        stats = split_stats[split]

        lines.append(
            f"| {split} | "
            f"{stats['rows']:,} | "
            f"{stats['seconds'] / 60:.2f} min | "
            f"{stats['seconds'] / 3600:.2f} h |"
        )

    lines.append("")

    # --------------------------------------------------------
    # Class distribution
    # --------------------------------------------------------

    lines.append("## Class Distribution")
    lines.append("")
    lines.append(
        "| Intent | Total | Train | Validation | Eval |"
    )
    lines.append(
        "|---|---:|---:|---:|---:|"
    )

    all_intents = sorted(
        class_counts,
        key=lambda x: (-class_counts[x], x),
    )

    for intent in all_intents:
        counts = []

        for split in SPLITS:
            counts.append(
                split_stats.get(
                    split,
                    {}
                )
                .get(
                    "classes",
                    Counter()
                )
                .get(
                    intent,
                    0,
                )
            )

        lines.append(
            f"| `{intent}` | "
            f"{class_counts[intent]:,} | "
            f"{counts[0]:,} | "
            f"{counts[1]:,} | "
            f"{counts[2]:,} |"
        )

    lines.append("")

    # --------------------------------------------------------
    # Columns
    # --------------------------------------------------------

    columns = ds[SPLITS[0]].column_names

    lines.append("## Columns")
    lines.append("")

    for column in columns:
        lines.append(f"- `{column}`")

    lines.append("")

    with open(
        OUTPUT,
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            "\n".join(lines)
            + "\n"
        )

    print(
        f"README written to: {OUTPUT}"
    )

    print(
        f"Total rows: {total_rows:,}"
    )

    print(
        f"Total duration: "
        f"{format_hours(total_seconds)}"
    )


if __name__ == "__main__":
    main()