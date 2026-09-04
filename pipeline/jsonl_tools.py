"""Make intent-dataset JSONL manifests portable by trimming audio paths to filenames only.

Example:
  python -m pipeline.jsonl_tools --input data.jsonl --output portable.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def iter_rows(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                yield line_number, json.loads(line)


def make_portable(input_path: Path, output_path: Path) -> None:
    if input_path.resolve() == output_path.resolve():
        raise ValueError("--output must differ from --input to keep conversion recoverable")

    converted = 0
    with output_path.open("w", encoding="utf-8") as output:
        for line_number, row in iter_rows(input_path):
            if "chunk_path" not in row:
                raise ValueError(f"line {line_number}: missing chunk_path")

            # Split path by slashes and keep only the filename
            row["chunk_path"] = str(row["chunk_path"]).split("/")[-1]

            output.write(json.dumps(row, ensure_ascii=False) + "\n")
            converted += 1

    print(f"Wrote {converted:,} portable rows to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input", type=Path, required=True, help="Input JSONL file")
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL file")

    args = parser.parse_args()
    make_portable(args.input, args.output)


if __name__ == "__main__":
    main()