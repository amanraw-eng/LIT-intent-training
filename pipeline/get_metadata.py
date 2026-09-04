# """Analyze intent-dataset JSONL manifest and output intent-wise statistics.

# Usage:
#   python analyze_intents.py --input data.jsonl [--output report.json]
# """
# from __future__ import annotations

# import argparse
# import json
# from collections import Counter, defaultdict
# from pathlib import Path


# def analyze_intents(input_path: Path) -> dict:
#     intents = defaultdict(lambda: {
#         "count": 0,
#         "seconds": 0.0,
#         "unknown_duration": 0,
#         "total_words": 0,
#     })
    
#     total_rows = 0
#     missing_intent_count = 0

#     with input_path.open(encoding="utf-8") as handle:
#         for line_number, line in enumerate(handle, 1):
#             line = line.strip()
#             if not line:
#                 continue
            
#             row = json.loads(line)
#             total_rows += 1
            
#             intent = str(row.get("intent") or "MISSING_INTENT")
#             if intent == "MISSING_INTENT":
#                 missing_intent_count += 1
                
#             data = intents[intent]
#             data["count"] += 1
            
#             # Duration parsing
#             duration = row.get("duration_s")
#             try:
#                 duration = float(duration)
#             except (TypeError, ValueError):
#                 duration = None
                
#             if duration is None:
#                 data["unknown_duration"] += 1
#             else:
#                 data["seconds"] += duration
                
#             # Transcript word count analysis
#             transcript = row.get("transcript")
#             if isinstance(transcript, str) and transcript.strip():
#                 data["total_words"] += len(transcript.strip().split())

#     # Build report
#     report = {
#         "file": str(input_path),
#         "total_rows": total_rows,
#         "total_intents": len(intents),
#         "missing_intent_rows": missing_intent_count,
#         "intents": {}
#     }

#     for intent, data in sorted(intents.items(), key=lambda x: x[1]["count"], reverse=True):
#         hours = data["seconds"] / 3600
#         avg_duration = (data["seconds"] / (data["count"] - data["unknown_duration"])) if data["count"] > data["unknown_duration"] else 0
#         avg_words = (data["total_words"] / data["count"]) if data["count"] > 0 else 0
        
#         report["intents"][intent] = {
#             "samples": data["count"],
#             "hours": round(hours, 4),
#             "total_seconds": round(data["seconds"], 2),
#             "avg_duration_s": round(avg_duration, 2),
#             "avg_words_per_transcript": round(avg_words, 2),
#             "unknown_duration_samples": data["unknown_duration"],
#         }

#     return report


# def main() -> None:
#     parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
#     parser.add_argument("--input", type=Path, required=True, help="Input JSONL manifest file")
#     parser.add_argument("--output", type=Path, default=None, help="Optional JSON report output file path")
    
#     args = parser.parse_args()
#     report = analyze_intents(args.input)
    
#     # Render report to console
#     rendered = json.dumps(report, ensure_ascii=False, indent=2)
#     print(rendered)
    
#     if args.output:
#         args.output.write_text(rendered + "\n", encoding="utf-8")
#         print(f"\nReport written to {args.output}")


# if __name__ == "__main__":
#     main()


"""Comprehensive Intent-Wise Dataset Analyzer.

Aggregates statistics for ALL unique intents present in JSONL manifest files.

Usage:
  python analyze_all_intents.py --input data.jsonl [--output report.json]
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def analyze_all_intents(input_path: Path) -> dict:
    intents = defaultdict(lambda: {
        "count": 0,
        "seconds": 0.0,
        "unknown_duration": 0,
        "total_words": 0,
    })
    
    total_rows = 0
    total_seconds = 0.0
    missing_intent_count = 0

    with input_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            
            row = json.loads(line)
            total_rows += 1
            
            # Fetch intent directly without hardcoded limits
            intent = str(row.get("intent") or "MISSING_INTENT").strip()
            if intent == "MISSING_INTENT":
                missing_intent_count += 1
                
            data = intents[intent]
            data["count"] += 1
            
            # Process duration
            duration = row.get("duration_s")
            try:
                duration = float(duration)
            except (TypeError, ValueError):
                duration = None
                
            if duration is None:
                data["unknown_duration"] += 1
            else:
                data["seconds"] += duration
                total_seconds += duration
                
            # Process transcript word count
            transcript = row.get("transcript")
            if isinstance(transcript, str) and transcript.strip():
                data["total_words"] += len(transcript.strip().split())

    # Compile intent-wise summary
    intent_summary = {}
    for intent, data in sorted(intents.items(), key=lambda x: x[1]["count"], reverse=True):
        valid_duration_count = data["count"] - data["unknown_duration"]
        avg_duration = (data["seconds"] / valid_duration_count) if valid_duration_count > 0 else 0.0
        avg_words = (data["total_words"] / data["count"]) if data["count"] > 0 else 0.0
        
        intent_summary[intent] = {
            "sample_count": data["count"],
            "hours": round(data["seconds"] / 3600, 4),
            "total_seconds": round(data["seconds"], 2),
            "avg_duration_s": round(avg_duration, 2),
            "avg_words_per_transcript": round(avg_words, 2),
            "missing_duration_count": data["unknown_duration"],
        }

    report = {
        "summary": {
            "file": str(input_path),
            "total_samples": total_rows,
            "total_unique_intents": len(intents),
            "total_hours": round(total_seconds / 3600, 4),
            "rows_missing_intent": missing_intent_count,
        },
        "intents": intent_summary
    }

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True, help="Input JSONL manifest file path")
    parser.add_argument("--output", type=Path, default=None, help="Optional output path to write JSON report")
    
    args = parser.parse_args()
    report = analyze_all_intents(args.input)
    
    # Render pretty JSON output
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print(f"\nFull report successfully written to {args.output}")


if __name__ == "__main__":
    main()