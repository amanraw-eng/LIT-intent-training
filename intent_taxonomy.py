"""Load and render the authoritative 17-intent taxonomy."""
from __future__ import annotations
import json
from pathlib import Path

TAXONOMY_PATH = Path(__file__).with_name("intent17.json")
with TAXONOMY_PATH.open(encoding="utf-8") as _file:
    TAXONOMY = json.load(_file)
INTENT_NAMES = tuple(item["name"] for item in TAXONOMY["intents"])
INTENT_SET = frozenset(INTENT_NAMES)


def prompt_taxonomy() -> str:
    blocks = []
    for item in TAXONOMY["intents"]:
        lines = [f"### {item['name']}", item["definition"], "Examples: " + "; ".join(item.get("examples", []))]
        if item.get("notes"):
            lines.append("Note: " + item["notes"])
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)
