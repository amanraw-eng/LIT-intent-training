import json
from collections import Counter
import sys
import os
from config import INTENTS_JSON_PATH, BASE_DIR


def load_intent_rows( cache_path):
    """intents.md is only ever parsed once - after that this reads the cached
    JSON. Delete cache_path to force a re-parse (e.g. after editing intents.md)."""
    if os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as f:
            return json.load(f)
    return "error occured"


INTENT_ROWS = load_intent_rows(INTENTS_JSON_PATH)
INTENT_NAMES = [r["name"] for r in INTENT_ROWS]

def count_intents(file_path):
    intent_counts = Counter()
    for intent in INTENT_NAMES:
        intent_counts[intent]=0

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
                
            try:
                data = json.loads(line)
                # Use .get() in case some rows are missing the 'intent' key
                intent = data.get("intent", "NO_INTENT_FOUND") 
                intent_counts[intent] += 1
            except json.JSONDecodeError:
                print("Skipped invalid JSON line.")

    # Print results sorted by highest count first
    print("\n--- Intent Counts ---")
    with open(f'{BASE_DIR}/intent_analytics.json', 'w+') as file:
        file.write("{")
        i=0
        for intent, count in intent_counts.most_common():
            if i==0:
                file.write(f'\n"{intent}": {count}')
            else:
                file.write(f',\n"{intent}": {count}')
            i+=1
            print(f"{intent}: {count}")
        file.write("\n}")


if __name__ == "__main__":
    # Replace 'data.jsonl' with your actual file name
    count_intents("/mnt/HDD8TB/aman_ws/stt/data/call_trascript_intent_data_v3/data.jsonl")