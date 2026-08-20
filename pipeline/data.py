from dotenv import load_dotenv
import os
import json
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from bson import ObjectId
from pymongo import MongoClient, ReadPreference, ASCENDING
from tqdm.auto import tqdm

load_dotenv('pipeline/.env')

# OUT = Path("shards")
OUT = f"{os.environ['SOURCE_DATA_DIR']}/SHARDS"
OUT.mkdir(exist_ok=True)
STATE = OUT / "_state.json"

SHARD_ROWS = 5_000
BATCH_SIZE = 100
WORK_SECS = 60
THROTTLE_SECS = 5.0
CHECKPOINT_SECS = 300

client = MongoClient(
    os.getenv("MONGO_DB_URI"),
    read_preference=ReadPreference.SECONDARY_PREFERRED,
    compressors="zstd",
)

valid_collections_file_path=os.environ['VALID_COLLECTION_FILE_PATH']
collections = json.load(open(valid_collections_file_path))

query = {
    "recording_url": {"$ne": None},
    "call_duration": {"$gt": 30},
    "messages.1": {"$exists": True},
}

projection = {
    "_id": 1,
    "conversation_id": 1,
    "recording_url": 1,
    "messages": 1,
    "call_duration": 1,
}

schema = pa.schema([
    ("db", pa.string()),
    ("oid", pa.string()),
    ("conversation_id", pa.string()),
    ("recording_url", pa.string()),
    ("call_duration", pa.float64()),
    ("messages", pa.string()),
])

state = json.load(open(STATE)) if STATE.exists() else {}


def to_row(doc, db_name):
    return {
        "db": db_name,
        "oid": str(doc["_id"]),
        "conversation_id": str(doc.get("conversation_id")),
        "recording_url": doc["recording_url"],
        "call_duration": float(doc["call_duration"]),
        "messages": json.dumps(doc["messages"], default=str),
    }


def checkpoint(db_name, rows, st):
    if rows:
        pq.write_table(
            pa.Table.from_pylist(rows, schema=schema),
            OUT / f"{db_name}__{st['shard']:05d}.parquet",
            compression="zstd",
        )
        st["shard"] += 1
    state[db_name] = st
    json.dump(state, open(STATE, "w"))


for db_name, count in tqdm(collections.items(), desc="dbs"):
    st = state.get(db_name, {"last_id": None, "shard": 0, "done": False})
    if st["done"]:
        continue

    coll = client[db_name]["conversation_history"]
    q = dict(query)
    if st["last_id"]:
        q["_id"] = {"$gt": ObjectId(st["last_id"])}

    rows = []
    kept = 0
    last_work = time.monotonic()
    last_ckpt = time.monotonic()

    cursor = coll.find(q, projection, batch_size=BATCH_SIZE).sort("_id", ASCENDING)
    bar = tqdm(cursor, total=None, desc=db_name, leave=False, unit="hit")

    for doc in bar:
        rows.append(to_row(doc, db_name))
        st["last_id"] = str(doc["_id"])
        kept += 1

        now = time.monotonic()

        if len(rows) >= SHARD_ROWS or now - last_ckpt > CHECKPOINT_SECS:
            checkpoint(db_name, rows, st)
            rows = []
            last_ckpt = now
            bar.set_postfix(shards=st["shard"], kept=kept)

        if now - last_work > WORK_SECS:
            time.sleep(THROTTLE_SECS)
            last_work = time.monotonic()
            last_ckpt += THROTTLE_SECS

    bar.close()
    cursor.close()

    st["done"] = True
    checkpoint(db_name, rows, st)