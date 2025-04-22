import json
from pymongo import MongoClient, InsertOne
from tqdm import tqdm
import os
from dotenv import load_dotenv

load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("MONGO_DB_NAME", "resume_db")
COLL_NAME = os.getenv("MONGO_COLLECTION_NAME", "resumes") + "_exp"
JSONL_PATH = "resumes_with_embeddings.jsonl"

cli = MongoClient(MONGO_URI)
coll = cli[DB_NAME][COLL_NAME]

BATCH_SIZE = 100
ops = []
inserted = 0
skipped = 0
total_lines = sum(1 for _ in open(JSONL_PATH, "r", encoding="utf-8"))

print(f"📄 Total documents in JSONL: {total_lines}")

with open(JSONL_PATH, "r", encoding="utf-8") as f:
    for line in tqdm(f, total=total_lines, desc="📦 Inserting from JSONL"):
        doc = json.loads(line)
        if coll.count_documents({"_id": doc["_id"]}, limit=1):
            skipped += 1
            continue
        ops.append(InsertOne(doc))
        inserted += 1
        if len(ops) >= BATCH_SIZE:
            coll.bulk_write(ops)
            ops.clear()

if ops:
    coll.bulk_write(ops)

print(f"\n✅ Insertion complete.")
print(f"🔢 Total in JSONL       : {total_lines}")
print(f"✅ Newly inserted       : {inserted}")
print(f"⏭️ Already in MongoDB  : {skipped}")
