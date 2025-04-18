import os, json, asyncio, httpx, logging, tiktoken
from pymongo import MongoClient, InsertOne
from dotenv import load_dotenv
from tqdm import tqdm

# -------- logging --------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# -------- env --------
load_dotenv()
AZURE_ENDPOINT       = os.getenv("AZURE_OPENAI_ENDPOINT").rstrip("/")
AZURE_API_KEY        = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_EMBED_VERSION  = os.getenv("AZURE_EMBEDDING_API_VERSION", "2024-08-01-preview")
AZURE_EMBED_DEPLOY   = os.getenv("AZURE_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002")
EMBED_URL            = f"{AZURE_ENDPOINT}/openai/deployments/{AZURE_EMBED_DEPLOY}/embeddings?api-version={AZURE_EMBED_VERSION}"
HEADERS              = {"Content-Type": "application/json", "api-key": AZURE_API_KEY}

MONGO_URI   = os.getenv("MONGO_URI")
DB_NAME     = os.getenv("MONGO_DB_NAME", "resume_db")
SRC_NAME    = os.getenv("MONGO_COLLECTION_NAME", "resumes")
DST_NAME    = SRC_NAME + "_exp"

# -------- tokenizer --------
tokenizer = tiktoken.get_encoding("cl100k_base")

def truncate_text(text: str, max_tokens: int = 8192) -> str:
    tokens = tokenizer.encode(text)
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
    return tokenizer.decode(tokens)

async def embed_one(text: str) -> list[float]:
    cleaned = truncate_text(text)
    body = {"input": cleaned}
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(EMBED_URL, headers=HEADERS, json=body)
        r.raise_for_status()
        return r.json()["data"][0]["embedding"]

# -------- main --------
async def main():
    cli = MongoClient(MONGO_URI)
    src = cli[DB_NAME][SRC_NAME]
    dst = cli[DB_NAME][DST_NAME]

    total = src.estimated_document_count()
    log.info("Copying %s docs  (%s.%s ➜ %s.%s)", total, DB_NAME, SRC_NAME, DB_NAME, DST_NAME)
    jsonl = open("resumes_with_embeddings.jsonl", "a", encoding="utf-8")  # switched to append mode

    ops, written = [], 0
    cursor = src.find({})

    for doc in tqdm(cursor, total=total):
        if dst.count_documents({"_id": doc["_id"]}, limit=1):  # ✅ skip already done
            log.info("⏭️ Skipping already processed: %s", doc.get("name", "Unknown"))
            continue

        try:
            # -------- prepare field texts --------
            summary_text = doc.get("summary", "")
            skills_text = " ".join(doc.get("skills", []))
            projects_text = " ".join([
                (str(p.get("title") or "") + " " + str(p.get("description") or ""))
                for p in doc.get("projects", [])
            ])
            experience_text = " ".join([
                f"{e.get('title') or ''} {e.get('company') or ''} {e.get('description') or ''}"
                for e in doc.get("experience", [])
            ])

            certifications_text = " ".join([
                (c.get("title", "") + " " + c.get("issuer", ""))
                for c in doc.get("certifications", [])
            ])
            education_text = " ".join([
                (e.get("degree", "") + " " + e.get("institution", ""))
                for e in doc.get("education", [])
            ])

            # -------- generate 6 separate embeddings --------
            doc["summary_embed"]        = await embed_one(summary_text)
            doc["skills_embed"]         = await embed_one(skills_text)
            doc["projects_embed"]       = await embed_one(projects_text)
            doc["experience_embed"]     = await embed_one(experience_text)
            doc["certifications_embed"] = await embed_one(certifications_text)
            doc["education_embed"]      = await embed_one(education_text)

            ops.append(InsertOne(doc))
            jsonl.write(json.dumps(doc, ensure_ascii=False) + "\n")
            written += 1

            if len(ops) >= 100:
                dst.bulk_write(ops); ops.clear()
                log.info("Inserted %s / %s resumes...", written, total)

        except Exception as e:
            log.error("❌ Embedding failed for '%s' — skipping (%s)", doc.get("name", "Unknown"), e)
            continue

    if ops:
        dst.bulk_write(ops)

    jsonl.close()
    log.info("✅ Finished. %s docs written to %s and JSONL.", written, DST_NAME)

if __name__ == "__main__":
    asyncio.run(main())
