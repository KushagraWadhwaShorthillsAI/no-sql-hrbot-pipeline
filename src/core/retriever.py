import os
import json
import argparse
import asyncio
import httpx
from pymongo import MongoClient
from .config import MONGO_URI, DB_NAME, COLLECTION_NAME
from dotenv import load_dotenv
from .search_builder import bm25_pipeline

load_dotenv()
PIPELINE_MODE = os.getenv("PIPELINE_MODE", "false").lower() == "true"
LOG_FILE = "logs/nosql_pipeline.log" if PIPELINE_MODE else "logs/nosql_retriever.log"

def log(msg):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")

def safe_json_load(text: str):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        print("❌ Failed to parse JSON from LLM response.")
        return None


class ResumeRetriever:
    def __init__(self):
        self.azure_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
        self.azure_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")

        self.gemini_key = os.getenv("GEMINI_API_KEY")
        self.gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

        self.collection = MongoClient(MONGO_URI)[DB_NAME][COLLECTION_NAME]

    async def call_azure_llm(self, prompt: str) -> str:
        headers = {
            "Content-Type": "application/json",
            "api-key": self.azure_key,
        }
        body = {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant that extracts keywords from natural language queries for robust MongoDB document search."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.2,
            "max_tokens": 6000,
        }
        url = f"{self.azure_endpoint}/openai/deployments/{self.azure_deployment}/chat/completions?api-version={self.azure_version}"

        async with httpx.AsyncClient(timeout=60) as client:
            print("🌐 Sending request to Azure...")
            response = await client.post(url, headers=headers, json=body)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]

    async def call_gemini_llm(self, prompt: str) -> str:
        if not self.gemini_key:
            raise ValueError("❌ GEMINI_API_KEY not set in .env")

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.gemini_model}:generateContent?key={self.gemini_key}"
        payload = {"contents": [{"parts": [{"text": prompt}]}]}
        headers = {"Content-Type": "application/json"}

        async with httpx.AsyncClient(timeout=60) as client:
            print("🌐 Sending request to Gemini...")
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            return response.json()["candidates"][0]["content"]["parts"][0]["text"]

    def clean_llm_response(self, text: str) -> str:
        cleaned = text.strip()
        if cleaned.startswith("```json") and cleaned.endswith("```"):
            return cleaned[7:-3].strip()
        elif cleaned.startswith("```") and cleaned.endswith("```"):
            return cleaned[3:-3].strip()
        return cleaned

    async def extract_keywords(self, query: str, use_gemini=False) -> list[str]:
        print("\n🧠 Extracting enriched keywords...")

        prompt = f"""
You are an intelligent keyword extraction and enrichment engine for a resume retrieval system using MongoDB.

Your job is to extract the most relevant keywords or phrases from a natural language HR query. These keywords will be used to build $regex-based MongoDB queries to find matching resumes from a structured NoSQL database.

--- Context ---
Each resume is a JSON object with fields like:
- name, summary, skills, experience (title, company, description), education, certifications, projects, etc.

--- Your Objective ---
From the query, extract:
✓ Core technical and non-technical keywords
✓ Job titles, skills, tools, responsibilities, locations (e.g., "team lead", "api testing", "onboarding process")
✓ Phrases describing HR actions or managerial work (e.g., "managed payroll", "conducted training")
✓ Domain-specific or industry terms (e.g., "e-commerce", "finance")
✓ Degrees, certifications (e.g., "MBA", "PMP certified")
✓ Named candidates (e.g., "rahul sharma") along with other relevant keywords
✓ If the query is explicitly asking about info of some particular candidate only give keywords of their names.

--- Enrichment Rules ---
✓ For each keyword/phrase, also include:
  - Related terms that people usually put in their resumes instead of the exact keywords extracted (e.g., "REST" → "rest api", "restful api")
  - Decomposed tokens for multi-word phrases (e.g., "rahul sharma" → "rahul", "sharma")
✓ Maintain all original keywords too.
✓ try to break the keywords into meanigful singular words as well that can help in better keyword search.

--- What to Avoid ---
✗ Don't include standalone generic verbs like "implemented", "built", "worked" in case of technical skills related query if required you can keep them for non tech queries
✓ Only keep such verbs if part of a meaningful phrase like "implemented payroll system"
✗ Don't include vague or overfit terms like "tools", "project", "experience", unless explicitly needed

--- Output Format ---
Return only a **valid JSON array** of strings. Example:
["api testing", "rest api", "payroll", "rahul sharma", "rahul", "sharma"]

--- Query ---
{query}
"""

        try:
            raw = await (self.call_gemini_llm(prompt) if use_gemini else self.call_azure_llm(prompt))
            print("📝 Raw LLM Response:", raw)
            cleaned = self.clean_llm_response(raw)
            print("✅ Cleaned Response:", cleaned)
            keywords = json.loads(cleaned) if cleaned.startswith("[") else []
            if not isinstance(keywords, list):
                print("⚠️ Keyword list is not a valid JSON array.")
                return []
            keywords = list(set(k.strip().lower() for k in keywords if isinstance(k, str) and k.strip()))
            log("🎯 Final Keywords: " + json.dumps(keywords, indent=2))
            return keywords
        except Exception as e:
            print(f"❌ Keyword extraction failed: {e}")
            return []



    def build_mongo_query(self, keywords: list[str]) -> dict:
        print("\n🔧 Building MongoDB OR query...")
        or_conditions = []
        fields = [
            "name","summary", "skills", "projects.description", "projects.title",
            "experience.description", "experience.title", "experience.company",
            "education.institution", "certifications.title", "certifications.issuer",
        ]

        for kw in keywords:
            for field in fields:
                or_conditions.append({field: {"$regex": kw, "$options": "i"}})

        query = {"$or": or_conditions} if or_conditions else {}
        # print("📄 Final Mongo Query:\n", json.dumps(query, indent=2))
        return query

    async def search(self, query: str, use_gemini: bool = False) -> dict:
        print(f"\n🔍 Original NL Query: \"{query}\"")
        keywords = await self.extract_keywords(query, use_gemini)
        if not keywords:
            return {
                "query": query,
                "keywords": [],
                "matched": [],
                "error": "Keyword extraction failed"
            }

        # mongo_query = self.build_mongo_query(keywords)
        # print("\n🔎 Searching in MongoDB...")
        # results = list(self.collection.find(mongo_query))
        pipeline = bm25_pipeline(keywords, k=300)
        print("\n🔎 Searching in MongoDB with BM25 pipeline...")
        results  = list(self.collection.aggregate(pipeline))
        
        print(f"\n📊 Found {len(results)} matching resumes:\n")
        for res in results:
            print(f" - {res.get('name')} | {res.get('email')}")

        return {
            "query": query,
            "keywords": keywords,
            "matched": [
               {**res, "_id": str(res["_id"])} for res in results
            ]
        }
    def log_query_result(self, query: str, keywords: list[str], matched: list[dict], log_path: str):
        log_data = {
            "query": query,
            "keywords": keywords,
            "matched_names": [res.get("name", "Not Provided") for res in matched],
        }
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_data, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True, help="Natural language resume query")
    parser.add_argument("--use-gemini", action="store_true", help="Use Gemini instead of Azure")
    args = parser.parse_args()

    retriever = ResumeRetriever()
    result = asyncio.run(retriever.search(args.query, use_gemini=args.use_gemini))

    # import pprint
    # pprint.pprint(result)

    log_path = "data/nosql_retrieval_logs.jsonl"
    retriever.log_query_result(
        query=result["query"],
        keywords=result["keywords"],
        matched=result["matched"],
        log_path=log_path
    )

