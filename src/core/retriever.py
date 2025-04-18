import os
import json
import argparse
import asyncio
import httpx
from pymongo import MongoClient
from .config import MONGO_URI, DB_NAME, COLLECTION_NAME
from dotenv import load_dotenv
from src.core.search_builder import bm25_pipeline
from collections import defaultdict

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

def apply_rrf(bm25_hits, vector_hits, k_rrf=60):
    """
    Applies Reciprocal Rank Fusion over two ranked lists (BM25 + Vector).
    Returns a unified list of resumes sorted by fused RRF score.
    """
    rank_scores = {}

    def score_doc(doc, rank, weight=1.0):
        _id = str(doc.get("_id", doc.get("email", doc.get("name", ""))))
        rank_scores.setdefault(_id, {
            "doc": doc,
            "rrf_score": 0,
            "sources": [],
        })
        rank_scores[_id]["rrf_score"] += weight / (k_rrf + rank)
        rank_scores[_id]["sources"].append(rank)

    for rank, doc in enumerate(bm25_hits):
        score_doc(doc, rank, weight=1.0)

    for rank, doc in enumerate(vector_hits):
        score_doc(doc, rank, weight=1.0)

    fused = list(rank_scores.values())
    fused.sort(key=lambda x: x["rrf_score"], reverse=True)

    return [
        {**entry["doc"], "rrf_score": entry["rrf_score"]}
        for entry in fused
    ]



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
["api testing", "rest api", "payroll", "rahul sharma", "rahul"]

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
            "name",
            "location",
            "summary",
            "skills",
            "experience.title",
            "experience.description",
            "experience.company",
            "projects.title",
            "projects.description",
            "education.degree",
            "education.institution",
            "education.year",
            "certifications.title",
            "certifications.issuer",
            # "languages",
            # "social_profiles"
        ]

        for kw in keywords:
            for field in fields:
                or_conditions.append({field: {"$regex": kw, "$options": "i"}})

        query = {"$or": or_conditions} if or_conditions else {}
        # print("📄 Final Mongo Query:\n", json.dumps(query, indent=2))
        return query

    async def search(self, query: str, override_keywords: list[str] = None, use_gemini: bool = False) -> dict:
        print(f"\n🔍 Original NL Query: in search \"{query}\"")

        if override_keywords:
            keywords = override_keywords
            print(f"📌 Using manual keywords: {keywords}")
        else:
            keywords = await self.extract_keywords(query, use_gemini)

        if not keywords:
            return {
                "query": query,
                "keywords": [],
                "matched": [],
                "error": "Keyword extraction failed"
            }

        pipeline = bm25_pipeline(keywords, top_k=10)
        print("\n🔎 Searching in MongoDB with BM25 pipeline...")
        results = list(self.collection.aggregate(pipeline))

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

    
    
    async def embed_query(self, query: str) -> list[float]:
        """
        Embed the query using Azure OpenAI embedding API.
        """
        url = f"{self.azure_endpoint}/openai/deployments/{os.getenv('AZURE_EMBEDDING_DEPLOYMENT')}/embeddings?api-version={os.getenv('AZURE_EMBEDDING_API_VERSION')}"
        headers = {"Content-Type": "application/json", "api-key": self.azure_key}
        body = {"input": query}
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(url, headers=headers, json=body)
            r.raise_for_status()
            return r.json()["data"][0]["embedding"]

    async def vector_search(self, query: str, k: int = 100) -> list[dict]:
        """
        Runs per-field knnBeta vector search and fuses results (via max score).
        """
        print(f"\n🧠 Embedding query for vector search: '{query}'")
        qvec = await self.embed_query(query)

        fields = [
            "summary_embed",
            "skills_embed",
            "projects_embed",
            "experience_embed",
            "certifications_embed",
            "education_embed"
        ]

        results_by_id = defaultdict(lambda: {"score_vec": 0.0, "sources": set()})

        for field in fields:
            print(f"🔍 Searching field: {field}")
            pipeline = [
                {
                    "$search": {
                        "index": "default_vector",
                        "knnBeta": {
                            "vector": qvec,
                            "path": field,
                            "k": k
                        }
                    }
                },
                { "$project": {
                    "_id": 1,
                    "name": 1,
                    "email": 1,
                    "summary": 1,
                    "score_vec": { "$meta": "searchScore" }
                }}
            ]

            for doc in self.collection.aggregate(pipeline):
                _id = str(doc["_id"])
                if doc["score_vec"] > results_by_id[_id]["score_vec"]:
                    results_by_id[_id].update({
                        "doc": doc,
                        "score_vec": doc["score_vec"],
                    })
                results_by_id[_id]["sources"].add(field)

        print(f"✅ Fused results from {len(fields)} fields. Found {len(results_by_id)} unique resumes.")

        # Convert and sort
        fused_results = [
            {**v["doc"], "score_vec": v["score_vec"], "matched_fields": list(v["sources"])}
            for v in results_by_id.values()
        ]
        fused_results.sort(key=lambda r: r["score_vec"], reverse=True)

        return fused_results
    
    async def hybrid_search(self, query: str, bm25_override: list[str] = None, top_n_each=200) -> dict:
        print(f"\n🤝 Running hybrid search for: '{query}'")

        # Step 1: BM25
        if bm25_override:
            keywords = bm25_override
            print(f"📌 Using manual BM25 keywords: {keywords}")
        else:
            keywords = await self.extract_keywords(query)
            print(f"🔍 Extracted BM25 keywords: {keywords}")

        pipeline = bm25_pipeline(keywords, k=top_n_each)
        print("\n🔎 Searching in MongoDB with BM25 pipeline...")
        bm25_results = list(self.collection.aggregate(pipeline))
        print(f"📚 BM25 retrieved {len(bm25_results)} resumes")

        # Step 2: Vector (always use the full query for semantically rich embeddings)
        vector_results = await self.vector_search(query, k=top_n_each)
        print(f"🧠 Vector search returned {len(vector_results)} resumes")

        # Step 3: RRF Fusion
        final = apply_rrf(bm25_results, vector_results)
        print(f"✅ RRF fusion produced {len(final)} unified results")

        return {
            "query": query,
            "bm25": bm25_results,
            "vector": vector_results,
            "fused": final
        }
    def keyword_search(self, keywords: list[str], top_k: int = 200) -> list[dict]:
        print("\n🔍 Running simple keyword match (non-regex)...")

        fields = [
            "name", "location", "summary", "skills",
            "experience.title", "experience.description", "experience.company",
            "projects.title", "projects.description",
            "education.degree", "education.institution", "education.year",
            "certifications.title", "certifications.issuer"
        ]

        or_conditions = []
        for field in fields:
            for kw in keywords:
                or_conditions.append({field: {"$regex": kw, "$options": "i"}})

        query = {"$or": or_conditions}
        print(f"🛠 Built MongoDB $or query with {len(or_conditions)} conditions")
        return list(self.collection.find(query).limit(top_k))

    
    def bm25_search_from_keywords(self, keywords: list[str], top_k: int = 100) -> list[dict]:
        """
        Run BM25-style search directly from pre-extracted keywords.
        """
        from .search_builder import bm25_pipeline
        pipeline = [bm25_pipeline(keywords)]
        results = list(self.collection.aggregate(pipeline))
        return results[:top_k]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True, help="Natural language resume query")
    parser.add_argument("--use-gemini", action="store_true", help="Use Gemini instead of Azure")
    parser.add_argument("--mode", choices=["bm25", "vector", "hybrid"], default="bm25", help="Retrieval mode (bm25 | vector | hybrid)")
    args = parser.parse_args()

    retriever = ResumeRetriever()

    if args.mode == "bm25":
        result = asyncio.run(retriever.search(args.query, use_gemini=args.use_gemini))
        log_path = "data/nosql_retrieval_logs.jsonl"
        retriever.log_query_result(
            query=result["query"],
            keywords=result["keywords"],
            matched=result["matched"],
            log_path=log_path
        )
    elif args.mode == "hybrid":
        result = asyncio.run(retriever.hybrid_search(args.query))
        
        print("\n🔍 Top 10 from BM25:")
        for doc in result["bm25"][:10]:
            print(f" - {doc.get('name')} | via BM25")

        print("\n🔍 Top 10 from Vector:")
        for doc in result["vector"][:10]:
            print(f" - {doc.get('name')} | via Vector")

        print("\n🔝 Top 10 from RRF Fusion:")
        for doc in result["fused"][:10]:
            print(f" - {doc.get('name')} | RRF score = {doc.get('rrf_score'):.4f}")

    else:
        results = asyncio.run(retriever.vector_search(args.query))
        for r in results[:10]:
            print(f" - {r['name']} | vec_score = {r['score_vec']:.4f} | fields = {', '.join(r['matched_fields'])}")

