
import asyncio
import os
os.environ["PIPELINE_MODE"] = "true" 
import json
from src.core.retriever import ResumeRetriever
from src.core.generator import AnswerGenerator

LOG_FILE = "logs/nosql_pipeline.log"
os.makedirs("logs", exist_ok=True)

def log_pipeline(msg: str):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")

class NoSQLQueryPipeline:
    def __init__(self, use_gemini: bool = False):
        self.use_gemini = use_gemini
        self.retriever = ResumeRetriever()
        self.generator = AnswerGenerator(use_gemini=use_gemini)

    async def run(self, query: str) -> dict:
        log_pipeline(f"🔍 Step 1: Starting retrieval for query: {query}")
        result = await self.retriever.search(query, use_gemini=self.use_gemini)
        resumes = result.get("matched", [])

        if not resumes:
            log_pipeline("❌ Step 1 Failed: No resumes retrieved.")
            return {
                "query": query,
                "keywords": result.get("keywords", []),
                "reranked_resumes": [],
                "answer": "No resumes retrieved for the given query."
            }

        log_pipeline(f"✅ Step 1 Complete: Retrieved {len(resumes)} resumes")

        log_pipeline("📌 Step 2: Starting reranking...")
        reranked = await self.generator.rerank_resumes(query, resumes)
        log_pipeline(f"✅ Step 2 Complete: {len(reranked)} resumes marked relevant after reranking.")

        log_pipeline("✏️ Step 3: Generating final answer...")
        answer = await self.generator.generate_answer(query, reranked)
        log_pipeline("✅ Step 3 Complete: Final answer generated.")

        return {
            "query": query,
            "keywords": result.get("keywords", []),
            "reranked_resumes": reranked,
            "answer": answer
        }

# CLI Testing
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True, help="Natural language query")
    parser.add_argument("--use-gemini", action="store_true", help="Use Gemini instead of Azure")
    args = parser.parse_args()

    pipeline = NoSQLQueryPipeline(use_gemini=args.use_gemini)
    output = asyncio.run(pipeline.run(args.query))
    print("\n================ FINAL ANSWER ================\n")
    print(output)
