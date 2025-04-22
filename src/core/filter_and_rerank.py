import os
import json
import asyncio
import httpx
from typing import List, Dict
from dotenv import load_dotenv
load_dotenv()

class FilterAndRerank:
    def __init__(self):
        self.api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
        self.api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")

    async def _call_llm(self, query: str, resumes: List[Dict]) -> str:
        url = f"{self.endpoint}/openai/deployments/{self.deployment}/chat/completions?api-version={self.api_version}"
        headers = {
            "Content-Type": "application/json",
            "api-key": self.api_key,
        }

        system_message = {
            "role": "system",
            "content": "You are a helpful assistant that filters resumes relevant to a search query."
        }

        user_message = {
            "role": "user",
            "content": (
                f"Given the following query:\n\n{query}\n\n"
                "And the following resumes:\n"
                f"{json.dumps(resumes, indent=2)}\n\n"
                "Instructions:\n"
                "1. Classify resumes as 'relevant' or 'irrelevant' strictly based on whether they should appear in search results for the query.\n"
                "2. Only include resumes in 'relevant' if they clearly or partially satisfy the query.\n"
                "3. Do not infer beyond the text.\n\n"
                "Return a JSON like this:\n"
                "{\n"
                "  \"relevant\": [\"Name1\", \"Name2\"],\n"
                "  \"irrelevant\": [\"Name3\", \"Name4\"],\n"
                "  \"reasons\": {\n"
                "    \"Name1\": \"Worked on CRM integration\",\n"
                "    \"Name3\": \"No mention of relevant experience\"\n"
                "  }\n"
                "}"
            )
        }

        body = {
            "messages": [system_message, user_message],
            "temperature": 0.2,
            "max_tokens": 6000
        }

        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(url, headers=headers, json=body)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]

    async def filter_and_rerank(self, query: str, resumes: List[Dict], top_k: int = None) -> Dict:
        try:
            response_str = await self._call_llm(query, resumes)
            response = json.loads(response_str)

            relevant_names = set(response.get("relevant", []))
            reasons = response.get("reasons", {})

            filtered = []
            for resume in resumes:
                name = resume.get("name", "").strip()
                if name in relevant_names:
                    resume["filter_reason"] = reasons.get(name, "")
                    filtered.append(resume)

            print(f"✅ Filtered {len(filtered)} relevant resumes out of {len(resumes)} total")

            return {
                "query": query,
                "filtered": filtered,
                "original_count": len(resumes),
                "filtered_count": len(filtered)
            }

        except Exception as e:
            print(f"❌ Filter failed: {e}")
            return {
                "query": query,
                "filtered": resumes[:top_k] if top_k else resumes,
                "original_count": len(resumes),
                "filtered_count": len(resumes[:top_k]) if top_k else len(resumes),
                "error": str(e)
            }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run LLM-based resume filtering for a query")
    parser.add_argument("--query", required=True, help="Search query")
    parser.add_argument("--input", required=True, help="Path to input JSON of resumes")
    parser.add_argument("--top_k", type=int, default=None)

    args = parser.parse_args()

    async def main():
        with open(args.input, "r", encoding="utf-8") as f:
            resumes = json.load(f)

        reranker = FilterAndRerank()
        result = await reranker.filter_and_rerank(args.query, resumes, top_k=args.top_k)
        print(json.dumps(result, indent=2, ensure_ascii=False))

    asyncio.run(main())
