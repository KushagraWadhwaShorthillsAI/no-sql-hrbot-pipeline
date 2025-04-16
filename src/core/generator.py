import os
import json
import argparse
import httpx
import asyncio
from dotenv import load_dotenv
from .retriever import ResumeRetriever

load_dotenv()

PIPELINE_MODE = os.getenv("PIPELINE_MODE", "false").lower() == "true"
LOG_FILE = "logs/nosql_pipeline.log" if PIPELINE_MODE else "logs/nosql_generator.log"

os.makedirs("logs", exist_ok=True)

def log(msg):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")

class AnswerGenerator:
    def __init__(self, use_gemini: bool = False):
        self.use_gemini = use_gemini

        self.azure_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
        self.azure_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")

        self.gemini_key = os.getenv("GEMINI_API_KEY")
        self.gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

    async def _call_azure(self, messages):
        url = f"{self.azure_endpoint}/openai/deployments/{self.azure_deployment}/chat/completions?api-version={self.azure_version}"
        headers = {"Content-Type": "application/json", "api-key": self.azure_key}
        body = {"messages": messages, "temperature": 0.3, "max_tokens": 1000}

        async with httpx.AsyncClient(timeout=60) as client:
            log("🌐 Sending to Azure...")
            response = await client.post(url, headers=headers, json=body)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]

    async def _call_gemini(self, prompt):
        if not self.gemini_key:
            raise ValueError("❌ GEMINI_API_KEY not set in .env")

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.gemini_model}:generateContent?key={self.gemini_key}"
        payload = {"contents": [{"parts": [{"text": prompt}]}]}
        headers = {"Content-Type": "application/json"}

        async with httpx.AsyncClient(timeout=60) as client:
            log("🌐 Sending to Gemini...")
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

    async def rerank_resumes(self, query: str, resumes: list[dict]) -> list[dict]:
        input_text = "\n\n".join(
            [f"Resume {i+1}:\n" + json.dumps(res, indent=2) for i, res in enumerate(resumes)]
        )
        prompt = f"""
You are an intelligent assistant designed to help HR professionals make sense of multiple candidate resumes based on a specific hiring query.

You are provided with:
1. A **natural language query** that an HR professional might ask (e.g., "List candidates who have worked with Python and C++", or "Find people with managerial experience in the education sector").
2. A list of **resumes (in JSON format)** that were already filtered and retrieved from MongoDB based on key terms extracted from the query. These resumes are the most likely matches to the HR query.

<query>
{query}
</query>

<resumes>
{input_text}
</resumes>

Your task is to:
- Rigorously evaluate each resume to determine if it directly or strongly supports answering the query.
- Perform intelligent filtering and re-ranking of resumes based on their relevance and closeness to the query intent.
- You must evaluate all resumes rigorously based on actual content (roles, experience, descriptions) and match them to the query intent.
- Relevancy is not compromisable — do not mistakenly classify a relevant resume as irrelevant.
- Be strict when assigning to "irrelevant_names", but lenient when considering resumes for "relevant_names" — if in doubt, default to relevance.
- Order the "relevant_names" list in descending order of their relevance to the query, from most suitable to least.
- Avoid hallucinations: base your decisions **strictly on the resume content**.

Guidelines:
- If the query is simple and broad (e.g., "List all Python developers"), then **most resumes might be relevant** and re-rank based on relevancy to the query. In such cases, list them all under "relevant_names" with minimal or no "irrelevant_names".
- If the query includes multiple filters (e.g., "Python devs with knowledge of C++ and ML frameworks"), you must find resumes that satisfy **all conditions**, and consider others as less relevant or irrelevant.
- For ambiguous or subjective queries (e.g., "candidates showing a clear career shift"), use your best analytical reasoning from the resume timeline, titles, and industries.
- If you're unsure about a resume, **default to keeping it relevant** rather than wrongly discarding it.

Return your answer in the following JSON format ONLY:
{{
  "relevant_names": ["Full Name 1", "Full Name 2"],
  "irrelevant_names": ["Full Name 3", "Full Name 4"]
}}

Remember:
- Do not explain.
- Do not return anything outside of the JSON block.
- All resumes provided must be categorized into one of the two lists. The total count of names in both lists **must equal** the total number of resumes provided.
- The two lists must be mutually exclusive and only include candidate "name" fields.


"""

        try:
            response_raw = await (
                self._call_gemini(prompt) if self.use_gemini else self._call_azure([{"role": "user", "content": prompt}])
            )
            log("🧾 Raw reranker response: " + response_raw)
            response_clean = self.clean_llm_response(response_raw)
            log("🧹 Cleaned reranker response: " + response_clean)

            parsed = json.loads(response_clean)
            relevant_names = parsed.get("relevant_names", [])
            irrelevant_names = parsed.get("irrelevant_names", [])

            relevant = [r for r in resumes if r["name"] in relevant_names]

            log("🔍 Resumes sent to reranker: " + str([r["name"] for r in resumes]))
            log("✅ Relevant resumes after rerank: " + str(relevant_names))
            log("❌ Irrelevant resumes after rerank: " + str(irrelevant_names))

            return relevant
        except Exception as e:
            log(f"⚠️ Reranking failed: {e}")
            return resumes

    async def generate_answer(self, query: str, reranked_resumes: list[dict]) -> str:
        content_summary = "\n\n".join([
            f"Resume {i+1}:\n" + json.dumps(res, indent=2)
            for i, res in enumerate(reranked_resumes)
        ])

        prompt = f"""
--- Role ---
You are an intelligent and precise assistant that helps HR professionals generate responses from structured resume data. You are not responsible for filtering or reranking — all resumes you are given have already been retrieved and prioritized using earlier stages.

--- Objective ---
Your task is to generate a clear, complete, and helpful answer to the user's query using **only the provided resume JSONs**. The response should be tailored to the query’s intent — whether it's to list candidates, summarize experiences, identify skillsets, or extract work patterns.

--STRICT GUIDELINES--
- You must rigorously assess each resume and include only those that clearly and explicitly satisfy the query requirements. Avoid vague inclusions — be precise and strict with your matching logic.

--- Resume Format ---
You will receive top-ranked resumes as structured JSON objects. Each resume may contain:
- name, email, phone, location
- summary, skills, education (degree, institution, year)
- experience (title, company, duration, location, description)
- certifications (title, issuer, year)
- projects (title, description, link)
- languages, social profiles

--- Query ---
{query}


--- Top-Ranked Resumes ---
{content_summary}

--- Response Rules ---
- ✅ Your answer must be fully grounded in the information from the provided resumes.
- ✅ Ensure all resumes are rigorously considered and no relevant profiles are skipped. Include every candidate that satisfies the query conditions.
- ✅ Mention candidate **full names** when referring to individual profiles.
- ✅ Use markdown formatting (headings, bullet lists, tables) **only if it improves clarity** and is appropriate for the query.
- ✅ Summarize only what is asked — e.g., tools used, certifications held, companies worked at, transitions made, etc.
- ✅ Redact personal details like phone numbers or emails.
- ✅ Be professional, direct, and accurate in tone.

- ❌ Do NOT infer or hallucinate skills, roles, or experience that are not in the resume.
- ❌ Do NOT speculate on candidate preferences, intentions, or strengths unless explicitly stated.
- ❌ Do NOT assume answers beyond what's available in the resume fields.
- ❌ Do NOT re-rank, discard, or comment on the relevance of resumes — that has already been handled.
- ❌ Do NOT list a candidate unless at least one specific field (e.g., experience, summary, skills, certifications) provides concrete evidence for inclusion.

--- Output ---
Respond in a helpful, factual tone. Use complete sentences or structured markdown formatting only when needed. Do not provide explanations about how the data was processed — just give the final answer based on resumes.

""".strip()


        messages = [
            {"role": "system", "content": "You are a helpful assistant that summarizes relevant candidate information based on a natural language query."},
            {"role": "user", "content": prompt}
        ]

        try:
            response = await (
                self._call_gemini(messages[-1]["content"]) if self.use_gemini else self._call_azure(messages)
            )
            log("📝 Final generated answer:\n" + response)
            return response
        except Exception as e:
            log(f"⚠️ Generation failed: {e}")
            return f"⚠️ Generation failed: {e}"

# === CLI TESTING ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True, help="Natural language query")
    parser.add_argument("--use-gemini", action="store_true", help="Use Gemini instead of Azure")
    args = parser.parse_args()

    async def run():
        print("🔍 Stage 1: Retrieving resumes...")
        retriever = ResumeRetriever()
        retrieval_result = await retriever.search(args.query, use_gemini=args.use_gemini)
        resumes = retrieval_result.get("matched", [])

        if not resumes:
            print("❌ No resumes retrieved. Aborting generation.")
            log("❌ No resumes retrieved. Aborting generation.")
            return

        print("✅ Stage 1 Complete: Resumes retrieved.")
        print("📊 Number of resumes retrieved:", len(resumes))

        print("\n📌 Stage 2: Reranking resumes...")
        generator = AnswerGenerator(use_gemini=args.use_gemini)
        reranked = await generator.rerank_resumes(args.query, resumes)

        print("✅ Stage 2 Complete: Resumes reranked.")
        print("📊 Number of resumes after rerank:", len(reranked))

        print("\n✏️ Stage 3: Generating final answer...")
        summary = await generator.generate_answer(args.query, reranked)

        print("✅ Stage 3 Complete: Final answer generated.")
        print("\n📝 FINAL GENERATED ANSWER:\n")
        print(summary)

    asyncio.run(run())
