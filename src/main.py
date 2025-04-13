import asyncio
from src.core.pipeline import NoSQLQueryPipeline

if __name__ == "__main__":
    query = input("🔍 Enter your HR query: ").strip()
    use_gemini = False  # Set to True if you want Gemini

    async def run():
        pipeline = NoSQLQueryPipeline(use_gemini=use_gemini)
        result = await pipeline.run(query)

        print("\n✅ FINAL ANSWER:\n")
        print(result["answer"])

        print("\n📋 RELEVANT CANDIDATES:")
        for resume in result["reranked_resumes"]:
            print(f"- {resume.get('name')} | {resume.get('email')}")

    asyncio.run(run())
