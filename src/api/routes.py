from fastapi import FastAPI, Request
from pydantic import BaseModel
from src.core.pipeline import NoSQLQueryPipeline

app = FastAPI()

class QueryRequest(BaseModel):
    query: str
    use_gemini: bool = False

@app.post("/query")
async def run_query(request: QueryRequest):
    pipeline = NoSQLQueryPipeline(use_gemini=request.use_gemini)
    result = await pipeline.run(request.query)
    if "error" in result:
        return {"error": result["error"], "keywords": result.get("keywords", [])}
    return {
        "query": request.query,
        "answer": result["answer"],
        "relevant_resumes": result["reranked_resumes"]
    }
